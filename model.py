"""
Medical LLM Model Architecture

Based on modded-nanogpt optimizations:
- Rotary Position Embeddings (RoPE)
- RMSNorm instead of LayerNorm
- QK-Norm for attention stability
- Deep architecture support (64-80 layers)
- Flash Attention support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    vocab_size: int = 100352  # tiktoken cl100k_base (100277) padded to multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # No bias in linear layers (modern practice)
    max_seq_length: int = 2048
    rope_base: float = 10000.0
    use_flash: bool = True
    qk_norm: bool = True  # QK normalization for stability


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_length: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_length)

    def _build_cache(self, seq_length: int):
        t = torch.arange(seq_length, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple:
        if seq_len > self.max_seq_length:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional QK-norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_flash = config.use_flash

        # QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Rotary embeddings
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_length, config.rope_base)

        # QK normalization for training stability
        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply QK normalization
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply rotary position embeddings
        cos, sin = self.rotary(q, T)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use Flash Attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale

            # Causal mask
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """Feed-forward network with SiLU activation (SwiGLU variant)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(config.n_embd * 8 / 3)  # SwiGLU uses 8/3 ratio
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # Round up to multiple of 64

        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MedicalLLM(nn.Module):
    """
    Medical Language Model.

    Architecture based on modded-nanogpt with modern improvements:
    - RoPE for position encoding
    - RMSNorm for normalization
    - SwiGLU activation
    - QK-Norm for stability
    - Optional deep architecture (64-80 layers)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (optional, saves parameters)
        # self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print(f"Model parameters: {self.get_num_params() / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (B, T)
            targets: Target token IDs for loss computation (B, T)

        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        B, T = input_ids.size()
        assert T <= self.config.max_seq_length, f"Sequence too long: {T} > {self.config.max_seq_length}"

        # Token embeddings
        x = self.transformer.wte(input_ids)
        x = self.transformer.drop(x)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final norm and LM head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_length else input_ids[:, -self.config.max_seq_length:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Preset configurations
def get_tiny_config() -> ModelConfig:
    """Tiny model for testing (~10M params)."""
    return ModelConfig(
        n_layer=6,
        n_head=6,
        n_embd=384,
        max_seq_length=512,
    )


def get_small_config() -> ModelConfig:
    """Small model (~85M params, like GPT-2 small)."""
    return ModelConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        max_seq_length=1024,
    )


def get_medium_config() -> ModelConfig:
    """Medium model (~350M params, like GPT-2 medium)."""
    return ModelConfig(
        n_layer=24,
        n_head=16,
        n_embd=1024,
        max_seq_length=2048,
    )


def get_baguette_config() -> ModelConfig:
    """Baguettotron-style deep model (~320M params, 80 layers)."""
    return ModelConfig(
        n_layer=80,
        n_head=8,
        n_embd=512,
        max_seq_length=2048,
    )


def get_large_config() -> ModelConfig:
    """Large model (~500M params for $20 budget)."""
    return ModelConfig(
        n_layer=36,
        n_head=16,
        n_embd=1280,
        max_seq_length=2048,
    )


if __name__ == "__main__":
    # Test model creation
    print("Testing model configurations...")

    for name, config_fn in [
        ("tiny", get_tiny_config),
        ("small", get_small_config),
        ("medium", get_medium_config),
        ("baguette", get_baguette_config),
    ]:
        config = config_fn()
        model = MedicalLLM(config)
        print(f"\n{name}: {model.get_num_params() / 1e6:.1f}M params")
        print(f"  Layers: {config.n_layer}, Heads: {config.n_head}, Dim: {config.n_embd}")

        # Test forward pass
        x = torch.randint(0, config.vocab_size, (2, 64))
        logits, loss = model(x, x)
        print(f"  Forward pass OK, logits shape: {logits.shape}")
