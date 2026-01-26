#!/usr/bin/env python3
"""
Simple chat CLI for Baguettotron (or a fine-tuned checkpoint).

Usage:
  python chat.py --model PleIAs/Baguettotron
  python chat.py --model ./medical_llm_output/final_model
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer, system_prompt, history, use_chat_template):
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": system_prompt}] + history
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback to a simple transcript
    lines = []
    if system_prompt:
        lines.append(f"System: {system_prompt}")
    for msg in history:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Chat with a medical LLM")
    parser.add_argument("--model", default="PleIAs/Baguettotron", help="Model id or local path")
    parser.add_argument("--system", default="You are a medical expert.", help="System prompt")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Use the tokenizer chat template if available")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    ).to(device)

    history = []

    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user_input})
        prompt = build_prompt(tokenizer, args.system, history, args.use_chat_template)
        inputs = tokenizer(prompt, return_tensors="pt")
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output[0][input_len:]
        assistant_reply = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Truncate if the model starts a new turn
        for marker in ["\nUser:", "\nSystem:", "\nAssistant:"]:
            if marker in assistant_reply:
                assistant_reply = assistant_reply.split(marker, 1)[0]
        assistant_reply = assistant_reply.strip()
        if not assistant_reply:
            full_text = tokenizer.decode(output[0], skip_special_tokens=True)
            assistant_reply = full_text.split("Assistant:", 1)[-1].strip()

        print(f"\nAssistant: {assistant_reply}\n")
        history.append({"role": "assistant", "content": assistant_reply})

    print("Bye.")


if __name__ == "__main__":
    main()
