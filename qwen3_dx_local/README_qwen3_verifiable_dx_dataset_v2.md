# Qwen3 Verifiable Diagnosis Dataset — v2 (Synthetic, Diagnostic-Only)

This dataset is **synthetic** and **diagnostic-only**. It is designed for **fine-tuning** small chat LLMs
(e.g., Qwen3-0.6B) on *exam-style diagnostic classification*.

## Safety / scope
- This is for **educational** classification behavior, not real-world medical care.
- Outputs are constrained to **diagnosis labels or letters only**.
- Includes an explicit abstain label: **Insufficient information (need more data)** to reduce overconfident guessing.

## Files
- `qwen3_verifiable_dx_dataset_v2_train.jsonl`
- `qwen3_verifiable_dx_dataset_v2_val.jsonl`
- `qwen3_verifiable_dx_dataset_v2_test.jsonl`

## Task types
- `dx_mcq_letter_v2`: 5-option multiple choice; assistant outputs **ONLY** `A/B/C/D/E`.
- `dx_mcq_label_v2`: 5-option list; assistant outputs **ONLY** the exact diagnosis text from the list.
- `dx_free_label_v2`: no options; assistant outputs **ONLY** the diagnosis label.
- `dx_abstain_options_v2`: underspecified/conflicting vignette; options include the abstain label; assistant outputs abstain label.
- `dx_abstain_free_v2`: underspecified/conflicting vignette; assistant outputs abstain label.

## Schema (each JSONL line)
Each line is a JSON object with:
- `id`, `created`, `task_type`, `dx_label`
- `messages`: standard chat format with system/user/assistant

## Counts (generated on 2026-01-28)
{
  "dx_mcq_letter_v2": 62400,
  "dx_mcq_label_v2": 26000,
  "dx_free_label_v2": 15600,
  "dx_abstain_options_v2": 10070,
  "dx_abstain_free_v2": 9930
}

Split sizes:
{
  "train": 117750,
  "val": 3135,
  "test": 3115
}
