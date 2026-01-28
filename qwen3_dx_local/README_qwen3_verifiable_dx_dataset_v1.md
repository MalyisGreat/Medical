# Qwen3 Verifiable Diagnosis Dataset (Synthetic) — v1

This is a **synthetic, exam-style** dataset intended for training/evaluating small chat LLMs on
**verifiable diagnostic classification** tasks (useful for RL with verifiable rewards such as GRPO).

## Safety note
This dataset is **NOT** for real-world medical diagnosis. It is designed for educational classification behavior.
If you deploy a model trained on it, keep clear disclaimers and strong safety/triage guardrails.

## Files
- `qwen3_verifiable_dx_dataset_v1_train.jsonl`
- `qwen3_verifiable_dx_dataset_v1_val.jsonl`
- `qwen3_verifiable_dx_dataset_v1_test.jsonl`
- `qwen3_verifiable_dx_dataset_v1.jsonl` (full)

## Task types
- `dx_mcq`: vignette + 5 options. Assistant outputs **ONLY** `A/B/C/D/E`.
- `dx_label`: vignette + allowed label list. Assistant outputs **ONLY** the label text.
- `dx_abstain_label`: intentionally underspecified vignette. Correct label is:
  `Insufficient information (need more data)`.
- `dx_triage_json`: assistant outputs single-line JSON: {"dx": "...", "triage": "..."}.
- `dx_triage_json_abstain`: underspecified vignette with JSON abstain output.

## JSONL schema
Each line is a JSON object like:
```json
{
  "id": "qwen3_dx_v1_000001",
  "created": "YYYY-MM-DD",
  "task_type": "dx_mcq",
  "dx_label": "Pneumonia",
  "triage": "URGENT",
  "choices": {"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."},
  "messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":"..."},
    {"role":"assistant","content":"B"}
  ]
}
```

## Why this is RL-friendly
- Rewards can be **exact match** on `A/B/C/D/E` or label text.
- JSON tasks are verifiable for both **parsing** and **field correctness**.
- Abstain tasks teach calibrated behavior: “don’t guess when info is missing.”

