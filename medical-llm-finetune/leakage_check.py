#!/usr/bin/env python3
"""
Check for overlap between training splits and benchmark/holdout splits.
"""

import argparse
import re
from typing import Iterable, List

from datasets import load_dataset


def _norm(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _limit(ds: Iterable, max_rows: int):
    if max_rows is None:
        for row in ds:
            yield row
    else:
        for i, row in enumerate(ds):
            if i >= max_rows:
                break
            yield row


def load_medqa(split: str):
    try:
        return load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    except Exception:
        return load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split=split)


def medqa_key(row) -> str:
    question = row.get("question", row.get("sent1", ""))
    if "options" in row:
        options = row["options"]
        if isinstance(options, dict):
            opts = [f"{k}:{options.get(k, '')}" for k in ["A", "B", "C", "D", "E"] if k in options]
        else:
            opts = [str(o) for o in options]
    else:
        opts = [row.get("ending0", ""), row.get("ending1", ""), row.get("ending2", ""), row.get("ending3", "")]
    return _norm(question + "||" + "||".join(opts))


def medmcqa_key(row) -> str:
    question = row.get("question", "")
    opts = [row.get("opa", ""), row.get("opb", ""), row.get("opc", ""), row.get("opd", "")]
    return _norm(question + "||" + "||".join(opts))


def load_pubmedqa_train():
    return load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")


def load_pubmedqa_test(prefer_bigbio: bool):
    if prefer_bigbio:
        try:
            return load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split="test")
        except Exception:
            pass
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    if "test" in ds:
        return ds["test"]
    if "validation" in ds:
        return ds["validation"]
    if "dev" in ds:
        return ds["dev"]
    raise ValueError("No test/validation split available for PubMedQA.")


def pubmedqa_key(row) -> str:
    question = row.get("question", row.get("QUESTION", ""))
    context = row.get("context", row.get("CONTEXTS", ""))
    if isinstance(context, dict):
        context = " ".join(context.get("contexts", []))
    if isinstance(context, list):
        context = " ".join(str(c) for c in context)
    return _norm(str(question) + "||" + str(context))


def check_overlap(name: str, train_keys: List[str], eval_keys: List[str], max_show: int = 5):
    train_set = set(train_keys)
    eval_set = set(eval_keys)
    overlap = train_set & eval_set
    print(f"{name}: train={len(train_set)} eval={len(eval_set)} overlap={len(overlap)}")
    if overlap:
        for i, item in enumerate(list(overlap)[:max_show]):
            print(f"  overlap[{i+1}]: {item[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Check for dataset leakage across splits")
    parser.add_argument("--max_train", type=int, default=None, help="Limit train rows (optional)")
    parser.add_argument("--max_eval", type=int, default=None, help="Limit eval rows (optional)")
    parser.add_argument("--prefer_bigbio_pubmedqa", action="store_true",
                        help="Prefer BigBio PubMedQA test split")
    args = parser.parse_args()

    # MedQA
    medqa_train = load_medqa("train")
    medqa_test = load_medqa("test")
    medqa_train_keys = [medqa_key(r) for r in _limit(medqa_train, args.max_train)]
    medqa_test_keys = [medqa_key(r) for r in _limit(medqa_test, args.max_eval)]
    check_overlap("MedQA", medqa_train_keys, medqa_test_keys)

    # MedMCQA
    medmcqa_train = load_dataset("openlifescienceai/medmcqa", split="train")
    medmcqa_val = load_dataset("openlifescienceai/medmcqa", split="validation")
    medmcqa_train_keys = [medmcqa_key(r) for r in _limit(medmcqa_train, args.max_train)]
    medmcqa_val_keys = [medmcqa_key(r) for r in _limit(medmcqa_val, args.max_eval)]
    check_overlap("MedMCQA", medmcqa_train_keys, medmcqa_val_keys)

    # PubMedQA
    pubmed_train = load_pubmedqa_train()
    try:
        pubmed_test = load_pubmedqa_test(prefer_bigbio=args.prefer_bigbio_pubmedqa)
    except Exception as e:
        print(f"PubMedQA: skipped ({e})")
        return

    pubmed_train_keys = [pubmedqa_key(r) for r in _limit(pubmed_train, args.max_train)]
    pubmed_test_keys = [pubmedqa_key(r) for r in _limit(pubmed_test, args.max_eval)]
    check_overlap("PubMedQA", pubmed_train_keys, pubmed_test_keys)


if __name__ == "__main__":
    main()
