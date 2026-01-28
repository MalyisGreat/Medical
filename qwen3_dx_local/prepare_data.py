#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as r, out_path.open("wb") as f:
        shutil.copyfileobj(r, f)

def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def make_messages(system: str, user: str, assistant: str):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]

SYSTEM_DX_MCQ = (
    "You are an exam-style medical vignette classifier for educational use only. "
    "Given a vignette and answer choices, output ONLY the single letter (A, B, C, D, or E) "
    "for the most likely diagnosis. Do not output any other text."
)
SYSTEM_DX_LABEL = (
    "You are an exam-style medical vignette classifier for educational use only. "
    "Given a vignette, output ONLY ONE diagnosis label from the provided allowed list. "
    "Do not output any other words."
)
SYSTEM_DX_JSON = (
    "You are an exam-style medical vignette classifier for educational use only. "
    "Output a single-line JSON object with exactly two keys: \"dx\" and \"triage\". "
    "\"dx\" must be one allowed diagnosis label. "
    "\"triage\" must be one of: EMERGENCY, URGENT, ROUTINE, SELF_CARE. "
    "Do not add any extra keys or commentary."
)

def normalize_infant_vignette(v: str) -> str:
    return re.sub(r"(\d+)-year-old infant", r"\1-month-old infant", v)

def build_conditions():
    # Condensed but still broad; you can expand easily by adding entries.
    conditions = [
        dict(label="Acute myocardial infarction (heart attack)", cat="cardio", triage="EMERGENCY",
             patterns=[
                 "A {age}-year-old {sex} has sudden crushing central chest pressure radiating to the left arm with sweating and nausea for {dur}.",
                 "A {age}-year-old {sex} with diabetes develops intense chest pressure with shortness of breath and diaphoresis lasting {dur}.",
             ]),
        dict(label="Stable angina", cat="cardio", triage="URGENT",
             patterns=[
                 "A {age}-year-old {sex} gets substernal chest pressure when climbing stairs that resolves within minutes of rest for the past {weeks} weeks.",
             ]),
        dict(label="Pericarditis", cat="cardio", triage="URGENT",
             patterns=[
                 "A {age}-year-old {sex} has sharp chest pain worse when lying flat and improves when sitting forward; started {dur} ago after a viral illness.",
             ]),
        dict(label="Aortic dissection", cat="cardio", triage="EMERGENCY",
             patterns=[
                 "A {age}-year-old {sex} has sudden severe tearing chest pain radiating to the back and feels faint.",
             ]),
        dict(label="Pulmonary embolism", cat="pulm", triage="EMERGENCY",
             patterns=[
                 "A {age}-year-old {sex} has sudden shortness of breath and pleuritic chest pain; recently had a long flight and has a swollen calf.",
             ]),
        dict(label="Pneumonia", cat="pulm", triage="URGENT",
             patterns=[
                 "A {age}-year-old {sex} has fever, productive cough, and shortness of breath for {days} days.",
             ]),
        dict(label="Spontaneous pneumothorax", cat="pulm", triage="EMERGENCY",
             patterns=[
                 "A {age}-year-old {sex} suddenly develops sharp one-sided chest pain and shortness of breath at rest; began minutes ago.",
             ]),
        dict(label="Ischemic stroke", cat="neuro", triage="EMERGENCY",
             patterns=[
                 "A {age}-year-old {sex} has sudden face droop and arm weakness on one side with slurred speech starting {mins} minutes ago.",
             ]),
        dict(label="Bell's palsy", cat="neuro", triage="URGENT",
             patterns=[
                 "A {age}-year-old {sex} has sudden facial weakness including the forehead on one side; no arm/leg weakness.",
             ]),
        dict(label="Subarachnoid hemorrhage", cat="neuro", triage="EMERGENCY",
             patterns=[
                 "A {age}-year-old {sex} has a sudden 'worst headache of life' with neck stiffness and vomiting.",
             ]),
        dict(label="Appendicitis", cat="gi", triage="URGENT",
             patterns=[
                 "A {age}-year-old {sex} has abdominal pain that started near the belly button and moved to the right lower abdomen with fever and loss of appetite.",
             ]),
        dict(label="Cholecystitis", cat="gi", triage="URGENT",
             patterns=[
                 "A {age}-year-old {sex} has right-upper abdominal pain after a fatty meal with fever and nausea; pain lasts hours.",
             ]),
        dict(label="Kidney stone (ureterolithiasis)", cat="renal", triage="URGENT",
             patterns=[
                 "A {age}-year-old {sex} has sudden severe flank pain radiating to the groin with nausea and blood in the urine.",
             ]),
        dict(label="Cystitis (bladder UTI)", cat="renal", triage="ROUTINE",
             patterns=[
                 "A {age}-year-old {sex} has burning with urination and urinary frequency for {days} days; no fever or flank pain.",
             ]),
        dict(label="Pyelonephritis (kidney infection)", cat="renal", triage="URGENT",
             patterns=[
                 "A {age}-year-old {sex} has fever, chills, flank pain, and painful urination.",
             ]),
        dict(label="Ectopic pregnancy", cat="obgyn", triage="EMERGENCY",
             patterns=[
                 "A {age}-year-old woman with a missed period has one-sided pelvic pain and spotting; feels dizzy.",
             ]),
        dict(label="Diabetic ketoacidosis (DKA)", cat="endo", triage="EMERGENCY",
             patterns=[
                 "A {age}-year-old {sex} with diabetes has vomiting, abdominal pain, deep rapid breathing, and fruity breath.",
             ]),
        dict(label="Hypoglycemia", cat="endo", triage="EMERGENCY",
             patterns=[
                 "A {age}-year-old {sex} with diabetes is sweaty, shaky, and confused after skipping a meal; improves after sugar.",
             ]),
        dict(label="Shingles (herpes zoster)", cat="derm", triage="ROUTINE",
             patterns=[
                 "A {age}-year-old {sex} has burning pain followed by a blistering rash in a stripe on one side of the torso.",
             ]),
        dict(label="Cellulitis", cat="derm", triage="URGENT",
             patterns=[
                 "A {age}-year-old {sex} has a warm, red, tender patch on the lower leg that is spreading with fever.",
             ]),
    ]
    # Simple pediatric add-ons
    conditions += [
        dict(label="Croup", cat="peds", triage="URGENT",
             patterns=["A {age}-year-old child has a barking cough and stridor worse at night after a cold."]),
        dict(label="Acute otitis media", cat="peds", triage="ROUTINE",
             patterns=["A {age}-year-old child has ear pain and fever after a cold; pulling at the ear."]),
    ]
    return conditions

def generate_vignette(cond, rng: random.Random) -> str:
    t = rng.choice(cond["patterns"])
    if "infant" in t:
        age = rng.choice([2,3,4,6,8,10,12])
        sex = "infant"
    elif "child" in t:
        age = rng.choice([2,3,4,5,6,7])
        sex = "child"
    else:
        age = rng.choice([18,22,27,32,38,45,52,60,68,75])
        sex = rng.choice(["man","woman","person"])

    v = t.format(
        age=age,
        sex=sex,
        dur=rng.choice(["20 minutes","1 hour","3 hours","12 hours"]),
        days=rng.choice([1,2,3,4,5,7,10]),
        weeks=rng.choice([2,3,4,6,8]),
        mins=rng.choice([10,20,30,45,60,90]),
    )
    v = normalize_infant_vignette(v)
    return v

def make_mcq_example(vignette: str, correct_label: str, labels_all: list[str], rng: random.Random):
    # Pick distractors
    distractors = [l for l in labels_all if l != correct_label]
    rng.shuffle(distractors)
    options = distractors[:4] + [correct_label]
    rng.shuffle(options)

    letters = ["A","B","C","D","E"]
    letter_map = {letters[i]: options[i] for i in range(5)}
    correct_letter = next(k for k,v in letter_map.items() if v == correct_label)

    prompt = (
        "Choose the single most likely diagnosis.\n\n"
        f"Vignette: {vignette}\n\nOptions:\n" +
        "\n".join([f"{L}. {letter_map[L]}" for L in letters]) +
        "\n\nAnswer with ONE letter only."
    )
    return prompt, correct_letter, letter_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data", help="Where to write dataset files")
    ap.add_argument("--download_url", type=str, default="", help="Optional: URL to a zip containing train/val/test jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_per_condition", type=int, default=240)
    ap.add_argument("--train_frac", type=float, default=0.95)
    ap.add_argument("--val_frac", type=float, default=0.025)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "qwen3_verifiable_dx_dataset_v1_train.jsonl"
    val_path   = out_dir / "qwen3_verifiable_dx_dataset_v1_val.jsonl"
    test_path  = out_dir / "qwen3_verifiable_dx_dataset_v1_test.jsonl"

    # If user provided a URL, download + extract.
    if args.download_url.strip():
        zip_path = out_dir / "dx_dataset.zip"
        print(f"[prepare_data] Downloading dataset zip -> {zip_path}")
        download_file(args.download_url.strip(), zip_path)
        print(f"[prepare_data] Extracting -> {out_dir}")
        extract_zip(zip_path, out_dir)
        print("[prepare_data] Done.")
        return

    # Otherwise, generate locally.
    rng = random.Random(args.seed)
    conditions = build_conditions()
    labels_all = [c["label"] for c in conditions]

    allowed_list_str = "\n".join([f"- {l}" for l in labels_all])
    abstain_label = "Insufficient information (need more data)"
    allowed_with_abstain = allowed_list_str + "\n- " + abstain_label

    ambiguous_vignettes = [
        "A {age}-year-old {sex} has chest discomfort that comes and goes. No other details are available.",
        "A {age}-year-old {sex} has abdominal pain. Location and associated symptoms are unknown.",
        "A {age}-year-old {sex} has dizziness. No vitals or timing are provided.",
        "A {age}-year-old {sex} has a headache. Onset and neurologic symptoms are not described.",
        "A {age}-year-old {sex} has shortness of breath. No trigger, fever, or exam findings are given.",
    ]

    records = []
    today = dt.date.today().isoformat()

    # dx_mcq (verifiable A-E)
    for cond in conditions:
        for _ in range(args.n_per_condition):
            v = generate_vignette(cond, rng)
            prompt, correct_letter, letter_map = make_mcq_example(v, cond["label"], labels_all, rng)
            records.append({
                "task_type": "dx_mcq",
                "dx_label": cond["label"],
                "triage": cond["triage"],
                "choices": letter_map,
                "messages": make_messages(SYSTEM_DX_MCQ, prompt, correct_letter),
            })

    # dx_label (exact label)
    label_prompt_templates = [
        "From the allowed diagnosis labels, pick the single best match.\n\nAllowed labels:\n{allowed}\n\nVignette: {v}\n\nOutput ONLY the label.",
        "Select the correct diagnosis label from this list.\n{allowed}\n\nCase: {v}\n\nReturn only the label text.",
    ]
    for cond in conditions:
        for _ in range(80):
            v = generate_vignette(cond, rng)
            q = rng.choice(label_prompt_templates).format(allowed=allowed_list_str, v=v)
            records.append({
                "task_type": "dx_label",
                "dx_label": cond["label"],
                "triage": cond["triage"],
                "messages": make_messages(SYSTEM_DX_LABEL, q, cond["label"]),
            })

    # abstain label
    for _ in range(1200):
        age = rng.choice([18,24,33,45,62,74])
        sex = rng.choice(["man","woman","person"])
        v = rng.choice(ambiguous_vignettes).format(age=age, sex=sex)
        q = "Pick the best label.\nAllowed labels:\n" + allowed_with_abstain + "\n\nVignette: " + v + "\n\nOutput ONLY one label."
        records.append({
            "task_type": "dx_abstain_label",
            "dx_label": abstain_label,
            "triage": "ROUTINE",
            "messages": make_messages(SYSTEM_DX_LABEL, q, abstain_label),
        })

    # dx+triage strict JSON
    for cond in conditions:
        for _ in range(40):
            v = generate_vignette(cond, rng)
            q = "Allowed diagnoses:\n" + allowed_list_str + "\n\nVignette: " + v + "\n\nOutput JSON only."
            a = json.dumps({"dx": cond["label"], "triage": cond["triage"]}, ensure_ascii=False)
            records.append({
                "task_type": "dx_triage_json",
                "dx_label": cond["label"],
                "triage": cond["triage"],
                "messages": make_messages(SYSTEM_DX_JSON, q, a),
            })

    # JSON abstain
    for _ in range(400):
        age = rng.choice([18,24,33,45,62,74])
        sex = rng.choice(["man","woman","person"])
        v = rng.choice(ambiguous_vignettes).format(age=age, sex=sex)
        q = "Allowed diagnoses:\n" + allowed_with_abstain + "\n\nVignette: " + v + "\n\nOutput JSON only."
        a = json.dumps({"dx": abstain_label, "triage": "ROUTINE"}, ensure_ascii=False)
        records.append({
            "task_type": "dx_triage_json_abstain",
            "dx_label": abstain_label,
            "triage": "ROUTINE",
            "messages": make_messages(SYSTEM_DX_JSON, q, a),
        })

    # Assign ids + shuffle
    rng.shuffle(records)
    for i, r in enumerate(records, start=1):
        r["id"] = f"qwen3_dx_v1_{i:06d}"
        r["created"] = today

    # Split
    n = len(records)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train = records[:n_train]
    val = records[n_train:n_train+n_val]
    test = records[n_train+n_val:]

    def write_jsonl(path: Path, rows: list[dict]):
        with path.open("w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)

    meta = {
        "created": today,
        "seed": args.seed,
        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        }
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[prepare_data] Generated dataset:")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
