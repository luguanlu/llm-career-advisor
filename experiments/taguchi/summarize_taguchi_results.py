import json
import csv
from pathlib import Path

plan = [
    ("T1", "30%",  "Baseline",     "short",  243),
    ("T2", "30%",  "BOHB-best",    "medium", 243),
    ("T3", "30%",  "BOHB+NEFTune", "full",   243),
    ("T4", "60%",  "Baseline",     "medium", 486),
    ("T5", "60%",  "BOHB-best",    "full",   486),
    ("T6", "60%",  "BOHB+NEFTune", "short",  486),
    ("T7", "100%", "Baseline",     "full",   810),
    ("T8", "100%", "BOHB-best",    "short",  810),
    ("T9", "100%", "BOHB+NEFTune", "medium", 810),
]

rows = []

for exp_id, data_scale, strategy, resource, samples in plan:
    out_dir = Path(f"saves/taguchi/llama3-8b/{exp_id}")
    train_path = out_dir / "train_results.json"
    eval_path = out_dir / "eval_results.json"

    train = json.loads(train_path.read_text()) if train_path.exists() else {}
    eval_data = json.loads(eval_path.read_text()) if eval_path.exists() else {}

    train_loss = train.get("train_loss", "")
    eval_loss = eval_data.get("eval_loss", "")

    if train_loss != "" and eval_loss != "":
        gap = eval_loss - train_loss
    else:
        gap = ""

    rows.append({
        "experiment_id": exp_id,
        "data_scale": data_scale,
        "train_samples": samples,
        "tuning_strategy": strategy,
        "training_resource": resource,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "generalization_gap": gap,
        "train_runtime": train.get("train_runtime", ""),
        "train_samples_per_second": train.get("train_samples_per_second", ""),
        "eval_runtime": eval_data.get("eval_runtime", ""),
        "output_dir": str(out_dir),
        "status": "done" if train_path.exists() and eval_path.exists() else "missing"
    })

out_path = Path("experiments/taguchi/taguchi_l9_results_summary.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {out_path}")
for row in rows:
    print(row)
