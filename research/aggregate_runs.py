"""Aggregate repeated lightweight-student runs into mean and standard deviation."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="research_outputs/yorubalitesent")
    args = parser.parse_args()
    metrics_paths = sorted(Path(args.root).glob("seed_*/metrics.json"))
    if not metrics_paths:
        raise SystemExit("No metrics.json files found. Run multiple seeds first.")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in metrics_paths]
    summary = {"runs": len(records), "seeds": [row["seed"] for row in records]}
    for metric in ("accuracy", "precision", "recall", "f1"):
        values = np.array([row["test"][metric] for row in records], dtype=float)
        summary[metric] = {"mean": round(float(values.mean()), 4), "std": round(float(values.std(ddof=1)), 4)}
    print(json.dumps(summary, indent=2))
    output = Path(args.root) / "aggregate_metrics.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
