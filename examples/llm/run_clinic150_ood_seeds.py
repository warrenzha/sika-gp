import argparse
import csv
import os
import statistics
import subprocess
import sys


def read_metrics(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames if reader.fieldnames else []


def summarize(rows, fieldnames):
    summary = {}
    for k in fieldnames:
        if k == "seed":
            continue
        vals = []
        for row in rows:
            v = row.get(k)
            if v is None or v == "":
                continue
            try:
                vals.append(float(v))
            except ValueError:
                pass
        if vals:
            summary[k] = {
                "mean": statistics.mean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run clinic150_ood.py across multiple seeds and aggregate metrics.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--log_csv", type=str, default=None, help="Optional override for metrics file. Defaults to bayes-specific name.")
    parser.add_argument("--script", type=str, default="clinic150_ood.py")
    parser.add_argument("--append", action="store_true", help="Keep existing log_csv instead of overwriting.")
    parser.add_argument("--bayes", type=str, default="bayes_linear", choices=["csgp", "bayes_linear", "svdkl"], help="Bayesian head to pass through")
    args, extra = parser.parse_known_args()

    # Default log file name depends on bayes choice if not explicitly provided
    log_csv = args.log_csv or f"clinic150_ood_{args.bayes}_runs.csv"

    if (not args.append) and os.path.exists(log_csv):
        os.remove(log_csv)

    for seed in args.seeds:
        cmd = [
            sys.executable,
            args.script,
            "--seed", str(seed),
            "--bayes", args.bayes,
            "--log_csv", log_csv,
            *extra,
        ]
        print(f"Running seed {seed}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    if not os.path.exists(log_csv):
        print("No metrics CSV found; nothing to summarize.")
        return

    rows, fields = read_metrics(log_csv)
    summary = summarize(rows, fields)

    print(f"\nAggregated metrics from {len(rows)} runs (file: {log_csv}):")
    for k, stats in summary.items():
        print(f"{k}: mean={stats['mean']:.4f} std={stats['std']:.4f} (n={stats['n']})")


if __name__ == "__main__":
    main()
