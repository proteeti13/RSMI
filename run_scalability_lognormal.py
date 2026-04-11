#!/usr/bin/env python3
"""
run_scalability_lognormal.py
=============================
Runs the RSMI point-query benchmark on the log-normal synthetic dataset
at four scalability sizes: 1 M, 10 M, 30 M, 60 M tuples.

Mirrors run_scalability_snap.py exactly — same metric parsing, same CSV/summary
format — so results are directly comparable with SNAP experiments.

Usage
-----
    # Step 1 — generate dataset (run once, ~5–15 min for 70 M samples):
    python3 generate_lognormal_triples.py \\
        --output  datasets/lognormal_triples.txt \\
        --samples 70000000 \\
        --mu 0.0 --sigma 1.0 --seed 42

    # Step 2 — run RSMI on all 4 sizes:
    python3 run_scalability_lognormal.py

    # Or specify a different dataset file:
    python3 run_scalability_lognormal.py \\
        --dataset datasets/lognormal_triples.txt

Outputs (written to results/ directory):
    results/rsmi_lognormal_scalability.csv
    results/rsmi_lognormal_scalability_summary.txt
    results/rsmi_lognormal_combined.csv     (extended schema, optional)

Distribution parameters (for reproducibility)
----------------------------------------------
    Distribution : log-normal(mu=0.0, sigma=1.0)
    Random seed  : 42
    Scale factor : 100_000
    Raw samples  : 70_000_000 (before dedup)

Metrics reported (parsed from RSMI stdout)
------------------------------------------
    build_time_s       Build time in seconds
    index_size_mb      In-memory index size in MB
    mean_latency_us    Mean point-query latency in microseconds
    p95_latency_us     P95 point-query latency in microseconds
    throughput_qps     Queries per second (100 k queries / total query time)
    avg_page_access    Average page accesses per query
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time


# ── Configuration ─────────────────────────────────────────────────────────────

EXP_BIN       = "./Exp"
DEFAULT_DS    = "datasets/lognormal_triples.txt"
DATASET_NAME  = "lognormal"
RESULTS_DIR   = "results"

# Scalability sizes (4-step, per thesis spec)
SIZES  = [1_000_000, 10_000_000, 30_000_000, 60_000_000]
LABELS = ["1M", "10M", "30M", "60M"]

# Dataset generation parameters (for provenance in output files)
GEN_MU      = 0.0
GEN_SIGMA   = 1.0
GEN_SEED    = 42
GEN_SCALE   = 100_000
GEN_SAMPLES = 70_000_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_lines(path: str) -> int:
    """Count lines in path without loading it into memory."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def run_exp(dataset: str, limit: int, label: str) -> dict | None:
    """
    Execute ./Exp for one (dataset, limit) pair.
    stderr streams live to terminal; stdout captured for metric parsing.
    Returns a dict of parsed metrics, or None on failure.
    """
    cmd = [EXP_BIN, "--dataset", dataset, "--limit", str(limit)]
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting run: {label} ({limit:,} tuples)",
          flush=True)
    print(f"  command: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    stdout, _ = proc.communicate()

    if proc.returncode != 0:
        print(f"  ERROR: Exp exited with code {proc.returncode}", flush=True)
        return None

    patterns = {
        "build_time_s":    r"Build Time \(s\):\s+([\d.]+)",
        "index_size_mb":   r"Index Size \(MB\):\s+([\d.]+)",
        "mean_latency_us": r"Mean Lookup Latency \(us\):\s+([\d.]+)",
        "p95_latency_us":  r"P95 Lookup Latency \(us\):\s+([\d.]+)",
        "throughput_qps":  r"Query Throughput \(q/s\):\s+([\d.]+)",
        "avg_page_access": r"Average Page Accesses:\s+([\d.]+)",
    }

    metrics = {}
    for key, pat in patterns.items():
        m = re.search(pat, stdout)
        if m:
            metrics[key] = float(m.group(1))
        else:
            print(f"  WARNING: could not parse '{key}'", flush=True)
            metrics[key] = float("nan")

    print(
        f"  done – build {metrics.get('build_time_s', float('nan')):.1f}s  "
        f"index {metrics.get('index_size_mb', float('nan')):.1f} MB  "
        f"mean {metrics.get('mean_latency_us', float('nan')):.3f} µs",
        flush=True,
    )
    return metrics


# ── Table formatter ───────────────────────────────────────────────────────────

def build_table(rows: list) -> str:
    """rows = list of (label, metrics_dict)"""
    col_w   = [14, 14, 15, 17, 8, 16, 15]
    headers = [
        "Dataset Size", "Build Time (s)", "Index Size (MB)",
        "Mean Latency (µs)", "P95 (µs)", "Throughput (q/s)", "Avg Page Access",
    ]

    sep    = "=" * 104
    dashes = "-" * 104

    def fmt_row(cells):
        parts = []
        for i, (c, w) in enumerate(zip(cells, col_w)):
            align = "<" if i == 0 else ">"
            parts.append(f"{c:{align}{w}}")
        return " | ".join(parts)

    lines = [
        sep,
        f"RSMI Scalability Summary ({DATASET_NAME}  –  log-normal mu={GEN_MU} sigma={GEN_SIGMA} seed={GEN_SEED})",
        sep,
        fmt_row(headers),
        dashes,
    ]
    for label, m in rows:
        cells = [
            label,
            f"{m['build_time_s']:.3f}",
            f"{m['index_size_mb']:.3f}",
            f"{m['mean_latency_us']:.3f}",
            f"{m['p95_latency_us']:.3f}",
            f"{m['throughput_qps']:.0f}",
            f"{m['avg_page_access']:.4f}",
        ]
        lines.append(fmt_row(cells))
    lines.append(sep)

    # Append provenance block
    lines += [
        "",
        "Generation parameters:",
        f"  distribution : log-normal(mu={GEN_MU}, sigma={GEN_SIGMA})",
        f"  random seed  : {GEN_SEED}",
        f"  scale factor : {GEN_SCALE:,}",
        f"  raw samples  : {GEN_SAMPLES:,}  (before sort/dedup)",
        "",
        "Commands:",
        f"  python3 generate_lognormal_triples.py \\",
        f"      --output datasets/lognormal_triples.txt \\",
        f"      --samples {GEN_SAMPLES} --mu {GEN_MU} --sigma {GEN_SIGMA} --seed {GEN_SEED}",
        f"  python3 run_scalability_lognormal.py",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RSMI 4-step scalability benchmark on the log-normal dataset."
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DS,
        help=f"Path to the lognormal triple file  [default: {DEFAULT_DS}]",
    )
    args = parser.parse_args()

    dataset = args.dataset

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if not os.path.exists(dataset):
        sys.exit(
            f"ERROR: dataset not found: {dataset}\n"
            f"       Run generate_lognormal_triples.py first."
        )
    if not os.path.exists(EXP_BIN):
        sys.exit("ERROR: ./Exp binary not found – run 'make' first.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_txt = os.path.join(RESULTS_DIR, "rsmi_lognormal_scalability_summary.txt")
    out_csv = os.path.join(RESULTS_DIR, "rsmi_lognormal_scalability.csv")
    out_combined = os.path.join(RESULTS_DIR, "rsmi_lognormal_combined.csv")

    # ── Count dataset lines ───────────────────────────────────────────────────
    print(f"Counting lines in {dataset} …", end=" ", flush=True)
    full_n = count_lines(dataset)
    print(f"{full_n:,} unique triples", flush=True)

    # ── Filter sizes that fit in the dataset ──────────────────────────────────
    valid_pairs = [(s, l) for s, l in zip(SIZES, LABELS) if s <= full_n]
    skipped     = [(s, l) for s, l in zip(SIZES, LABELS) if s > full_n]
    for s, l in skipped:
        print(
            f"  WARNING: dataset has only {full_n:,} tuples — "
            f"skipping {l} ({s:,}) run.",
            flush=True,
        )

    if not valid_pairs:
        sys.exit("ERROR: dataset too small for any requested benchmark size.")

    # ── Run all valid sizes ───────────────────────────────────────────────────
    rows = []
    for limit, label in valid_pairs:
        m = run_exp(dataset, limit, label)
        if m is None:
            sys.exit(f"Aborting: run failed for {label} ({limit:,} tuples)")
        rows.append((label, limit, m))

    # ── Print and save summary table ──────────────────────────────────────────
    table_rows = [(label, m) for label, _, m in rows]
    table = build_table(table_rows)
    print("\n" + table)

    with open(out_txt, "w") as f:
        f.write(table + "\n")
    print(f"\nSummary saved → {out_txt}")

    # ── Save standard CSV ─────────────────────────────────────────────────────
    csv_cols = [
        "dataset_name", "dataset_size",
        "build_time_s", "index_size_mb",
        "mean_latency_us", "p95_latency_us",
        "throughput_qps", "avg_page_access",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_cols)
        for label, limit, m in rows:
            writer.writerow([
                DATASET_NAME,
                limit,
                f"{m['build_time_s']:.3f}",
                f"{m['index_size_mb']:.3f}",
                f"{m['mean_latency_us']:.3f}",
                f"{m['p95_latency_us']:.3f}",
                f"{m['throughput_qps']:.0f}",
                f"{m['avg_page_access']:.4f}",
            ])
    print(f"CSV saved       → {out_csv}")

    # ── Save extended combined CSV (optional thesis schema) ───────────────────
    combined_cols = [
        "dataset_name", "distribution", "size",
        "build_time_s", "index_size_mb",
        "mean_latency", "p95_latency",
        "throughput", "correctness",
        "max_pred_error", "avg_refine_window",
    ]
    with open(out_combined, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(combined_cols)
        for label, limit, m in rows:
            writer.writerow([
                DATASET_NAME,
                f"log-normal(mu={GEN_MU},sigma={GEN_SIGMA})",
                limit,
                f"{m['build_time_s']:.3f}",
                f"{m['index_size_mb']:.3f}",
                f"{m['mean_latency_us']:.3f}",
                f"{m['p95_latency_us']:.3f}",
                f"{m['throughput_qps']:.0f}",
                "N/A",   # correctness not reported by Exp binary
                "N/A",   # max_pred_error not reported by Exp binary
                "N/A",   # avg_refine_window not reported by Exp binary
            ])
    print(f"Combined CSV    → {out_combined}")


if __name__ == "__main__":
    main()
