#!/usr/bin/env python3
"""
run_scalability_synthetic.py
==============================
Benchmarks RSMI on synthetic datasets at three scale checkpoints:
  1M  →  5M  →  10M
  (uniform_sparse already completed separately at all 5 sizes)

For each dataset, results are saved to two files in results/:
  <name>_scalability_summary.txt   — formatted table (same layout as run_scalability_snap.py)
  <name>_scalability.csv           — raw CSV for downstream analysis

After all datasets complete, a consolidated cross-dataset summary table
(mean latency at every size) is printed to stdout.

The ./Exp binary must already be compiled.  Do not attempt to recompile.
Failed or timed-out runs are recorded as NaN — the script never aborts early.

Usage:
    python3 run_scalability_synthetic.py
"""

import subprocess
import sys
import os
import csv
import re
import time
import math

EXP_BIN     = "./Exp"
RESULTS_DIR = "results"
SIZES       = [1_000_000, 5_000_000, 10_000_000]   # default for full runs
LABELS      = ["1M", "5M", "10M"]

DATASETS = [
    # uniform_sparse, uniform_dense, uniform_matched — fully completed, excluded
    # normal + lognormal: run all three sizes from scratch
    {"name": "normal",    "path": "datasets/normal_60M.txt",
     "sizes": [1_000_000, 5_000_000, 10_000_000], "labels": ["1M", "5M", "10M"]},
    {"name": "lognormal", "path": "datasets/lognormal_60M.txt",
     "sizes": [1_000_000, 5_000_000, 10_000_000], "labels": ["1M", "5M", "10M"]},
]

METRIC_PATTERNS = {
    "build_time_s":    r"Build Time \(s\):\s+([\d.]+)",
    "index_size_mb":   r"Index Size \(MB\):\s+([\d.]+)",
    "mean_latency_us": r"Mean Lookup Latency \(us\):\s+([\d.]+)",
    "p95_latency_us":  r"P95 Lookup Latency \(us\):\s+([\d.]+)",
    "throughput_qps":  r"Query Throughput \(q/s\):\s+([\d.]+)",
    "avg_page_access": r"Average Page Accesses:\s+([\d.]+)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(val, fmt):
    """Format a metric value; return 'N/A' if NaN."""
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    return format(val, fmt)


def run_exp(dataset, limit):
    """
    Run ./Exp --dataset <dataset> --limit <limit>.
    stderr inherits the terminal for live progress.
    stdout is captured and parsed for metrics.
    Returns a dict of six float metrics (NaN on parse failure / non-zero exit).
    """
    cmd = [EXP_BIN, "--dataset", dataset, "--limit", str(limit)]

    print(f"\n[{time.strftime('%H:%M:%S')}]  START  {limit:,} tuples", flush=True)
    print(f"  cmd : {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=None,      # inherit terminal → live output
                            text=True)
    stdout, _ = proc.communicate()

    if proc.returncode != 0:
        print(f"  ERROR: ./Exp exited with code {proc.returncode}", flush=True)
        return {k: float("nan") for k in METRIC_PATTERNS}

    metrics = {}
    for key, pat in METRIC_PATTERNS.items():
        m = re.search(pat, stdout)
        if m:
            metrics[key] = float(m.group(1))
        else:
            print(f"  WARNING: could not parse '{key}'", flush=True)
            metrics[key] = float("nan")

    print(f"  done — build {_fmt(metrics['build_time_s'], '.1f')}s  "
          f"index {_fmt(metrics['index_size_mb'], '.1f')} MB  "
          f"mean {_fmt(metrics['mean_latency_us'], '.3f')} µs",
          flush=True)
    return metrics


# ── Table formatter (identical layout to run_scalability_snap.py) ─────────────

def build_table(dataset_name, rows):
    """
    rows = list of (label, metrics_dict)
    Returns a formatted string table.
    """
    col_w   = [14, 14, 15, 17, 8, 16, 15]
    headers = ["Dataset Size", "Build Time (s)", "Index Size (MB)",
               "Mean Latency (µs)", "P95 (µs)", "Throughput (q/s)", "Avg Page Access"]
    sep     = "=" * 104
    dashes  = "-" * 104

    def fmt_row(cells):
        parts = []
        for i, (c, w) in enumerate(zip(cells, col_w)):
            align = "<" if i == 0 else ">"
            parts.append(f"{c:{align}{w}}")
        return " | ".join(parts)

    lines = [
        sep,
        f"RSMI Scalability Summary ({dataset_name})",
        sep,
        fmt_row(headers),
        dashes,
    ]

    for label, m in rows:
        cells = [
            label,
            _fmt(m["build_time_s"],    ".3f"),
            _fmt(m["index_size_mb"],   ".3f"),
            _fmt(m["mean_latency_us"], ".3f"),
            _fmt(m["p95_latency_us"],  ".3f"),
            _fmt(m["throughput_qps"],  ".0f"),
            _fmt(m["avg_page_access"], ".4f"),
        ]
        lines.append(fmt_row(cells))

    lines.append(sep)
    return "\n".join(lines)


# ── Per-dataset output ────────────────────────────────────────────────────────

def save_results(name, rows, sizes):
    """
    Write <name>_scalability_summary.txt and <name>_scalability.csv to results/.
    rows   = [(label, metrics_dict), ...]
    sizes  = [int, ...]   — raw numeric sizes matching rows
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    txt_path = os.path.join(RESULTS_DIR, f"{name}_scalability_summary.txt")
    csv_path = os.path.join(RESULTS_DIR, f"{name}_scalability.csv")

    # ── txt ───────────────────────────────────────────────────────────────────
    table = build_table(name, rows)
    with open(txt_path, "w") as f:
        f.write(table + "\n")
    print(f"\n  Summary saved → {txt_path}", flush=True)

    # ── csv ───────────────────────────────────────────────────────────────────
    csv_cols = ["dataset_name", "dataset_size", "build_time_s", "index_size_mb",
                "mean_latency_us", "p95_latency_us", "throughput_qps", "avg_page_access"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_cols)
        for (label, m), size in zip(rows, sizes):
            writer.writerow([
                name,
                size,
                _fmt(m["build_time_s"],    ".3f"),
                _fmt(m["index_size_mb"],   ".3f"),
                _fmt(m["mean_latency_us"], ".3f"),
                _fmt(m["p95_latency_us"],  ".3f"),
                _fmt(m["throughput_qps"],  ".0f"),
                _fmt(m["avg_page_access"], ".4f"),
            ])
    print(f"  CSV saved       → {csv_path}", flush=True)


# ── Cross-dataset summary table ───────────────────────────────────────────────

def print_cross_dataset_summary(all_results):
    """
    all_results = {dataset_name: [(label, metrics_dict), ...]}
    Prints a table with rows=datasets, columns=sizes, values=mean_latency_us.
    """
    col_w_name  = 20
    col_w_val   = 10

    print(f"\n{'='*104}", flush=True)
    print("Cross-Dataset Summary — Mean Lookup Latency (µs) at each scale checkpoint", flush=True)
    print("=" * 104, flush=True)

    # Header row
    hdr_cells = [f"{'Dataset':<{col_w_name}}"] + [f"{lbl:>{col_w_val}}" for lbl in LABELS]
    print(" | ".join(hdr_cells), flush=True)
    print("-" * 104, flush=True)

    for ds_name, rows in all_results.items():
        cells = [f"{ds_name:<{col_w_name}}"]
        for label, m in rows:
            val = m.get("mean_latency_us", float("nan"))
            cells.append(f"{_fmt(val, '.3f'):>{col_w_val}}")
        print(" | ".join(cells), flush=True)

    print("=" * 104, flush=True)

    # Also print throughput table
    print(f"\n{'='*104}", flush=True)
    print("Cross-Dataset Summary — Query Throughput (q/s) at each scale checkpoint", flush=True)
    print("=" * 104, flush=True)
    print(" | ".join(hdr_cells), flush=True)
    print("-" * 104, flush=True)
    for ds_name, rows in all_results.items():
        cells = [f"{ds_name:<{col_w_name}}"]
        for label, m in rows:
            val = m.get("throughput_qps", float("nan"))
            cells.append(f"{_fmt(val, '.0f'):>{col_w_val}}")
        print(" | ".join(cells), flush=True)
    print("=" * 104, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Sanity checks
    if not os.path.exists(EXP_BIN):
        sys.exit(f"ERROR: '{EXP_BIN}' not found — run 'make' first.")

    missing = [ds["path"] for ds in DATASETS if not os.path.exists(ds["path"])]
    if missing:
        print("WARNING: the following dataset files are missing:", flush=True)
        for p in missing:
            print(f"  {p}", flush=True)
        print("Run generate_synthetic_datasets.py first.\n", flush=True)
        # Do not abort — remaining datasets will still run.

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72, flush=True)
    print("RSMI Scalability Benchmark — Synthetic Datasets", flush=True)
    print(f"Datasets  : {len(DATASETS)}", flush=True)
    print(f"Sizes     : {[f'{s:,}' for s in SIZES]}", flush=True)
    print(f"Total runs: {len(DATASETS) * len(SIZES)}", flush=True)
    print("=" * 72, flush=True)

    all_results = {}   # {name: [(label, metrics), ...]}

    for ds in DATASETS:
        name   = ds["name"]
        dspath = ds["path"]

        print(f"\n{'='*72}", flush=True)
        print(f"  Dataset: {name}  ({dspath})", flush=True)
        print(f"{'='*72}", flush=True)

        # Use per-dataset sizes/labels if specified, else fall back to globals
        ds_sizes  = ds.get("sizes",  SIZES)
        ds_labels = ds.get("labels", LABELS)

        if not os.path.exists(dspath):
            print(f"  SKIP: file not found — recording NaN for all sizes.", flush=True)
            nan_m = {k: float("nan") for k in METRIC_PATTERNS}
            rows  = [(lbl, nan_m) for lbl in ds_labels]
            save_results(name, rows, ds_sizes)
            all_results[name] = rows
            continue

        rows = []
        for size, label in zip(ds_sizes, ds_labels):
            m = run_exp(dspath, size)
            rows.append((label, m))

        # Print and save per-dataset table
        table = build_table(name, rows)
        print(f"\n{table}", flush=True)
        save_results(name, rows, ds_sizes)
        all_results[name] = rows

    # Cross-dataset summary
    print_cross_dataset_summary(all_results)
    print(f"\nAll benchmarks complete.  Results written to {RESULTS_DIR}/\n",
          flush=True)


if __name__ == "__main__":
    main()
