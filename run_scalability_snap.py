#!/usr/bin/env python3
"""
run_scalability_snap.py
========================
Runs the RSMI scalability benchmark on one SNAP-derived triple file.

Mirrors run_scalability.py exactly — same metric parsing, same output format,
same prefix-based subset methodology — so results are directly comparable
with Wiki-Vote and other models.

Usage:
    python3 run_scalability_snap.py --dataset datasets/roadnet_ca_triples.txt \\
                                    --name    roadnet_ca

    python3 run_scalability_snap.py --dataset datasets/web_google_triples.txt \\
                                    --name    web_google

Outputs (written to the current directory):
    <name>_scalability_summary.txt
    <name>_scalability.csv
"""

import subprocess
import sys
import os
import csv
import re
import time
import argparse

EXP_BIN   = "./Exp"
SIZES     = [1_000_000, 2_500_000]   # full dataset is appended after line-count
MAX_FULL  = 5_000_000                # skip FULL run if dataset exceeds this size


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_lines(path: str) -> int:
    """Count lines without loading the whole file into memory."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def run_exp(dataset: str, limit: int) -> dict:
    """
    Run ./Exp for one dataset size and return a dict of parsed metrics.
    stderr streams live to the terminal so you see progress in real time.
    stdout is captured for metric parsing after the run completes.
    """
    cmd = [EXP_BIN,
           "--dataset", dataset,
           "--limit",   str(limit)]

    label = f"{limit:,}"
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting run: {label} tuples", flush=True)
    print(f"  command: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=None,       # inherit terminal → live output
                            text=True)
    stdout, _ = proc.communicate()

    if proc.returncode != 0:
        print(f"  ERROR: Exp exited with code {proc.returncode}", flush=True)
        return None

    # ── Parse the six metrics (identical patterns to run_scalability.py) ──────
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

    print(f"  done – build {metrics.get('build_time_s', '?'):.1f}s  "
          f"index {metrics.get('index_size_mb', '?'):.1f} MB  "
          f"mean {metrics.get('mean_latency_us', '?'):.3f} µs",
          flush=True)
    return metrics


# ── Table formatter (identical layout to run_scalability.py) ──────────────────

def build_table(dataset_name: str, rows: list) -> str:
    """rows = list of (label, metrics_dict)"""
    col_w   = [14, 14, 15, 17, 8, 16, 15]
    headers = ["Dataset Size", "Build Time (s)", "Index Size (MB)",
               "Mean Latency (µs)", "P95 (µs)", "Throughput (q/s)", "Avg Page Access"]

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
        f"RSMI Scalability Summary ({dataset_name})",
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
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RSMI scalability benchmark on a SNAP-derived triple file."
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to the triple file (e.g. datasets/roadnet_ca_triples.txt)")
    parser.add_argument("--name",    required=True,
                        help="Short dataset name used in output filenames and table header "
                             "(e.g. roadnet_ca or web_google)")
    args = parser.parse_args()

    dataset = args.dataset
    name    = args.name
    out_txt = f"{name}_scalability_summary.txt"
    out_csv = f"{name}_scalability.csv"

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if not os.path.exists(dataset):
        sys.exit(f"ERROR: dataset not found: {dataset}\n"
                 f"       Run generate_snap_triples.py first.")
    if not os.path.exists(EXP_BIN):
        sys.exit(f"ERROR: Exp binary not found – run 'make' first.")

    # ── Auto-detect full dataset size ─────────────────────────────────────────
    print(f"Counting lines in {dataset} …", end=" ", flush=True)
    full_n = count_lines(dataset)
    print(f"{full_n:,} tuples", flush=True)

    # Filter thresholds that are smaller than the dataset
    valid_sizes = [s for s in SIZES if s <= full_n]
    skipped     = [s for s in SIZES if s > full_n]
    if skipped:
        for s in skipped:
            print(f"  WARNING: dataset has only {full_n:,} tuples — "
                  f"skipping {s:,} run.", flush=True)

    if full_n > MAX_FULL:
        print(f"  NOTE: full dataset ({full_n:,} tuples) exceeds safe RAM limit "
              f"({MAX_FULL:,}). Skipping FULL run — only 1M and 2.5M will run.",
              flush=True)
        all_sizes = valid_sizes
        labels    = [f"{n:,}" for n in all_sizes]
    else:
        all_sizes = valid_sizes + [full_n]
        labels    = [f"{n:,}" if n != full_n else "FULL" for n in all_sizes]

    # If the full dataset equals one of the thresholds, avoid a duplicate run
    # (deduplicate while preserving order)
    seen = set()
    deduped = []
    for size, label in zip(all_sizes, labels):
        if size not in seen:
            seen.add(size)
            deduped.append((size, label))
    all_sizes, labels = zip(*deduped) if deduped else ([], [])

    # ── Run all sizes ─────────────────────────────────────────────────────────
    rows = []
    for limit, label in zip(all_sizes, labels):
        m = run_exp(dataset, limit)
        if m is None:
            sys.exit(f"Aborting: run failed for {label}")
        rows.append((label, m))

    # ── Print and save summary table ──────────────────────────────────────────
    table = build_table(name, rows)
    print("\n" + table)

    with open(out_txt, "w") as f:
        f.write(table + "\n")
    print(f"\nSummary saved → {out_txt}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_cols = ["dataset_name", "dataset_size", "build_time_s", "index_size_mb",
                "mean_latency_us", "p95_latency_us", "throughput_qps", "avg_page_access"]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_cols)
        for (label, m), limit in zip(rows, all_sizes):
            writer.writerow([
                name,
                limit,
                f"{m['build_time_s']:.3f}",
                f"{m['index_size_mb']:.3f}",
                f"{m['mean_latency_us']:.3f}",
                f"{m['p95_latency_us']:.3f}",
                f"{m['throughput_qps']:.0f}",
                f"{m['avg_page_access']:.4f}",
            ])
    print(f"CSV saved       → {out_csv}")


if __name__ == "__main__":
    main()
