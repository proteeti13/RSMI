#!/usr/bin/env python3
"""
RSMI Scalability Benchmark – Wiki-Vote triples
===============================================
Runs RSMI on three dataset sizes (1M, 2.5M, full) in sequence.
Each run is a prefix of the sorted file, so datasets are nested.

Outputs:
  rsmi_wikivote_scalability_summary.txt  – formatted table
  rsmi_wikivote_scalability.csv          – raw metrics, one row per size
"""

import subprocess
import sys
import os
import csv
import re
import time

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET   = "datasets/wiki_vote_triples.txt"
EXP_BIN   = "./Exp"
OUT_TXT   = "rsmi_wikivote_scalability_summary.txt"
OUT_CSV   = "rsmi_wikivote_scalability.csv"

# ── Dataset sizes to benchmark ────────────────────────────────────────────────
SIZES = [1_000_000, 2_500_000]   # full dataset appended after line-count


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_lines(path: str) -> int:
    """Count lines without loading the whole file into memory."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def run_exp(limit: int) -> dict:
    """
    Run ./Exp for one dataset size and return a dict of parsed metrics.
    Always passes --limit explicitly so the run is self-documenting.
    """
    cmd = [EXP_BIN,
           "--dataset", DATASET,
           "--limit",   str(limit)]

    label = f"{limit:,}"
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting run: {label} tuples", flush=True)
    print(f"  command: {' '.join(cmd)}", flush=True)

    # stderr streams live to terminal (so you see "Loaded …" and any errors immediately)
    # stdout is captured for metric parsing after the run completes
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=None,          # inherit terminal → live output
                            text=True)
    stdout, _ = proc.communicate()

    if proc.returncode != 0:
        print(f"  ERROR: Exp exited with code {proc.returncode}", flush=True)
        return None

    # ── Parse the six metrics from formatted output ──────────────────────────
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


def build_table(rows: list) -> str:
    """rows = list of (label, metrics_dict)"""
    col_w = [14, 14, 15, 17, 8, 16, 15]
    headers = ["Dataset Size", "Build Time (s)", "Index Size (MB)",
               "Mean Latency (µs)", "P95 (µs)", "Throughput (q/s)", "Avg Page Access"]

    sep   = "=" * 104
    dashes = "-" * 104

    def fmt_row(cells):
        parts = []
        for i, (c, w) in enumerate(zip(cells, col_w)):
            align = "<" if i == 0 else ">"
            parts.append(f"{c:{align}{w}}")
        return " | ".join(parts)

    lines = [
        sep,
        "RSMI Scalability Summary (wiki_vote_triples)",
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
    # Sanity checks
    if not os.path.exists(DATASET):
        sys.exit(f"ERROR: dataset not found: {DATASET}")
    if not os.path.exists(EXP_BIN):
        sys.exit(f"ERROR: Exp binary not found – run 'make' first.")

    # Auto-detect full dataset size
    print(f"Counting lines in {DATASET} …", end=" ", flush=True)
    full_n = count_lines(DATASET)
    print(f"{full_n:,} tuples", flush=True)

    all_sizes = SIZES + [full_n]
    labels = [f"{n:,}" if n != full_n else "FULL" for n in all_sizes]

    # ── Run all three sizes ───────────────────────────────────────────────────
    rows = []
    for limit, label in zip(all_sizes, labels):
        m = run_exp(limit)
        if m is None:
            sys.exit(f"Aborting: run failed for {label}")
        rows.append((label, m))

    # ── Print and save summary table ──────────────────────────────────────────
    table = build_table(rows)
    print("\n" + table)

    with open(OUT_TXT, "w") as f:
        f.write(table + "\n")
    print(f"\nSummary saved → {OUT_TXT}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_cols = ["dataset_size", "build_time_s", "index_size_mb",
                "mean_latency_us", "p95_latency_us", "throughput_qps", "avg_page_access"]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_cols)
        for (label, m), limit in zip(rows, all_sizes):
            writer.writerow([
                limit,
                f"{m['build_time_s']:.3f}",
                f"{m['index_size_mb']:.3f}",
                f"{m['mean_latency_us']:.3f}",
                f"{m['p95_latency_us']:.3f}",
                f"{m['throughput_qps']:.0f}",
                f"{m['avg_page_access']:.4f}",
            ])
    print(f"CSV saved       → {OUT_CSV}")


if __name__ == "__main__":
    main()
