#!/usr/bin/env python3
"""
generate_lognormal_triples.py
==============================
Generates a large synthetic 3D dataset of (SourceID, Hop1_ID, Hop2_ID) triples
drawn from a log-normal distribution, for RSMI scalability experiments.

Distribution
------------
    Each coordinate is independently drawn as:
        X = exp(N(mu, sigma^2))   →   log-normal(mu, sigma)
    Converted to a non-negative integer by:
        int_val = floor(X * SCALE)
    where SCALE=100_000 maps the bulk of the log-normal mass to integers in
    the range [~5_000 .. ~2_000_000], giving ample space for 60 M unique triples.

Pipeline
--------
    1. Generate raw (src, h1, h2) tuples in chunks → stream to temp file
    2. External Unix sort --unique (disk-based, no giant RAM footprint)
    3. Assign sequential offsets 0..N-1
    4. Validate first 100 k rows
    5. Print summary (sizes, head/tail)

Output format (same as SNAP triples / RSMI input):
    SourceID Hop1_ID Hop2_ID Offset

Usage
-----
    # Defaults: 70 M samples, mu=0, sigma=1, seed=42
    python3 generate_lognormal_triples.py \\
        --output datasets/lognormal_triples.txt

    # Explicit parameters
    python3 generate_lognormal_triples.py \\
        --output  datasets/lognormal_triples.txt \\
        --samples 70000000 \\
        --mu      0.0 \\
        --sigma   1.0 \\
        --seed    42

Notes
-----
- Run once; all four RSMI sizes (1M / 10M / 30M / 60M) are served via
  --limit N in the Exp binary from this single sorted file.
- Memory footprint: O(chunk_size) numpy arrays (~120 MB for 5M-row chunks).
- Disk usage: ~3–5 GB for the raw temp file + final output.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


# ── Default parameters ────────────────────────────────────────────────────────

DEFAULT_SAMPLES    = 70_000_000   # generate this many raw samples
DEFAULT_MU         = 0.0          # log-normal location parameter
DEFAULT_SIGMA      = 1.0          # log-normal scale parameter
DEFAULT_SEED       = 42           # reproducibility
DEFAULT_CHUNK_SIZE = 5_000_000    # rows per generation chunk (RAM budget)
SCALE              = 100_000      # multiply log-normal value → integer


# ── Step 1: generate raw triples in chunks ────────────────────────────────────

def generate_raw_to_file(n_samples: int, mu: float, sigma: float,
                          seed: int, chunk_size: int, tmp_path: str) -> int:
    """
    Draw n_samples log-normal triples and stream them to tmp_path.
    Each line: 'src h1 h2\\n'
    Returns the number of raw lines written.
    """
    rng = np.random.default_rng(seed)
    written = 0
    t0 = time.time()

    print(f"  Parameters : mu={mu}, sigma={sigma}, seed={seed}, SCALE={SCALE:,}",
          flush=True)
    print(f"  Chunks of {chunk_size:,} rows  →  {n_samples:,} total samples",
          flush=True)

    with open(tmp_path, "w", buffering=8 * 1024 * 1024) as f:
        remaining = n_samples
        while remaining > 0:
            batch = min(chunk_size, remaining)

            # Draw 3 independent log-normal variates per row
            src = np.floor(
                np.exp(rng.normal(mu, sigma, batch)) * SCALE
            ).astype(np.int64)
            h1 = np.floor(
                np.exp(rng.normal(mu, sigma, batch)) * SCALE
            ).astype(np.int64)
            h2 = np.floor(
                np.exp(rng.normal(mu, sigma, batch)) * SCALE
            ).astype(np.int64)

            # Stream to file — avoid per-row Python formatting overhead
            lines = (
                np.column_stack([src, h1, h2])
                .astype(str)
            )
            for row in lines:
                f.write(f"{row[0]} {row[1]} {row[2]}\n")

            written    += batch
            remaining  -= batch
            elapsed     = time.time() - t0
            print(f"    … {written:,} raw samples written  ({elapsed:.0f}s elapsed)",
                  flush=True)

    print(f"  Raw file complete: {written:,} rows  →  {tmp_path}", flush=True)
    return written


# ── Step 2: external sort + dedup ─────────────────────────────────────────────

def sort_and_dedup(tmp_raw: str, tmp_sorted: str):
    """
    Use the Unix external sort to sort lexicographically-by-field and dedup.
    Uses 1 GB RAM cap; spills to the output directory.
    """
    tmp_dir = os.path.dirname(os.path.abspath(tmp_sorted))
    print(f"  Sorting + deduplicating  (sort temp dir: {tmp_dir}) …", flush=True)
    cmd = [
        "sort",
        "-u",                          # deduplicate identical lines
        "-k1,1n", "-k2,2n", "-k3,3n", # numeric sort on each field
        "--buffer-size=1G",
        f"--temporary-directory={tmp_dir}",
        "-o", tmp_sorted,
        tmp_raw,
    ]
    subprocess.run(cmd, check=True)
    print("  Sort complete.", flush=True)


# ── Step 3: assign offsets ────────────────────────────────────────────────────

def assign_offsets(sorted_path: str, out_path: str) -> int:
    """
    Read sorted file, append sequential offset, write output.
    O(1) RAM.  Returns unique triple count.
    """
    print(f"  Assigning offsets  →  {out_path} …", flush=True)
    count = 0
    with open(sorted_path, "r", buffering=8 * 1024 * 1024) as fin, \
         open(out_path,    "w", buffering=8 * 1024 * 1024) as fout:
        for line in fin:
            line = line.rstrip()
            if not line:
                continue
            fout.write(f"{line} {count}\n")
            count += 1
            if count % 5_000_000 == 0:
                print(f"    … {count:,} unique triples written", flush=True)
    print(f"  Offset assignment done: {count:,} unique triples.", flush=True)
    return count


# ── Step 4: validate ──────────────────────────────────────────────────────────

def validate(out_path: str, check_rows: int = 100_000):
    """
    Stream through check_rows rows and verify:
      - offset == row index
      - lexicographic sort order holds
    """
    print(f"  Validating first {check_rows:,} rows …", flush=True)
    prev = None
    with open(out_path, "r") as f:
        for i, line in enumerate(f):
            if i >= check_rows:
                break
            parts = line.split()
            assert len(parts) == 4, f"Row {i}: expected 4 fields, got {len(parts)}"
            u, v, w, off = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            assert off == i,  f"Row {i}: offset={off} expected {i}"
            curr = (u, v, w)
            if prev is not None:
                assert prev <= curr, f"Row {i}: sort violation {prev} > {curr}"
            prev = curr
    print(f"  Validation PASSED.", flush=True)


# ── Summary ───────────────────────────────────────────────────────────────────

SCALABILITY_SIZES = [1_000_000, 10_000_000, 30_000_000, 60_000_000]

def print_summary(unique_count: int, out_path: str,
                  mu: float, sigma: float, seed: int, n_samples: int):
    n = unique_count
    print(f"\n{'='*64}")
    print(f"  Distribution        : log-normal(mu={mu}, sigma={sigma})")
    print(f"  Random seed         : {seed}")
    print(f"  Scale factor        : {SCALE:,}")
    print(f"  Raw samples drawn   : {n_samples:,}")
    print(f"  Unique triples      : {n:,}")
    print(f"  Output file         : {out_path}")
    print(f"{'='*64}")

    print(f"\n  Scalability support:")
    for t in SCALABILITY_SIZES:
        if n >= t:
            status = "YES"
        else:
            status = f"NO  (only {n:,} unique triples available)"
        print(f"    {t:>12,} tuples  →  {status}")
    print(f"    {'FULL':>12}           →  YES  ({n:,} unique triples)")

    print("\n  First 5 rows (SourceID Hop1_ID Hop2_ID Offset):")
    with open(out_path) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            p = line.split()
            print(f"    {p[0]:>10}  {p[1]:>10}  {p[2]:>10}  {p[3]:>12}")

    print("\n  Last 5 rows:")
    result = subprocess.run(["tail", "-5", out_path], capture_output=True, text=True)
    for line in result.stdout.strip().splitlines():
        p = line.split()
        print(f"    {p[0]:>10}  {p[1]:>10}  {p[2]:>10}  {p[3]:>12}")

    print(f"\n  Offset range : 0 .. {n - 1:,}")
    print(f"{'='*64}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate RSMI-compatible log-normal 3D triples dataset."
    )
    parser.add_argument("--output",  default="datasets/lognormal_triples.txt",
                        help="Output triple file path  [default: datasets/lognormal_triples.txt]")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Raw samples to draw (before dedup)  [default: {DEFAULT_SAMPLES:,}]")
    parser.add_argument("--mu",      type=float, default=DEFAULT_MU,
                        help=f"Log-normal mu parameter  [default: {DEFAULT_MU}]")
    parser.add_argument("--sigma",   type=float, default=DEFAULT_SIGMA,
                        help=f"Log-normal sigma parameter  [default: {DEFAULT_SIGMA}]")
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility  [default: {DEFAULT_SEED}]")
    parser.add_argument("--chunk",   type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Rows per generation chunk  [default: {DEFAULT_CHUNK_SIZE:,}]")
    args = parser.parse_args()

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_dir  = os.path.dirname(os.path.abspath(out_path))

    tmp_raw    = os.path.join(out_dir, "_tmp_lognormal_raw.txt")
    tmp_sorted = os.path.join(out_dir, "_tmp_lognormal_sorted.txt")

    t_start = time.time()

    try:
        # ── 1. Generate raw samples ─────────────────────────────────────────
        print(f"[1/4] Generating {args.samples:,} raw log-normal samples …", flush=True)
        raw_count = generate_raw_to_file(
            args.samples, args.mu, args.sigma,
            args.seed, args.chunk, tmp_raw
        )

        # ── 2. External sort + dedup ────────────────────────────────────────
        print(f"[2/4] External sort + dedup …", flush=True)
        sort_and_dedup(tmp_raw, tmp_sorted)
        os.remove(tmp_raw)

        # ── 3. Assign offsets ───────────────────────────────────────────────
        print(f"[3/4] Assigning offsets …", flush=True)
        unique_count = assign_offsets(tmp_sorted, out_path)
        os.remove(tmp_sorted)

        # ── 4. Validate ─────────────────────────────────────────────────────
        print(f"[4/4] Validating output …", flush=True)
        validate(out_path)

        elapsed = time.time() - t_start
        print(f"\nDataset generated in {elapsed:.1f}s", flush=True)

        # Warn if dataset is too small for desired scalability sizes
        for t in SCALABILITY_SIZES:
            if unique_count < t:
                print(f"  WARNING: only {unique_count:,} unique triples — "
                      f"cannot run {t:,}-tuple benchmark.", flush=True)

        # ── Summary ─────────────────────────────────────────────────────────
        print_summary(unique_count, out_path,
                      args.mu, args.sigma, args.seed, args.samples)

    finally:
        for p in [tmp_raw, tmp_sorted]:
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()
