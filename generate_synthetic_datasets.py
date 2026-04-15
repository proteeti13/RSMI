#!/usr/bin/env python3
"""
generate_synthetic_datasets.py
================================
Generates five 3D synthetic datasets, each containing exactly 60,000,000
unique triples, for the RSMI learned-index benchmark suite.

Triple format (4 space-separated columns per line):
    SourceID  Hop1_ID  Hop2_ID  Offset

Sorted lexicographically by (SourceID, Hop1_ID, Hop2_ID).
Offsets are sequential integers 0 .. N-1.

Datasets generated:
  1. uniform_sparse   — Uniform [1, 10_000_000)   → datasets/uniform_sparse_60M.txt
  2. uniform_dense    — Uniform [1,    500_000)   → datasets/uniform_dense_60M.txt
  3. uniform_matched  — Uniform [1,  1_000_000)   → datasets/uniform_matched_60M.txt
  4. normal           — Normal(500_000, 166_667), clamp [1, 999_999]
                                                   → datasets/normal_60M.txt
  5. lognormal        — Lognormal(0, 1), SCALE=1_000_000/P99, clamp [1, 999_999]
                                                   → datasets/lognormal_60M.txt

Requirements: numpy + standard library only.
Seed: 42 throughout for full reproducibility.
"""

import numpy as np
import sys
import os
import time

# ── Constants ─────────────────────────────────────────────────────────────────
SEED        = 42
TARGET      = 60_000_000
BATCH_SIZE  = 5_000_000      # rows generated per batch
DATASETS_DIR = "datasets"

# ── Dataset configs ───────────────────────────────────────────────────────────
DATASETS = [
    {
        "name":          "uniform_sparse",
        "output":        "datasets/uniform_sparse_60M.txt",
        "description":   "Uniform Sparse — each dim from Uniform[1, 10_000_000)",
        "type":          "uniform",
        "low":           1,
        "high":          10_000_000,
        "density_check": False,
    },
    {
        "name":          "uniform_dense",
        "output":        "datasets/uniform_dense_60M.txt",
        "description":   "Uniform Dense — each dim from Uniform[1, 500_000)",
        "type":          "uniform",
        "low":           1,
        "high":          500_000,
        "density_check": False,
    },
    {
        "name":          "uniform_matched",
        "output":        "datasets/uniform_matched_60M.txt",
        "description":   "Uniform Matched — each dim from Uniform[1, 1_000_000)",
        "type":          "uniform",
        "low":           1,
        "high":          1_000_000,
        "density_check": True,
    },
    {
        "name":          "normal",
        "output":        "datasets/normal_60M.txt",
        "description":   "Normal — mu=500_000, sigma=166_667, clamp [1, 999_999]",
        "type":          "normal",
        "mu":            500_000.0,
        "sigma":         166_667.0,
        "density_check": True,
    },
    {
        "name":          "lognormal",
        "output":        "datasets/lognormal_60M.txt",
        "description":   "Lognormal — mu=0.0, sigma=1.0, SCALE=1_000_000/P99(pilot), clamp [1, 999_999]",
        "type":          "lognormal",
        "mu":            0.0,
        "sigma":         1.0,
        "density_check": True,
    },
]


# ── Per-distribution batch generators ────────────────────────────────────────

def gen_uniform_batch(rng, n, low, high):
    """3 independent Uniform integer columns → (n, 3) int32."""
    return np.column_stack([
        rng.integers(low, high, size=n, dtype=np.int32),
        rng.integers(low, high, size=n, dtype=np.int32),
        rng.integers(low, high, size=n, dtype=np.int32),
    ])


def gen_normal_batch(rng, n, mu, sigma):
    """3 independent Normal columns, clamped to [1, 999_999] → (n, 3) int32."""
    cols = []
    for _ in range(3):
        raw  = rng.normal(mu, sigma, size=n)
        vals = np.clip(np.floor(raw).astype(np.int64), 1, 999_999).astype(np.int32)
        cols.append(vals)
    return np.column_stack(cols)


def compute_lognormal_scale(rng, mu, sigma, pilot_n=200_000):
    """
    Draw pilot_n lognormal samples, compute P99, return (scale, p99).
    scale = 1_000_000 / P99  so the 99th percentile maps to ~1_000_000 before clamping.
    """
    pilot = rng.lognormal(mean=mu, sigma=sigma, size=pilot_n)
    p99   = float(np.percentile(pilot, 99))
    scale = 1_000_000.0 / p99
    return scale, p99


def gen_lognormal_batch(rng, n, mu, sigma, scale):
    """3 independent Lognormal columns, scaled and clamped [1, 999_999] → (n, 3) int32."""
    cols = []
    for _ in range(3):
        raw  = rng.lognormal(mean=mu, sigma=sigma, size=n)
        vals = np.clip(np.floor(raw * scale).astype(np.int64), 1, 999_999).astype(np.int32)
        cols.append(vals)
    return np.column_stack(cols)


# ── Core accumulation / dedup loop ────────────────────────────────────────────

def collect_unique_triples(gen_fn, target=TARGET, batch_size=BATCH_SIZE):
    """
    Accumulate exactly `target` unique, lexicographically sorted triples.

    Strategy:
      1. Generate target × 1.05 rows in batches of `batch_size`.
      2. Consolidate (np.unique, axis=0) every 3 batches to control memory.
      3. If still short after the initial pass, add extra batches until satisfied.

    Returns: numpy array (target, 3), dtype int32, sorted.
    """
    initial_need = int(target * 1.05)   # 5 % over-generation buffer
    n_initial    = (initial_need + batch_size - 1) // batch_size

    print(f"  Initial pass: {n_initial} batch(es) of up to {batch_size:,} rows "
          f"({initial_need:,} total — 5% buffer)…", flush=True)

    chunks      = []
    rows_queued = 0

    for i in range(n_initial):
        this_n = min(batch_size, initial_need - rows_queued)
        chunks.append(gen_fn(this_n))
        rows_queued += this_n

        # Periodic consolidation every 3 batches to cap peak memory
        if (i + 1) % 3 == 0:
            merged = np.unique(np.vstack(chunks), axis=0)
            print(f"  [batch {i+1:2d}] {rows_queued:,} generated → "
                  f"{len(merged):,} unique so far", flush=True)
            chunks = [merged]

    # Final merge + dedup of whatever remains in chunks
    arr = np.unique(np.vstack(chunks), axis=0)
    print(f"  Initial pass done: {rows_queued:,} generated → {len(arr):,} unique",
          flush=True)

    # Top-up rounds (rare; handles extremely high-collision edge cases)
    topup = 0
    while len(arr) < target:
        topup += 1
        deficit = target - len(arr)
        print(f"  Top-up round {topup}: need {deficit:,} more unique rows; "
              f"generating {batch_size:,}…", flush=True)
        extra = gen_fn(batch_size)
        arr   = np.unique(np.vstack([arr, extra]), axis=0)
        print(f"  After top-up {topup}: {len(arr):,} unique rows", flush=True)

    return arr[:target]   # already sorted by np.unique


# ── Validation ────────────────────────────────────────────────────────────────

def validate(arr, name, density_check=False):
    """
    Run 6 validation checks on the (60M, 3) int32 array.
    Prints PASSED / FAILED for each check.
    Returns (all_passed: bool, density_fraction: float or None).
    """
    SEP = "─" * 72
    print(f"\n{SEP}", flush=True)
    print(f"  Validation: {name}", flush=True)
    print(SEP, flush=True)

    all_ok  = True
    density = None

    # ── Check 1: row count ────────────────────────────────────────────────────
    n = len(arr)
    ok = (n == TARGET)
    all_ok = all_ok and ok
    print(f"  [1] Row count      : {n:,}  →  {'PASSED' if ok else 'FAILED'}", flush=True)

    # ── Check 2: per-column min/max ───────────────────────────────────────────
    for i, col in enumerate(["SourceID", "Hop1_ID ", "Hop2_ID "]):
        cmin = int(arr[:, i].min())
        cmax = int(arr[:, i].max())
        print(f"  [2] {col}  min={cmin:>12,}   max={cmax:>12,}", flush=True)

    # ── Check 3: P99 across all three columns combined ────────────────────────
    p99 = float(np.percentile(arr.flatten().astype(np.int64), 99))
    print(f"  [3] P99 (all cols) : {p99:,.1f}", flush=True)

    # ── Check 4: density in [400_000, 600_000] on SourceID (col 0) ───────────
    if density_check:
        mask    = (arr[:, 0] >= 400_000) & (arr[:, 0] <= 600_000)
        density = float(mask.sum()) / n
        print(f"  [4] Density [400K–600K] on SourceID: "
              f"{density:.4f}  ({density * 100:.2f}%)", flush=True)
    else:
        print(f"  [4] Density check  : N/A (not applicable for this dataset)", flush=True)

    # ── Check 5: first 3 and last 3 rows ─────────────────────────────────────
    print(f"  [5] First 3 rows:", flush=True)
    for r in arr[:3]:
        print(f"        {int(r[0]):>12,}  {int(r[1]):>12,}  {int(r[2]):>12,}", flush=True)
    print(f"  [5] Last 3 rows:", flush=True)
    for r in arr[-3:]:
        print(f"        {int(r[0]):>12,}  {int(r[1]):>12,}  {int(r[2]):>12,}", flush=True)

    # ── Check 6: lexicographic sort — 1,000 random consecutive pairs ──────────
    rng_chk    = np.random.default_rng(9999)
    idx        = rng_chk.integers(0, n - 1, size=1000)
    violations = 0
    for i in idx:
        a, b = arr[i], arr[i + 1]
        for k in range(3):
            if a[k] < b[k]:
                break
            if a[k] > b[k]:
                violations += 1
                break
    ok = (violations == 0)
    all_ok = all_ok and ok
    status = "PASSED" if ok else f"FAILED — {violations} violation(s)"
    print(f"  [6] Sort check (1,000 pairs): {status}", flush=True)

    print(f"\n  ► Overall: {'PASSED ✓' if all_ok else 'FAILED ✗'}", flush=True)
    print(SEP, flush=True)

    return all_ok, density


# ── Write to file ─────────────────────────────────────────────────────────────

def write_to_file(arr, output_path, chunk_size=2_000_000):
    """
    Write (N, 3) int32 array to a text file, appending sequential offsets.
    Writes in chunks via np.savetxt to avoid peak-memory spikes.
    """
    n = len(arr)
    print(f"  Writing {n:,} rows to {output_path} …", flush=True)
    t0 = time.time()

    with open(output_path, "wb") as f:
        offset = 0
        while offset < n:
            end   = min(offset + chunk_size, n)
            chunk = arr[offset:end]                        # (chunk_n, 3) int32
            offs  = np.arange(offset, end, dtype=np.int64).reshape(-1, 1)
            block = np.hstack([chunk.astype(np.int64), offs])  # (chunk_n, 4)
            np.savetxt(f, block, fmt="%d", delimiter=" ")
            offset = end
            if offset % (chunk_size * 5) == 0 or offset == n:
                pct = 100 * offset / n
                print(f"    {offset:,} / {n:,} rows written ({pct:.0f}%)…", flush=True)

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"  Write complete: {size_mb:.1f} MB in {elapsed:.1f}s", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Compute lognormal SCALE first so it can be printed at top-of-run
    ln_cfg              = next(d for d in DATASETS if d["type"] == "lognormal")
    ln_scale, ln_p99    = compute_lognormal_scale(rng, ln_cfg["mu"], ln_cfg["sigma"])
    ln_cfg["scale"]     = ln_scale

    print("=" * 72, flush=True)
    print("Synthetic Dataset Generation — 5 datasets × 60,000,000 unique triples", flush=True)
    print("=" * 72, flush=True)
    print(f"\nLognormal SCALE (computed from 200,000 pilot samples):", flush=True)
    print(f"  mu=0.0, sigma=1.0  →  raw P99 = {ln_p99:.6f}", flush=True)
    print(f"  SCALE = 1,000,000 / {ln_p99:.6f} = {ln_scale:.4f}\n", flush=True)

    # Accumulators for the final summary tables
    summary_ranges  = []   # [(name, [col_mins], [col_maxes])]
    summary_density = []   # [(name, fraction)]  — density-check datasets only

    # ── Main loop ─────────────────────────────────────────────────────────────
    for ds in DATASETS:
        name   = ds["name"]
        output = ds["output"]
        dtype  = ds["type"]

        print(f"\n{'='*72}", flush=True)
        print(f"  Dataset : {name}", flush=True)
        print(f"  Desc    : {ds['description']}", flush=True)
        print(f"  Output  : {output}", flush=True)
        print(f"{'='*72}", flush=True)

        t_start = time.time()

        # Build per-dataset generator closure (captures params, uses shared rng)
        if dtype == "uniform":
            lo, hi = ds["low"], ds["high"]
            gen_fn = lambda n, _lo=lo, _hi=hi: gen_uniform_batch(rng, n, _lo, _hi)

        elif dtype == "normal":
            mu, sigma = ds["mu"], ds["sigma"]
            gen_fn = lambda n, _mu=mu, _sg=sigma: gen_normal_batch(rng, n, _mu, _sg)

        elif dtype == "lognormal":
            mu, sigma, scale = ds["mu"], ds["sigma"], ds["scale"]
            gen_fn = lambda n, _mu=mu, _sg=sigma, _sc=scale: \
                gen_lognormal_batch(rng, n, _mu, _sg, _sc)

        else:
            sys.exit(f"ERROR: unknown dataset type '{dtype}'")

        # ── Generate ──────────────────────────────────────────────────────────
        arr = collect_unique_triples(gen_fn, target=TARGET, batch_size=BATCH_SIZE)

        # ── Validate ──────────────────────────────────────────────────────────
        ok, density = validate(arr, name, density_check=ds["density_check"])
        if not ok:
            sys.exit(f"ABORTING: validation failed for dataset '{name}'.")

        # ── Write ─────────────────────────────────────────────────────────────
        write_to_file(arr, output)

        elapsed = time.time() - t_start
        print(f"\n  Total time for '{name}': {elapsed:.1f}s", flush=True)

        # Collect for final summary tables
        col_mins  = [int(arr[:, i].min()) for i in range(3)]
        col_maxes = [int(arr[:, i].max()) for i in range(3)]
        summary_ranges.append((name, col_mins, col_maxes))
        if ds["density_check"] and density is not None:
            summary_density.append((name, density))

        del arr   # free RAM before next dataset

    # ── Final summary tables ──────────────────────────────────────────────────

    W = 100
    print(f"\n\n{'='*W}", flush=True)
    print("Coordinate Range Summary — All Five Datasets", flush=True)
    print("=" * W, flush=True)
    hdr = (f"{'Dataset':<20} | {'Src min':>10} | {'Src max':>12} | "
           f"{'H1 min':>10} | {'H1 max':>12} | "
           f"{'H2 min':>10} | {'H2 max':>12}")
    print(hdr, flush=True)
    print("-" * W, flush=True)
    for name, mins, maxes in summary_ranges:
        print(f"{name:<20} | {mins[0]:>10,} | {maxes[0]:>12,} | "
              f"{mins[1]:>10,} | {maxes[1]:>12,} | "
              f"{mins[2]:>10,} | {maxes[2]:>12,}", flush=True)
    print("=" * W, flush=True)

    if summary_density:
        print(f"\n{'='*60}", flush=True)
        print("Density Comparison — fraction of SourceID in [400K, 600K]", flush=True)
        print(f"{'='*60}", flush=True)
        for name, d in summary_density:
            bar_len = int(d * 40)
            bar     = "#" * bar_len + "." * (40 - bar_len)
            print(f"  {name:<20}: {d:.4f}  ({d*100:5.2f}%)  [{bar}]", flush=True)
        print("=" * 60, flush=True)

    print(f"\nAll five datasets generated successfully.\n", flush=True)


if __name__ == "__main__":
    main()
