"""
generate_snap_triples.py
------------------------
Converts a SNAP-format edge list into 2-hop path triples for RSMI experiments.

Memory-efficient: streams triples to a temp file, uses Unix external sort
(no giant in-memory set). Safe for 60M+ triple datasets like web-Google.

Supports:
  --directed    use edges as-is (e.g. web-Google)
  --undirected  mirror every edge u->v to also add v->u (e.g. roadNet-CA)

Output format (4 space-separated columns, numerically sorted):
    SourceID Hop1_ID Hop2_ID Offset

Usage:
    python3 generate_snap_triples.py \
        --input  datasets/roadNet-CA.txt/roadNet-CA.txt \
        --output datasets/roadnet_ca_triples.txt \
        --undirected

    python3 generate_snap_triples.py \
        --input  datasets/web-Google.txt/web-Google.txt \
        --output datasets/web_google_triples.txt \
        --directed
"""

import sys
import os
import argparse
import subprocess
import tempfile
from collections import defaultdict


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_edges(path, undirected: bool):
    """
    Parse a SNAP edge list, skipping comment lines.
    If undirected, add the reverse edge for every (u, v).
    Returns a list of (u, v) int tuples.
    """
    raw = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            raw.append((u, v))
            if undirected and u != v:
                raw.append((v, u))
    return raw


def build_adjacency(edges):
    """Build adjacency list. Deduplicates neighbours."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
    return adj


# ── Triple generation — streaming, no in-memory set ───────────────────────────

def generate_triples_to_file(adj, tmp_path: str) -> int:
    """
    Enumerate all valid 2-hop paths u->v->w and write each as
    a text line to tmp_path.  No deduplication yet — that is
    delegated to the external sort step.

    Uses an 8 MB write buffer to keep I/O fast.
    Returns the raw (pre-dedup) triple count.
    """
    raw_count = 0
    print("  Generating triples and streaming to temp file …", flush=True)
    with open(tmp_path, "w", buffering=8 * 1024 * 1024) as f:
        for u, neighbours_u in adj.items():
            for v in neighbours_u:
                if v in adj:
                    for w in adj[v]:
                        f.write(f"{u} {v} {w}\n")
                        raw_count += 1
                        if raw_count % 5_000_000 == 0:
                            print(f"    … {raw_count:,} raw triples written",
                                  flush=True)
    print(f"  Raw triples written : {raw_count:,}", flush=True)
    return raw_count


def sort_and_dedup(tmp_path: str, sorted_path: str):
    """
    Use Unix external sort (disk-based merge sort) to sort numerically
    and deduplicate.  --buffer-size 1G limits RAM used by sort.
    The temporary directory is set to the same folder as the output
    so sort spills land near the output file.
    """
    tmp_dir = os.path.dirname(os.path.abspath(sorted_path))
    print(f"  Sorting and deduplicating with Unix sort …", flush=True)
    print(f"  (sort temp dir: {tmp_dir})", flush=True)
    cmd = [
        "sort",
        "-u",                          # deduplicate identical lines
        "-k1,1n", "-k2,2n", "-k3,3n", # numeric sort on each field
        "--buffer-size=1G",            # cap RAM usage
        f"--temporary-directory={tmp_dir}",
        "-o", sorted_path,
        tmp_path,
    ]
    subprocess.run(cmd, check=True)
    print("  Sort complete.", flush=True)


# ── Assign offsets while streaming — no full list in RAM ──────────────────────

def assign_offsets_streaming(sorted_path: str, out_path: str) -> int:
    """
    Read the sorted deduplicated file line by line and append the offset.
    Writes directly to out_path.  RAM usage: O(1).
    Returns the total number of unique triples written.
    """
    print(f"  Assigning offsets and writing output …", flush=True)
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
    return count


# ── Lightweight validation (checks first/last rows + monotonicity sample) ─────

def validate_output(out_path: str, n: int):
    """
    Validate without loading the whole file:
      - first and last few rows
      - spot-check that offset matches line number
      - confirm sort order on first 100k rows
    """
    print("  Validating output (streaming) …", flush=True)
    prev = None
    with open(out_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.split()
            u, v, w, off = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            assert off == i, f"Row {i}: offset={off} expected {i}"
            curr = (u, v, w)
            if prev is not None:
                assert prev < curr, (
                    f"Row {i}: {curr} not > previous {prev} — sort violated"
                )
            prev = curr
            if i >= 100_000:   # spot-check first 100k rows is enough
                break
    print("  Validation PASSED (first 100k rows checked).", flush=True)


# ── Safety-check summary ───────────────────────────────────────────────────────

THRESHOLDS = [1_000_000, 2_500_000]

def print_summary(edge_count, raw_count, unique_count, out_path, undirected):
    n = unique_count
    print(f"\n{'='*64}")
    print(f"  Graph mode          : "
          f"{'UNDIRECTED (edges mirrored)' if undirected else 'DIRECTED (as-is)'}")
    print(f"  Input edges         : {edge_count:,}"
          f"  ({'after mirroring' if undirected else 'original'})")
    print(f"  Raw path triples    : {raw_count:,}  (before dedup)")
    print(f"  Unique triples      : {n:,}")
    print(f"  Output file         : {out_path}")
    print(f"{'='*64}")

    if n > 10_000_000:
        print(f"\n  *** NOTE: {n:,} triples is large (>10M).")
        print(f"      RSMI build will be slow. Consider using --limit.")

    print(f"\n  Scalability support:")
    for t in THRESHOLDS:
        status = "YES" if n >= t else f"NO  (only {n:,} available)"
        print(f"    {t:>12,} tuples  →  {status}")
    print(f"    {'FULL':>12}           →  YES  ({n:,} tuples)")

    # Print first/last 5 rows from the output file
    print("\n  First 5 rows (SourceID Hop1_ID Hop2_ID Offset):")
    with open(out_path) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            p = line.split()
            print(f"    {p[0]:>8}  {p[1]:>8}  {p[2]:>8}  {p[3]:>10}")

    print("\n  Last 5 rows:")
    # tail the file without reading all of it
    result = subprocess.run(["tail", "-5", out_path], capture_output=True, text=True)
    for line in result.stdout.strip().splitlines():
        p = line.split()
        print(f"    {p[0]:>8}  {p[1]:>8}  {p[2]:>8}  {p[3]:>10}")

    print(f"\n  Offset range : 0 .. {n - 1:,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate RSMI-compatible 2-hop triples from a SNAP edge list."
    )
    parser.add_argument("--input",  required=True,  help="Input SNAP edge file")
    parser.add_argument("--output", required=True,  help="Output triples file")
    graph_type = parser.add_mutually_exclusive_group(required=True)
    graph_type.add_argument("--directed",   action="store_true",
                            help="Treat graph as directed (use edges as-is)")
    graph_type.add_argument("--undirected", action="store_true",
                            help="Treat graph as undirected (mirror every edge)")
    args = parser.parse_args()

    undirected = args.undirected

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: input file not found: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Temp files live next to the output file
    out_dir   = os.path.dirname(os.path.abspath(args.output))
    tmp_raw    = os.path.join(out_dir, "_tmp_raw_triples.txt")
    tmp_sorted = os.path.join(out_dir, "_tmp_sorted_triples.txt")

    try:
        # ── Step 1: Read edges ──────────────────────────────────────────────
        print(f"[1/5] Reading edges from : {args.input}", flush=True)
        edges = read_edges(args.input, undirected)
        edge_count = len(edges)
        print(f"      Edges loaded       : {edge_count:,}"
              f"  ({'after mirroring' if undirected else 'directed, as-is'})",
              flush=True)

        # ── Step 2: Build adjacency list ────────────────────────────────────
        print(f"[2/5] Building adjacency list …", flush=True)
        adj = build_adjacency(edges)
        print(f"      Unique source nodes: {len(adj):,}", flush=True)

        # ── Step 3: Stream triples to temp file ─────────────────────────────
        print(f"[3/5] Streaming raw triples to disk …", flush=True)
        raw_count = generate_triples_to_file(adj, tmp_raw)

        # Free adjacency list memory before sort
        del adj
        del edges

        if raw_count == 0:
            sys.exit("ERROR: no triples generated. Check input file.")

        # ── Step 4: External sort + dedup ───────────────────────────────────
        print(f"[4/5] External sort + dedup …", flush=True)
        sort_and_dedup(tmp_raw, tmp_sorted)
        os.remove(tmp_raw)   # free disk space

        # ── Step 5: Assign offsets ──────────────────────────────────────────
        print(f"[5/5] Assigning offsets …", flush=True)
        unique_count = assign_offsets_streaming(tmp_sorted, args.output)
        os.remove(tmp_sorted)

        # ── Safety checks ───────────────────────────────────────────────────
        for t in THRESHOLDS:
            if unique_count < t:
                print(f"\n  *** WARNING: only {unique_count:,} unique triples — "
                      f"below the {t:,} threshold.")

        # ── Validate ────────────────────────────────────────────────────────
        validate_output(args.output, unique_count)

        # ── Summary ─────────────────────────────────────────────────────────
        print_summary(edge_count, raw_count, unique_count, args.output, undirected)

    finally:
        # Clean up temp files even if script crashes
        for p in [tmp_raw, tmp_sorted]:
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()
