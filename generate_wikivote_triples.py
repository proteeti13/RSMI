"""
generate_wikivote_triples.py
----------------------------
Converts a SNAP-format directed edge list into 2-hop path triples.

Each output row represents a valid 2-hop path:
    SourceID -> Hop1_ID -> Hop2_ID

Output format (4 space-separated columns, lexicographically sorted):
    SourceID Hop1_ID Hop2_ID Offset

Usage:
    python3 generate_wikivote_triples.py Wiki-Vote.txt wiki_vote_triples.txt
"""

import sys
import os
from collections import defaultdict


def read_edges(path):
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
    return edges


def build_adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
    return adj


def generate_triples(adj):
    triples = set()
    for u, neighbors_u in adj.items():
        for v in neighbors_u:
            if v in adj:
                for w in adj[v]:
                    triples.add((u, v, w))
    return triples


def assign_offsets(triples):
    sorted_triples = sorted(triples)   # lex order: u -> v -> w
    return [(u, v, w, i) for i, (u, v, w) in enumerate(sorted_triples)]


def validate(records):
    for i, (u, v, w, offset) in enumerate(records):
        assert offset == i, (
            f"Row {i}: offset={offset} but expected {i}. "
            f"Triple=({u},{v},{w})"
        )
        if i > 0:
            prev = records[i - 1][:3]
            curr = (u, v, w)
            assert prev < curr, (
                f"Row {i}: triple {curr} is not strictly greater than "
                f"previous {prev}. Lex sort violated."
            )


def write_output(records, out_path):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        for u, v, w, offset in records:
            f.write(f"{u} {v} {w} {offset}\n")


def print_summary(edges, raw_count, records, out_path):
    n = len(records)
    print(f"\n{'='*60}")
    print(f"Input edges read    : {len(edges)}")
    print(f"Raw triples         : {raw_count}")
    print(f"Unique triples      : {n}")
    print(f"Output file         : {out_path}")
    print(f"{'='*60}")

    print("\nFirst 10 rows (SourceID Hop1_ID Hop2_ID Offset):")
    for u, v, w, offset in records[:10]:
        print(f"  {u:>7}  {v:>7}  {w:>7}  {offset:>8}")

    print("\nLast 10 rows:")
    for u, v, w, offset in records[-10:]:
        print(f"  {u:>7}  {v:>7}  {w:>7}  {offset:>8}")

    print(f"\n  Offset range : 0 .. {n - 1}")
    print(f"  Validation   : PASSED")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 generate_wikivote_triples.py <input.txt> <output.txt>")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]

    print(f"Reading edges from: {in_path}")
    edges = read_edges(in_path)
    print(f"Edges loaded: {len(edges)}")

    adj = build_adjacency(edges)
    print("Adjacency list built.")

    print("Generating 2-hop triples...")
    raw_triples = generate_triples(adj)
    raw_count = len(raw_triples)        # already unique (set)
    print(f"Unique triples generated: {raw_count}")

    print("Sorting and assigning offsets...")
    records = assign_offsets(raw_triples)

    print("Validating...")
    validate(records)

    print(f"Writing to: {out_path}")
    write_output(records, out_path)

    print_summary(edges, raw_count, records, out_path)


if __name__ == "__main__":
    main()
