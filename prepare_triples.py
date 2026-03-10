"""
prepare_triples.py
------------------
Generates the input dataset for a 3D learned index experiment.

Each record maps a triple to its rank in lexicographic order:

    f(SourceID, Hop1_ID, Hop2_ID) -> Offset

Output format (4 space-separated columns per line):

    SourceID Hop1_ID Hop2_ID Offset

Sort order:
    ORDER BY SourceID ASC, Hop1_ID ASC, Hop2_ID ASC

Offset:
    0 for the first triple after sorting
    1 for the second
    ...
    N-1 for the last

Example:
    (1,2,3) -> offset 0  ->  line: "1 2 3 0"
    (1,2,7) -> offset 1  ->  line: "1 2 7 1"
    (1,5,2) -> offset 2  ->  line: "1 5 2 2"
    (2,1,1) -> offset 3  ->  line: "2 1 1 3"

Duplicate policy:
    This script generates UNIQUE triples only (using a set).
    If your real data contains duplicates, see the note at the bottom.

Usage:
    python3 prepare_triples.py
"""

import random
import os


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def generate_unique_triples(n, max_id, rng):
    """
    Draw n unique (SourceID, Hop1_ID, Hop2_ID) triples uniformly at random.
    Uniqueness is guaranteed via a set. Raises if the ID space is too small.
    """
    id_space = max_id ** 3
    if id_space < n:
        raise ValueError(
            f"Cannot draw {n} unique triples from a space of {id_space} "
            f"(max_id={max_id}). Increase max_id."
        )

    triples = set()
    while len(triples) < n:
        src = rng.randint(1, max_id)
        h1  = rng.randint(1, max_id)
        h2  = rng.randint(1, max_id)
        triples.add((src, h1, h2))

    return triples


def assign_offsets(triples):
    """
    Sort triples lexicographically and assign offset = row index.

    Returns a list of (SourceID, Hop1_ID, Hop2_ID, Offset) tuples.
    Python's default tuple sort is lexicographic, so no key needed.
    """
    sorted_triples = sorted(triples)          # lex order: SourceID -> Hop1_ID -> Hop2_ID
    return [(src, h1, h2, offset)
            for offset, (src, h1, h2) in enumerate(sorted_triples)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(records):
    """
    Check two invariants:
      1. Triples are strictly lexicographically sorted (no ties because unique).
      2. Each offset == its row position.

    Raises AssertionError with a descriptive message on failure.
    """
    for i, (src, h1, h2, offset) in enumerate(records):

        # Invariant 1: offset == row position
        assert offset == i, (
            f"Row {i}: offset field is {offset} but expected {i}. "
            f"Triple = ({src}, {h1}, {h2})"
        )

        # Invariant 2: strictly lex-sorted relative to previous row
        if i > 0:
            prev_src, prev_h1, prev_h2, _ = records[i - 1]
            prev = (prev_src, prev_h1, prev_h2)
            curr = (src, h1, h2)
            assert prev < curr, (
                f"Row {i}: triple {curr} is NOT strictly greater than "
                f"previous triple {prev}. Lex sort violated."
            )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_dataset(records, out_path):
    """Write records to out_path, one record per line, 4 space-separated columns."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        for src, h1, h2, offset in records:
            f.write(f"{src} {h1} {h2} {offset}\n")


def print_summary(records, out_path):
    n = len(records)
    print(f"\n{'='*60}")
    print(f"Dataset: {out_path}")
    print(f"Records: {n}")
    print(f"{'='*60}")

    print("\nFirst 5 rows (SourceID Hop1_ID Hop2_ID Offset):")
    for src, h1, h2, offset in records[:5]:
        print(f"  {src:>6}  {h1:>6}  {h2:>6}  {offset:>6}")

    print("\nLast 5 rows:")
    for src, h1, h2, offset in records[-5:]:
        print(f"  {src:>6}  {h1:>6}  {h2:>6}  {offset:>6}")

    print(f"\n  Offset range : 0 .. {n-1}")
    print(f"  First triple : ({records[0][0]}, {records[0][1]}, {records[0][2]})")
    print(f"  Last triple  : ({records[-1][0]}, {records[-1][1]}, {records[-1][2]})")
    print(f"  Validation   : PASSED")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def make_dataset(n, max_id, out_path, rng):
    print(f"\nGenerating {n} unique triples (max_id={max_id}) ...")

    raw     = generate_unique_triples(n, max_id, rng)
    records = assign_offsets(raw)      # sort + number
    validate(records)                  # assert invariants
    write_dataset(records, out_path)
    print_summary(records, out_path)


if __name__ == "__main__":
    rng = random.Random(42)            # fixed seed for reproducibility

    make_dataset(
        n       = 10_000,
        max_id  = 5_000,               # SourceID, Hop1_ID, Hop2_ID each in [1, 5000]
        out_path= "datasets/triples_10k.txt",
        rng     = rng,
    )

    make_dataset(
        n       = 50_000,
        max_id  = 15_000,              # larger space keeps collision rate low
        out_path= "datasets/triples_50k.txt",
        rng     = rng,
    )

    print("\nBoth datasets written and validated.\n")


# ---------------------------------------------------------------------------
# NOTE ON DUPLICATES
# ---------------------------------------------------------------------------
# This script enforces unique triples (set-based generation).
# If your real graph data contains duplicate 2-hop paths:
#
#   Option A (recommended): deduplicate before indexing.
#       The learned index maps one key -> one offset, so duplicates are
#       ambiguous. Deduplication is semantically correct.
#
#   Option B: allow duplicates, assign them consecutive offsets.
#       Sort the list (not set) and enumerate. Ties are broken arbitrarily
#       by Python's stable sort. Two identical triples get adjacent offsets.
#       The validation step above must be relaxed to allow prev <= curr
#       instead of prev < curr.
#
# Example (Option B, one extra line):
#   sorted_triples = sorted(list(triples_with_dupes))   # list, not set
#   records = [(s,h1,h2,i) for i,(s,h1,h2) in enumerate(sorted_triples)]
# ---------------------------------------------------------------------------
