"""
Diagnostic script for CCA classification collapse issue.
Run this in the SAME directory structure as your app.py
(i.e. it expects ./data/GSE*_series_matrix.txt to exist).

This does NOT modify your pipeline. It only inspects metadata
and prints a report. Paste the full printed output back to Claude.

Usage:
    python diagnose_collapse.py
"""

import os
import re
import sys
from collections import Counter

DATA_DIR = "data"

FILES = {
    "GSE76297": "GSE76297_series_matrix.txt",
    "GSE132305": "GSE132305_series_matrix.txt",
    "GSE32225": "GSE32225_series_matrix.txt",
}


def read_meta_lines(path):
    """Read only metadata lines (!Sample_*), skip the giant expression table."""
    meta = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_"):
                parts = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")]
                key = parts[0]
                meta.setdefault(key, []).append(parts[1:])
    return meta


def extract_label_from_title(title):
    t = title.lower().strip()
    if t.endswith("_bd"):
        return 0
    elif t.endswith("_ecca"):
        return 1
    elif t.startswith("ctrl"):
        return 0
    elif t.startswith("ccbcn") or t.startswith("ccm") or t.startswith("ccny"):
        return 1
    elif any(k in t for k in ["normal", "control", "benign"]):
        return 0
    elif any(k in t for k in ["tumor", "cca", "cancer"]):
        return 1
    else:
        return -1


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: '{DATA_DIR}' directory not found. Run this script from "
              f"the same folder where app.py expects ./data/ to exist.")
        sys.exit(1)

    for gse_name, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        print("=" * 70)
        print(f"{gse_name}  ({fname})")
        print("=" * 70)

        if not os.path.exists(path):
            print(f"  !! FILE NOT FOUND: {path}")
            continue

        meta = read_meta_lines(path)

        geo_accession = meta.get("!Sample_geo_accession", [[]])[0]
        titles = meta.get("!Sample_title", [[]])[0]

        n = len(geo_accession) if geo_accession else len(titles)
        print(f"  N samples: {n}")

        # Labels from title (same logic as app.py)
        labels = [extract_label_from_title(t) for t in titles] if titles else []
        label_counts = Counter(labels)
        print(f"  Label counts from title-parsing: {dict(label_counts)}  "
              f"(1=CCA, 0=normal, -1=unlabeled/dropped)")

        # Print every !Sample_* field name available, so we know what metadata exists
        print(f"\n  Available metadata fields ({len(meta)}):")
        for key in sorted(meta.keys()):
            print(f"    {key}")

        # For fields likely to carry batch/technical info, print value distribution
        # split by label (0 vs 1) to check confounding
        interesting_keys = [
            k for k in meta.keys()
            if any(kw in k.lower() for kw in [
                "platform", "submission", "date", "characteristics",
                "source_name", "extract_protocol", "hyb_protocol",
                "scan_protocol", "label_protocol", "growth_protocol",
                "supplementary_file", "series_id", "contact"
            ])
        ]

        print(f"\n  Checking {len(interesting_keys)} potentially batch-related "
              f"fields for confounding with label:")

        for key in interesting_keys:
            values = meta[key]
            # values is a list of lists (each row = one field occurrence across samples)
            for row_i, row in enumerate(values):
                if len(row) != n:
                    continue
                # group values by label
                by_label = {0: [], 1: []}
                for v, lab in zip(row, labels):
                    if lab in (0, 1):
                        by_label[lab].append(v)

                uniq0 = set(by_label[0])
                uniq1 = set(by_label[1])
                overlap = uniq0 & uniq1

                # Flag if this field perfectly separates the two classes
                # (i.e. no overlap in values between normal and CCA samples)
                if uniq0 and uniq1 and not overlap:
                    print(f"\n    ⚠️  PERFECT SEPARATION on '{key}' (row {row_i}):")
                    print(f"        normal (n={len(by_label[0])}) values: "
                          f"{sorted(uniq0)[:5]}{'...' if len(uniq0) > 5 else ''}")
                    print(f"        CCA    (n={len(by_label[1])}) values: "
                          f"{sorted(uniq1)[:5]}{'...' if len(uniq1) > 5 else ''}")
                elif uniq0 or uniq1:
                    print(f"\n    '{key}' (row {row_i}): "
                          f"normal has {len(uniq0)} unique values, "
                          f"CCA has {len(uniq1)} unique values, "
                          f"{len(overlap)} shared")

        print()

    print("=" * 70)
    print("DONE. Paste everything above back to Claude.")
    print("=" * 70)


if __name__ == "__main__":
    main()
