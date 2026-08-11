"""
Quick check: print raw !Sample_title values from GSE76297 series matrix,
so we can confirm the EXACT string format before fixing extract_labels().

Usage:
    python check_titles.py
"""

import os

DATA_DIR = "data"
FNAME = "GSE76297_series_matrix.txt"


def main():
    path = os.path.join(DATA_DIR, FNAME)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                parts = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")]
                titles = parts[1:]
                break
            if line.startswith("!series_matrix_table_begin"):
                print("Reached table without finding !Sample_title -- unexpected.")
                return

    print(f"Total titles found: {len(titles)}")
    print("\nFirst 20 raw titles:")
    for t in titles[:20]:
        print(f"  {repr(t)}")

    print("\nAll UNIQUE title patterns containing 'tumor' (case-insensitive), "
          "deduplicated by replacing digits/IDs with '#':")
    import re
    seen = set()
    for t in titles:
        if "tumor" in t.lower():
            generic = re.sub(r"\d+", "#", t)
            if generic not in seen:
                seen.add(generic)
                print(f"  pattern: {generic!r}   (example: {t!r})")

    # Count how many contain "non-tumor" / "non tumor" / "nontumor" in some form
    lowered = [t.lower() for t in titles]
    non_tumor_variants = ["non-tumor", "non tumor", "nontumor", "non_tumor"]
    for variant in non_tumor_variants:
        count = sum(1 for t in lowered if variant in t)
        print(f"\nCount containing {variant!r}: {count}")

    tumor_total = sum(1 for t in lowered if "tumor" in t)
    print(f"\nTotal containing 'tumor' (any form): {tumor_total}")
    print(f"Total titles: {len(titles)}")


if __name__ == "__main__":
    main()
