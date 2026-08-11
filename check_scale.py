"""
Standalone check: confirms the scale-mismatch diagnosis and
shows what auto_log2_transform() does to each dataset.
Run this BEFORE touching app.py's pipeline, just to sanity
check the fix on your actual data files.

Usage:
    python check_scale.py
"""

import numpy as np
import pandas as pd


def parse_series_matrix_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            start_idx = i + 1
            break

    headers = [x.strip('"') for x in lines[start_idx].split("\t")]
    sample_ids = headers[1:]
    rows = []

    for line in lines[start_idx + 1:]:
        if line.startswith("!series_matrix_table_end"):
            break
        fields = [x.strip('"') for x in line.split("\t")]
        if len(fields) == len(headers):
            rows.append(fields)

    expr_data = np.array([r[1:] for r in rows], dtype=float)
    return pd.DataFrame(expr_data.T, index=sample_ids,
                         columns=[r[0] for r in rows])


def describe(name, arr):
    arr = arr[~np.isnan(arr)]
    print(f"{name:12s}  min={arr.min():10.2f}  "
          f"median={np.median(arr):10.2f}  "
          f"max={arr.max():12.2f}")


def auto_log2_transform(df, name=""):
    vals = df.values.astype(float)
    max_val = np.nanmax(vals)
    if max_val > 100:
        print(f"[{name}] -> raw/linear scale, applying log2(x+1)")
        return np.log2(df.clip(lower=0) + 1)
    print(f"[{name}] -> already log2-scale, left as-is")
    return df


if __name__ == "__main__":
    files = {
        "GSE76297": "data/GSE76297_series_matrix.txt",
        "GSE132305": "data/GSE132305_series_matrix.txt",
        "GSE32225": "data/GSE32225_series_matrix.txt",
    }

    print("=" * 70)
    print("BEFORE log2 fix (raw values as parsed from series matrix)")
    print("=" * 70)
    dfs = {}
    for name, path in files.items():
        df = parse_series_matrix_raw(path)
        dfs[name] = df
        describe(name, df.values)

    print()
    print("=" * 70)
    print("AFTER auto_log2_transform()")
    print("=" * 70)
    for name, df in dfs.items():
        fixed = auto_log2_transform(df, name)
        describe(name, fixed.values)
