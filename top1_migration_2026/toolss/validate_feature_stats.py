#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Offline feature distribution check for dumped 2026 observations.

Usage:
  python3 top1_migration_2026/toolss/validate_feature_stats.py raw2/*.json raw/*.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_ppo.feature.top1_feature_builder import Top1FeatureBuilder  # noqa: E402


def load_observation(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "observation" in data:
        return data["observation"]
    if isinstance(data, dict) and "frame_state" in data:
        return data
    raise ValueError(f"{path}: cannot find observation/frame_state")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="dumped json observation files")
    parser.add_argument("--saturation", type=float, default=0.05, help="warn when >= this ratio is clipped at +/-1")
    args = parser.parse_args()

    builder = Top1FeatureBuilder()
    rows = []
    bad_files = []
    for name in args.files:
        path = pathlib.Path(name)
        try:
            rows.append(builder.build_observation(load_observation(path)))
        except Exception as exc:
            bad_files.append((str(path), str(exc)))

    if not rows:
        print("no valid observations")
        return 2

    arr = np.vstack(rows)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    stds = arr.std(axis=0)
    saturation = ((np.isclose(arr, 1.0) | np.isclose(arr, -1.0)).sum(axis=0) / arr.shape[0])

    out_of_range = np.where((mins < -1.05) | (maxs > 1.05))[0]
    flat = np.where(stds <= 1e-8)[0]
    saturated = np.where(saturation >= args.saturation)[0]

    print(f"files_ok={len(rows)} files_bad={len(bad_files)} feature_dim={arr.shape[1]}")
    print(f"out_of_range_cols={out_of_range[:80].tolist()} count={len(out_of_range)}")
    print(f"flat_cols={flat[:80].tolist()} count={len(flat)}")
    print(f"saturated_cols={saturated[:80].tolist()} count={len(saturated)}")
    if bad_files:
        print("bad_files:")
        for path, err in bad_files[:20]:
            print(f"  {path}: {err}")
    return 1 if len(out_of_range) else 0


if __name__ == "__main__":
    raise SystemExit(main())
