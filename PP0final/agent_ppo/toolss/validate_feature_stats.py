#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Validate the 1280-dim 2024-baseline feature layout."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_ppo.conf.conf import Config  # noqa: E402
from agent_ppo.feature.layout import FEATURE_BLOCK_SLICES, FEATURE_DIM, LEGACY_PREFIX_SIZES, block_slice  # noqa: E402
from agent_ppo.feature.top1_feature_builder import Top1FeatureBuilder  # noqa: E402


MODEL_BRANCHES = {
    "hero_frd": 384,
    "hero_emy": 384,
    "hero_main": 32,
    "soldier_frd": 128,
    "soldier_emy": 128,
    "organ_frd": 64,
    "organ_emy": 64,
    "global": 96,
}


def load_observation(path):
    payload = json.loads(Path(path).read_text())
    return payload.get("observation", payload)


def validate_layout_contract():
    assert FEATURE_DIM == Config.FEATURE_DIM, (FEATURE_DIM, Config.FEATURE_DIM)
    assert sum(MODEL_BRANCHES.values()) == Config.FEATURE_DIM
    assert Config.SERI_VEC_SPLIT_SHAPE == [(Config.FEATURE_DIM,), (Config.LEGAL_ACTION_DIM,)]
    assert LEGACY_PREFIX_SIZES == {
        "hero_frd": 235,
        "hero_emy": 235,
        "hero_main": 14,
        "soldier_slot": 18,
        "organ_slot": 18,
        "global": 25,
    }
    for name, expected_size in MODEL_BRANCHES.items():
        start, end = block_slice(name)
        assert end - start == expected_size, (name, start, end, expected_size)


def validate_observation(path):
    feature = Top1FeatureBuilder().build_observation(load_observation(path))
    assert len(feature) == FEATURE_DIM, (path, len(feature), FEATURE_DIM)
    for name, (_start, end) in FEATURE_BLOCK_SLICES.items():
        assert end <= FEATURE_DIM, (name, end, FEATURE_DIM)
    return len(feature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="*", help="raw observation JSON files to validate")
    args = parser.parse_args()

    validate_layout_contract()
    for json_file in args.json_files:
        validate_observation(json_file)

    print(f"feature layout ok: feature={FEATURE_DIM}, legal={Config.LEGAL_ACTION_DIM}, seri={Config.SERI_VEC_DIM}")
    if args.json_files:
        print(f"validated observations: {len(args.json_files)}")


if __name__ == "__main__":
    main()
