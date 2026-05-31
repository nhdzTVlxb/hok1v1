#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Single source of truth for the 2026 handcrafted observation layout."""

from collections import OrderedDict


TARGET_NONE = 0
TARGET_ENEMY_HERO = 1
TARGET_SELF_HERO = 2
TARGET_ENEMY_SOLDIER_START = 3
TARGET_ENEMY_SOLDIER_COUNT = 4
TARGET_ENEMY_TOWER = 7
TARGET_MONSTER = 8

TARGET_ENTITY_COUNT = 9
TARGET_ENTITY_DIM = 32
TARGET_FEATURE_DIM = TARGET_ENTITY_COUNT * TARGET_ENTITY_DIM

FEATURE_BLOCK_SIZES = OrderedDict(
    [
        ("hero_frd", 24),
        ("hero_emy", 18),
        ("skill", 21),
        ("tower", 16),
        ("soldier_summary", 24),
        ("monster", 12),
        ("cake", 8),
        ("unit_slots", 56),
        ("projectile_slots", 70),
        ("combat_risk", 20),
        ("objective", 20),
        ("global", 12),
        ("status_macro", 64),
        ("strategic_macro", 128),
        ("legal_summary", 15),
        ("ability_control", 64),
        ("lane_entity", 256),
        ("tower_wave", 96),
        ("projectile_detail", 160),
        ("position_bucket", 64),
        ("buff_mark_detail", 64),
        ("economy_phase", 36),
        ("hero_attr_detail", 64),
        ("skill_detail", 112),
    ]
)


def _build_slices(block_sizes):
    start = 0
    slices = {}
    for name, size in block_sizes.items():
        end = start + size
        slices[name] = (start, end)
        start = end
    return slices, start


FEATURE_BLOCK_SLICES, MAIN_FEATURE_DIM = _build_slices(FEATURE_BLOCK_SIZES)
FEATURE_DIM = MAIN_FEATURE_DIM + TARGET_FEATURE_DIM
TARGET_FEATURE_SLICE = (MAIN_FEATURE_DIM, FEATURE_DIM)


def block_slice(name):
    return FEATURE_BLOCK_SLICES[name]


def assemble_feature_blocks(blocks):
    """Concatenate named blocks and fail loudly on any layout drift."""
    block_map = OrderedDict(blocks)
    expected_names = list(FEATURE_BLOCK_SIZES.keys())
    actual_names = list(block_map.keys())
    if actual_names != expected_names:
        raise ValueError(f"Feature block order/name mismatch: expected {expected_names}, got {actual_names}")

    feature = []
    for name, expected_size in FEATURE_BLOCK_SIZES.items():
        values = list(block_map[name] or [])
        actual_size = len(values)
        if actual_size != expected_size:
            raise ValueError(f"Feature block {name!r} expected {expected_size} dims, got {actual_size}")
        feature.extend(values)
    if len(feature) != MAIN_FEATURE_DIM:
        raise ValueError(f"Main feature expected {MAIN_FEATURE_DIM} dims, got {len(feature)}")
    return feature


def target_slot_name(index):
    if index == TARGET_NONE:
        return "none"
    if index == TARGET_ENEMY_HERO:
        return "enemy_hero"
    if index == TARGET_SELF_HERO:
        return "self_hero"
    if TARGET_ENEMY_SOLDIER_START <= index < TARGET_ENEMY_SOLDIER_START + TARGET_ENEMY_SOLDIER_COUNT:
        return f"enemy_soldier_{index - TARGET_ENEMY_SOLDIER_START}"
    if index == TARGET_ENEMY_TOWER:
        return "enemy_tower"
    if index == TARGET_MONSTER:
        return "monster"
    return "unknown"
