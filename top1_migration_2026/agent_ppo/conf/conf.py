#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Compact PPO configuration for the first debuggable 2026 1v1 feature version.
"""


class GameConfig:
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 6.0,
        "money": 0.004,
        "exp": 0.003,
        "ep_rate": 0.1,
        "death": -2.5,
        "kill": 5.0,
        "last_hit": 0.45,
        "forward": 0.02,
        "danger": -0.7,
        "idle": -0.02,
        "attack_hit": 0.14,
        "skill_hit": 0.30,
        "skill_misuse": -0.03,
        "monster_last_hit": 2.2,
        "hero_lane_advantage": 0.12,
        "monster_contest": 0.22,
        "tower_chase_risk": -0.6,
        "attack_power": 0.0,
        "attack_speed": 0.0,
        "tower_attack": 0.25,
        "safe_push": 0.08,
        "safe_tower_damage": 1.5,
        "finish_push": 0.5,
        "post_fight_recall": 0.2,
        "recall_use": 0.8,
        "rune_approach": 0.08,
        "rune_pickup": 1.2,
        "enemy_wave_overextend": -0.25,
        "lane_anchor": 0.08,
        "grass_ambush": 0.12,
        "stutter_step": 0.12,
        "fight_risk": -0.15,
    }
    REMOVE_FORWARD_AFTER = 1200
    TIME_SCALE_ARG = 8000
    REWARD_WITHOUT_TIME_SCALE = {
        "death",
        "kill",
        "danger",
        "idle",
        "attack_hit",
        "skill_hit",
        "skill_misuse",
        "monster_last_hit",
        "hero_lane_advantage",
        "monster_contest",
        "tower_chase_risk",
        "attack_power",
        "attack_speed",
        "tower_attack",
        "safe_push",
        "safe_tower_damage",
        "finish_push",
        "post_fight_recall",
        "recall_use",
        "rune_approach",
        "rune_pickup",
        "enemy_wave_overextend",
        "lane_anchor",
        "grass_ambush",
        "stutter_step",
        "fight_risk",
    }
    MODEL_SAVE_INTERVAL = 1800
    CAMP_HEROES = [112, 133]

    HERO_SUMMONER_SKILL = {
        112: (80102, 80110, 80103),
        133: (80102, 80110, 80103),
    }
    LOW_HP_HEAL_THRESHOLD = 0.75
    LOW_HP_RECOVER_THRESHOLD = 0.85
    LOW_EP_RECOVER_THRESHOLD = 0.50
    FORCE_LEAVE_BASE_HP_THRESHOLD = 0.80
    POST_FIGHT_RECALL_HP_THRESHOLD_EARLY = 0.50
    POST_FIGHT_RECALL_HP_THRESHOLD_LATE = 0.30
    POST_FIGHT_PUSH_HP_THRESHOLD_LATE = 0.05
    POST_FIGHT_PUSH_LEVEL = 9
    POST_FIGHT_PUSH_MONEY = 6000
    POST_FIGHT_PUSH_PHY_VAMP = 0.10
    RECALL_BUTTON = 9
    RECALL_NEAR_TOWER_RADIUS = 9500.0
    ENEMY_RETURN_HOLD_HP_THRESHOLD = 0.85
    BASE_RADIUS = 14000.0
    MID_LANE_TARGET = (0.0, 0.0)
    CHOSEN_SUMMONER_BUTTON = 8
    RECOVER_BUTTON = 7
    SELF_TARGET_INDEX = 2
    PASSIVE_BUTTONS = {0, 1, 7, 8, 9, 10, 11}
    BERSERK_SKILL_ID = 80110
    HEAL_SUMMONER_SKILL_ID = 80102
    STUN_SUMMONER_SKILL_ID = 80103
    STUN_SUMMONER_RANGE = 6000.0
    FIGHT_RETREAT_HP_GAP = 0.18
    FIGHT_RETREAT_LOW_HP = 0.35
    FIGHT_WITHOUT_SUMMONER_MIN_HP_GAP = 0.10
    TOWER_EARLY_PUSH_MONEY = 5500
    TOWER_FULL_PUSH_MONEY = 6000
    RUNE_LOW_HP_THRESHOLD = 0.80
    RUNE_SEARCH_RADIUS = 25000.0
    RUNE_PICKUP_RADIUS = 6000.0
    GRASS_AMBUSH_FRAMES = 1800


class DimConfig:
    DIM_OF_FEATURE = [320]


class Config:
    NETWORK_NAME = "network"
    LSTM_DROPOUT = 0
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DIM_PUBLIC = 256

    FEATURE_DIM = DimConfig.DIM_OF_FEATURE[0]
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    LEGAL_ACTION_FLAT_DIM = sum(LABEL_SIZE_LIST)

    DATA_SPLIT_SHAPE = [
        FEATURE_DIM + LEGAL_ACTION_FLAT_DIM,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_FLAT_DIM,)]

    INIT_LEARNING_RATE_START = 1e-4
    TARGET_LR = 1e-5
    TARGET_STEP = 10000
    BETA_START = 0.01
    LOG_EPSILON = 1e-6
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]

    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001
    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(FEATURE_DIM + LEGAL_ACTION_FLAT_DIM) * LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [12 * LSTM_TIME_STEPS],
        [16 * LSTM_TIME_STEPS],
        [16 * LSTM_TIME_STEPS],
        [16 * LSTM_TIME_STEPS],
        [16 * LSTM_TIME_STEPS],
        [9 * LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_UNIT_SIZE],
        [LSTM_UNIT_SIZE],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95
    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
