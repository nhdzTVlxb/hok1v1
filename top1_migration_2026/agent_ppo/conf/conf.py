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
        "death": -2.0,
        "kill": 2.0,
        "last_hit": 0.4,
        "forward": 0.02,
        "danger": -0.2,
        "idle": -0.02,
        "tower_attack": 0.25,
        "safe_push": 0.04,
    }
    REMOVE_FORWARD_AFTER = 1200
    TIME_SCALE_ARG = 8000
    REWARD_WITHOUT_TIME_SCALE = {"death", "kill", "danger", "idle", "tower_attack", "safe_push"}
    MODEL_SAVE_INTERVAL = 1800
    CAMP_HEROES = [112, 133]

    HERO_SUMMONER_SKILL = {
        112: 80102,
        133: 80102,
    }
    LOW_HP_HEAL_THRESHOLD = 0.75
    CHOSEN_SUMMONER_BUTTON = 8
    SELF_TARGET_INDEX = 2
    PASSIVE_BUTTONS = {0, 1, 7, 8, 9, 10, 11}


class DimConfig:
    DIM_OF_FEATURE = [128]


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
