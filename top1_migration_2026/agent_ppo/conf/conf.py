#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Top1-style PPO configuration adapted for the 2026 1v1 task.
"""


class GameConfig:
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 8.0,
        "money": 0.004,
        "exp": 0.004,
        "ep_rate": 0.75,
        "death": -1.0,
        "kill": -0.6,
        "last_hit": 0.5,
        "forward": 0.01,
    }
    REMOVE_FORWARD_AFTER = 1000
    TIME_SCALE_ARG = 8000
    REWARD_WITHOUT_TIME_SCALE = set()
    MODEL_SAVE_INTERVAL = 1800
    CAMP_HEROES = [112, 133]

    # Deterministic summoner skill policy used by Agent.init_config.
    # 80102 = Heal. Pair this with Agent's low-hp action override.
    HERO_SUMMONER_SKILL = {
        112: 80102,
        133: 80102,
    }
    LOW_HP_HEAL_THRESHOLD = 0.75
    CHOSEN_SUMMONER_BUTTON = 8
    SELF_TARGET_INDEX = 2
    PASSIVE_BUTTONS = {0, 1, 7, 8, 9, 10, 11}


class Args:
    RELATIVE_DISTANCE_UNIT_SIZE = 600
    RELATIVE_DISTANCE_MAX_SIZE = 24600
    DIM_RELATIVE_DISTANCE = (RELATIVE_DISTANCE_MAX_SIZE // RELATIVE_DISTANCE_UNIT_SIZE + 2) * 2 + 1
    WHOLE_DISTANCE_UNIT_SIZE = 5000
    WHOLE_DISTANCE_MAX_SIZE = int(9e4)
    DIM_WHOLE_DISTANCE = (WHOLE_DISTANCE_MAX_SIZE // WHOLE_DISTANCE_UNIT_SIZE) * 2 + 2
    DIM_DISTANCE = DIM_RELATIVE_DISTANCE + DIM_WHOLE_DISTANCE

    HP_UNIT_SIZE = 100
    HP_MAX_SIZE = 3000

    # 2026 heroes are Luban No.7 and Di Renjie. The exact 2026 buff/mark ids
    # should still be debugged on-platform; these buckets cover the likely passive
    # mark ids and reserve an unknown bucket instead of baking 2025 hero ids in.
    MARK_ID_LAYERS = {
        11200: 5,
        13300: 5,
    }
    DIM_MARK = sum(v + 1 for v in MARK_ID_LAYERS.values()) + 1
    DIM_UNIT = DIM_DISTANCE + HP_MAX_SIZE // HP_UNIT_SIZE + 3 + DIM_MARK

    HERO_CONFIG_ID = [112, 133]
    HERO_BEHAVE = [
        "State_Dead",
        "State_Idle",
        "Direction_Move",
        "Normal_Attack",
        "State_Revive",
        "UseSkill_1",
        "UseSkill_2",
        "UseSkill_3",
    ]
    EP_UNIT_SIZE = 30
    EP_MAX_SIZE = 240
    CD_UNIT_SIZE = 1000
    CD_MAX_SIZE = 10000
    LEVEL_MAX = 15
    MONEY_UNIT_SIZE = 20
    MONEY_MAX_SIZE = 300
    COMMON_BUFFS = [90015, 10000, 10010, 11001, 11002, 11010, 11111]
    HERO_BUFF_PREFIXES = (112, 133)
    DIM_BUFF = len(COMMON_BUFFS) + 12 + 1

    DIM_HERO = (
        DIM_UNIT
        + 1
        + len(HERO_BEHAVE) + 1
        + EP_MAX_SIZE // EP_UNIT_SIZE + 2
        + (CD_MAX_SIZE // CD_UNIT_SIZE + 4) * 5
        + LEVEL_MAX
        + MONEY_MAX_SIZE // MONEY_UNIT_SIZE + 3
        + 1
        + 2
        + DIM_BUFF
    )

    SOLDIER_MAX_NUM = 4
    SOLDIER_BEHAVE = ["State_Dead", "Attack_Path"]
    SOLDIER_CONFIG_ID = [[6801, 6804], [6800, 6803], [6802, 6805]]
    DIM_SOLDIER = DIM_UNIT + len(SOLDIER_BEHAVE) + 1 + len(SOLDIER_CONFIG_ID) + 2
    DIM_SOLDIERS = DIM_SOLDIER * SOLDIER_MAX_NUM

    MONSTER_BEHAVE = ["State_Dead", "State_Auto", "State_Revive", "State_Born"]
    DIM_MONSTER = DIM_UNIT + len(MONSTER_BEHAVE) + 1

    DIM_ORGAN = DIM_UNIT + 3 + 2

    BULLET_MAX_NUM = 10
    BULLET_SLOT = [0, 1, 2, 3, "other"]
    DIM_BULLET = len(BULLET_SLOT) + DIM_DISTANCE
    DIM_BULLETS = DIM_BULLET * BULLET_MAX_NUM

    DIM_ALL_UNITS = DIM_HERO * 2 + DIM_SOLDIERS * 2 + DIM_MONSTER + DIM_ORGAN * 2
    DIM_ALL = DIM_ALL_UNITS + DIM_BULLETS


class DimConfig:
    DIM_OF_HERO_FRD = [Args.DIM_HERO]
    DIM_OF_HERO_EMY = [Args.DIM_HERO]
    DIM_OF_SOLDIER_1_4 = [Args.DIM_SOLDIER] * Args.SOLDIER_MAX_NUM
    DIM_OF_SOLDIER_5_8 = [Args.DIM_SOLDIER] * Args.SOLDIER_MAX_NUM
    DIM_OF_MONSTER = [Args.DIM_MONSTER]
    DIM_OF_ORGAN_1 = [Args.DIM_ORGAN]
    DIM_OF_ORGAN_2 = [Args.DIM_ORGAN]
    DIM_OF_BULLET_1_9 = [Args.DIM_BULLET] * 9
    DIM_OF_BULLET_10 = [Args.DIM_BULLET]


class Config:
    NETWORK_NAME = "network"
    LSTM_DROPOUT = 0
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DIM_PUBLIC = 512

    DATA_SPLIT_SHAPE = [
        Args.DIM_ALL + 85,
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
    SERI_VEC_SPLIT_SHAPE = [(Args.DIM_ALL,), (85,)]
    INIT_LEARNING_RATE_START = 1e-4
    TARGET_LR = 1e-5
    TARGET_STEP = 10000
    BETA_START = 0.0
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]

    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001
    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(Args.DIM_ALL + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95
    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
