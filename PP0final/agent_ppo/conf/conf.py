#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    CAMP_HEROES = [
        [{"hero_id": 112, "select_skill": 80110}],
        [{"hero_id": 133, "select_skill": 80110}],
    ]
    HERO_SUMMONER_SKILL = {
        112: (80110,),
        133: (80110,),
    }
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 10.0,
        "money": 0.006,
        "exp": 0.006,
        "lv_rwd": 0.20,
        "ep_rate": 0.75,
        "death": -1.1,
        "kill": 0.0,
        "kd_gap": 0.20,
        "last_hit": 0.5,
        "forward": 0.01,
        "safe_tower_damage": 0.70,
        "tower_hp_gap": 0.90,
        "enemy_tower_low": 0.35,
        "defend_tower_clear": 0.10,
        "finish_push": 0.60,
        "tower_attack": 0.15,
        "safe_push": 0.06,
        "monster_last_hit": 0.25,
        "hero_lane_advantage": 0.04,
        "monster_contest": 0.04,
        "post_fight_recall": 0.16,
        "recall_use": 0.04,
        "rune_approach": 0.02,
        "rune_pickup": 0.18,
        "enemy_rune_after_kill": 0.04,
        "attack_hit": 0.03,
        "skill_hit": 0.08,
        "skill_misuse": -0.03,
        "retaliate_hit": 0.05,
        "danger": -0.18,
        "tower_chase_risk": -0.12,
        "attack_power": 0.02,
        "attack_speed": 0.02,
        "fight_risk": -0.08,
        "early_bad_fight": -0.05,
        "stutter_step": 0.02,
        "grass_ambush": 0.03,
        "grass_engage": 0.04,
        "enemy_wave_overextend": 0.04,
        "lane_anchor": 0.02,
    }
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    REMOVE_FORWARD_AFTER = 1000
    TIME_SCALE_ARG = 8000
    REWARD_WITHOUT_TIME_SCALE = {}
    REWARD_LATE_PHASE_FRAME = 3800
    REWARD_EARLY_SCALE = {
        "safe_tower_damage": 0.80,
        "tower_hp_gap": 0.80,
        "finish_push": 0.70,
    }
    REWARD_LATE_SCALE = {
        "tower_hp_point": 1.15,
        "safe_tower_damage": 1.35,
        "tower_hp_gap": 1.35,
        "defend_tower_clear": 1.15,
        "finish_push": 1.60,
    }
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800
    FORCE_LEAVE_BASE_HP_THRESHOLD = 0.90
    BASE_RADIUS = 14000.0
    MID_LANE_TARGET = (0.0, 0.0)
    OPENING_GRASS_TARGET = (1600.0, -9200.0)
    OPENING_FORCE_UNTIL_FRAME = 3200
    OPENING_TARGET_RADIUS = 8500.0
    OPENING_LANE_TARGET = (0.0, 0.0)
    CHOSEN_SUMMONER_BUTTON = 8
    SELF_TARGET_INDEX = 2
    BERSERK_SKILL_ID = 80110
    BERSERK_HP_THRESHOLD = 0.50
    BERSERK_MIN_COMBAT_FRAMES = 30
    RECOVER_BUTTON = 7
    RECALL_BUTTON = 9
    PASSIVE_BUTTONS = {0, 1, 7, 8, 9, 10, 11}
    HEAL_SUMMONER_SKILL_ID = 80102
    STUN_SUMMONER_SKILL_ID = 80103
    STUN_SUMMONER_RANGE = 6000.0
    LOW_HP_HEAL_THRESHOLD = 0.70
    LOW_HP_RECOVER_THRESHOLD = 0.85
    LOW_EP_RECOVER_THRESHOLD = 0.50
    POST_FIGHT_RECALL_HP_THRESHOLD_EARLY = 0.50
    POST_FIGHT_RECALL_HP_THRESHOLD_LATE = 0.26
    POST_FIGHT_PUSH_HP_THRESHOLD_LATE = 0.05
    POST_FIGHT_PUSH_LEVEL = 7
    POST_FIGHT_PUSH_MONEY = 4500
    POST_FIGHT_PUSH_PHY_VAMP = 0.10
    RECALL_NEAR_TOWER_RADIUS = 9500.0
    ENEMY_RETURN_HOLD_HP_THRESHOLD = 0.85
    BASE_HEAL_EXIT_HP = 0.98
    BASE_HEAL_DEFEND_HP = 0.55
    CRITICAL_HOME_HP_THRESHOLD = 0.16
    SUPPLY_RETURN_HOME_HP_THRESHOLD = 0.16
    SAFE_RECALL_HP_THRESHOLD = 0.24
    RECALL_EXIT_HP = 0.45
    RECALL_DEFEND_EXIT_HP = 0.30
    ENDGAME_MONEY = 6000
    LATE_GAME_FRAME = 3800
    ENDGAME_MIN_HP = 0.26
    ENDGAME_IGNORE_RECALL_HP = 0.32
    ENDGAME_RECALL_HP = 0.12
    ADVANTAGE_RECALL_HP = 0.18
    ADVANTAGE_MONEY_GAP = 1000
    ADVANTAGE_LEVEL_GAP = 1
    TOWER_EARLY_PUSH_MONEY = 4200
    TOWER_FULL_PUSH_MONEY = 5000
    RUNE_LOW_HP_THRESHOLD = 0.80
    RUNE_SEARCH_RADIUS = 32000.0
    RUNE_PICKUP_RADIUS = 6000.0
    ENEMY_RUNE_AFTER_KILL_RADIUS = 36000.0
    MONSTER_MIN_HP_THRESHOLD = 0.35
    POST_KILL_MONSTER_MAX_MONEY = 5200
    POST_KILL_MONSTER_RADIUS = 30000.0
    MONSTER_FOCUS_MAX_MONEY = 6500
    MONSTER_REWARD_DECAY_KILLS = 3
    MONSTER_REWARD_MIN_SCALE = 0.10
    EARLY_LANE_DISCIPLINE_LEVEL = 4
    EARLY_LANE_DISCIPLINE_FRAME = 2200
    EARLY_CENTER_CROSS_MARGIN = 1500.0
    GRASS_AMBUSH_FRAMES = 7200
    GRASS_AMBUSH_MIN_HP = 0.45
    GRASS_AMBUSH_MAX_HP_GAP = 0.18


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    DIM_OF_HERO_FRD = [113]
    DIM_OF_HERO_EMY = [117]
    DIM_OF_SOLDIER_1_6 = [24, 24, 24, 24, 24, 24]
    DIM_OF_SOLDIER_7_12 = [24, 24, 24, 24, 24, 24]
    DIM_OF_ORGAN_1_3 = [17, 17, 17]
    DIM_OF_ORGAN_4_6 = [17, 17, 17]
    DIM_OF_GLOBAL_INFO = [509]
    DIM_OF_FEATURE = [1129]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    FEATURE_DIM = DimConfig.DIM_OF_FEATURE[0]
    LEGAL_ACTION_FLAT_DIM = 85
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
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(FEATURE_DIM + LEGAL_ACTION_FLAT_DIM) * LSTM_TIME_STEPS],
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

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
