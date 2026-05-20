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
        "tower_hp_point": 10.0,
        "money": 0.0045,
        "exp": 0.0045,
        "ep_rate": 0.45,
        "death": -1.0,
        "kill": 1.7,
        "last_hit": 0.6,
        "forward": 0.012,
        "monster_last_hit": 0.5,
        "monster_contest": 0.025,
        "safe_tower_damage": 0.45,
        "tower_hp_gap": 0.55,
        "defend_tower_clear": 0.12,
    }
    REMOVE_FORWARD_AFTER = 1000
    TIME_SCALE_ARG = 8000
    REWARD_WITHOUT_TIME_SCALE = {
    }
    REWARD_LATE_PHASE_FRAME = 3800
    REWARD_EARLY_SCALE = {
        "money": 1.05,
        "exp": 1.05,
        "last_hit": 1.10,
        "monster_last_hit": 1.10,
        "monster_contest": 1.05,
        "safe_tower_damage": 0.90,
        "tower_hp_gap": 0.90,
        "forward": 1.0,
    }
    REWARD_LATE_SCALE = {
        "money": 0.85,
        "exp": 0.85,
        "monster_last_hit": 0.40,
        "monster_contest": 0.35,
        "tower_hp_point": 1.25,
        "kill": 1.20,
        "safe_tower_damage": 1.45,
        "tower_hp_gap": 1.45,
        "defend_tower_clear": 1.20,
    }
    MODEL_SAVE_INTERVAL = 1800
    CAMP_HEROES = [112, 133]

    HERO_SUMMONER_SKILL = {
        112: (80102, 80110),
        133: (80102, 80110),
    }
    LOW_HP_HEAL_THRESHOLD = 0.70
    LOW_HP_RECOVER_THRESHOLD = 0.85
    LOW_EP_RECOVER_THRESHOLD = 0.50
    FORCE_LEAVE_BASE_HP_THRESHOLD = 0.80
    OPENING_GRASS_TARGET = (1600.0, -9200.0)
    POST_FIGHT_RECALL_HP_THRESHOLD_EARLY = 0.50
    POST_FIGHT_RECALL_HP_THRESHOLD_LATE = 0.26
    POST_FIGHT_PUSH_HP_THRESHOLD_LATE = 0.05
    POST_FIGHT_PUSH_LEVEL = 9
    POST_FIGHT_PUSH_MONEY = 6000
    POST_FIGHT_PUSH_PHY_VAMP = 0.10
    RECALL_BUTTON = 9
    RECALL_NEAR_TOWER_RADIUS = 9500.0
    ENEMY_RETURN_HOLD_HP_THRESHOLD = 0.85
    BASE_RADIUS = 14000.0
    BASE_HEAL_EXIT_HP = 0.98
    BASE_HEAL_DEFEND_HP = 0.55
    MID_LANE_TARGET = (0.0, 0.0)
    CHOSEN_SUMMONER_BUTTON = 8
    RECOVER_BUTTON = 7
    SELF_TARGET_INDEX = 2
    PASSIVE_BUTTONS = {0, 1, 7, 8, 9, 10, 11}
    BERSERK_SKILL_ID = 80110
    BERSERK_MIN_COMBAT_FRAMES = 60
    BERSERK_HP_THRESHOLD = 0.70
    HEAL_SUMMONER_SKILL_ID = 80102
    STUN_SUMMONER_SKILL_ID = 80103
    STUN_SUMMONER_RANGE = 6000.0
    FIGHT_RETREAT_HP_GAP = 0.18
    FIGHT_RETREAT_LOW_HP = 0.35
    FIGHT_WITHOUT_SUMMONER_MIN_HP_GAP = 0.10
    FIGHT_ADVANTAGE_ENGAGE_SCORE = 0.18
    FIGHT_ADVANTAGE_RETREAT_SCORE = -0.25
    FIGHT_ADVANTAGE_STRONG_SCORE = 0.45
    FIGHT_RETALIATE_SCORE = -0.08
    FIGHT_COMMIT_SCORE = 0.08
    FIGHT_MONEY_SCALE = 2500.0
    FIGHT_MIN_HEALTHY_HP = 0.38
    EARLY_LANE_DISCIPLINE_LEVEL = 4
    EARLY_LANE_DISCIPLINE_FRAME = 2200
    EARLY_CENTER_CROSS_MARGIN = 1500.0
    EARLY_TRADE_MEMORY_SCALE = 1500.0
    EARLY_TRADE_BAD_SCORE = -0.18
    EARLY_BAD_MATCHUP_FRAME = 1800
    DI_VS_LUBAN_EARLY_SCORE_PENALTY = 0.36
    LUBAN_VS_DI_EARLY_FRAME = 1800
    LUBAN_PRESS_GRASS_MIN_LEVEL = 2
    LUBAN_PRESS_GRASS_RADIUS = 22000.0
    LUBAN_PROBE_GRASS_RANGE = 24000.0
    LUBAN_PROBE_GRASS_FRAMES_AFTER_LOST = 180
    LUBAN_PRESS_IGNORE_LANE_RADIUS = 7600.0
    ENDGAME_MONEY = 6000
    LATE_GAME_FRAME = 3800
    ENDGAME_MIN_HP = 0.26
    ENDGAME_IGNORE_RECALL_HP = 0.32
    ENDGAME_RECALL_HP = 0.12
    ADVANTAGE_RECALL_HP = 0.18
    ADVANTAGE_MONEY_GAP = 1000
    ADVANTAGE_LEVEL_GAP = 1
    RECALL_LANE_THREAT_RADIUS = 14000.0
    SUSTAIN_RESOURCE_RADIUS = 26000.0
    LUBAN_SKILL3_DANGER_RADIUS = 9000.0
    ACTION_RULES_LIGHTWEIGHT = True
    TOWER_EARLY_PUSH_MONEY = 5500
    TOWER_FULL_PUSH_MONEY = 6000
    RUNE_LOW_HP_THRESHOLD = 0.80
    RUNE_SEARCH_RADIUS = 32000.0
    RUNE_PICKUP_RADIUS = 6000.0
    ENEMY_RUNE_AFTER_KILL_RADIUS = 36000.0
    ENEMY_RUNE_ESCAPE_FRAMES = 90
    ENEMY_RUNE_ESCAPE_PICK_RADIUS = 6200.0
    RETURN_WAVE_RADIUS = 4200.0
    CRITICAL_HOME_HP_THRESHOLD = 0.16
    SUPPLY_RETURN_HOME_HP_THRESHOLD = 0.16
    SAFE_RECALL_HP_THRESHOLD = 0.24
    RECALL_EXIT_HP = 0.45
    RECALL_DEFEND_EXIT_HP = 0.30
    RETREAT_BASE_LOCK_FRAMES = 45
    DEFEND_TOWER_LOCK_FRAMES = 45
    SAFE_PUSH_LOCK_FRAMES = 120
    COMMIT_FIGHT_LOCK_FRAMES = 30
    MOVE_OSCILLATION_WINDOW = 36
    MOVE_OSCILLATION_LIMIT = 3
    MOVE_OSCILLATION_LOCK_FRAMES = 45
    SUPPLY_RUNE_HP_THRESHOLD = 0.62
    MONSTER_MIN_HP_THRESHOLD = 0.35
    POST_KILL_MONSTER_MAX_MONEY = 5200
    POST_KILL_MONSTER_RADIUS = 30000.0
    MONSTER_FOCUS_MAX_MONEY = 6500
    MONSTER_REWARD_DECAY_KILLS = 3
    MONSTER_REWARD_MIN_SCALE = 0.10
    TOWER_SAFE_ANCHOR_OFFSET = 8500.0
    ENEMY_TOWER_FORBID_HERO_HP = 0.45
    ENEMY_TOWER_DIVE_ENEMY_HP = 0.15
    ENEMY_TOWER_DIVE_SCORE = 0.55
    CONTROL_CLEANSE_FRAMES = 45
    GRASS_AMBUSH_FRAMES = 7200
    GRASS_AMBUSH_MIN_HP = 0.45
    GRASS_AMBUSH_MAX_HP_GAP = 0.18
    GRASS_AMBUSH_ENTRY_RADIUS = 18000.0
    GRASS_AMBUSH_EXIT_FRAMES = 90
    GRASS_POINTS = [
        (-5200.0, -9000.0),
        (-2600.0, -9800.0),
        (2600.0, -9200.0),
        (7200.0, -5200.0),
        (9800.0, -1200.0),
        (9000.0, 3200.0),
        (4500.0, 8200.0),
        (-1200.0, 9800.0),
        (-7200.0, 5200.0),
        (-9800.0, 1200.0),
    ]

    # Debug-only deterministic observation collection. These are disabled by
    # default and can also be enabled by environment variables in train_test.
    DEBUG_AGENT = False
    DEBUG_MAX_RUN_EPISODES = 4
    DEBUG_TOTAL_FRAMES = 7200


class DimConfig:
    DIM_OF_FEATURE = [1536]


class Config:
    NETWORK_NAME = "network"
    LSTM_DROPOUT = 0
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DIM_PUBLIC = 512
    MULTI_HEAD = True
    HERO_HEAD_IDS = [112, 133]
    HERO_ID_FEATURE_SLICE = (0, 2)
    TARGET_ENTITY_COUNT = 9
    TARGET_ENTITY_DIM = 32
    TARGET_FEATURE_DIM = TARGET_ENTITY_COUNT * TARGET_ENTITY_DIM

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
