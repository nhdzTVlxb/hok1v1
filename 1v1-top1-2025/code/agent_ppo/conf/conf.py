#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 10.0,
        "money": 4e-3,
        "exp": 4e-3,
        "ep_rate": 0.75,
        "death": -1.0,
        "kill": -0.6,  # kill-death是击杀一次的奖励
        "last_hit": 0.5,  # 当前的dead_action不完整 (bug)
        "forward": 0.01,
    }
    REMOVE_FORWARD_AFTER = 1000  # 一定帧数后删除forward奖励
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 8000
    # 跳过衰减的奖励
    REWARD_WITHOUT_TIME_SCALE = {
        # "hp_point",
        # "tower_hp_point",
        # "death",
        # "kill",
    }
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800
    # 训练的英雄阵容
    CAMP_HEROES = [
        [169],  # 后羿
        [173],  # 李元芳
        [174],  # 虞姬
    ]

    """ DEBUG """
    # 是否使用固定动作的agent调试obs
    debug_agent: bool = False
    debug_max_run_episodes: int = 1
    # 是否保存obs信息到json文件
    debug_frames: bool = False
    # 从56开始一个step为6帧
    debug_max_save_frame_no: int = 56 + 6 * 8000
    # debug总帧数 (4800升到4级, 2300升到3级)
    debug_total_frames = 4000

class Args:
    ### Observation Builder 配置 ###
    # 单位通用特征
    RELATIVE_DISTANCE_UNIT_SIZE = 600  # 相对距离单位大小 (向上取整)
    RELATIVE_DISTANCE_MAX_SIZE = 24600  # 相对距离最大大小
    DIM_RELATIVE_DISTANCE = (RELATIVE_DISTANCE_MAX_SIZE // RELATIVE_DISTANCE_UNIT_SIZE + 2) * 2 + 1
    WHOLE_DISTANCE_UNIT_SIZE = 5000  # 全局距离单位大小 (向下取整)
    WHOLE_DISTANCE_MAX_SIZE = int(9e4)  # 全局距离最大大小
    DIM_WHOLE_DISTANCE = (WHOLE_DISTANCE_MAX_SIZE // WHOLE_DISTANCE_UNIT_SIZE) * 2 + 2
    DIM_DISTANCE = DIM_RELATIVE_DISTANCE + DIM_WHOLE_DISTANCE
    HP_UNIT_SIZE = 100  # 生命值单位大小
    HP_MAX_SIZE = 2400  # 生命值最大大小 (向上取整)
    MARK_ID_LAYERS = {  # 每个mark的最大层数 (debug_agent获得)
        16900: 3,  # 后羿普攻次数被动
        16901: 2,  # 后裔被动相关
        17300: 4,  # 李元芳一技能标记
        17310: 1,  # 李元芳一技能相关
    }
    DIM_MARK = sum([v + 1 for v in MARK_ID_LAYERS.values()]) + 1  # 所有层数都加上0层, 以及一个未知空位
    DIM_UNIT = int(
        DIM_DISTANCE +
        HP_MAX_SIZE // HP_UNIT_SIZE + 3 +
        DIM_MARK
    )
    # 英雄专属特征
    HERO_CONFIG_ID = [169, 173, 174]  # 映射config id到编号
    HERO_BEHAVE = ['State_Dead', 'State_Idle',
    'Direction_Move', 'Normal_Attack', 'State_Revive',
    'UseSkill_1', 'UseSkill_2', 'UseSkill_3']  # 映射behave到编号
    EP_UNIT_SIZE = 30  # 法术单位大小
    EP_MAX_SIZE = 240  # 法术最大大小 (向下取整)
    CD_UNIT_SIZE = 1  # 冷却单位大小
    CD_MAX_SIZE = 10  # 冷却最大大小 (向上取整)
    LEVEL_MAX = 15  # 最大等级
    MONEY_UNIT_SIZE = 20  # 金币获得单位大小
    MONEY_MAX_SIZE = 300  # 金币获得最大大小 (向下取整)
    BUFFS = [  # 53种
        90015,  # 可能是泉水的回复buff
        10000,  # 点回复技能时候产生的buff (1.2s先消失)
        10010,  # 回复技能产生的恢复buff (5.8s)
        11001,  # 可能是加速buff
        11002,  # 可能是减速buff
        11010,  # 可能是净化buff
        11111,  # 可能是某种通用buff (虞姬)
        911220, 911290, 914110, 914210, 914211, 914250,  # 一些未知buff (6,)
        # 后裔 (10,)
        169000, 169010, 169020, 169040,
        169100,
        169900, 169901, 169910, 169920, 169963,
        # 李元芳 (20,)
        173000, 173040,
        173101, 173110, 173120, 173150, 173151, 173152, 173153, 173154, 173155, 173159, 173160, 173170, 173173,
        173250,
        173920, 173950, 173990, 173999,
        174000, 174010, 174090,
        # 虞姬 (10,)
        174100,
        174250, 174260,
        174360,
        174910, 174920, 174950,
    ]
    DIM_BUFF = len(BUFFS) + 1
    DIM_HERO = (
        DIM_UNIT + 1 + len(HERO_BEHAVE) + 1 +
        EP_MAX_SIZE // EP_UNIT_SIZE + 2 +
        (CD_MAX_SIZE // CD_UNIT_SIZE + 4) * 5 +
        LEVEL_MAX + 
        MONEY_MAX_SIZE // MONEY_UNIT_SIZE + 3 +  # 离散化金钱变化, 金钱变化是否在0~20, 总金钱
        1 +  # 是否在草丛
        2 +  # 是否在塔攻击范围内, 是否为塔的攻击目标
        DIM_BUFF
    )
    # 小兵专属特征
    SOLDIER_MAX_NUM = 4  # 考虑的最大小兵数目
    SOLDIER_BEHAVE = ['State_Dead', 'Attack_Path']  # 映射behave到编号
    SOLDIER_CONFIG_ID = [[6801, 6804], [6800, 6803], [6802, 6805]]  # 近战, 远程, 炮车
    DIM_SOLDIER = (
        DIM_UNIT + len(SOLDIER_BEHAVE) + 1 + len(SOLDIER_CONFIG_ID) +
        2  # 是否在塔的攻击范围内, 是否为塔的攻击目标
    )
    DIM_SOLDIERS = DIM_SOLDIER * SOLDIER_MAX_NUM
    # 河蟹专属特征
    RIVER_CRAB_BEHAVE = ['State_Dead', 'State_Auto', 'State_Revive', 'State_Born']
    DIM_RIVER_CRAB = DIM_UNIT + len(RIVER_CRAB_BEHAVE) + 1
    # 防御塔专属特征
    DIM_ORGAN = DIM_UNIT + 3 + 2  # (3,)攻击目标, (2,)塔后是否有血包, 血包生成剩余时间
    # 全部单位特征
    DIM_ALL_UNITS = (
        DIM_HERO * 2 + DIM_SOLDIERS * 2 +
        DIM_RIVER_CRAB + DIM_ORGAN * 2
    )
    # 子弹特征
    BULLET_MAX_NUM = 10  # 最大子弹数量, 9个英雄子弹, 1个防御塔子弹
    BULLET_SLOT = ['SLOT_SKILL_0', 'SLOT_SKILL_1', 'SLOT_SKILL_2', 'SLOT_SKILL_3', 'SLOT_SKILL_VALID']
    DIM_BULLET = (
        len(BULLET_SLOT) +
        DIM_DISTANCE
    )
    DIM_BULLETS = DIM_BULLET * BULLET_MAX_NUM
    # 全部特征
    DIM_ALL = DIM_ALL_UNITS + DIM_BULLETS

# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    # main camp hero
    DIM_OF_HERO_FRD = [Args.DIM_HERO]
    # enemy camp hero
    DIM_OF_HERO_EMY = [Args.DIM_HERO]
    # main camp soldier
    DIM_OF_SOLDIER_1_4 = [Args.DIM_SOLDIER] * 4
    # enemy camp soldier
    DIM_OF_SOLDIER_5_8 = [Args.DIM_SOLDIER] * 4
    # river crab
    DIM_OF_RIVER_CRAB = [Args.DIM_RIVER_CRAB]
    # main camp organ
    DIM_OF_ORGAN_1 = [Args.DIM_ORGAN]
    # enemy camp organ
    DIM_OF_ORGAN_2 = [Args.DIM_ORGAN]
    # bullet
    DIM_OF_BULLET_1_9 = [Args.DIM_BULLET] * 9
    DIM_OF_BULLET_10 = [Args.DIM_BULLET]

# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_DROPOUT = 0
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DIM_PUBLIC = 512  # 将LSTM和旁路MLP的结果合并后执行MLP到512维
    MULTI_HEAD = True  # 是否使用三个英雄对应的输出头网络

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
    INIT_LEARNING_RATE_START = 1e-5
    BETA_START = 0
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

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])

if __name__ == '__main__':
    print(Config.SAMPLE_DIM)
    print(Args.DIM_ALL)
    print(Args.DIM_DISTANCE)
    print(Args.DIM_UNIT - Args.DIM_DISTANCE)
