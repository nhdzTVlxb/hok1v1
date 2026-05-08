#!/usr/bin/env python4
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
from torch.nn import ModuleDict
import torch.nn.functional as F

import numpy as np
from math import ceil, floor
from collections import OrderedDict
from typing import Dict, List, Tuple

from agent_ppo.conf.conf import DimConfig, Config, Args


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # feature configure parameter
        # 特征配置参数
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.dim_public = Config.DIM_PUBLIC
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        # NETWORK DIM
        # 网络维度
        self.single_hero_feature_dim = int(DimConfig.DIM_OF_HERO_EMY[0])
        self.single_soldier_feature_dim = int(DimConfig.DIM_OF_SOLDIER_1_4[0])
        self.single_organ_feature_dim = int(DimConfig.DIM_OF_ORGAN_1[0])
        self.single_river_crab_feature_dim = int(DimConfig.DIM_OF_RIVER_CRAB[0])
        self.single_bullet_feature_dim = int(DimConfig.DIM_OF_BULLET_1_9[0])

        self.all_hero_feature_dim = (
            int(np.sum(DimConfig.DIM_OF_HERO_FRD)) +
            int(np.sum(DimConfig.DIM_OF_HERO_EMY))
        )
        self.all_soldier_feature_dim = (
            int(np.sum(DimConfig.DIM_OF_SOLDIER_1_4)) +
            int(np.sum(DimConfig.DIM_OF_SOLDIER_5_8))
        )
        self.all_organ_feature_dim = (
            int(np.sum(DimConfig.DIM_OF_ORGAN_1)) +
            int(np.sum(DimConfig.DIM_OF_ORGAN_2))
        )
        self.all_bullet_feature_dim = (
            int(np.sum(DimConfig.DIM_OF_BULLET_1_9)) +
            int(np.sum(DimConfig.DIM_OF_BULLET_10))
        )

        """public position"""
        fc_position_list = [Args.DIM_DISTANCE, 128, 64]
        self.position_delta_dim = 64 - Args.DIM_DISTANCE  # 压缩后的位置维度64
        self.position_mlp = MLP(fc_position_list, "position_mlp")

        """public unit"""
        fc_unit_no_pos_list = [Args.DIM_UNIT - Args.DIM_DISTANCE, 64, 32]
        self.unit_delta_dim = 64 + 32 - Args.DIM_UNIT  # 压缩后的unit维度64+32
        self.unit_no_pos_mlp = MLP(fc_unit_no_pos_list, "unit_no_pos_mlp")

        # 构建网络 (相同特征维度输入, 前面两个层共享, 最后一个层单独计算)
        """hero module"""
        fc_hero_dim_list = [self.single_hero_feature_dim + self.unit_delta_dim, 512, 256, 128]
        self.hero_mlp = MLP(fc_hero_dim_list[:-1], "hero_mlp", non_linearity_last=True)
        self.hero_frd_fc = make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1])
        self.hero_emy_fc = make_fc_layer(fc_hero_dim_list[-2], fc_hero_dim_list[-1])

        """soldier module"""
        fc_soldier_dim_list = [self.single_soldier_feature_dim + self.unit_delta_dim, 128, 64, 32]
        self.soldier_mlp = MLP(fc_soldier_dim_list[:-1], "soldier_mlp", non_linearity_last=True)
        self.soldier_frd_fc = make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1])
        self.soldier_emy_fc = make_fc_layer(fc_soldier_dim_list[-2], fc_soldier_dim_list[-1])

        """river crab module"""
        fc_river_crab_list = [self.single_river_crab_feature_dim + self.unit_delta_dim, 128, 64, 32]
        self.river_crab_mlp = MLP(fc_river_crab_list, "river_crab_mlp")

        """organ module"""
        fc_organ_dim_list = [self.single_organ_feature_dim + self.unit_delta_dim, 128, 64, 32]
        self.organ_mlp = MLP(fc_organ_dim_list[:-1], "organ_mlp", non_linearity_last=True)
        self.organ_frd_fc = make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1])
        self.organ_emy_fc = make_fc_layer(fc_organ_dim_list[-2], fc_organ_dim_list[-1])

        """bullet module"""
        fc_bullet_list = [self.single_bullet_feature_dim + self.position_delta_dim, 64, 64, 32]
        self.bullet_mlp = MLP(fc_bullet_list[:-1], "bullet_mlp", non_linearity_last=True)
        self.bullet_hero_fc = make_fc_layer(fc_bullet_list[-2], fc_bullet_list[-1])
        self.bullet_organ_fc = make_fc_layer(fc_bullet_list[-2], fc_bullet_list[-1])

        """public concat"""
        concat_dim = (
            fc_hero_dim_list[-1] * 2 +    # 128*2
            fc_soldier_dim_list[-1] * 2 +    # 32*2
            fc_river_crab_list[-1] +    # 32
            fc_organ_dim_list[-1] * 2 +    # 32*2
            fc_bullet_list[-1] * 2      # 32 * 2
        )    # sum=480
        fc_concat_dim_list = [concat_dim, self.lstm_unit_size]
        self.concat_mlp = MLP(fc_concat_dim_list, "concat_mlp", non_linearity_last=True)
        fc_concat_other_mlp_list = [concat_dim, 512, self.lstm_unit_size]    # MLP与LSTM输出维度一致
        self.concate_mlp_other = MLP(fc_concat_other_mlp_list, "concat_other_mlp")
        fc_lstm_and_linear_mlp = [fc_concat_dim_list[-1] + fc_concat_other_mlp_list[-1], self.dim_public]
        self.lstm_and_linear_mlp = MLP(fc_lstm_and_linear_mlp, 'lstm_and_linear_mlp', non_linearity_last=True)

        """public lstm"""
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=Config.LSTM_DROPOUT,
            bidirectional=False,
        )

        if Config.MULTI_HEAD:    # 使用多输出头
            """output label 动作网络, 前5维动作 (3英雄各一个)"""
            self.label_mlps = nn.ModuleList([ModuleDict(
                {
                    f"hero_label{label_index}_mlp": MLP(
                        [self.dim_public, self.label_size_list[label_index]],
                        f"hero_label{label_index}_mlp",
                    )
                    for label_index in range(len(self.label_size_list) - 1)
                }
            ) for _ in range(3)])
            # 用于预测target的embed特征 (将lstm的输出转为自注意力中的value, 3英雄各一个)
            self.lstm_tar_embed_mlps = nn.ModuleList([
                make_fc_layer(self.dim_public, self.target_embed_dim) for _ in range(3)
            ])
            # 将只经过mlp的特征后32维提取出来的特征, 转化为自注意力中的query, 3英雄各一个
            self.target_embed_mlps = nn.ModuleList([
                make_fc_layer(32, self.target_embed_dim, use_bias=False) for _ in range(3)
            ])

            """output value (三英雄各一个)"""
            self.value_mlps = nn.ModuleList([
                MLP([self.dim_public, 64, 1], "hero_value_mlp") for _ in range(3)
            ])
        else:    # 使用单输出头
            """output label 动作网络, 前5维动作"""
            self.label_mlp = ModuleDict(
                {
                    f"hero_label{label_index}_mlp": MLP(
                        [self.dim_public, self.label_size_list[label_index]],
                        f"hero_label{label_index}_mlp",
                    )
                    for label_index in range(len(self.label_size_list) - 1)
                }
            )
            # 用于预测target的embed特征 (将lstm的输出转为自注意力中的value)
            self.lstm_tar_embed_mlp = make_fc_layer(self.dim_public, self.target_embed_dim)
            # 将只经过mlp的特征后32维提取出来的特征, 转化为自注意力中的query
            self.target_embed_mlp = make_fc_layer(32, self.target_embed_dim, use_bias=False)

            """output value"""
            self.value_mlp = MLP([self.dim_public, 64, 1], "hero_value_mlp")

    def process_sub_feature(self, x, mlp, is_unit: bool):
        """用于降维共享特征, mlp为处理x降维后特征的网络, is_unit表示是否为单位特征, 否则为子弹特征"""
        ret = [x]
        dim_suffix = Args.DIM_DISTANCE
        if is_unit:
            unit_no_pos = self.unit_no_pos_mlp(x[..., -Args.DIM_UNIT:-Args.DIM_DISTANCE])
            ret.append(unit_no_pos)
            dim_suffix = Args.DIM_UNIT
        pos = self.position_mlp(x[..., -Args.DIM_DISTANCE:])
        ret.append(pos)
        ret[0] = x[..., :-dim_suffix]
        return mlp(torch.cat(ret, dim=-1))

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init, hero_split_nums = data_list

        result_list = []

        # 特征向量分割 (B*L, Args.DIM_ALL)
        feature_vec_split_list = feature_vec.split(
            [
                self.all_hero_feature_dim,    # 英雄
                self.all_soldier_feature_dim,    # 小兵
                self.single_river_crab_feature_dim,    # 河蟹
                self.all_organ_feature_dim,    # 防御塔
                self.all_bullet_feature_dim    # 子弹
            ],
            dim=1,
        )
        hero_vec_list = feature_vec_split_list[0].split(    # 英雄
            [
                int(np.sum(DimConfig.DIM_OF_HERO_FRD)),
                int(np.sum(DimConfig.DIM_OF_HERO_EMY)),
            ],
            dim=1,
        )
        soldier_vec_list = feature_vec_split_list[1].split(    # 小兵
            [
                int(np.sum(DimConfig.DIM_OF_SOLDIER_1_4)),
                int(np.sum(DimConfig.DIM_OF_SOLDIER_5_8)),
            ],
            dim=1,
        )
        river_crab_tensor = feature_vec_split_list[2]    # 河蟹
        organ_vec_list = feature_vec_split_list[3].split(    # 防御塔
            [
                int(np.sum(DimConfig.DIM_OF_ORGAN_1)),
                int(np.sum(DimConfig.DIM_OF_ORGAN_2)),
            ],
            dim=1,
        )
        bullet_vec_list = feature_vec_split_list[4].split(    # 子弹
            [
                int(np.sum(DimConfig.DIM_OF_BULLET_1_9)),
                int(np.sum(DimConfig.DIM_OF_BULLET_10)),
            ],
            dim=1,
        )

        """ DEBUG 对每个划分出的维度去重查看是否正确划分 (可以修改obs_builder来固定每个维度的数值) """
        # from pprint import pprint
        # fn = lambda x: torch.unique(x)
        # debug_data = {
        #     'our_hero': fn(hero_vec_list[0]),
        #     'enemy_hero': fn(hero_vec_list[1]),
        #     'our_soldier': fn(soldier_vec_list[0]),
        #     'enemy_soldier': fn(soldier_vec_list[1]),
        #     'crab': fn(river_crab_tensor),
        #     'our_organ': fn(organ_vec_list[0]),
        #     'enemy_organ': fn(organ_vec_list[1]),
        #     'hero_bullet': fn(bullet_vec_list[0]),
        #     'tower_bullet': fn(bullet_vec_list[1])
        # }
        # print("[DEBUG] debug_data:")
        # pprint(debug_data)

        _hero_frd = hero_vec_list[0].split(DimConfig.DIM_OF_HERO_FRD, dim=1)
        _hero_emy = hero_vec_list[1].split(DimConfig.DIM_OF_HERO_EMY, dim=1)
        _soldier_1_4 = soldier_vec_list[0].split(DimConfig.DIM_OF_SOLDIER_1_4, dim=1)
        _soldier_5_8 = soldier_vec_list[1].split(DimConfig.DIM_OF_SOLDIER_5_8, dim=1)
        _organ_1 = organ_vec_list[0].split(DimConfig.DIM_OF_ORGAN_1, dim=1)
        _organ_2 = organ_vec_list[1].split(DimConfig.DIM_OF_ORGAN_2, dim=1)
        _bullet_1_9 = bullet_vec_list[0].split(DimConfig.DIM_OF_BULLET_1_9, dim=1)
        _bullet_10 = bullet_vec_list[1].split(DimConfig.DIM_OF_BULLET_10, dim=1)

        tar_embed_list = []    # 获取全连接后对应特征的 (B*L, 9, 32), 再转为query

        # 敌方英雄
        hero_emy_result_list = []
        for index in range(len(_hero_emy)):
            hero_emy_mlp_out = self.process_sub_feature(_hero_emy[index], self.hero_mlp, True)
            hero_emy_fc_out = self.hero_emy_fc(hero_emy_mlp_out)
            _, split_1 = hero_emy_fc_out.split([96, 32], dim=1)
            tar_embed_list.append(split_1)    # target 1
            hero_emy_result_list.append(hero_emy_fc_out)
        hero_emy_concat_result = torch.cat(hero_emy_result_list, dim=1)

        # 我方英雄
        hero_frd_result_list = []
        for index in range(len(_hero_frd)):
            hero_frd_mlp_out = self.process_sub_feature(_hero_frd[index], self.hero_mlp, True)
            hero_frd_fc_out = self.hero_frd_fc(hero_frd_mlp_out)
            _, split_1 = hero_frd_fc_out.split([96, 32], dim=1)
            tar_embed_list.append(split_1)    # target 2
            hero_frd_result_list.append(hero_frd_fc_out)
        hero_frd_concat_result = torch.cat(hero_frd_result_list, dim=1)

        # 我方士兵
        soldier_frd_result_list = []
        for index in range(len(_soldier_1_4)):
            soldier_frd_mlp_out = self.process_sub_feature(_soldier_1_4[index], self.soldier_mlp, True)
            soldier_frd_fc_out = self.soldier_frd_fc(soldier_frd_mlp_out)
            soldier_frd_result_list.append(soldier_frd_fc_out)
        soldier_frd_concat_result = torch.cat(soldier_frd_result_list, dim=1)
        reshape_frd_soldier = soldier_frd_concat_result.reshape(-1, 4, 32)
        soldier_frd_concat_result, _ = reshape_frd_soldier.max(dim=1)

        # 敌方士兵
        soldier_emy_result_list = []
        for index in range(len(_soldier_5_8)):
            soldier_emy_mlp_out = self.process_sub_feature(_soldier_5_8[index], self.soldier_mlp, True)
            soldier_emy_fc_out = self.soldier_emy_fc(soldier_emy_mlp_out)
            soldier_emy_result_list.append(soldier_emy_fc_out)
            tar_embed_list.append(soldier_emy_fc_out)    # target 3,4,5,6
        soldier_emy_concat_result = torch.cat(soldier_emy_result_list, dim=1)
        reshape_emy_soldier = soldier_emy_concat_result.reshape(-1, 4, 32)
        soldier_emy_concat_result, _ = reshape_emy_soldier.max(dim=1)

        # 河蟹
        river_crab_result = self.process_sub_feature(river_crab_tensor, self.river_crab_mlp, True)

        # 我方防御塔
        organ_frd_result_list = []
        for index in range(len(_organ_1)):
            organ_frd_mlp_out = self.process_sub_feature(_organ_1[index], self.organ_mlp, True)
            organ_frd_fc_out = self.organ_frd_fc(organ_frd_mlp_out)
            organ_frd_result_list.append(organ_frd_fc_out)
        organ_frd_concat_result = torch.cat(organ_frd_result_list, dim=1)

        # 敌方防御塔
        organ_emy_result_list = []
        for index in range(len(_organ_2)):
            organ_emy_mlp_out = self.process_sub_feature(_organ_2[index], self.organ_mlp, True)
            organ_emy_fc_out = self.organ_emy_fc(organ_emy_mlp_out)
            organ_emy_result_list.append(organ_emy_fc_out)
            tar_embed_list.append(organ_emy_fc_out)    # target 7
        organ_emy_concat_result = torch.cat(organ_emy_result_list, dim=1)

        # 英雄子弹
        bullet_hero_result_list = []
        for index in range(len(_bullet_1_9)):
            bullet_hero_mlp_out = self.process_sub_feature(_bullet_1_9[index], self.bullet_mlp, False)
            bullet_hero_fc_out = self.bullet_hero_fc(bullet_hero_mlp_out)
            bullet_hero_result_list.append(bullet_hero_fc_out)
        bullet_hero_concat_result = torch.cat(bullet_hero_result_list, dim=1)
        reshape_bullet = bullet_hero_concat_result.reshape(-1, 9, 32)
        bullet_hero_concat_result, _ = reshape_bullet.max(dim=1)

        # 防御塔子弹
        bullet_organ_result_list = []
        for index in range(len(_bullet_10)):
            bullet_organ_mlp_out = self.process_sub_feature(_bullet_10[index], self.bullet_mlp, False)
            bullet_organ_fc_out = self.bullet_organ_fc(bullet_organ_mlp_out)
            bullet_organ_result_list.append(bullet_organ_fc_out)
        bullet_organ_concat_result = torch.cat(bullet_organ_result_list, dim=1)

        tar_embed_0 = 0.1 * torch.ones_like(tar_embed_list[-1]).to(feature_vec.device)
        tar_embed_list.insert(0, tar_embed_0)    # target 0 填充

        tar_embed_8 = 0.1 * torch.ones_like(tar_embed_list[-1]).to(feature_vec.device)
        tar_embed_list.append(tar_embed_8)    # target 8 填充

        tar_embedding = torch.stack(tar_embed_list, dim=1)    # (B*L, 9, 32)

        concat_result = torch.cat(
            [
                hero_frd_concat_result,
                hero_emy_concat_result,
                soldier_frd_concat_result,
                soldier_emy_concat_result,
                river_crab_result,
                organ_frd_concat_result,
                organ_emy_concat_result,
                bullet_hero_concat_result,
                bullet_organ_concat_result,
            ],
            dim=1,
        )

        # 公共LSTM
        fc_public_result = self.concat_mlp(concat_result)
        reshape_fc_public_result = fc_public_result.reshape(-1, self.lstm_time_steps, self.lstm_unit_size)    # (B, L, N)
        lstm_initial_state_in = [    # 第一个隐藏状态 h0，第二个细胞状态 c0
            lstm_hidden_init.unsqueeze(0),    # (1, B, N)
            lstm_cell_init.unsqueeze(0),    # (1, B, N)
        ]
        lstm_outputs, state = self.lstm(reshape_fc_public_result, lstm_initial_state_in)
        lstm_hidden_output = state[0]
        lstm_cell_output = state[1]
        lstm_outputs = lstm_outputs.reshape(-1, self.lstm_unit_size)
        # 公共全连接层
        public_mlp_result = self.concate_mlp_other(concat_result)

        # concat两个部分, 做全连接
        lstm_outputs = torch.cat([lstm_outputs, public_mlp_result], dim=-1)
        lstm_outputs = self.lstm_and_linear_mlp(lstm_outputs)

        # (B*L, N)
        reshape_lstm_outputs_result = lstm_outputs.reshape(-1, self.dim_public)

        if Config.MULTI_HEAD:    # 多输出头
            hero_split_nums = [x * self.lstm_time_steps for x in hero_split_nums]
            lstm_outputs_list = reshape_lstm_outputs_result.split(hero_split_nums, dim=0)    # (Bi*L, N)
            tar_embedding_list = tar_embedding.split(hero_split_nums, dim=0)    # (Bi*L, 9, 32)

            result_list = [[] for _ in range(7)]
            # 当前英雄对应的编号 (用不同的输出头网络)
            for i in range(3):
                label_mlp = self.label_mlps[i]
                lstm_tar_embed_mlp = self.lstm_tar_embed_mlps[i]
                target_embed_mlp = self.target_embed_mlps[i]
                value_mlp = self.value_mlps[i]

                # 前5个Action预测
                for label_index in range(len(self.label_size_list[:-1])):
                    label_mlp_out = label_mlp[f"hero_label{label_index}_mlp"](lstm_outputs_list[i])
                    result_list[label_index].append(label_mlp_out)

                # 制作value
                lstm_tar_embed_result = lstm_tar_embed_mlp(lstm_outputs_list[i])    # (Bi*L, N_e), N_e表示embed维度32
                reshape_label_result = lstm_tar_embed_result.reshape(-1, self.target_embed_dim, 1)    # (Bi*L, N_e, 1)

                # 制作query
                ulti_tar_embedding = target_embed_mlp(tar_embedding_list[i])    # (Bi*L, 9, N_e)
                ulti_tar_embedding = nn.functional.softmax(ulti_tar_embedding, dim=-1)    # 对query最后一个维度做softmax

                # 计算自注意力, 得到最后一个target action预测
                label_result = torch.matmul(ulti_tar_embedding, reshape_label_result)    # (Bi*L, 9, 1)
                target_output_dim = int(np.prod(label_result.shape[1:]))
                reshape_label_result = label_result.reshape(-1, target_output_dim)    # (Bi*L, 9)
                result_list[-2].append(reshape_label_result)

                # 输出价值
                value_result = value_mlp(lstm_outputs_list[i])
                result_list[-1].append(value_result)

            for i in range(len(result_list)):    # 合并
                result_list[i] = torch.cat(result_list[i], dim=0)

        else:    # 单输出头
            result_list = []
            # 前5个Action预测
            for label_index in range(len(self.label_size_list[:-1])):
                label_mlp_out = self.label_mlp[f"hero_label{label_index}_mlp"](reshape_lstm_outputs_result)
                result_list.append(label_mlp_out)

            # 制作value
            lstm_tar_embed_result = self.lstm_tar_embed_mlp(reshape_lstm_outputs_result)    # (B*L, N_e), N_e表示embed维度32
            reshape_label_result = lstm_tar_embed_result.reshape(-1, self.target_embed_dim, 1)    # (B*L, N_e, 1)

            # 制作query
            ulti_tar_embedding = self.target_embed_mlp(tar_embedding)    # (B*L, 9, N_e)
            ulti_tar_embedding = nn.functional.softmax(ulti_tar_embedding, dim=-1)    # 对query最后一个维度做softmax

            # 计算自注意力, 得到最后一个target action预测
            label_result = torch.matmul(ulti_tar_embedding, reshape_label_result)    # (B*L, 9, 1)
            target_output_dim = int(np.prod(label_result.shape[1:]))
            reshape_label_result = label_result.reshape(-1, target_output_dim)    # (B*L, 9)
            result_list.append(reshape_label_result)

            # 输出价值
            value_result = self.value_mlp(reshape_lstm_outputs_result)
            result_list.append(value_result)

        # 准备推理图
        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)    # (B*L, 12+16*4+9), 这里flatten也是莫名其妙
        value = result_list[-1]

        if inference:    # 只有推理时候需要用到连续的lstm状态信息
            return [logits, value, lstm_cell_output, lstm_hidden_output]
        else:    # 训练时候不需要用到lstm状态信息
            return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        # squeeze tensor
        # 压缩张量
        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]

        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)

        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        # loss of value net
        # 值网络的损失
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = 0.5 * torch.mean(torch.square(reward - fc2_value_result_squeezed), dim=0)
        # new_advantage = reward - fc2_value_result_squeezed
        # self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        # for entropy loss calculate
        # 用于熵损失计算
        label_logits_subtract_max_list = []
        label_sum_exp_logits_list = []
        label_probability_list = []

        epsilon = 1e-5

        # policy loss: ppo clip loss
        # 策略损失：PPO剪辑损失
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = torch.tensor(0.0)
                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])

                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary

                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_logits_subtract_max_list.append(label_logits_subtract_max)

                label_exp_logits = (
                    legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                )

                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
                label_sum_exp_logits_list.append(label_sum_exp_logits)

                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * 1
                ) / torch.maximum(torch.sum((weight_list[task_index].float()) * 1), torch.tensor(1.0))

                self.policy_cost = self.policy_cost + temp_policy_loss

        # cross entropy loss
        # 交叉熵损失
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )

                temp_entropy_loss = -torch.sum(
                    (temp_entropy_loss * weight_list[task_index].float() * 1)
                ) / torch.maximum(torch.sum(weight_list[task_index].float() * 1), torch.tensor(1.0))

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = torch.tensor(0.0)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = torch.tensor(0.0)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """Wrapper function to create and initialize a linear layer

    Args:
        in_features (int): ``in_features``
        out_features (int): ``out_features``

    Returns:
        nn.Linear: the initialized linear layer
    """
    """ 创建并初始化线性层的包装函数

    参数:
        in_features (int): 输入特征数
        out_features (int): 输出特征数

    返回:
        nn.Linear: 初始化的线性层
    """
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        """Create a MLP object

        Args:
            fc_feat_dim_list (List[int]): ``in_features`` of the first linear layer followed by
                ``out_features`` of each linear layer
            name (str): human-friendly name, serving as prefix of each comprising layers
            non_linearity (nn.Module, optional): the activation function to use. Defaults to nn.ReLU.
            non_linearity_last (bool, optional): whether to append a activation function in the end.
                Defaults to False.
        """
        """ 创建一个MLP对象

        参数:
            fc_feat_dim_list (List[int]): 第一个线性层的输入特征数，后续每个线性层的输出特征数
            name (str): 人类友好的名称，作为每个组成层的前缀
            non_linearity (nn.Module, optional): 要使用的激活函数。默认为 nn.ReLU。
            non_linearity_last (bool, optional): 是否在最后附加一个激活函数。默认为 False。
        """
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)

def load_pretrain_single_head(model_multi: Model, state_dict_single: dict):
    """
    单输出头single与多输出头输出头multi差异如下:
    single [module_key].fc_layers.[layer_name].[weight,bias]
    multi  [module_key]s.[0,1,2].fc_layers.[layer_name].[weight,bias]
    因此只需要按照.进行分割, 替换前两个字符串即可
    """
    state_dict = model_multi.state_dict()
    same_dict = {k: v for k, v in state_dict_single.items() if k in state_dict and v.shape == state_dict[k].shape}
    from pprint import pprint
    head_module_keys = {
        'label_mlps': 'label_mlp',
        'lstm_tar_embed_mlps': 'lstm_tar_embed_mlp',
        'target_embed_mlps': 'target_embed_mlp',
        'value_mlps': 'value_mlp',
    }
    diff_keys = set(state_dict.keys()) - set(same_dict.keys())
    for key in diff_keys:
        module, _, *suffix = key.split('.')
        load_module = head_module_keys[module]
        load_key = '.'.join([load_module] + suffix)
        same_dict[key] = state_dict_single[load_key].clone()
    model_multi.load_state_dict(same_dict)

def test_model(model: Model):
    # 检测模型
    # model.set_train_mode()
    model.set_eval_mode()
    B, L = 16, model.lstm_time_steps
    x = torch.zeros((B*L, Args.DIM_ALL))
    lstm_hidden, lstm_cell = torch.zeros((B, Config.LSTM_UNIT_SIZE)), torch.zeros((B, Config.LSTM_UNIT_SIZE))
    hero_split_nums = [5, 8, 3]
    data_list = [x, lstm_hidden, lstm_cell, hero_split_nums]
    y = model(data_list, inference=True)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape)
    # torch.save(model.state_dict(), 'test_model.pkl')
    # torch.onnx.export(model, data_list, './onnx_model.onnx')

if __name__ == '__main__':
    model = Model()
    # state_dict_single = torch.load(r"/home/yy/Downloads/v1_2_3_260128/ckpt/model.ckpt-260128.pkl")
    # load_pretrain_single_head(model, state_dict_single)
    test_model(model)
