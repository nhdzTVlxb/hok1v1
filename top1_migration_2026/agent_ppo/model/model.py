#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Entity-structured PPO model adapted from the 2025 top1 solution.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict
from typing import List

from agent_ppo.conf.conf import DimConfig, Config, Args


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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
        self.var_beta = self.m_var_beta
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.single_hero_feature_dim = int(DimConfig.DIM_OF_HERO_EMY[0])
        self.single_soldier_feature_dim = int(DimConfig.DIM_OF_SOLDIER_1_4[0])
        self.single_organ_feature_dim = int(DimConfig.DIM_OF_ORGAN_1[0])
        self.single_monster_feature_dim = int(DimConfig.DIM_OF_MONSTER[0])
        self.single_bullet_feature_dim = int(DimConfig.DIM_OF_BULLET_1_9[0])

        self.all_hero_feature_dim = int(np.sum(DimConfig.DIM_OF_HERO_FRD) + np.sum(DimConfig.DIM_OF_HERO_EMY))
        self.all_soldier_feature_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_1_4) + np.sum(DimConfig.DIM_OF_SOLDIER_5_8))
        self.all_organ_feature_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1) + np.sum(DimConfig.DIM_OF_ORGAN_2))
        self.all_bullet_feature_dim = int(np.sum(DimConfig.DIM_OF_BULLET_1_9) + np.sum(DimConfig.DIM_OF_BULLET_10))

        self.position_delta_dim = 64 - Args.DIM_DISTANCE
        self.position_mlp = MLP([Args.DIM_DISTANCE, 128, 64], "position_mlp")

        self.unit_delta_dim = 64 + 32 - Args.DIM_UNIT
        self.unit_no_pos_mlp = MLP([Args.DIM_UNIT - Args.DIM_DISTANCE, 64, 32], "unit_no_pos_mlp")

        self.hero_mlp = MLP([self.single_hero_feature_dim + self.unit_delta_dim, 512, 256], "hero_mlp", non_linearity_last=True)
        self.hero_frd_fc = make_fc_layer(256, 128)
        self.hero_emy_fc = make_fc_layer(256, 128)

        self.soldier_mlp = MLP([self.single_soldier_feature_dim + self.unit_delta_dim, 128, 64], "soldier_mlp", non_linearity_last=True)
        self.soldier_frd_fc = make_fc_layer(64, 32)
        self.soldier_emy_fc = make_fc_layer(64, 32)

        self.monster_mlp = MLP([self.single_monster_feature_dim + self.unit_delta_dim, 128, 64, 32], "monster_mlp")

        self.organ_mlp = MLP([self.single_organ_feature_dim + self.unit_delta_dim, 128, 64], "organ_mlp", non_linearity_last=True)
        self.organ_frd_fc = make_fc_layer(64, 32)
        self.organ_emy_fc = make_fc_layer(64, 32)

        self.bullet_mlp = MLP([self.single_bullet_feature_dim + self.position_delta_dim, 64, 64], "bullet_mlp", non_linearity_last=True)
        self.bullet_hero_fc = make_fc_layer(64, 32)
        self.bullet_organ_fc = make_fc_layer(64, 32)

        concat_dim = 128 * 2 + 32 * 2 + 32 + 32 * 2 + 32 * 2
        self.concat_mlp = MLP([concat_dim, self.lstm_unit_size], "concat_mlp", non_linearity_last=True)
        self.concat_mlp_other = MLP([concat_dim, 512, self.lstm_unit_size], "concat_other_mlp")
        self.lstm_and_linear_mlp = MLP([self.lstm_unit_size * 2, self.dim_public], "lstm_and_linear_mlp", non_linearity_last=True)

        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=Config.LSTM_DROPOUT,
            bidirectional=False,
        )

        self.label_mlp = ModuleDict(
            {
                f"hero_label{label_index}_mlp": MLP([self.dim_public, self.label_size_list[label_index]], f"hero_label{label_index}_mlp")
                for label_index in range(len(self.label_size_list) - 1)
            }
        )
        self.lstm_tar_embed_mlp = make_fc_layer(self.dim_public, self.target_embed_dim)
        self.target_embed_mlp = make_fc_layer(32, self.target_embed_dim, use_bias=False)
        self.value_mlp = MLP([self.dim_public, 64, 1], "hero_value_mlp")

    def process_sub_feature(self, x, mlp, is_unit):
        ret = [x]
        dim_suffix = Args.DIM_DISTANCE
        if is_unit:
            unit_no_pos = self.unit_no_pos_mlp(x[..., -Args.DIM_UNIT : -Args.DIM_DISTANCE])
            ret.append(unit_no_pos)
            dim_suffix = Args.DIM_UNIT
        pos = self.position_mlp(x[..., -Args.DIM_DISTANCE :])
        ret.append(pos)
        ret[0] = x[..., :-dim_suffix]
        return mlp(torch.cat(ret, dim=-1))

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        feature_vec_split_list = feature_vec.split(
            [
                self.all_hero_feature_dim,
                self.all_soldier_feature_dim,
                self.single_monster_feature_dim,
                self.all_organ_feature_dim,
                self.all_bullet_feature_dim,
            ],
            dim=1,
        )
        hero_vec_list = feature_vec_split_list[0].split([int(np.sum(DimConfig.DIM_OF_HERO_FRD)), int(np.sum(DimConfig.DIM_OF_HERO_EMY))], dim=1)
        soldier_vec_list = feature_vec_split_list[1].split([int(np.sum(DimConfig.DIM_OF_SOLDIER_1_4)), int(np.sum(DimConfig.DIM_OF_SOLDIER_5_8))], dim=1)
        monster_tensor = feature_vec_split_list[2]
        organ_vec_list = feature_vec_split_list[3].split([int(np.sum(DimConfig.DIM_OF_ORGAN_1)), int(np.sum(DimConfig.DIM_OF_ORGAN_2))], dim=1)
        bullet_vec_list = feature_vec_split_list[4].split([int(np.sum(DimConfig.DIM_OF_BULLET_1_9)), int(np.sum(DimConfig.DIM_OF_BULLET_10))], dim=1)

        _hero_frd = hero_vec_list[0].split(DimConfig.DIM_OF_HERO_FRD, dim=1)
        _hero_emy = hero_vec_list[1].split(DimConfig.DIM_OF_HERO_EMY, dim=1)
        _soldier_frd = soldier_vec_list[0].split(DimConfig.DIM_OF_SOLDIER_1_4, dim=1)
        _soldier_emy = soldier_vec_list[1].split(DimConfig.DIM_OF_SOLDIER_5_8, dim=1)
        _organ_frd = organ_vec_list[0].split(DimConfig.DIM_OF_ORGAN_1, dim=1)
        _organ_emy = organ_vec_list[1].split(DimConfig.DIM_OF_ORGAN_2, dim=1)
        _bullet_hero = bullet_vec_list[0].split(DimConfig.DIM_OF_BULLET_1_9, dim=1)
        _bullet_organ = bullet_vec_list[1].split(DimConfig.DIM_OF_BULLET_10, dim=1)

        tar_embed_list = []

        hero_emy_result_list = []
        for hero in _hero_emy:
            out = self.hero_emy_fc(self.process_sub_feature(hero, self.hero_mlp, True))
            _, target_part = out.split([96, 32], dim=1)
            tar_embed_list.append(target_part)
            hero_emy_result_list.append(out)
        hero_emy_concat_result = torch.cat(hero_emy_result_list, dim=1)

        hero_frd_result_list = []
        for hero in _hero_frd:
            out = self.hero_frd_fc(self.process_sub_feature(hero, self.hero_mlp, True))
            _, target_part = out.split([96, 32], dim=1)
            tar_embed_list.append(target_part)
            hero_frd_result_list.append(out)
        hero_frd_concat_result = torch.cat(hero_frd_result_list, dim=1)

        soldier_frd_result_list = []
        for soldier in _soldier_frd:
            soldier_frd_result_list.append(self.soldier_frd_fc(self.process_sub_feature(soldier, self.soldier_mlp, True)))
        soldier_frd_concat_result = torch.cat(soldier_frd_result_list, dim=1).reshape(-1, Args.SOLDIER_MAX_NUM, 32).max(dim=1).values

        soldier_emy_result_list = []
        for soldier in _soldier_emy:
            out = self.soldier_emy_fc(self.process_sub_feature(soldier, self.soldier_mlp, True))
            soldier_emy_result_list.append(out)
            tar_embed_list.append(out)
        soldier_emy_concat_result = torch.cat(soldier_emy_result_list, dim=1).reshape(-1, Args.SOLDIER_MAX_NUM, 32).max(dim=1).values

        monster_result = self.process_sub_feature(monster_tensor, self.monster_mlp, True)

        organ_frd_result_list = []
        for organ in _organ_frd:
            organ_frd_result_list.append(self.organ_frd_fc(self.process_sub_feature(organ, self.organ_mlp, True)))
        organ_frd_concat_result = torch.cat(organ_frd_result_list, dim=1)

        organ_emy_result_list = []
        for organ in _organ_emy:
            out = self.organ_emy_fc(self.process_sub_feature(organ, self.organ_mlp, True))
            organ_emy_result_list.append(out)
            tar_embed_list.append(out)
        organ_emy_concat_result = torch.cat(organ_emy_result_list, dim=1)

        bullet_hero_result_list = []
        for bullet in _bullet_hero:
            bullet_hero_result_list.append(self.bullet_hero_fc(self.process_sub_feature(bullet, self.bullet_mlp, False)))
        bullet_hero_concat_result = torch.cat(bullet_hero_result_list, dim=1).reshape(-1, 9, 32).max(dim=1).values

        bullet_organ_result_list = []
        for bullet in _bullet_organ:
            bullet_organ_result_list.append(self.bullet_organ_fc(self.process_sub_feature(bullet, self.bullet_mlp, False)))
        bullet_organ_concat_result = torch.cat(bullet_organ_result_list, dim=1)

        target_pad = 0.1 * torch.ones_like(tar_embed_list[-1]).to(feature_vec.device)
        tar_embed_list.insert(0, target_pad)
        tar_embed_list.append(0.1 * torch.ones_like(tar_embed_list[-1]).to(feature_vec.device))
        tar_embedding = torch.stack(tar_embed_list, dim=1)

        concat_result = torch.cat(
            [
                hero_frd_concat_result,
                hero_emy_concat_result,
                soldier_frd_concat_result,
                soldier_emy_concat_result,
                monster_result,
                organ_frd_concat_result,
                organ_emy_concat_result,
                bullet_hero_concat_result,
                bullet_organ_concat_result,
            ],
            dim=1,
        )

        fc_public_result = self.concat_mlp(concat_result)
        reshape_fc_public_result = fc_public_result.reshape(-1, self.lstm_time_steps, self.lstm_unit_size)
        lstm_outputs, state = self.lstm(
            reshape_fc_public_result,
            [lstm_hidden_init.unsqueeze(0), lstm_cell_init.unsqueeze(0)],
        )
        lstm_hidden_output, lstm_cell_output = state
        lstm_outputs = lstm_outputs.reshape(-1, self.lstm_unit_size)
        public_mlp_result = self.concat_mlp_other(concat_result)
        public_result = self.lstm_and_linear_mlp(torch.cat([lstm_outputs, public_mlp_result], dim=-1)).reshape(-1, self.dim_public)

        result_list = []
        for label_index in range(len(self.label_size_list[:-1])):
            result_list.append(self.label_mlp[f"hero_label{label_index}_mlp"](public_result))

        lstm_tar_embed_result = self.lstm_tar_embed_mlp(public_result)
        target_value = lstm_tar_embed_result.reshape(-1, self.target_embed_dim, 1)
        target_query = nn.functional.softmax(self.target_embed_mlp(tar_embedding), dim=-1)
        target_label = torch.matmul(target_query, target_value).reshape(-1, self.label_size_list[-1])
        result_list.append(target_label)

        value_result = self.value_mlp(public_result)
        result_list.append(value_result)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]

        if inference:
            return [logits, value, lstm_cell_output, lstm_hidden_output]
        return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        reward = data_list[1].reshape(-1, self.data_split_shape[1]).squeeze(dim=1)
        advantage = data_list[2].reshape(-1, self.data_split_shape[2]).squeeze(dim=1)
        frame_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3]).squeeze(dim=1)

        label_list = []
        for shape_index in range(len(self.label_size_list)):
            label_list.append(data_list[3 + shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long().squeeze(dim=1))

        old_label_probability_list = []
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list.append(
                data_list[3 + len(self.label_size_list) + shape_index].reshape(
                    -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
                )
            )

        weight_list = []
        for shape_index in range(len(self.label_size_list)):
            weight_list.append(
                data_list[3 + 2 * len(self.label_size_list) + shape_index]
                .reshape(-1, self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index])
                .squeeze(dim=1)
            )

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        legal_action_flag_list = torch.split(split_feature_legal_action.reshape(-1, np.prod(self.seri_vec_split_shape[1])), self.label_size_list, dim=1)

        value_result = rst_list[-1].squeeze(dim=1)
        new_advantage = reward - value_result
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        label_result = rst_list[:-1]
        label_probability_list = []
        epsilon = 1e-5
        self.policy_cost = torch.tensor(0.0, device=value_result.device)

        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                boundary = torch.pow(torch.tensor(10.0, device=value_result.device), torch.tensor(20.0, device=value_result.device))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])
                legal_action_mask = (1 - legal_action_flag_list[task_index]) * boundary
                label_logits_subtract_max = torch.clamp(
                    label_result[task_index] - torch.max(label_result[task_index] - legal_action_mask, dim=1, keepdim=True).values,
                    -boundary,
                    1,
                )
                label_exp_logits = legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                label_probability = label_exp_logits / label_exp_logits.sum(1, keepdim=True)
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                ratio = torch.exp(torch.log(policy_p + epsilon) - torch.log(old_policy_p))
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                denom = torch.maximum(torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0, device=value_result.device))
                self.policy_cost = self.policy_cost - torch.sum(
                    torch.minimum(surr1, surr2) * weight_list[task_index].float() * frame_is_train
                ) / denom

        entropy_loss_list = []
        for task_index, label_probability in enumerate(label_probability_list):
            temp_entropy_loss = -torch.sum(
                label_probability * legal_action_flag_list[task_index] * torch.log(label_probability + epsilon),
                dim=1,
            )
            denom = torch.maximum(torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0, device=value_result.device))
            entropy_loss_list.append(-torch.sum(temp_entropy_loss * weight_list[task_index].float() * frame_is_train) / denom)

        self.entropy_cost = torch.stack(entropy_loss_list).sum()
        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
        return self.loss, [self.loss, [self.value_cost, self.policy_cost, self.entropy_cost]]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
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
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            self.fc_layers.add_module(f"{name}_fc{i + 1}", make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1]))
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_non_linear{i + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
