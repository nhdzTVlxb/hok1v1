#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Compact PPO model for the 128-dim handcrafted feature vector.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict

from agent_ppo.conf.conf import Config


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
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.feature_dim = Config.FEATURE_DIM
        self.target_entity_count = Config.TARGET_ENTITY_COUNT
        self.target_entity_dim = Config.TARGET_ENTITY_DIM
        self.target_feature_dim = Config.TARGET_FEATURE_DIM
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.multi_head = getattr(Config, "MULTI_HEAD", False)
        self.hero_head_ids = getattr(Config, "HERO_HEAD_IDS", [112, 133])
        self.hero_id_feature_slice = getattr(Config, "HERO_ID_FEATURE_SLICE", (0, len(self.hero_head_ids)))

        self.hero_frd_slice = (0, 24)
        self.hero_emy_slice = (24, 42)
        self.organ_slice = (63, 79)
        self.lane_entity_slice = (642, 898)
        self.tower_wave_slice = (898, 994)
        self.projectile_detail_slice = (994, 1154)
        self.target_slice = (self.feature_dim - self.target_feature_dim, self.feature_dim)
        self.hero_frd_mlp = MLP([24, 64, 64], "hero_frd_mlp", non_linearity_last=True)
        self.hero_emy_mlp = MLP([18, 64, 64], "hero_emy_mlp", non_linearity_last=True)
        self.soldier_mlp = MLP([Config.TARGET_ENTITY_DIM, 64, 64], "soldier_entity_mlp", non_linearity_last=True)
        self.organ_mlp = MLP([8, 64, 64], "organ_entity_mlp", non_linearity_last=True)
        self.tower_wave_mlp = MLP([96, 128, 64], "tower_wave_mlp", non_linearity_last=True)
        self.projectile_mlp = MLP([16, 64, 64], "projectile_entity_mlp", non_linearity_last=True)
        self.structured_mlp = MLP([64 * 8, self.dim_public], "structured_mlp", non_linearity_last=True)
        self.feature_mlp = MLP([self.feature_dim + self.dim_public, self.dim_public, self.lstm_unit_size], "feature_mlp", non_linearity_last=True)
        self.side_mlp = MLP([self.feature_dim + self.dim_public, self.dim_public], "side_mlp", non_linearity_last=True)
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=Config.LSTM_DROPOUT,
            bidirectional=False,
        )
        self.public_mlp = MLP([self.lstm_unit_size + self.dim_public, self.dim_public], "public_mlp", non_linearity_last=True)
        if self.multi_head:
            self.label_mlp = ModuleDict(
                {
                    f"hero{head_idx}_label{label_idx}_mlp": MLP(
                        [self.dim_public, 128, size],
                        f"hero{head_idx}_label{label_idx}_mlp",
                    )
                    for head_idx in range(len(self.hero_head_ids))
                    for label_idx, size in enumerate(self.label_size_list[:-1])
                }
            )
        else:
            self.label_mlp = ModuleDict(
                {
                    f"hero_label{idx}_mlp": MLP([self.dim_public, 128, size], f"hero_label{idx}_mlp")
                    for idx, size in enumerate(self.label_size_list[:-1])
                }
            )
        self.target_query_mlp = MLP([self.dim_public, 128, self.target_embed_dim], "target_query_mlp")
        self.target_entity_mlp = MLP(
            [self.target_entity_dim, 64, self.target_embed_dim],
            "target_entity_mlp",
            non_linearity_last=True,
        )
        self.value_mlp = MLP([self.dim_public, 128, 1], "hero_value_mlp")

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        structured_embed = self._structured_feature_embed(feature_vec)
        model_input = torch.cat([feature_vec, structured_embed], dim=1)
        embed = self.feature_mlp(model_input)
        side_embed = self.side_mlp(model_input)

        if inference:
            seq = embed.reshape(-1, 1, self.lstm_unit_size)
        else:
            seq = embed.reshape(-1, self.lstm_time_steps, self.lstm_unit_size)
            side_embed = side_embed.reshape(-1, self.lstm_time_steps, self.dim_public)

        lstm_outputs, state = self.lstm(
            seq,
            [lstm_hidden_init.unsqueeze(0), lstm_cell_init.unsqueeze(0)],
        )
        lstm_hidden_output, lstm_cell_output = state
        lstm_flat = lstm_outputs.reshape(-1, self.lstm_unit_size)
        side_flat = side_embed.reshape(-1, self.dim_public)
        public_result = self.public_mlp(torch.cat([lstm_flat, side_flat], dim=1))

        result_list = self._policy_head_outputs(public_result, feature_vec)
        target_features = feature_vec[:, -self.target_feature_dim :].reshape(
            -1, self.target_entity_count, self.target_entity_dim
        )
        target_embed = self.target_entity_mlp(target_features.reshape(-1, self.target_entity_dim)).reshape(
            -1, self.target_entity_count, self.target_embed_dim
        )
        target_query = self.target_query_mlp(public_result).reshape(-1, self.target_embed_dim, 1)
        target_logits = torch.matmul(target_embed, target_query).reshape(-1, self.target_entity_count)
        result_list.append(target_logits)
        value_result = self.value_mlp(public_result)
        result_list.append(value_result)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value, lstm_cell_output, lstm_hidden_output]
        return result_list

    def _structured_feature_embed(self, feature_vec):
        hero_frd = self.hero_frd_mlp(feature_vec[:, self.hero_frd_slice[0] : self.hero_frd_slice[1]])
        hero_emy = self.hero_emy_mlp(feature_vec[:, self.hero_emy_slice[0] : self.hero_emy_slice[1]])

        organ = feature_vec[:, self.organ_slice[0] : self.organ_slice[1]].reshape(-1, 2, 8)
        organ_embed = self.organ_mlp(organ.reshape(-1, 8)).reshape(-1, 2, 64)
        organ_frd = organ_embed[:, 0, :]
        organ_emy = organ_embed[:, 1, :]

        lane = feature_vec[:, self.lane_entity_slice[0] : self.lane_entity_slice[1]].reshape(-1, 8, Config.TARGET_ENTITY_DIM)
        soldier_embed = self.soldier_mlp(lane.reshape(-1, Config.TARGET_ENTITY_DIM)).reshape(-1, 8, 64)
        soldier_frd = soldier_embed[:, :4, :].max(dim=1).values
        soldier_emy = soldier_embed[:, 4:, :].max(dim=1).values

        tower_wave = self.tower_wave_mlp(feature_vec[:, self.tower_wave_slice[0] : self.tower_wave_slice[1]])

        projectile = feature_vec[:, self.projectile_detail_slice[0] : self.projectile_detail_slice[1]].reshape(-1, 10, 16)
        projectile_embed = self.projectile_mlp(projectile.reshape(-1, 16)).reshape(-1, 10, 64)
        projectile_pool = projectile_embed.max(dim=1).values

        structured = torch.cat(
            [
                hero_frd,
                hero_emy,
                soldier_frd,
                soldier_emy,
                organ_frd,
                organ_emy,
                tower_wave,
                projectile_pool,
            ],
            dim=1,
        )
        return self.structured_mlp(structured)

    def _hero_head_weights(self, feature_vec):
        start, end = self.hero_id_feature_slice
        weights = feature_vec[:, start:end]
        if weights.shape[1] != len(self.hero_head_ids):
            return None
        weights = torch.clamp(weights, 0.0, 1.0)
        total = weights.sum(dim=1, keepdim=True)
        default = torch.zeros_like(weights)
        default[:, 0] = 1.0
        return torch.where(total > 0.0, weights / torch.clamp(total, min=1e-6), default)

    def _policy_head_outputs(self, public_result, feature_vec):
        if not self.multi_head:
            return [
                self.label_mlp[f"hero_label{idx}_mlp"](public_result)
                for idx in range(len(self.label_size_list) - 1)
            ]

        head_weights = self._hero_head_weights(feature_vec)
        if head_weights is None:
            return [
                self.label_mlp[f"hero0_label{idx}_mlp"](public_result)
                for idx in range(len(self.label_size_list) - 1)
            ]
        outputs = []
        for label_idx, _ in enumerate(self.label_size_list[:-1]):
            head_logits = [
                self.label_mlp[f"hero{head_idx}_label{label_idx}_mlp"](public_result)
                for head_idx in range(len(self.hero_head_ids))
            ]
            stacked = torch.stack(head_logits, dim=1)
            outputs.append((stacked * head_weights.unsqueeze(-1)).sum(dim=1))
        return outputs

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
                data_list[3 + len(self.label_size_list) + shape_index]
                .reshape(-1, self.data_split_shape[3 + len(self.label_size_list) + shape_index])
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
        legal_action_flag_list = torch.split(
            split_feature_legal_action.reshape(-1, np.prod(self.seri_vec_split_shape[1])),
            self.label_size_list,
            dim=1,
        )

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
