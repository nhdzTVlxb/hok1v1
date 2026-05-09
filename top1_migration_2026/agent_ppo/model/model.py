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

        self.feature_mlp = MLP([self.feature_dim, 256, self.lstm_unit_size], "feature_mlp", non_linearity_last=True)
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=Config.LSTM_DROPOUT,
            bidirectional=False,
        )
        self.public_mlp = MLP([self.lstm_unit_size, self.dim_public], "public_mlp", non_linearity_last=True)
        self.label_mlp = ModuleDict(
            {
                f"hero_label{idx}_mlp": MLP([self.dim_public, 128, size], f"hero_label{idx}_mlp")
                for idx, size in enumerate(self.label_size_list)
            }
        )
        self.value_mlp = MLP([self.dim_public, 128, 1], "hero_value_mlp")

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        embed = self.feature_mlp(feature_vec)

        if inference:
            seq = embed.reshape(-1, 1, self.lstm_unit_size)
        else:
            seq = embed.reshape(-1, self.lstm_time_steps, self.lstm_unit_size)

        lstm_outputs, state = self.lstm(
            seq,
            [lstm_hidden_init.unsqueeze(0), lstm_cell_init.unsqueeze(0)],
        )
        lstm_hidden_output, lstm_cell_output = state
        public_result = self.public_mlp(lstm_outputs.reshape(-1, self.lstm_unit_size))

        result_list = [
            self.label_mlp[f"hero_label{idx}_mlp"](public_result)
            for idx in range(len(self.label_size_list))
        ]
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
