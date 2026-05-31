#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""2024-style typed PPO network adapted to Phase 1 2026 features."""

from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict

from agent_ppo.conf.conf import Config, DimConfig


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        self.hero_frd_dim = int(np.sum(DimConfig.DIM_OF_HERO_FRD))
        self.hero_emy_dim = int(np.sum(DimConfig.DIM_OF_HERO_EMY))
        self.soldier_frd_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_1_6))
        self.soldier_emy_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_7_12))
        self.organ_frd_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1_3))
        self.organ_emy_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_4_6))
        self.global_feature_dim = int(np.sum(DimConfig.DIM_OF_GLOBAL_INFO))

        self.all_hero_feature_dim = self.hero_frd_dim + self.hero_emy_dim
        self.all_soldier_feature_dim = self.soldier_frd_dim + self.soldier_emy_dim
        self.all_organ_feature_dim = self.organ_frd_dim + self.organ_emy_dim

        self.hero_frd_mlp = MLP([self.hero_frd_dim, 512, 256, 128], "hero_frd_mlp")
        self.hero_emy_mlp = MLP([self.hero_emy_dim, 512, 256, 128], "hero_emy_mlp")

        self.soldier_mlp = MLP([24, 64, 64], "soldier_mlp", non_linearity_last=True)
        self.soldier_frd_fc = make_fc_layer(64, 32)
        self.soldier_emy_fc = make_fc_layer(64, 32)

        self.organ_mlp = MLP([17, 64, 64], "organ_mlp", non_linearity_last=True)
        self.organ_frd_fc = make_fc_layer(64, 32)
        self.organ_emy_fc = make_fc_layer(64, 32)

        concat_dim = 128 + 128 + 32 + 32 + 32 + 32 + self.global_feature_dim
        self.concat_mlp = MLP([concat_dim, 512], "concat_mlp", non_linearity_last=True)

        self.gru = torch.nn.GRU(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.label_mlp = ModuleDict(
            {
                f"hero_label{label_index}_mlp": MLP(
                    [self.lstm_unit_size, 128, self.label_size_list[label_index]],
                    f"hero_label{label_index}_mlp",
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )
        self.gru_tar_embed_mlp = MLP([self.lstm_unit_size, 128, self.target_embed_dim], "gru_tar_embed_mlp")
        self.value_mlp = MLP([self.lstm_unit_size, 128, 1], "hero_value_mlp")
        self.target_embed_mlp = make_fc_layer(32, self.target_embed_dim, use_bias=False)

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        result_list = []

        hero_part, soldier_part, organ_part, global_info = feature_vec.split(
            [
                self.all_hero_feature_dim,
                self.all_soldier_feature_dim,
                self.all_organ_feature_dim,
                self.global_feature_dim,
            ],
            dim=1,
        )
        hero_frd, hero_emy = hero_part.split([self.hero_frd_dim, self.hero_emy_dim], dim=1)
        soldier_frd, soldier_emy = soldier_part.split([self.soldier_frd_dim, self.soldier_emy_dim], dim=1)
        organ_frd, organ_emy = organ_part.split([self.organ_frd_dim, self.organ_emy_dim], dim=1)

        hero_frd_result = self.hero_frd_mlp(hero_frd)
        hero_emy_result = self.hero_emy_mlp(hero_emy)
        tar_embed_list = [hero_emy_result[:, -32:], hero_frd_result[:, -32:]]

        soldier_frd_slots = soldier_frd.split(DimConfig.DIM_OF_SOLDIER_1_6, dim=1)
        soldier_emy_slots = soldier_emy.split(DimConfig.DIM_OF_SOLDIER_7_12, dim=1)
        soldier_frd_embed = []
        soldier_emy_embed = []
        for slot in soldier_frd_slots:
            soldier_frd_embed.append(self.soldier_frd_fc(self.soldier_mlp(slot)))
        for slot_index, slot in enumerate(soldier_emy_slots):
            embed = self.soldier_emy_fc(self.soldier_mlp(slot))
            soldier_emy_embed.append(embed)
            if slot_index < 4:
                tar_embed_list.append(embed)
        soldier_frd_pool = torch.stack(soldier_frd_embed, dim=1).max(dim=1).values
        soldier_emy_pool = torch.stack(soldier_emy_embed, dim=1).max(dim=1).values

        organ_frd_slots = organ_frd.split(DimConfig.DIM_OF_ORGAN_1_3, dim=1)
        organ_emy_slots = organ_emy.split(DimConfig.DIM_OF_ORGAN_4_6, dim=1)
        organ_frd_embed = []
        organ_emy_embed = []
        for slot in organ_frd_slots:
            organ_frd_embed.append(self.organ_frd_fc(self.organ_mlp(slot)))
        for slot in organ_emy_slots:
            organ_emy_embed.append(self.organ_emy_fc(self.organ_mlp(slot)))
        organ_frd_pool = torch.stack(organ_frd_embed, dim=1).max(dim=1).values
        organ_emy_pool = torch.stack(organ_emy_embed, dim=1).max(dim=1).values
        tar_embed_list.append(organ_emy_pool)

        tar_embed_0 = 0.1 * torch.ones_like(tar_embed_list[-1]).to(feature_vec.device)
        tar_embed_8 = 0.1 * torch.ones_like(tar_embed_list[-1]).to(feature_vec.device)
        tar_embed_list.insert(0, tar_embed_0)
        tar_embed_list.append(tar_embed_8)

        concat_result = torch.cat(
            [
                soldier_frd_pool,
                soldier_emy_pool,
                organ_frd_pool,
                organ_emy_pool,
                hero_frd_result,
                hero_emy_result,
                global_info,
            ],
            dim=1,
        )
        fc_public_result = self.concat_mlp(concat_result)
        gru_input = fc_public_result.reshape(-1, self.lstm_time_steps, self.lstm_unit_size)
        gru_initial_state = lstm_hidden_init.unsqueeze(0)
        gru_outputs, state = self.gru(gru_input, gru_initial_state)
        gru_outputs = torch.cat([gru_outputs[:, idx, :] for idx in range(gru_outputs.size(1))], dim=1)

        self.lstm_hidden_output = state
        self.lstm_cell_output = lstm_cell_init.unsqueeze(0)
        gru_flat = gru_outputs.reshape(-1, self.lstm_unit_size)

        for label_index in range(len(self.label_size_list) - 1):
            result_list.append(self.label_mlp[f"hero_label{label_index}_mlp"](gru_flat))

        gru_tar_embed_result = self.gru_tar_embed_mlp(gru_flat)
        tar_embedding = self.target_embed_mlp(torch.stack(tar_embed_list, dim=1))
        target_logits = torch.matmul(tar_embedding, gru_tar_embed_result.reshape(-1, self.target_embed_dim, 1))
        result_list.append(target_logits.reshape(-1, int(np.prod(target_logits.shape[1:]))))

        value_result = self.value_mlp(gru_flat)
        result_list.append(value_result)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
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

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [weight.squeeze(dim=1) for weight in usq_weight_list]
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

        value_squeezed = value_result.squeeze(dim=1)
        new_advantage = reward - value_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        label_probability_list = []
        epsilon = 1e-5
        self.policy_cost = torch.tensor(0.0, device=value_result.device)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                boundary = torch.pow(torch.tensor(10.0, device=value_result.device), torch.tensor(20.0, device=value_result.device))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])
                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary
                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(label_result[task_index] - legal_action_flag_list_max_mask, dim=1, keepdim=True).values,
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
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * weight_list[task_index].float() * frame_is_train
                ) / torch.maximum(torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0, device=value_result.device))
                self.policy_cost = self.policy_cost + temp_policy_loss

        entropy_loss_list = []
        current_entropy_loss_index = 0
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                entropy = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )
                entropy = -torch.sum(entropy * weight_list[task_index].float() * frame_is_train) / torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0, device=value_result.device)
                )
                entropy_loss_list.append(entropy)
                current_entropy_loss_index += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0, device=value_result.device))

        self.entropy_cost = torch.tensor(0.0, device=value_result.device)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element
        self.entropy_cost_list = entropy_loss_list
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
    nn.init.orthogonal(fc_layer.weight)
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
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i + 1}", fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_non_linear{i + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
