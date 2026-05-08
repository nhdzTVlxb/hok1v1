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

from agent_ppo.model.model_single import Model as SingleModel
from agent_ppo.conf.conf import DimConfig, Config, Args


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.models = nn.ModuleList([SingleModel() for _ in range(3)])
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.m_var_beta = Config.BETA_START
        self.var_beta = self.m_var_beta

    def forward(self, data_list, inference=False):
        # 在train时L=LSTM_TIME_STEPS, 在eval时L=1
        # 三个向量的shape=(B*L, N), (B, lstm_N), (B, lstm_N), 最后nums=[n1, n2, n3]
        feature_vec, lstm_hidden_init, lstm_cell_init, hero_split_nums = data_list
        hero_split_nums_L = [x * self.lstm_time_steps for x in hero_split_nums]
        feature_vecs = feature_vec.split(hero_split_nums_L, dim=0)  # (Bi*L, N)
        lstm_hidden_inits = lstm_hidden_init.split(hero_split_nums, dim=0)  # (Bi, lstm_N)
        lstm_cell_inits = lstm_cell_init.split(hero_split_nums, dim=0)  # (Bi, lstm_N)
        result_list = [[] for _ in range(4 if inference else 7)]
        for i in range(3):
            rs = self.models[i]([feature_vecs[i], lstm_hidden_inits[i], lstm_cell_inits[i]], inference)
            for j, x in enumerate(rs):
                result_list[j].append(x)
        for i in range(len(result_list)):
            if inference and i >= 2:  # LSTM输出维度为(1, Bi, lstm_N)
                result_list[i] = torch.cat(result_list[i], dim=1)
            else:
                result_list[i] = torch.cat(result_list[i], dim=0)
        return result_list  # 推理输出4个, 训练输出7个

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
        for model in self.models:
            model.set_train_mode()
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        for model in self.models:
            model.set_eval_mode()
        self.eval()

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
    test_model(model)
