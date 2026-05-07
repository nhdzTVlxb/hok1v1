#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
import os
import time
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0

        self.logger = logger
        self.monitor = monitor

        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS

        self.last_report_monitor_time = 0

    def learn(self, list_sample_data):
        _input_datas = torch.stack([t.npdata for t in list_sample_data], dim=0)
        batch_size = _input_datas.shape[0]
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        
        # Correctly reshape data to preserve batch and time dimensions
        # 不再将所有数据展平，而是保留batch和time维度
        for i, (data, shape) in enumerate(zip(data_list, self.data_split_shape)):
            # The last two elements are LSTM states, which don't have a time dimension
            # 最后两个是LSTM初始状态，没有时间维度
            if i < len(self.data_split_shape) - 2:
                data_list[i] = data.reshape(batch_size, self.lstm_time_steps, shape).float()
            else:
                data_list[i] = data.reshape(batch_size, shape).float()

        seri_vec = data_list[0]
        
        try:
            feature, legal_action = seri_vec.split(
                [
                    np.prod(self.seri_vec_split_shape[0]),
                    np.prod(self.seri_vec_split_shape[1]),
                ],
                dim=2,
            )
            self.logger.info(f"Feature shape after split: {feature.shape}")
            self.logger.info(f"Legal action shape after split: {legal_action.shape}")
        except Exception as e:
            self.logger.error(f"Error in seri_vec split: {e}")
            self.logger.error(f"seri_vec.shape: {seri_vec.shape}")
            raise
        
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(batch_size, self.lstm_time_steps, self.seri_vec_split_shape[0][0])
        
        # LSTM states are already correctly shaped [batch_size, lstm_unit_size]
        lstm_hidden_state = init_lstm_hidden
        lstm_cell_state = init_lstm_cell

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # grad clip
        # 梯度剪裁
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        # update the learning rate
        # 更新学习率
        self.scheduler.step(self.train_step)

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now
