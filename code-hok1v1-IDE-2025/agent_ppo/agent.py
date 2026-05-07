#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    reset_wrapper,
    load_opponent_agent_wrapper,
)

from agent_ppo.conf.conf import Config
from kaiwu_agent.utils.common_func import attached
from agent_ppo.feature.reward_process import GameRewardManager
from torch.optim.lr_scheduler import LambdaLR
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.feature_process import FeatureProcess


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        # 配置信息
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        # env info
        # 环境信息
        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None

        # learning info
        # 学习信息
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # tools
        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

        # flag
        self.at_low_health = False

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        # Define learning rate decay function
        # 定义学习率衰减函数
        if step > self.target_step:
            return self.target_lr / self.lr
        else:
            return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    @reset_wrapper
    def reset(self, observation):
        # Reset function, called at the beginning of each episode
        # 重置函数，每局开始时调用
        self.hero_camp = observation["player_camp"]
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(f"PLAYERCAMP_{self.hero_camp}")

    def _model_inference(self, list_obs_data):
        # Using the network for inference
        # 使用网络进行推理
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.numpy())

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, d_prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    @predict_wrapper
    def predict(self, observation):
        # Prediction function, usually called during training
        # Returns a random sampling action
        # 预测函数，通常在训练时调用，返回随机采样动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        return action

    @exploit_wrapper
    def exploit(self, observation):
        # Exploitation function, usually called during evaluation
        # Returns the action with the highest probability
        # 利用函数，在评估时调用，返回最大概率动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        d_action = self.action_process(observation, act_data, False)
        # if self.at_low_health:
        #     self.logger.info(f"at low health actions: {d_action}")
        # elif d_action[0] == 2:
        #     self.logger.info(f"normal move: {d_action}")
        return d_action

    def observation_process(self, observation):
        feature = self.feature_processes.process_feature(observation)
        feature_vec, legal_action = (
            feature,
            observation["legal_action"],
        )
        
        # Check if agent health is low and modify legal actions
        # legal_action = self._modify_legal_action_for_low_health(observation, legal_action)
        
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    def _should_go_back_to_spring(self, feature_vec):
        """
        Check if agent should go back to spring based on two conditions:
        1. Main hero hp is less than 0.25 compared to enemy hp
        2. Main hero is in spring and hp is not full
        """
        try:
            # Extract information from processed feature vector
            # You'll need to determine the correct indices based on your feature processing
            # These are placeholder indices - adjust based on your actual feature structure
            
            # Assuming feature vector contains these values at specific indices:
            your_hero_hp = feature_vec[17]  # Main hero HP normalized [0,1]
            if your_hero_hp == 0:
                return False  # Dead heroes should not go back to spring
            your_hero_max_hp = feature_vec[18]
            enemy_hero_hp = feature_vec[65+17]  # Enemy hero HP normalized [0,1]

            # Distance to spring or spring indicator (might be pre-calculated in feature)
            distance_to_spring_x = feature_vec[-2]  # Distance to spring
            distance_to_spring_z = feature_vec[-1]
            is_in_spring = (distance_to_spring_x + distance_to_spring_z) < 0.2  # Threshold for being "in spring"
            
            # Condition 1: Main hero hp is less than 0.25 compared to enemy hp
            condition1 = your_hero_hp < (enemy_hero_hp * 0.25)
            
            # Condition 2: Main hero is in spring and hp is not full
            condition2 = is_in_spring and your_hero_hp < your_hero_max_hp
            
            # Return True if either condition is met
            return condition1 or condition2
            
        except (IndexError, TypeError) as e:
            # If feature processing fails, log error and return False
            self.logger.warning(f"Failed to process features for spring check: {e}")
            return False

    def _modify_legal_action_for_low_health(self, observation, legal_action):
        """
        Modify legal action mask when agent should go back to spring
        Only allow movement towards spring
        """
        frame_state = observation["frame_state"]
        
        # Extract hero information from frame state
        main_hero = None
        enemy_hero = None
        spring = None
        
        # Convert hero_camp number to proper string format for comparison
        hero_camp_str = f"PLAYERCAMP_{self.hero_camp}"
        
        # Find main hero and enemy hero
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == hero_camp_str:
                main_hero = hero
            else:
                enemy_hero = hero

        # Find spring (base) for main hero's camp
        for npc in frame_state["npc_states"]:
            if (npc["sub_type"] == "ACTOR_SUB_TOWER_SPRING" and 
                npc["camp"] == hero_camp_str):
                spring = npc
                break

        if not main_hero or not enemy_hero or not spring:
            return legal_action

        # Check if should go back to spring using extracted hero data
        feature_vec = self.feature_processes.process_feature(observation)
        if not self._should_go_back_to_spring(feature_vec):
            self.at_low_health = False
            return legal_action
        # Set low health flag
        self.at_low_health = True
        try:
            # Get hero and spring positions
            main_hero_x = main_hero["actor_state"]["location"]["x"]
            main_hero_z = main_hero["actor_state"]["location"]["z"]
            spring_x = spring["location"]["x"] * 1.2
            spring_z = spring["location"]["z"] * 1.2
            
            # Transform coordinates for camp 2 (similar to feature processing)
            if hero_camp_str == "PLAYERCAMP_2":
                main_hero_x = -main_hero_x
                main_hero_z = -main_hero_z
                spring_x = -spring_x
                spring_z = -spring_z
            
            # Calculate direction vector to spring
            direction_x = spring_x - main_hero_x
            direction_z = spring_z - main_hero_z
            
            # Normalize the direction vector
            magnitude = np.sqrt(direction_x**2 + direction_z**2)
            if magnitude == 0:
                # Already at spring, stay in place
                normalized_x = 0
                normalized_z = 0
            else:
                normalized_x = direction_x / magnitude
                normalized_z = direction_z / magnitude
            
            # Find the closest direction from 16x16 grid
            move_x_idx, move_z_idx = self._find_closest_direction(normalized_x, normalized_z)
            
            # Create a copy of legal_action to modify
            modified_legal_action = legal_action.copy()
            
            # Split legal action according to action structure
            split_indices = [sum(self.legal_action_size[:i+1]) for i in range(len(self.legal_action_size)-1)]
            legal_actions_split = np.split(modified_legal_action, split_indices)
            
            # Check if Move button has valid targets in the original legal action
            target_legal_reshaped = legal_actions_split[5].reshape([12, 9])
            move_targets = target_legal_reshaped[2]  # Row 2 = Move targets
            
            if np.sum(move_targets) == 0:
                # Move has no valid targets, don't modify legal actions
                self.logger.info("Move button has no valid targets, skipping legal action modification")
                return legal_action
            
            # Button actions: only allow Move (index 2)
            button_legal = legal_actions_split[0]
            button_legal[:] = 0
            button_legal[2] = 1
            
            # Move directions - use the closest direction
            move_x_legal = legal_actions_split[1]
            move_x_legal[:] = 0
            move_x_legal[move_x_idx] = 1
            
            move_z_legal = legal_actions_split[2]
            move_z_legal[:] = 0
            move_z_legal[move_z_idx] = 1
            
            # Get current HP for logging
            main_hero_hp = main_hero["actor_state"]["hp"]
            main_hero_max_hp = main_hero["actor_state"]["max_hp"]
            enemy_hero_hp = enemy_hero["actor_state"]["hp"]
            enemy_hero_max_hp = enemy_hero["actor_state"]["max_hp"]
            
            main_hp_ratio = main_hero_hp / main_hero_max_hp if main_hero_max_hp > 0 else 0
            enemy_hp_ratio = enemy_hero_hp / enemy_hero_max_hp if enemy_hero_max_hp > 0 else 0
            
            self.logger.info(f"Low health retreat: Main HP: {main_hp_ratio:.2f}, Enemy HP: {enemy_hp_ratio:.2f}. "
                            f"Target direction: ({normalized_x:.2f}, {normalized_z:.2f}), "
                            f"Chosen grid: X:{move_x_idx}, Z:{move_z_idx}")

            # Disable skill directions
            skill_x_legal = legal_actions_split[3]
            skill_x_legal[:] = 0
            skill_x_legal[move_x_idx] = 1
            skill_z_legal = legal_actions_split[4]
            skill_z_legal[:] = 0
            skill_z_legal[move_z_idx] = 1
            
            # CRITICAL FIX: Preserve Move targets, zero out other button targets
            target_legal_reshaped = legal_actions_split[5].reshape([12, 9])
            
            # Zero out targets for all buttons except Move (index 2)
            for i in range(12):
                if i != 2:  # Not Move button
                    target_legal_reshaped[i, :] = 0
            
            # Flatten back
            legal_actions_split[5] = target_legal_reshaped.flatten()
            
            # Reconstruct the legal action array
            modified_legal_action = np.concatenate(legal_actions_split)
            
            return modified_legal_action

        except (IndexError, TypeError, KeyError) as e:
            # If processing fails, log error and return original legal action
            self.logger.warning(f"Failed to modify legal actions for spring retreat: {e}")
            return legal_action

    def _find_closest_direction(self, target_x, target_z):
        """
        Find the closest direction in the 16x16 grid to the target direction vector
        """
        # Generate all possible direction vectors for 16x16 grid
        # Grid coordinates range from 0-15, with center at 8
        center = 8
        best_dot_product = -2  # Initialize to value less than minimum possible dot product (-1)
        best_x_idx = center
        best_z_idx = center
        
        for x_idx in range(16):
            for z_idx in range(16):
                # Convert grid indices to direction vector
                grid_x = (x_idx - center) / 8.0  # Normalize to [-1, 1] range
                grid_z = (z_idx - center) / 8.0  # Normalize to [-1, 1] range
                
                # Skip center point (no movement)
                if grid_x == 0 and grid_z == 0:
                    continue
                
                # Normalize the grid direction vector
                grid_magnitude = np.sqrt(grid_x**2 + grid_z**2)
                grid_x_norm = grid_x / grid_magnitude
                grid_z_norm = grid_z / grid_magnitude
                
                # Calculate dot product (cosine similarity) with target direction
                dot_product = target_x * grid_x_norm + target_z * grid_z_norm
                
                # Keep track of the best match
                if dot_product > best_dot_product:
                    best_dot_product = dot_product
                    best_x_idx = x_idx
                    best_z_idx = z_idx
        
        return best_x_idx, best_z_idx

    def action_process(self, observation, act_data, is_stochastic):
        if is_stochastic:
            # Use stochastic sampling action
            # 采用随机采样动作 action
            return act_data.action
        else:
            # Use the action with the highest probability
            # 采用最大概率动作 d_action
            return act_data.d_action

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    @load_opponent_agent_wrapper
    def load_opponent_agent(self, id="1"):
        # Framework provides loading opponent agent function, no need to implement function content
        # 框架提供的加载对手模型功能，无需实现函数内容
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional list
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_prob_list += list(probs)

        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))
