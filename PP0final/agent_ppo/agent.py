#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import random
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.conf.conf import Config
from agent_ppo.conf.conf import GameConfig
from agent_ppo.feature.reward_process import GameRewardManager
from torch.optim.lr_scheduler import LambdaLR
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.phase1_feature_builder import Phase1FeatureBuilder, camp_id, runtime_id, hp_ratio
from agent_ppo.utils_feature_audit import FeatureAuditDumper


# Available summoner skills / 可选召唤师技能
SUMMONER_SKILL_MAP = {
    80102: "治疗",
    80109: "疾跑",
    80104: "惩击",
    80108: "终结",
    80110: "狂暴",
    80105: "干扰",
    80103: "晕眩",
    80107: "净化",
    80121: "弱化",
    80115: "闪现",
}
SUMMONER_SKILL_IDS = list(SUMMONER_SKILL_MAP.keys())


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
        self.episode_idx = 0
        self.agent_idx = 0
        self.feature_builder = Phase1FeatureBuilder(logger)
        self.feature_auditor = FeatureAuditDumper(logger)
        self.last_action_source = "init"

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)
        self._try_load_partial_pretrain()

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        # Define learning rate decay function
        # 定义学习率衰减函数
        if step > self.target_step:
            return self.target_lr / self.lr
        else:
            return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data):
        # Select summoner skill for each hero based on hero lineup of both camps
        # 根据双方阵营英雄阵容，为己方每个英雄选择召唤师技能
        my_heroes = config_data.get("my_heroes", [])
        select_skills = {}
        for hero_id in my_heroes:
            skill_id = GameConfig.BERSERK_SKILL_ID
            select_skills[hero_id] = skill_id
        return select_skills

    def _try_load_partial_pretrain(self):
        model_path = os.environ.get("PARTIAL_PRETRAIN_MODEL", "").strip()
        if not model_path:
            model_dir = os.environ.get("PARTIAL_PRETRAIN_DIR", "").strip()
            model_id = os.environ.get("PARTIAL_PRETRAIN_ID", "").strip()
            if model_dir and model_id:
                model_path = os.path.join(model_dir, f"model.ckpt-{model_id}.pkl")
        if not model_path:
            return
        if not os.path.exists(model_path):
            if self.logger:
                self.logger.warning(f"partial pretrain skipped, file not found: {model_path}")
            return

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            if self.logger:
                self.logger.warning(f"partial pretrain skipped, unsupported checkpoint format: {model_path}")
            return

        current = self.model.state_dict()
        matched = {}
        skipped = []
        for key, value in checkpoint.items():
            if key in current and tuple(current[key].shape) == tuple(value.shape):
                matched[key] = value
            else:
                skipped.append(key)
        current.update(matched)
        self.model.load_state_dict(current, strict=True)
        if self.logger:
            self.logger.info(
                f"partial pretrain loaded from {model_path}, matched={len(matched)}, skipped={len(skipped)}"
            )

    def reset(self, observation):
        # Reset function, called at the beginning of each episode
        # 重置函数，每局开始时调用
        self.hero_camp = observation["camp"]
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_builder.reset()
        self.combat_start_frame = None
        self.episode_idx += 1

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

    def predict(self, observation):
        # Prediction function, usually called during training
        # Returns a random sampling action
        # 预测函数，通常在训练时调用，返回随机采样动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        return action

    def exploit(self, observation):
        # Exploitation function, usually called during evaluation
        # Returns the action with the highest probability
        # 利用函数，在评估时调用，返回最大概率动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        d_action = self.action_process(observation, act_data, False)
        return d_action

    def observation_process(self, observation):
        feature = self.feature_builder.build_observation(observation)
        self.feature_auditor.dump(self.episode_idx, self.agent_idx, observation, feature)
        feature_vec, legal_action = (
            feature,
            observation["legal_action"],
        )
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    def dump_feature_audit(self, observation, force=False):
        feature = self.feature_builder.build_observation(observation)
        if force:
            frame_state = (observation or {}).get("frame_state", {}) or {}
            frame_no = int(frame_state.get("frame_no", frame_state.get("frameNo", 0)) or 0)
            if hasattr(self.feature_auditor, "frames"):
                self.feature_auditor.frames.add(frame_no)
        self.feature_auditor.dump(self.episode_idx, self.agent_idx, observation, feature)

    def action_process(self, observation, act_data, is_stochastic):
        leave_base_action = self._force_leave_base_action(observation)
        if leave_base_action is not None:
            self.last_action_source = "force_leave_base"
            return leave_base_action

        opening_action = self._opening_lane_action(observation)
        if opening_action is not None:
            self.last_action_source = "opening_lane"
            return opening_action

        berserk_action = self._berserk_action(observation)
        if berserk_action is not None:
            self.last_action_source = "berserk"
            return berserk_action

        if is_stochastic:
            # Use stochastic sampling action
            # 采用随机采样动作 action
            self.last_action_source = "policy_stochastic"
            return act_data.action
        else:
            # Use the action with the highest probability
            # 采用最大概率动作 d_action
            self.last_action_source = "policy_deterministic"
            return act_data.d_action

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def _split_legal_actions(self, observation):
        legal_action = np.array(observation.get("legal_action", []), dtype=np.float32)
        if legal_action.size != sum(self.legal_action_size):
            return None
        split_points = [sum(self.legal_action_size[: index + 1]) for index in range(len(self.legal_action_size))]
        return np.split(legal_action, split_points[:-1])

    def _main_hero_state(self, observation):
        for hero in observation.get("frame_state", {}).get("hero_states", []) or []:
            if runtime_id(hero) == self.player_id or camp_id(hero.get("camp")) == camp_id(self.hero_camp):
                return hero
        return None

    def _enemy_hero_state(self, observation):
        for hero in observation.get("frame_state", {}).get("hero_states", []) or []:
            if camp_id(hero.get("camp")) != camp_id(self.hero_camp):
                return hero
        return None

    def _actor_pos(self, actor):
        loc = actor.get("location", {}) if isinstance(actor, dict) else {}
        if isinstance(loc, dict):
            return [float(loc.get("x", 0) or 0), float(loc.get("z", 0) or 0)]
        if isinstance(loc, (list, tuple)) and len(loc) >= 3:
            return [float(loc[0] or 0), float(loc[2] or 0)]
        return [0.0, 0.0]

    def _is_unseen_actor(self, actor):
        x, z = self._actor_pos(actor)
        return abs(x) >= 100000 or abs(z) >= 100000

    def _our_base_tower(self, observation, hero):
        towers = []
        for npc in observation.get("frame_state", {}).get("npc_states", []) or []:
            if camp_id(npc.get("camp")) == camp_id(hero.get("camp")) and npc.get("sub_type") in (23, 24):
                towers.append(npc)
        if not towers:
            for npc in observation.get("frame_state", {}).get("npc_states", []) or []:
                if camp_id(npc.get("camp")) == camp_id(hero.get("camp")) and npc.get("actor_type") == 2:
                    towers.append(npc)
        return min(towers, key=lambda tower: np.linalg.norm(np.array(self._actor_pos(hero)) - np.array(self._actor_pos(tower))), default=None)

    def _point_for_action(self, point):
        x, z = float(point[0]), float(point[1])
        if camp_id(self.hero_camp) == 2 and abs(x) < 100000 and abs(z) < 100000:
            return [-x, -z]
        return [x, z]

    def _delta_action_16x16(self, center, target):
        delta = np.array(target, dtype=np.float32) - np.array(center, dtype=np.float32)
        max_abs = float(np.max(np.abs(delta)))
        if max_abs <= 1e-6:
            return 8, 8
        grid = np.ceil(delta / max_abs * 7).astype(np.int32) + np.array([8, 8], dtype=np.int32)
        grid = np.clip(grid, 0, 15)
        return int(grid[0]), int(grid[1])

    def _move_to_point_action(self, legal_actions, center, target):
        if legal_actions[0][2] <= 0:
            return None
        move_x, move_z = self._delta_action_16x16(self._point_for_action(center), self._point_for_action(target))
        if legal_actions[1][move_x] <= 0:
            move_x = int(np.argmax(legal_actions[1]))
        if legal_actions[2][move_z] <= 0:
            move_z = int(np.argmax(legal_actions[2]))
        return [2, move_x, move_z, 0, 0, 0]

    def _force_leave_base_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero or hp_ratio(hero) < GameConfig.FORCE_LEAVE_BASE_HP_THRESHOLD:
            return None
        base = self._our_base_tower(observation, hero)
        if base is None:
            return None
        if np.linalg.norm(np.array(self._actor_pos(hero)) - np.array(self._actor_pos(base))) > GameConfig.BASE_RADIUS:
            return None
        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None
        return self._move_to_point_action(legal_actions, self._actor_pos(hero), GameConfig.OPENING_GRASS_TARGET)

    def _opening_lane_action(self, observation):
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = int(frame_state.get("frame_no", frame_state.get("frameNo", 0)) or 0)
        if frame_no > GameConfig.OPENING_FORCE_UNTIL_FRAME:
            return None
        hero = self._main_hero_state(observation)
        if not hero or hp_ratio(hero) < GameConfig.FORCE_LEAVE_BASE_HP_THRESHOLD:
            return None
        hero_pos = np.array(self._point_for_action(self._actor_pos(hero)), dtype=np.float32)
        opening_target = getattr(GameConfig, "OPENING_LANE_TARGET", GameConfig.OPENING_GRASS_TARGET)
        target = np.array(self._point_for_action(opening_target), dtype=np.float32)
        if np.linalg.norm(hero_pos - target) <= GameConfig.OPENING_TARGET_RADIUS:
            return None
        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None
        return self._move_to_point_action(legal_actions, self._actor_pos(hero), opening_target)

    def _summoner_skill_ready(self, hero):
        for slot in ((hero or {}).get("skill_state", {}) or {}).get("slot_states", []) or []:
            if int(slot.get("slot_type", -1) or -1) == 6:
                return float(slot.get("cooldown", 0) or 0) <= 0 and bool(slot.get("usable", True))
        return False

    def _in_combat(self, observation, hero, enemy):
        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        target = hero.get("attack_target", 0) == runtime_id(enemy) or enemy.get("attack_target", 0) == runtime_id(hero)
        close = np.linalg.norm(np.array(self._actor_pos(hero)) - np.array(self._actor_pos(enemy))) <= float(hero.get("attack_range", 0) or 0) + 4000
        if target or close:
            if self.combat_start_frame is None:
                self.combat_start_frame = frame_no
            return frame_no - self.combat_start_frame >= GameConfig.BERSERK_MIN_COMBAT_FRAMES
        self.combat_start_frame = None
        return False

    def _legal_action_with_target(self, legal_actions, button, target):
        if button >= len(legal_actions[0]) or legal_actions[0][button] <= 0:
            return None
        action = [button]
        for index in range(1, len(self.label_size_list) - 1):
            legal = legal_actions[index]
            preferred = 8 if index in (1, 2, 3, 4) else 0
            if preferred >= len(legal) or legal[preferred] <= 0:
                preferred = int(np.argmax(legal))
            action.append(preferred)
        target_legal = legal_actions[-1].reshape([self.legal_action_size[0], self.label_size_list[-1]])[button]
        if target >= len(target_legal) or target_legal[target] <= 0:
            if np.max(target_legal) <= 0:
                return None
            target = int(np.argmax(target_legal))
        action.append(target)
        return action

    def _berserk_action(self, observation):
        hero = self._main_hero_state(observation)
        enemy = self._enemy_hero_state(observation)
        if not hero or not enemy or self._is_unseen_actor(enemy):
            return None
        if hp_ratio(hero) > GameConfig.BERSERK_HP_THRESHOLD:
            return None
        if not self._summoner_skill_ready(hero):
            return None
        if not self._in_combat(observation, hero, enemy):
            return None
        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None
        return self._legal_action_with_target(legal_actions, GameConfig.CHOSEN_SUMMONER_BUTTON, GameConfig.SELF_TARGET_INDEX)

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

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
        legal_split_size = [sum(self.legal_action_size[: index + 1]) for index in range(len(self.legal_action_size))]
        legal_actions = np.split(legal_action, legal_split_size[:-1])
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
