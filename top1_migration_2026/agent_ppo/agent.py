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
import math
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.conf.conf import Config, GameConfig
from agent_ppo.feature.reward_process import GameRewardManager
from torch.optim.lr_scheduler import LambdaLR
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.top1_feature_builder import Top1FeatureBuilder


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


def game_config(name, default):
    return getattr(GameConfig, name, default)


def camp_id(camp):
    if isinstance(camp, str):
        if camp in ("0", "1", "2"):
            value = int(camp)
            return 1 if value == 0 else value
        if camp[-1:].isdigit():
            return int(camp[-1])
    if isinstance(camp, int):
        return 1 if camp == 0 else camp
    return camp


def is_neutral_camp(camp):
    return camp == 0 or camp == "0"


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
        self.last_hero_combat_frame = -100000
        self.last_enemy_dead_frame = -100000
        self.last_enemy_alive_frame = -100000
        self.selected_summoner_skill_by_hero = {}
        self.focus_monster_id = 0
        self.focus_monster_until_frame = -100000
        self.luban_combo_index = 0
        self.luban_pending_attack_until_frame = -100000
        self.last_enemy_skill2_used_frame = -100000
        self.last_enemy_skill3_used_frame = -100000
        self.last_enemy_skill3_hit_frame = -100000
        self.last_enemy_skill2_used_times = 0
        self.last_enemy_skill3_used_times = 0
        self.combat_start_frame = -100000
        self.last_control_frame = -100000
        self.supply_intent_until_frame = -100000
        self.grass_intent_until_frame = -100000

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

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

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
            skill_cfg = GameConfig.HERO_SUMMONER_SKILL.get(hero_id, 80115)
            if isinstance(skill_cfg, (list, tuple)):
                skill_id = int(np.random.choice(skill_cfg))
            else:
                skill_id = int(skill_cfg)
            select_skills[hero_id] = skill_id
            self.selected_summoner_skill_by_hero[hero_id] = skill_id
        return select_skills

    def reset(self, observation):
        # Reset function, called at the beginning of each episode
        # 重置函数，每局开始时调用
        self.hero_camp = camp_id(observation.get("camp", observation.get("player_camp", 1)))
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(self.player_id, self.hero_camp)
        self.feature_builder = Top1FeatureBuilder(self.logger)
        self.last_hero_combat_frame = -100000
        self.last_enemy_dead_frame = -100000
        self.last_enemy_alive_frame = -100000
        self.focus_monster_id = 0
        self.focus_monster_until_frame = -100000
        self.luban_combo_index = 0
        self.luban_pending_attack_until_frame = -100000
        self.last_enemy_skill2_used_frame = -100000
        self.last_enemy_skill3_used_frame = -100000
        self.last_enemy_skill3_hit_frame = -100000
        self.last_enemy_skill2_used_times = 0
        self.last_enemy_skill3_used_times = 0
        self.combat_start_frame = -100000
        self.last_control_frame = -100000
        self.supply_intent_until_frame = -100000
        self.grass_intent_until_frame = -100000

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
            np_output.append(output.detach().cpu().numpy())

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
        feature_vec, legal_action = (
            feature,
            observation["legal_action"],
        )
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    def action_process(self, observation, act_data, is_stochastic):
        self._update_combat_memory(observation)

        critical_home_action = self._critical_low_hp_home_action(observation)
        if critical_home_action is not None:
            if is_stochastic:
                act_data.action = critical_home_action
            else:
                act_data.d_action = critical_home_action
            return critical_home_action

        retreat_action = self._tower_retreat_action(observation)
        if retreat_action is not None:
            if is_stochastic:
                act_data.action = retreat_action
            else:
                act_data.d_action = retreat_action
            return retreat_action

        fight_retreat_action = self._fight_disadvantage_retreat_action(observation)
        if fight_retreat_action is not None:
            if is_stochastic:
                act_data.action = fight_retreat_action
            else:
                act_data.d_action = fight_retreat_action
            return fight_retreat_action

        leave_base_action = self._force_leave_base_action(observation)
        if leave_base_action is not None:
            if is_stochastic:
                act_data.action = leave_base_action
            else:
                act_data.d_action = leave_base_action
            return leave_base_action

        heal_action = self._low_hp_heal_action(observation)
        if heal_action is not None:
            if is_stochastic:
                act_data.action = heal_action
            else:
                act_data.d_action = heal_action
            return heal_action

        recover_action = self._low_hp_recover_action(observation)
        if recover_action is not None:
            if is_stochastic:
                act_data.action = recover_action
            else:
                act_data.d_action = recover_action
            return recover_action

        cleanse_action = self._control_cleanse_action(observation)
        if cleanse_action is not None:
            if is_stochastic:
                act_data.action = cleanse_action
            else:
                act_data.d_action = cleanse_action
            return cleanse_action

        supply_action = self._supply_action(observation)
        if supply_action is not None:
            if is_stochastic:
                act_data.action = supply_action
            else:
                act_data.d_action = supply_action
            return supply_action

        summoner_action = self._combat_summoner_action(observation)
        if summoner_action is not None:
            if is_stochastic:
                act_data.action = summoner_action
            else:
                act_data.d_action = summoner_action
            return summoner_action

        hero_specific_action = self._hero_specific_combat_action(observation)
        if hero_specific_action is not None:
            if is_stochastic:
                act_data.action = hero_specific_action
            else:
                act_data.d_action = hero_specific_action
            return hero_specific_action

        post_fight_action = self._post_fight_macro_action(observation)
        if post_fight_action is not None:
            if is_stochastic:
                act_data.action = post_fight_action
            else:
                act_data.d_action = post_fight_action
            return post_fight_action

        grass_action = self._grass_ambush_action(observation)
        if grass_action is not None:
            if is_stochastic:
                act_data.action = grass_action
            else:
                act_data.d_action = grass_action
            return grass_action

        monster_action = self._continue_monster_action(observation)
        if monster_action is not None:
            if is_stochastic:
                act_data.action = monster_action
            else:
                act_data.d_action = monster_action
            return monster_action

        tactical_action = self._tactical_active_action(observation)
        if tactical_action is not None:
            if is_stochastic:
                act_data.action = tactical_action
            else:
                act_data.d_action = tactical_action
            return tactical_action

        tower_action = self._forced_safe_tower_attack_action(observation)
        if tower_action is not None:
            if is_stochastic:
                act_data.action = tower_action
            else:
                act_data.d_action = tower_action
            return tower_action

        if is_stochastic:
            # Use stochastic sampling action
            # 采用随机采样动作 action
            action = act_data.action
        else:
            # Use the action with the highest probability
            # 采用最大概率动作 d_action
            action = act_data.d_action

        corrected_action = self._guard_misused_skill_action(observation, action)
        if corrected_action is not None:
            if is_stochastic:
                act_data.action = corrected_action
            else:
                act_data.d_action = corrected_action
            return corrected_action

        fallback_action = self._fallback_active_action(observation, action)
        if fallback_action is not None:
            if is_stochastic:
                act_data.action = fallback_action
            else:
                act_data.d_action = fallback_action
            return fallback_action

        return action

    def _low_hp_heal_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        if self._summoner_skill_id(hero) != game_config("HEAL_SUMMONER_SKILL_ID", 80102):
            return None

        hp = float(hero.get("hp", 0) or 0)
        max_hp = float(hero.get("max_hp", 0) or 0)
        if max_hp <= 0 or hp / max_hp > game_config("LOW_HP_HEAL_THRESHOLD", 0.75):
            return None

        legal_action = np.array(observation.get("legal_action", []), dtype=np.float32)
        if legal_action.size == 0:
            return None

        split_points = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, split_points[:-1])
        button = game_config("CHOSEN_SUMMONER_BUTTON", 8)
        if legal_actions[0][button] <= 0:
            return None

        action = [button]
        for index in range(1, len(self.label_size_list) - 1):
            legal = legal_actions[index]
            preferred = min(15, self.label_size_list[index] - 1)
            action.append(preferred if legal[preferred] > 0 else int(np.argmax(legal)))

        target_legal = legal_actions[-1].reshape([self.legal_action_size[0], self.label_size_list[-1]])[button]
        target = game_config("SELF_TARGET_INDEX", 2)
        if target_legal[target] <= 0:
            if np.max(target_legal) <= 0:
                return None
            target = int(np.argmax(target_legal))
        action.append(target)
        return action

    def _combat_summoner_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero:
            return None

        skill_id = self._summoner_skill_id(hero)
        if skill_id != game_config("BERSERK_SKILL_ID", 80110):
            return None
        enemy = self._enemy_hero_state(observation)
        if not enemy or self._enemy_dead(enemy) or self._is_unseen_actor(enemy):
            return None
        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        if not self._can_fight_enemy_hero(observation):
            return None
        if self._hp_ratio(hero) > game_config("BERSERK_HP_THRESHOLD", 0.50):
            if frame_no - self.combat_start_frame < game_config("BERSERK_MIN_COMBAT_FRAMES", 60):
                return None
        if self._hp_ratio(hero) - self._hp_ratio(enemy) < -0.10 or self._hp_ratio(hero) < 0.30:
            return None
        legal_actions = self._split_legal_actions(observation)
        button = game_config("CHOSEN_SUMMONER_BUTTON", 8)
        if legal_actions is None or legal_actions[0][button] <= 0:
            return None

        return self._legal_action_with_target(legal_actions, button, game_config("SELF_TARGET_INDEX", 2))

    def _fight_disadvantage_retreat_action(self, observation):
        hero = self._main_hero_state(observation)
        enemy = self._enemy_hero_state(observation)
        if not hero or not enemy or self._enemy_dead(enemy) or self._is_unseen_actor(enemy):
            return None
        if not self._enemy_threatening_hero(observation, hero, enemy):
            return None

        hero_hp = self._hp_ratio(hero)
        enemy_hp = self._hp_ratio(enemy)
        hp_gap = hero_hp - enemy_hp
        selected_summoner = self._summoner_skill_id(hero)
        combat_summoner = selected_summoner == game_config("BERSERK_SKILL_ID", 80110)
        summoner_unready = combat_summoner and not self._summoner_skill_ready(hero)

        bad_hp = hp_gap <= -game_config("FIGHT_RETREAT_HP_GAP", 0.18)
        low_hp_bad = hero_hp <= game_config("FIGHT_RETREAT_LOW_HP", 0.35) and enemy_hp > 0.25
        no_summoner_caution = summoner_unready and hp_gap < game_config("FIGHT_WITHOUT_SUMMONER_MIN_HP_GAP", 0.10)
        if not (bad_hp or low_hp_bad or no_summoner_caution):
            return None

        if self._can_finish_enemy_without_tower_risk(observation, hero, enemy):
            return None

        recall = self._safe_recall_action(observation)
        if recall is not None and hero_hp <= game_config("POST_FIGHT_RECALL_HP_THRESHOLD_LATE", 0.30):
            return recall

        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None
        our_tower = self._nearest_our_tower(observation, hero)
        if our_tower is None:
            return None
        target = self._own_tower_safe_anchor(hero, our_tower)
        if math.dist(self._actor_pos(hero), target) <= 2500.0:
            return None
        return self._move_to_point_action(legal_actions, self._actor_pos(hero), target)

    def _enemy_threatening_hero(self, observation, hero, enemy):
        hero_id = hero.get("runtime_id", hero.get("player_id"))
        enemy_id = enemy.get("runtime_id", enemy.get("player_id"))
        if hero.get("attack_target") == enemy_id or enemy.get("attack_target") == hero_id:
            return True
        for hit in enemy.get("hit_target_info", []) or []:
            if hit.get("hit_target") == hero_id:
                return True
        dist = math.dist(self._actor_pos(hero), self._actor_pos(enemy))
        threat_range = max(
            float(hero.get("attack_range", 0) or 0),
            float(enemy.get("attack_range", 0) or 0),
        ) + 4500.0
        return dist <= threat_range

    def _can_finish_enemy_without_tower_risk(self, observation, hero, enemy):
        hero_hp = self._hp_ratio(hero)
        enemy_hp = self._hp_ratio(enemy)
        if enemy_hp > 0.16 or hero_hp <= 0.40:
            return False
        enemy_tower = self._nearest_enemy_tower(observation, hero)
        if enemy_tower is not None and self._in_enemy_tower_range(hero, enemy_tower):
            tower_target = enemy_tower.get("attack_target", 0)
            return tower_target not in (hero.get("runtime_id"), hero.get("player_id"))
        return True

    def _summoner_skill_id(self, hero):
        for slot in (hero.get("skill_state", {}) or {}).get("slot_states", []) or []:
            if int(slot.get("slot_type", -1) or -1) == 6:
                return int(slot.get("configId", slot.get("config_id", 0)) or 0)
        return game_config("HEAL_SUMMONER_SKILL_ID", 80102)

    def _summoner_skill_ready(self, hero):
        for slot in (hero.get("skill_state", {}) or {}).get("slot_states", []) or []:
            if int(slot.get("slot_type", -1) or -1) == 6:
                return float(slot.get("cooldown", slot.get("cooldown_time", 0)) or 0) <= 0
        return False

    def _low_hp_recover_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        hp = float(hero.get("hp", 0) or 0)
        max_hp = float(hero.get("max_hp", 0) or 0)
        ep = float(hero.get("ep", 0) or 0)
        max_ep = float(hero.get("max_ep", 0) or 0)
        low_hp = max_hp > 0 and hp / max_hp <= game_config("LOW_HP_RECOVER_THRESHOLD", 0.85)
        low_ep = max_ep > 0 and ep / max_ep <= game_config("LOW_EP_RECOVER_THRESHOLD", 0.50)
        if not low_hp and not low_ep:
            return None
        legal_actions = self._split_legal_actions(observation)
        button = game_config("RECOVER_BUTTON", 7)
        if legal_actions is None or legal_actions[0][button] <= 0:
            return None
        return self._legal_action_with_target(legal_actions, button, game_config("SELF_TARGET_INDEX", 2))

    def _critical_low_hp_home_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero or self._hp_ratio(hero) > game_config("CRITICAL_HOME_HP_THRESHOLD", 0.20):
            return None
        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None

        our_base = self._our_base_tower(observation, hero)
        our_tower = self._nearest_our_tower(observation, hero)
        anchor = our_base or our_tower
        if anchor is None:
            return None

        if math.dist(self._actor_pos(hero), self._actor_pos(anchor)) <= game_config("RECALL_NEAR_TOWER_RADIUS", 9500.0):
            recall = self._recall_action(legal_actions)
            if recall is not None:
                self.supply_intent_until_frame = int(observation.get("frame_state", {}).get("frame_no", 0) or 0) + 180
                return recall
        if legal_actions[0][2] <= 0:
            return None
        self.supply_intent_until_frame = int(observation.get("frame_state", {}).get("frame_no", 0) or 0) + 180
        return self._move_to_point_action(legal_actions, self._actor_pos(hero), self._actor_pos(anchor))

    def _hero_specific_combat_action(self, observation):
        hero = self._main_hero_state(observation)
        enemy = self._enemy_hero_state(observation)
        if not hero or not enemy or self._enemy_dead(enemy) or self._is_unseen_actor(enemy):
            self.luban_pending_attack_until_frame = -100000
            return None
        if not self._can_fight_enemy_hero(observation) and not self._in_hero_combat_window(observation, 45):
            self.luban_pending_attack_until_frame = -100000
            return None

        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None

        hero_id = self._hero_config_id(hero)
        if hero_id == 133:
            return self._di_renjie_combat_action(observation, legal_actions, hero, enemy)
        if hero_id == 112:
            return self._luban_combo_action(observation, legal_actions, hero, enemy)
        return None

    def _luban_combo_action(self, observation, legal_actions, hero, enemy):
        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        if self.luban_pending_attack_until_frame >= frame_no:
            attack = self._legal_action_with_target(legal_actions, 3, 1, strict_target=True)
            if attack is not None:
                self.luban_pending_attack_until_frame = -100000
                return attack

        self.luban_pending_attack_until_frame = -100000
        combo = [6, 4, 5] if int(hero.get("level", 1) or 1) >= 4 else [4, 5]
        for offset in range(len(combo)):
            button = combo[(self.luban_combo_index + offset) % len(combo)]
            if button == 4:
                action = self._legal_directed_action_flexible_target(legal_actions, button, hero, enemy, (1, 0, 2))
            else:
                action = self._legal_directed_action_with_actor(legal_actions, button, hero, enemy, 1)
            if action is not None:
                self.luban_combo_index = (self.luban_combo_index + offset + 1) % len(combo)
                self.luban_pending_attack_until_frame = frame_no + 30
                return action
        return None

    def _di_renjie_combat_action(self, observation, legal_actions, hero, enemy):
        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        skill2 = self._di_renjie_skill2_action(legal_actions, hero, enemy)
        if skill2 is not None and (
            frame_no - self.last_enemy_skill3_hit_frame <= 45
            or frame_no - self.last_enemy_skill3_used_frame <= 30
        ):
            return skill2

        skill3_ready = self._legal_directed_action_with_actor(legal_actions, 6, hero, enemy, 1)
        enemy_hp = self._hp_ratio(enemy)
        enemy_used_skill2 = frame_no - self.last_enemy_skill2_used_frame <= 180
        if skill3_ready is not None and (enemy_used_skill2 or enemy_hp < 0.50):
            return skill3_ready

        if skill2 is not None and self._in_hero_combat_window(observation, 45):
            return skill2
        return None

    def _force_leave_base_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        hp = float(hero.get("hp", 0) or 0)
        max_hp = float(hero.get("max_hp", 0) or 0)
        if max_hp <= 0 or hp / max_hp <= game_config("FORCE_LEAVE_BASE_HP_THRESHOLD", 0.80):
            return None

        our_base = self._our_base_tower(observation, hero)
        if our_base is None:
            return None
        if math.dist(self._actor_pos(hero), self._actor_pos(our_base)) > game_config("BASE_RADIUS", 14000.0):
            return None

        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None or legal_actions[0][2] <= 0:
            return None
        target = game_config("OPENING_GRASS_TARGET", None)
        if target is None:
            target = game_config("MID_LANE_TARGET", (0.0, 0.0))
        return self._move_to_point_action(
            legal_actions, self._actor_pos(hero), target
        )

    def _post_fight_macro_action(self, observation):
        hero = self._main_hero_state(observation)
        enemy = self._enemy_hero_state(observation)
        if not hero:
            return None

        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None

        enemy_dead = self._enemy_dead(enemy)
        enemy_flee = self._enemy_fleeing_to_tower(observation, hero, enemy)
        if not enemy_dead and not enemy_flee:
            if self._enemy_recently_returned(observation) and self._hp_ratio(hero) < game_config(
                "ENEMY_RETURN_HOLD_HP_THRESHOLD", 0.85
            ):
                return self._hold_own_tower_action(observation, legal_actions)
            return None

        enemy_soldier = self._nearest_enemy_soldier(observation, hero)
        if enemy_soldier is not None:
            clear_action = self._clear_lane_action(
                observation, legal_actions, hero, enemy_soldier, allow_all_skills=enemy_dead
            )
            if clear_action is not None:
                return clear_action

        if enemy_dead and self._should_finish_push_after_fight(observation, hero, enemy):
            tower_action = self._safe_tower_attack_action(observation, legal_actions)
            if tower_action is not None:
                return tower_action
            enemy_tower = self._nearest_enemy_tower(observation, hero)
            if enemy_tower is not None:
                return self._move_to_point_action(legal_actions, self._actor_pos(hero), self._actor_pos(enemy_tower))

        if self._should_recall_after_fight(observation, hero, enemy):
            return self._return_tower_or_recall_action(observation, legal_actions, hero)
        return None

    def _enemy_dead(self, enemy):
        if not enemy:
            return False
        return float(enemy.get("hp", 0) or 0) <= 0 or float(enemy.get("revive_time", 0) or 0) > 0

    def _should_finish_push_after_fight(self, observation, hero, enemy):
        if self._hp_ratio(hero) <= game_config("SUPPLY_RETURN_HOME_HP_THRESHOLD", 0.20):
            return False
        if not self._enemy_dead(enemy):
            return False
        return self._wave_can_support_push(observation, hero, enemy)

    def _should_recall_after_fight(self, observation, hero, enemy):
        hp_rate = self._hp_ratio(hero)
        level = int(hero.get("level", 1) or 1)
        if level >= game_config("POST_FIGHT_PUSH_LEVEL", 9):
            if self._should_finish_push_after_fight(observation, hero, enemy):
                return False
            return hp_rate < game_config("POST_FIGHT_RECALL_HP_THRESHOLD_LATE", 0.30)
        if hp_rate >= game_config("POST_FIGHT_RECALL_HP_THRESHOLD_EARLY", 0.50):
            return False
        return self._used_recovery_tools(hero)

    def _used_recovery_tools(self, hero):
        for slot in hero.get("skill_state", {}).get("slot_states", []) or []:
            slot_type = int(slot.get("slot_type", -1) or -1)
            if slot_type in (5, 6) and int(slot.get("usedTimes", slot.get("used_times", 0)) or 0) > 0:
                return True
        return False

    def _wave_can_support_push(self, observation, hero, enemy):
        frame_state = observation.get("frame_state", {})
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        enemy_tower = self._nearest_enemy_tower(observation, hero)
        if enemy_tower is None:
            return False
        our_soldiers = [
            npc
            for npc in frame_state.get("npc_states", [])
            if self._is_soldier_actor(npc)
            and camp_id(npc.get("camp")) == player_camp
            and float(npc.get("hp", 0) or 0) > 0
            and not self._is_unseen_actor(npc)
        ]
        if len(our_soldiers) < 2:
            return False
        tower_target = enemy_tower.get("attack_target", 0)
        our_soldier_ids = {npc.get("runtime_id", npc.get("player_id")) for npc in our_soldiers}
        tower_tanking_minion = tower_target in our_soldier_ids and tower_target != 0
        near_tower_minions = [
            npc for npc in our_soldiers if math.dist(self._actor_pos(npc), self._actor_pos(enemy_tower)) <= 12000.0
        ]
        if not tower_tanking_minion and len(near_tower_minions) < 2:
            return False
        revive_time = float((enemy or {}).get("revive_time", 0) or 0)
        enemy_return_slack = revive_time <= 0 or revive_time >= 60 or len(near_tower_minions) >= 3
        return enemy_return_slack

    def _enemy_fleeing_to_tower(self, observation, hero, enemy):
        if not enemy or self._enemy_dead(enemy) or self._is_unseen_actor(enemy):
            return False
        enemy_tower = self._nearest_enemy_tower(observation, hero)
        if enemy_tower is None:
            return False
        hero_to_enemy = math.dist(self._actor_pos(hero), self._actor_pos(enemy))
        enemy_to_tower = math.dist(self._actor_pos(enemy), self._actor_pos(enemy_tower))
        hero_range = float(hero.get("attack_range", 0) or 0) + 2500.0
        return hero_to_enemy > hero_range and enemy_to_tower <= float(enemy_tower.get("attack_range", 0) or 0) + 3500.0

    def _enemy_recently_returned(self, observation):
        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        return frame_no - self.last_enemy_dead_frame <= 240 and frame_no - self.last_enemy_alive_frame <= 30

    def _clear_lane_action(self, observation, legal_actions, hero, target, allow_all_skills=False):
        target_index = self._target_index_for_actor(observation, target)
        buttons = (4, 5, 6) if allow_all_skills else (4,)
        for button in buttons:
            skill = self._legal_directed_action_with_actor(legal_actions, button, hero, target, target_index)
            if skill is not None:
                return skill
        attack = self._legal_action_with_target(legal_actions, 3, target_index, strict_target=True)
        if attack is not None:
            return attack
        return self._move_to_point_action(legal_actions, self._actor_pos(hero), self._actor_pos(target))

    def _return_tower_or_recall_action(self, observation, legal_actions, hero):
        if self._enemy_soldiers_near_own_tower(observation, hero):
            target = self._nearest_enemy_soldier(observation, hero)
            if target is not None:
                action = self._clear_lane_action(observation, legal_actions, hero, target)
                if action is not None:
                    return action
        our_tower = self._nearest_our_tower(observation, hero)
        if our_tower is None:
            return None
        if math.dist(self._actor_pos(hero), self._actor_pos(our_tower)) <= game_config("RECALL_NEAR_TOWER_RADIUS", 9500.0):
            recall = self._recall_action(legal_actions)
            if recall is not None:
                return recall
        target = self._own_tower_safe_anchor(hero, our_tower)
        if math.dist(self._actor_pos(hero), target) <= 2500.0:
            return None
        return self._move_to_point_action(legal_actions, self._actor_pos(hero), target)

    def _hold_own_tower_action(self, observation, legal_actions):
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        enemy_soldier = self._nearest_enemy_soldier(observation, hero)
        if enemy_soldier is not None and math.dist(self._actor_pos(hero), self._actor_pos(enemy_soldier)) <= 9000.0:
            action = self._clear_lane_action(observation, legal_actions, hero, enemy_soldier)
            if action is not None:
                return action
        our_tower = self._nearest_our_tower(observation, hero)
        if our_tower is not None:
            target = self._own_tower_safe_anchor(hero, our_tower)
            if math.dist(self._actor_pos(hero), target) <= 2500.0:
                return None
            return self._move_to_point_action(legal_actions, self._actor_pos(hero), target)
        return None

    def _recall_action(self, legal_actions):
        button = game_config("RECALL_BUTTON", 9)
        if button >= len(legal_actions[0]) or legal_actions[0][button] <= 0:
            return None
        return self._legal_action_with_target(legal_actions, button, game_config("SELF_TARGET_INDEX", 2))

    def _safe_recall_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        hp_rate = self._hp_ratio(hero)
        if hp_rate > game_config("POST_FIGHT_RECALL_HP_THRESHOLD_EARLY", 0.50):
            return None
        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None
        enemy = self._enemy_hero_state(observation)
        if enemy and not self._enemy_dead(enemy) and not self._is_unseen_actor(enemy):
            if math.dist(self._actor_pos(hero), self._actor_pos(enemy)) <= 15000.0:
                return None
        our_tower = self._nearest_our_tower(observation, hero)
        if our_tower is None:
            return None
        if self._enemy_soldiers_near_own_tower(observation, hero):
            target = self._nearest_enemy_soldier(observation, hero)
            return self._clear_lane_action(observation, legal_actions, hero, target) if target else None
        if math.dist(self._actor_pos(hero), self._actor_pos(our_tower)) <= game_config("RECALL_NEAR_TOWER_RADIUS", 9500.0):
            return self._recall_action(legal_actions)
        target = self._own_tower_safe_anchor(hero, our_tower)
        if math.dist(self._actor_pos(hero), target) <= 2500.0:
            return None
        return self._move_to_point_action(legal_actions, self._actor_pos(hero), target)

    def _nearest_cake(self, observation, hero):
        cakes = observation.get("frame_state", {}).get("cakes", []) or []
        visible = [cake for cake in cakes if not self._is_unseen_actor(cake)]
        if not visible:
            return None
        hero_pos = self._actor_pos(hero)
        return min(visible, key=lambda cake: math.dist(hero_pos, self._actor_pos(cake)))

    def _nearest_own_cake(self, observation, hero):
        cakes = observation.get("frame_state", {}).get("cakes", []) or []
        visible = [cake for cake in cakes if not self._is_unseen_actor(cake)]
        if not visible:
            return None
        our_tower = self._nearest_our_tower(observation, hero)
        enemy_tower = self._nearest_enemy_tower(observation, hero)
        if our_tower is not None and enemy_tower is not None:
            own_side = [
                cake for cake in visible
                if math.dist(self._actor_pos(cake), self._actor_pos(our_tower))
                <= math.dist(self._actor_pos(cake), self._actor_pos(enemy_tower))
            ]
            if own_side:
                visible = own_side
        hero_pos = self._actor_pos(hero)
        return min(visible, key=lambda cake: math.dist(hero_pos, self._actor_pos(cake)))

    def _low_hp_rune_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero or self._hp_ratio(hero) > game_config("RUNE_LOW_HP_THRESHOLD", 0.85):
            return None
        if self._enemy_threat_nearby(observation, hero, 14000.0):
            return None
        cake = self._nearest_own_cake(observation, hero)
        if cake is None:
            return None
        dist = math.dist(self._actor_pos(hero), self._actor_pos(cake))
        if dist > game_config("RUNE_SEARCH_RADIUS", 25000.0):
            return None
        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None or legal_actions[0][2] <= 0:
            return None
        return self._move_to_point_action(legal_actions, self._actor_pos(hero), self._actor_pos(cake))

    def _supply_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        hp_rate = self._hp_ratio(hero)
        if self._in_hero_combat_window(observation, 45) and hp_rate > 0.30:
            return None
        if hp_rate <= game_config("SUPPLY_RUNE_HP_THRESHOLD", 0.78):
            rune_action = self._low_hp_rune_action(observation)
            if rune_action is not None:
                self.supply_intent_until_frame = frame_no + 120
                return rune_action
        if hp_rate <= game_config("SUPPLY_RETURN_HOME_HP_THRESHOLD", 0.20) or frame_no <= self.supply_intent_until_frame:
            recall_action = self._safe_recall_action(observation)
            if recall_action is not None:
                self.supply_intent_until_frame = frame_no + 120
                return recall_action
        return None

    def _control_cleanse_action(self, observation):
        hero = self._main_hero_state(observation)
        enemy = self._enemy_hero_state(observation)
        if not hero or self._hero_config_id(hero) != 133:
            return None
        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        control_recent = frame_no - self.last_control_frame <= game_config("CONTROL_CLEANSE_FRAMES", 45)
        skill3_recent = frame_no - self.last_enemy_skill3_hit_frame <= game_config("CONTROL_CLEANSE_FRAMES", 45)
        if not control_recent and not skill3_recent:
            return None
        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None
        return self._di_renjie_skill2_action(legal_actions, hero, enemy)

    def _lane_boundary_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        our_soldiers = self._soldiers_by_side(observation, hero, same_side=True)
        enemy_soldiers = self._soldiers_by_side(observation, hero, same_side=False)
        if len(our_soldiers) < 1 or len(enemy_soldiers) < 2:
            return None
        if self._allowed_to_cross_enemy_wave(observation, hero):
            return None

        hero_pos = self._actor_pos(hero)
        enemy_cluster = self._centroid(enemy_soldiers[:3])
        our_cluster = self._centroid(our_soldiers[:3])
        if enemy_cluster is None or our_cluster is None:
            return None
        dist_enemy_wave = math.dist(hero_pos, enemy_cluster)
        dist_our_wave = math.dist(hero_pos, our_cluster)
        unsupported_in_enemy_wave = dist_enemy_wave <= 8000.0 and dist_our_wave >= dist_enemy_wave + 2000.0
        if not unsupported_in_enemy_wave:
            return None

        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None
        target = min(enemy_soldiers, key=lambda npc: math.dist(hero_pos, self._actor_pos(npc)))
        if math.dist(hero_pos, self._actor_pos(target)) <= float(hero.get("attack_range", 0) or 0) + 3000.0:
            target_index = self._target_index_for_actor(observation, target)
            skill = self._legal_directed_action_with_actor(legal_actions, 4, hero, target, target_index)
            if skill is not None:
                return skill
            attack = self._legal_action_with_target(legal_actions, 3, target_index, strict_target=True)
            if attack is not None:
                return attack
        return self._move_to_point_action(legal_actions, hero_pos, our_cluster)

    def _continue_monster_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        if self._hp_ratio(hero) < game_config("MONSTER_MIN_HP_THRESHOLD", 0.35):
            return None
        money = float(hero.get("money_cnt", hero.get("money", 0)) or 0)
        if money >= game_config("MONSTER_FOCUS_MAX_MONEY", 6500):
            self.focus_monster_id = 0
            return None
        if self._enemy_soldiers_near_own_tower(observation, hero):
            return None
        if self._enemy_hero_threatening_jungle(observation, hero):
            return None

        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        target = self._current_or_focused_monster(observation, hero, frame_no)
        if target is None:
            return None

        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None

        self.focus_monster_id = int(target.get("runtime_id", target.get("player_id", 0)) or 0)
        self.focus_monster_until_frame = frame_no + 120
        target_index = self._target_index_for_actor(observation, target)
        skill = self._legal_directed_action_with_actor(legal_actions, 4, hero, target, target_index)
        if skill is not None:
            return skill
        attack = self._legal_action_with_target(legal_actions, 3, target_index, strict_target=True)
        if attack is not None:
            return attack
        if legal_actions[0][2] > 0:
            return self._move_to_point_action(legal_actions, self._actor_pos(hero), self._actor_pos(target))
        return None

    def _grass_ambush_action(self, observation):
        hero = self._main_hero_state(observation)
        enemy = self._enemy_hero_state(observation)
        if not hero or not enemy or self._enemy_dead(enemy) or self._is_unseen_actor(enemy):
            return None
        if self._hp_ratio(hero) < game_config("GRASS_AMBUSH_MIN_HP", 0.45):
            return None
        if abs(self._hp_ratio(hero) - self._hp_ratio(enemy)) > game_config("GRASS_AMBUSH_MAX_HP_GAP", 0.18):
            return None
        if self._enemy_soldiers_near_own_tower(observation, hero):
            return None
        lane_target = self._best_lane_target_actor(observation)
        if lane_target is not None and math.dist(self._actor_pos(hero), self._actor_pos(lane_target)) <= 8500.0:
            return None

        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None

        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        if bool(hero.get("is_in_grass", False)):
            if frame_no > self.grass_intent_until_frame:
                return None
            if self._can_fight_enemy_hero(observation):
                action = self._legal_directed_action_flexible_target(legal_actions, 4, hero, enemy, (1, 0, 2))
                if action is not None:
                    self.grass_intent_until_frame = -100000
                    return action
                attack = self._legal_action_with_target(legal_actions, 3, 1, strict_target=True)
                if attack is not None:
                    self.grass_intent_until_frame = -100000
                    return attack
            return None

        grass = self._nearest_grass_point(hero)
        if grass is None:
            return None
        dist_grass = math.dist(self._actor_pos(hero), grass)
        dist_enemy = math.dist(self._actor_pos(hero), self._actor_pos(enemy))
        if dist_grass > game_config("GRASS_AMBUSH_ENTRY_RADIUS", 18000.0):
            return None
        if dist_enemy > float(hero.get("attack_range", 0) or 0) + 9000.0:
            return None
        if legal_actions[0][2] <= 0:
            return None
        self.grass_intent_until_frame = frame_no + game_config("GRASS_AMBUSH_EXIT_FRAMES", 90)
        return self._move_to_point_action(legal_actions, self._actor_pos(hero), grass)

    def _soldiers_by_side(self, observation, hero, same_side):
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        hero_pos = self._actor_pos(hero)
        soldiers = [
            npc
            for npc in observation.get("frame_state", {}).get("npc_states", [])
            if self._is_soldier_actor(npc)
            and float(npc.get("hp", 0) or 0) > 0
            and not self._is_unseen_actor(npc)
            and ((camp_id(npc.get("camp")) == player_camp) == same_side)
        ]
        soldiers.sort(key=lambda npc: math.dist(hero_pos, self._actor_pos(npc)))
        return soldiers

    def _centroid(self, actors):
        if not actors:
            return None
        points = [self._actor_pos(actor) for actor in actors]
        return [sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)]

    def _allowed_to_cross_enemy_wave(self, observation, hero):
        enemy = self._enemy_hero_state(observation)
        if not enemy or self._enemy_dead(enemy) or self._is_unseen_actor(enemy):
            return False
        enemy_tower = self._nearest_enemy_tower(observation, hero)
        if enemy_tower is not None and self._unsafe_enemy_tower_zone(observation, hero, enemy_tower):
            return False
        return self._hp_ratio(hero) - self._hp_ratio(enemy) > 0.10 and self._enemy_fleeing_to_tower(
            observation, hero, enemy
        )

    def _enemy_hero_threatening_jungle(self, observation, hero):
        enemy = self._enemy_hero_state(observation)
        if not enemy or self._enemy_dead(enemy) or self._is_unseen_actor(enemy):
            return False
        if self._hp_ratio(hero) - self._hp_ratio(enemy) >= 0.20:
            return False
        return math.dist(self._actor_pos(hero), self._actor_pos(enemy)) <= float(hero.get("attack_range", 0) or 0) + 4500.0

    def _current_or_focused_monster(self, observation, hero, frame_no):
        monsters = [
            npc
            for npc in observation.get("frame_state", {}).get("npc_states", [])
            if self._is_monster_actor(npc)
            and float(npc.get("hp", 0) or 0) > 0
            and not self._is_unseen_actor(npc)
        ]
        if not monsters:
            self.focus_monster_id = 0
            return None

        by_id = {int(npc.get("runtime_id", npc.get("player_id", 0)) or 0): npc for npc in monsters}
        attack_target = int(hero.get("attack_target", 0) or 0)
        if attack_target in by_id:
            return by_id[attack_target]

        if self.focus_monster_id in by_id and frame_no <= self.focus_monster_until_frame:
            target = by_id[self.focus_monster_id]
            if math.dist(self._actor_pos(hero), self._actor_pos(target)) <= 22000.0:
                return target
        nearest = min(monsters, key=lambda actor: math.dist(self._actor_pos(hero), self._actor_pos(actor)), default=None)
        if nearest is not None and math.dist(self._actor_pos(hero), self._actor_pos(nearest)) <= 24000.0:
            return nearest
        return None

    def _enemy_soldiers_near_own_tower(self, observation, hero):
        our_tower = self._nearest_our_tower(observation, hero)
        if our_tower is None:
            return False
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        for npc in observation.get("frame_state", {}).get("npc_states", []):
            if (
                self._is_soldier_actor(npc)
                and camp_id(npc.get("camp")) != player_camp
                and float(npc.get("hp", 0) or 0) > 0
                and math.dist(self._actor_pos(npc), self._actor_pos(our_tower)) <= 12000.0
            ):
                return True
        return False

    def _hp_ratio(self, actor):
        return float(actor.get("hp", 0) or 0) / max(float(actor.get("max_hp", 1) or 1), 1.0)

    def _main_hero_state(self, observation):
        player_id = observation.get("player_id", self.player_id)
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        for hero in observation.get("frame_state", {}).get("hero_states", []):
            if hero.get("runtime_id", hero.get("player_id")) == player_id:
                return hero
        for hero in observation.get("frame_state", {}).get("hero_states", []):
            if camp_id(hero.get("camp")) == player_camp:
                return hero
        return None

    def _fallback_active_action(self, observation, action):
        if not action or action[0] not in game_config("PASSIVE_BUTTONS", {0, 1, 7, 8, 9, 10, 11}):
            return None

        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None

        attack_action = self._attack_fallback_action(observation, legal_actions)
        if attack_action is not None:
            return attack_action
        return self._move_fallback_action(observation, legal_actions)

    def _forced_safe_tower_attack_action(self, observation):
        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None
        return self._safe_tower_attack_action(observation, legal_actions)

    def _tactical_active_action(self, observation):
        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None

        hero = self._main_hero_state(observation)
        enemy = self._enemy_hero_state(observation)
        if not hero:
            return None

        # Hero skills are only fired when a visible target exists.  Skill 1 can
        # clear lane, but a valid enemy hero target always has priority.
        if enemy and self._can_fight_enemy_hero(observation):
            skill2 = self._di_renjie_skill2_action(legal_actions, hero, enemy)
            if skill2 is not None:
                return skill2
            for button in (6, 5, 4):
                action = self._legal_directed_action_with_actor(legal_actions, button, hero, enemy, 1)
                if action is not None:
                    return action
            action = self._legal_action_with_target(legal_actions, 3, 1, strict_target=True)
            if action is not None:
                return action

        # When the wave is gone and the enemy is still clearing, fight if HP is
        # close and we do not need to chase into tower.  Otherwise contest monster.
        if self._enemy_wave_cleared(observation):
            if enemy and self._hp_gap(hero, enemy) <= 0.15 and self._can_fight_enemy_hero(observation):
                action = self._legal_directed_action_with_actor(legal_actions, 4, hero, enemy, 1)
                if action is not None:
                    return action
                action = self._legal_action_with_target(legal_actions, 3, 1, strict_target=True)
                if action is not None:
                    return action
            monster = self._nearest_monster(observation, hero)
            if monster is not None:
                target_index = self._target_index_for_actor(observation, monster)
                action = self._legal_directed_action_with_actor(legal_actions, 4, hero, monster, target_index)
                if action is not None:
                    return action
                action = self._legal_action_with_target(legal_actions, 3, target_index, strict_target=True)
                if action is not None:
                    return action

        # If there has been no hero interaction for about 3 seconds, prefer
        # clearing lane with skill 1, then normal attacks on low HP lane units.
        if not self._in_hero_combat_window(observation, 90):
            lane_target = self._best_lane_target_actor(observation)
            if lane_target is not None:
                target = self._target_index_for_actor(observation, lane_target)
                action = self._legal_directed_action_with_actor(legal_actions, 4, hero, lane_target, target)
                if action is not None:
                    return action
                action = self._legal_action_with_target(legal_actions, 3, target, strict_target=True)
                if action is not None:
                    return action
        return None

    def _guard_misused_skill_action(self, observation, action):
        if not action:
            return None
        button = action[0]
        if button not in (4, 5, 6):
            return None

        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None:
            return None

        hero = self._main_hero_state(observation)
        enemy = self._enemy_hero_state(observation)
        if hero and enemy and self._can_fight_enemy_hero(observation):
            if button == 5:
                replace = self._di_renjie_skill2_action(legal_actions, hero, enemy)
                if replace is not None:
                    return replace
            replace = self._legal_directed_action_with_actor(legal_actions, button, hero, enemy, 1)
            if replace is not None:
                return replace
            replace = self._legal_action_with_target(legal_actions, 3, 1, strict_target=True)
            if replace is not None:
                return replace

        if button in (5, 6):
            return self._move_fallback_action(observation, legal_actions)

        if hero and self._enemy_wave_cleared(observation):
            monster = self._nearest_monster(observation, hero)
            if monster is not None:
                target = self._target_index_for_actor(observation, monster)
                replace = self._legal_directed_action_with_actor(legal_actions, 4, hero, monster, target)
                if replace is not None:
                    return replace
                replace = self._legal_action_with_target(legal_actions, 3, target, strict_target=True)
                if replace is not None:
                    return replace

        lane_target = self._best_lane_target_actor(observation)
        if lane_target is not None:
            hero = hero or self._main_hero_state(observation)
            target = self._target_index_for_actor(observation, lane_target)
            replace = self._legal_directed_action_with_actor(legal_actions, 4, hero, lane_target, target)
            if replace is not None:
                return replace
            replace = self._legal_action_with_target(legal_actions, 3, target, strict_target=True)
            if replace is not None:
                return replace
        return self._move_fallback_action(observation, legal_actions)

    def _split_legal_actions(self, observation):
        legal_action = np.array(observation.get("legal_action", []), dtype=np.float32)
        if legal_action.size != sum(self.legal_action_size):
            return None
        split_points = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        return np.split(legal_action, split_points[:-1])

    def _legal_action_with_target(self, legal_actions, button, target, strict_target=False):
        if button >= len(legal_actions[0]) or legal_actions[0][button] <= 0:
            return None
        action = [button]
        for index in range(1, len(self.label_size_list) - 1):
            legal = legal_actions[index]
            preferred = 8 if button in (5, 6) else 0
            if preferred >= len(legal) or legal[preferred] <= 0:
                preferred = int(np.argmax(legal))
            action.append(preferred)
        target_legal = legal_actions[-1].reshape([self.legal_action_size[0], self.label_size_list[-1]])[button]
        if target >= len(target_legal) or target_legal[target] <= 0:
            if strict_target:
                return None
            if np.max(target_legal) <= 0:
                return None
            target = int(np.argmax(target_legal))
        action.append(target)
        return action

    def _legal_directed_action_with_actor(self, legal_actions, button, hero, target_actor, target_index):
        if hero is None or target_actor is None:
            return None
        if button >= len(legal_actions[0]) or legal_actions[0][button] <= 0:
            return None
        if self._is_unseen_actor(target_actor) or float(target_actor.get("hp", 0) or 0) <= 0:
            return None

        target_legal = legal_actions[-1].reshape([self.legal_action_size[0], self.label_size_list[-1]])[button]
        if target_index >= len(target_legal) or target_legal[target_index] <= 0:
            return None

        action = [button]
        move_x, move_z = self._delta_action_16x16(
            self._actor_pos_for_action(hero), self._actor_pos_for_action(target_actor)
        )
        for index in range(1, len(self.label_size_list) - 1):
            legal = legal_actions[index]
            preferred = move_x if index == 3 else move_z if index == 4 else 0
            if preferred >= len(legal) or legal[preferred] <= 0:
                if np.max(legal) <= 0:
                    return None
                preferred = int(np.argmax(legal))
            action.append(preferred)
        action.append(target_index)
        return action

    def _legal_directed_action_flexible_target(self, legal_actions, button, hero, target_actor, target_candidates):
        if hero is None or target_actor is None:
            return None
        if button >= len(legal_actions[0]) or legal_actions[0][button] <= 0:
            return None
        if self._is_unseen_actor(target_actor) or float(target_actor.get("hp", 0) or 0) <= 0:
            return None

        action = [button]
        move_x, move_z = self._delta_action_16x16(
            self._actor_pos_for_action(hero), self._actor_pos_for_action(target_actor)
        )
        for index in range(1, len(self.label_size_list) - 1):
            legal = legal_actions[index]
            preferred = move_x if index == 3 else move_z if index == 4 else 0
            if preferred >= len(legal) or legal[preferred] <= 0:
                if np.max(legal) <= 0:
                    return None
                preferred = int(np.argmax(legal))
            action.append(preferred)

        target_legal = legal_actions[-1].reshape([self.legal_action_size[0], self.label_size_list[-1]])[button]
        for target in target_candidates:
            if target < len(target_legal) and target_legal[target] > 0:
                action.append(target)
                return action
        if np.max(target_legal) <= 0:
            return None
        action.append(int(np.argmax(target_legal)))
        return action

    def _di_renjie_skill2_action(self, legal_actions, hero, enemy):
        if self._hero_config_id(hero) != 133:
            return None
        button = 5
        if button >= len(legal_actions[0]) or legal_actions[0][button] <= 0:
            return None
        target = game_config("SELF_TARGET_INDEX", 2)
        action = self._legal_action_with_target(legal_actions, button, target, strict_target=False)
        if action is not None:
            return action
        return self._legal_action_with_target(legal_actions, button, 1, strict_target=False)

    def _hero_config_id(self, hero):
        return int(hero.get("config_id", hero.get("configId", 0)) or 0)

    def _slot_used_times(self, hero, slot_type):
        for slot in (hero.get("skill_state", {}) or {}).get("slot_states", []) or []:
            if int(slot.get("slot_type", -1) or -1) == slot_type:
                return int(slot.get("usedTimes", slot.get("used_times", 0)) or 0)
        return 0

    def _attack_fallback_action(self, observation, legal_actions):
        button = 3
        if legal_actions[0][button] <= 0:
            return None
        target_legal = legal_actions[-1].reshape([self.legal_action_size[0], self.label_size_list[-1]])[button]

        tower_action = self._safe_tower_attack_action(observation, legal_actions)
        if tower_action is not None:
            return tower_action

        for target in [1, 3, 4, 5, 6, 7]:
            if target_legal[target] > 0:
                return [button, 0, 0, 0, 0, target]
        return None

    def _safe_tower_attack_action(self, observation, legal_actions):
        button = 3
        if legal_actions[0][button] <= 0:
            return None
        hero = self._main_hero_state(observation)
        target_legal = legal_actions[-1].reshape([self.legal_action_size[0], self.label_size_list[-1]])[button]
        if self._safe_enemy_tower_target(observation) is not None and target_legal[7] > 0:
            return [button, 0, 0, 0, 0, 7]
        return None

    def _move_fallback_action(self, observation, legal_actions):
        button = 2
        if legal_actions[0][button] <= 0:
            return None

        hero = self._main_hero_state(observation)
        target = self._preferred_move_target(observation)
        if not hero or target is None:
            return None

        return self._move_to_point_action(legal_actions, self._actor_pos(hero), target)

    def _move_to_point_action(self, legal_actions, center, target):
        move_x, move_z = self._delta_action_16x16(
            self._point_for_action(center), self._point_for_action(target)
        )
        return self._move_to_grid_action(legal_actions, move_x, move_z)

    def _move_to_grid_action(self, legal_actions, move_x, move_z):
        if legal_actions[0][2] <= 0:
            return None
        if legal_actions[1][move_x] <= 0:
            move_x = int(np.argmax(legal_actions[1]))
        if legal_actions[2][move_z] <= 0:
            move_z = int(np.argmax(legal_actions[2]))
        return [2, move_x, move_z, 0, 0, 0]

    def _preferred_move_target(self, observation):
        frame_state = observation.get("frame_state", {})
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        hero_pos = self._actor_pos(hero)

        candidates = []
        if self._enemy_wave_cleared(observation):
            enemy = self._enemy_hero_state(observation)
            if enemy and self._hp_gap(hero, enemy) <= 0.15 and self._can_fight_enemy_hero(observation):
                return self._actor_pos(enemy)
            monster = self._nearest_monster(observation, hero)
            if monster is not None:
                return self._actor_pos(monster)

        for other in frame_state.get("hero_states", []):
            if (
                camp_id(other.get("camp")) != player_camp
                and float(other.get("hp", 0) or 0) > 0
                and not self._is_unseen_actor(other)
            ):
                candidates.append(other)
        for npc in frame_state.get("npc_states", []):
            if camp_id(npc.get("camp")) != player_camp and float(npc.get("hp", 0) or 0) > 0 and not self._is_unseen_actor(npc):
                candidates.append(npc)
        safe_tower = self._safe_enemy_tower_target(observation)
        if safe_tower is not None:
            return self._actor_pos(safe_tower)
        if not candidates:
            money = float(hero.get("money_cnt", hero.get("money", 0)) or 0)
            if money < game_config("TOWER_EARLY_PUSH_MONEY", 5500):
                monster = self._nearest_monster(observation, hero)
                if monster is not None:
                    return self._actor_pos(monster)
                return game_config("MID_LANE_TARGET", (0.0, 0.0))
            enemy_tower = self._nearest_enemy_tower(observation, hero)
            if enemy_tower is not None:
                return self._actor_pos(enemy_tower)
            return [0.0, 0.0]
        return self._actor_pos(min(candidates, key=lambda actor: math.dist(hero_pos, self._actor_pos(actor))))

    def _tower_retreat_action(self, observation):
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        enemy_tower = self._nearest_enemy_tower(observation, hero)
        if enemy_tower is None:
            return None
        hero_id = hero.get("runtime_id", hero.get("player_id"))
        tower_target = enemy_tower.get("attack_target", 0)
        hp_rate = self._hp_ratio(hero)
        if tower_target != hero_id and (
            hp_rate > game_config("FIGHT_RETREAT_LOW_HP", 0.35)
            or not self._unsafe_enemy_tower_zone(observation, hero, enemy_tower)
        ):
            return None

        legal_actions = self._split_legal_actions(observation)
        if legal_actions is None or legal_actions[0][2] <= 0:
            return None

        hero_pos = np.array(self._actor_pos(hero), dtype=np.float32)
        tower_pos = np.array(self._actor_pos(enemy_tower), dtype=np.float32)
        away = hero_pos - tower_pos
        if float(np.max(np.abs(away))) <= 1e-6:
            our_tower = self._nearest_our_tower(observation, hero)
            if our_tower is None:
                return None
            target = self._actor_pos(our_tower)
        else:
            target = (hero_pos + away / max(float(np.linalg.norm(away)), 1.0) * 10000.0).tolist()

        return self._move_to_point_action(legal_actions, self._actor_pos(hero), target)

    def _unsafe_enemy_tower_zone(self, observation, hero, enemy_tower):
        tower_range = float(enemy_tower.get("attack_range", 0) or 0) + 500.0
        if math.dist(self._actor_pos(hero), self._actor_pos(enemy_tower)) > tower_range:
            return False
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        tower_target = enemy_tower.get("attack_target", 0)
        our_soldier_ids = {
            npc.get("runtime_id", npc.get("player_id"))
            for npc in observation.get("frame_state", {}).get("npc_states", [])
            if self._is_soldier_actor(npc)
            and camp_id(npc.get("camp")) == player_camp
            and float(npc.get("hp", 0) or 0) > 0
        }
        return not (tower_target in our_soldier_ids and tower_target != 0)

    def _own_tower_safe_anchor(self, hero, our_tower):
        tower_pos = np.array(self._actor_pos(our_tower), dtype=np.float32)
        direction = tower_pos.copy()
        if float(np.linalg.norm(direction)) <= 1.0:
            direction = np.array(self._actor_pos(hero), dtype=np.float32) - tower_pos
        norm = max(float(np.linalg.norm(direction)), 1.0)
        offset = game_config("TOWER_SAFE_ANCHOR_OFFSET", 8500.0)
        return (tower_pos + direction / norm * offset).tolist()

    def _enemy_threat_nearby(self, observation, hero, radius):
        enemy = self._enemy_hero_state(observation)
        if enemy and not self._enemy_dead(enemy) and not self._is_unseen_actor(enemy):
            if math.dist(self._actor_pos(hero), self._actor_pos(enemy)) <= radius:
                return True
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        for npc in observation.get("frame_state", {}).get("npc_states", []):
            if (
                self._is_soldier_actor(npc)
                and camp_id(npc.get("camp")) != player_camp
                and float(npc.get("hp", 0) or 0) > 0
                and not self._is_unseen_actor(npc)
                and math.dist(self._actor_pos(hero), self._actor_pos(npc)) <= radius * 0.65
            ):
                return True
        return False

    def _actor_pos(self, actor):
        loc = actor.get("location", {}) if isinstance(actor, dict) else {}
        if not loc and isinstance(actor, dict):
            loc = ((actor.get("collider", {}) or {}).get("location", {}) or {})
        if isinstance(loc, dict):
            return [float(loc.get("x", 0) or 0), float(loc.get("z", 0) or 0)]
        if isinstance(loc, (list, tuple)) and len(loc) >= 3:
            return [float(loc[0] or 0), float(loc[2] or 0)]
        return [0.0, 0.0]

    def _mirror_action_space(self):
        return camp_id(self.hero_camp) == 2

    def _point_for_action(self, point):
        x, z = float(point[0]), float(point[1])
        if self._mirror_action_space() and abs(x) < 100000 and abs(z) < 100000:
            return [-x, -z]
        return [x, z]

    def _actor_pos_for_action(self, actor):
        return self._point_for_action(self._actor_pos(actor))

    def _is_unseen_actor(self, actor):
        x, z = self._actor_pos(actor)
        return abs(x) >= 100000 or abs(z) >= 100000

    def _is_soldier_actor(self, actor):
        if is_neutral_camp(actor.get("camp")):
            return False
        return actor.get("actor_type") == 1 and actor.get("sub_type") in (1, 11, "ACTOR_SUB_SOLDIER")

    def _is_tower_actor(self, actor):
        return actor.get("actor_type") == 2 and actor.get("sub_type") in (21, 23, 24, "ACTOR_SUB_TOWER")

    def _is_monster_actor(self, actor):
        return (
            actor.get("actor_type") in (3, "ACTOR_TYPE_MONSTER")
            or actor.get("sub_type") in ("ACTOR_SUB_MONSTER", "ACTOR_SUB_NEUTRAL_MONSTER")
            or (is_neutral_camp(actor.get("camp")) and actor.get("actor_type") == 1 and actor.get("sub_type") == 0)
        )

    def _enemy_hero_state(self, observation):
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        for hero in observation.get("frame_state", {}).get("hero_states", []):
            if camp_id(hero.get("camp")) != player_camp:
                return hero
        return None

    def _hp_gap(self, a, b):
        def ratio(actor):
            return float(actor.get("hp", 0) or 0) / max(float(actor.get("max_hp", 1) or 1), 1.0)

        return abs(ratio(a) - ratio(b))

    def _update_combat_memory(self, observation):
        hero = self._main_hero_state(observation)
        enemy = self._enemy_hero_state(observation)
        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        if enemy:
            if self._enemy_dead(enemy):
                self.last_enemy_dead_frame = frame_no
            elif not self._is_unseen_actor(enemy):
                self.last_enemy_alive_frame = frame_no
        if not hero or not enemy:
            return
        enemy_id = enemy.get("runtime_id", enemy.get("player_id"))
        hero_id = hero.get("runtime_id", hero.get("player_id"))
        hit_enemy = any((hit.get("hit_target") == enemy_id) for hit in hero.get("hit_target_info", []) or [])
        enemy_hits = enemy.get("hit_target_info", []) or []
        enemy_hit_us = any((hit.get("hit_target") == hero_id) for hit in enemy_hits)
        enemy_skill2_used_times = self._slot_used_times(enemy, 2)
        enemy_skill3_used_times = self._slot_used_times(enemy, 3)
        if enemy_skill2_used_times > self.last_enemy_skill2_used_times:
            self.last_enemy_skill2_used_frame = frame_no
        if enemy_skill3_used_times > self.last_enemy_skill3_used_times:
            self.last_enemy_skill3_used_frame = frame_no
        self.last_enemy_skill2_used_times = enemy_skill2_used_times
        self.last_enemy_skill3_used_times = enemy_skill3_used_times
        if any((hit.get("hit_target") == hero_id and int(hit.get("slot_type", -1) or -1) == 3) for hit in enemy_hits):
            self.last_enemy_skill3_hit_frame = frame_no
        fighting = hero.get("attack_target") == enemy_id or enemy.get("attack_target") == hero_id or hit_enemy or enemy_hit_us
        if fighting:
            self.last_hero_combat_frame = frame_no
            if frame_no - self.combat_start_frame > 90:
                self.combat_start_frame = frame_no
        elif frame_no - self.last_hero_combat_frame > 90:
            self.combat_start_frame = -100000

        if self._controlled_or_action_blocked(hero):
            self.last_control_frame = frame_no

    def _controlled_or_action_blocked(self, hero):
        if not hero:
            return False
        mode = int(hero.get("behav_mode", 0) or 0)
        if mode in (10, 11, 12, 13, 14, 18, 19, 20, 21, 22):
            return True
        abilities = hero.get("abilities", []) or []
        if isinstance(abilities, (list, tuple)) and len(abilities) >= 6:
            if bool(abilities[1]) is False and bool(abilities[5]) is False and self._hp_ratio(hero) > 0:
                return True
        buff_state = hero.get("buff_state", {}) or {}
        for buff in (buff_state.get("buff_skills", []) or []) + (buff_state.get("buff_marks", []) or []):
            buff_id = str(buff.get("configId", buff.get("config_id", "")))
            if any(token in buff_id for token in ("stun", "dizzy", "control", "freeze", "silence")):
                return True
        return False

    def _in_hero_combat_window(self, observation, frames):
        frame_no = int(observation.get("frame_state", {}).get("frame_no", 0) or 0)
        return frame_no - self.last_hero_combat_frame <= frames

    def _enemy_wave_cleared(self, observation):
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        for npc in observation.get("frame_state", {}).get("npc_states", []):
            if (
                self._is_soldier_actor(npc)
                and camp_id(npc.get("camp")) != player_camp
                and float(npc.get("hp", 0) or 0) > 0
            ):
                return False
        return True

    def _can_fight_enemy_hero(self, observation):
        hero = self._main_hero_state(observation)
        enemy = self._enemy_hero_state(observation)
        if not hero or not enemy or float(enemy.get("hp", 0) or 0) <= 0 or self._is_unseen_actor(enemy):
            return False
        enemy_tower = self._nearest_enemy_tower(observation, hero)
        if enemy_tower is not None and self._in_enemy_tower_range(hero, enemy_tower):
            hero_hp_ratio = float(hero.get("hp", 0) or 0) / max(float(hero.get("max_hp", 1) or 1), 1.0)
            enemy_hp_ratio = float(enemy.get("hp", 0) or 0) / max(float(enemy.get("max_hp", 1) or 1), 1.0)
            hp_advantage = hero_hp_ratio - enemy_hp_ratio
            can_kill = (hp_advantage > 0.25 or enemy_hp_ratio <= 0.18) and float(enemy.get("hp", 0) or 0) < float(
                hero.get("phy_atk", 0) or 0
            ) * 3
            tower_not_targeting = enemy_tower.get("attack_target", 0) != hero.get("runtime_id", hero.get("player_id"))
            return bool(can_kill and tower_not_targeting and hero_hp_ratio > 0.55)
        hero_hp_ratio = float(hero.get("hp", 0) or 0) / max(float(hero.get("max_hp", 1) or 1), 1.0)
        enemy_hp_ratio = float(enemy.get("hp", 0) or 0) / max(float(enemy.get("max_hp", 1) or 1), 1.0)
        if hero_hp_ratio < 0.35 and enemy_hp_ratio > 0.25:
            return False
        hero_range = float(hero.get("attack_range", 0) or 0) + 3500.0
        return math.dist(self._actor_pos(hero), self._actor_pos(enemy)) <= hero_range

    def _in_enemy_tower_range(self, hero, enemy_tower):
        tower_range = float(enemy_tower.get("attack_range", 0) or 0) + 1000.0
        return math.dist(self._actor_pos(hero), self._actor_pos(enemy_tower)) <= tower_range

    def _nearest_monster(self, observation, hero):
        monsters = [
            npc
            for npc in observation.get("frame_state", {}).get("npc_states", [])
            if self._is_monster_actor(npc) and float(npc.get("hp", 0) or 0) > 0 and not self._is_unseen_actor(npc)
        ]
        if not monsters:
            return None
        hero_pos = self._actor_pos(hero)
        return min(monsters, key=lambda actor: math.dist(hero_pos, self._actor_pos(actor)))

    def _nearest_grass_point(self, hero):
        points = game_config("GRASS_POINTS", [])
        if not points:
            return None
        hero_pos = self._actor_pos(hero)
        return list(min(points, key=lambda point: math.dist(hero_pos, point)))

    def _target_index_for_actor(self, observation, actor):
        if self._is_tower_actor(actor):
            return 7
        if self._is_monster_actor(actor):
            return 8
        return 3

    def _best_lane_target_actor(self, observation):
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        low_hp_soldiers = [
            npc
            for npc in observation.get("frame_state", {}).get("npc_states", [])
            if self._is_soldier_actor(npc)
            and camp_id(npc.get("camp")) != player_camp
            and float(npc.get("hp", 0) or 0) > 0
            and float(npc.get("hp", 0) or 0) / max(float(npc.get("max_hp", 1) or 1), 1.0) <= 0.55
        ]
        if low_hp_soldiers:
            hero = self._main_hero_state(observation)
            if hero:
                hero_pos = self._actor_pos(hero)
                return min(low_hp_soldiers, key=lambda npc: math.dist(hero_pos, self._actor_pos(npc)))
            return low_hp_soldiers[0]
        return None

    def _nearest_enemy_soldier(self, observation, hero):
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        soldiers = [
            npc
            for npc in observation.get("frame_state", {}).get("npc_states", [])
            if self._is_soldier_actor(npc)
            and camp_id(npc.get("camp")) != player_camp
            and float(npc.get("hp", 0) or 0) > 0
            and not self._is_unseen_actor(npc)
        ]
        if not soldiers:
            return None
        hero_pos = self._actor_pos(hero)
        return min(soldiers, key=lambda npc: math.dist(hero_pos, self._actor_pos(npc)))

    def _nearest_enemy_tower(self, observation, hero):
        frame_state = observation.get("frame_state", {})
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        towers = [
            npc
            for npc in frame_state.get("npc_states", [])
            if self._is_tower_actor(npc)
            and camp_id(npc.get("camp")) != player_camp
            and float(npc.get("hp", 0) or 0) > 0
            and not self._is_unseen_actor(npc)
        ]
        if not towers:
            return None
        hero_pos = self._actor_pos(hero)
        return min(towers, key=lambda tower: math.dist(hero_pos, self._actor_pos(tower)))

    def _nearest_our_tower(self, observation, hero):
        frame_state = observation.get("frame_state", {})
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        towers = [
            npc
            for npc in frame_state.get("npc_states", [])
            if self._is_tower_actor(npc)
            and camp_id(npc.get("camp")) == player_camp
            and float(npc.get("hp", 0) or 0) > 0
            and not self._is_unseen_actor(npc)
        ]
        if not towers:
            return None
        hero_pos = self._actor_pos(hero)
        return min(towers, key=lambda tower: math.dist(hero_pos, self._actor_pos(tower)))

    def _our_base_tower(self, observation, hero):
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        base_towers = [
            npc
            for npc in observation.get("frame_state", {}).get("npc_states", [])
            if self._is_tower_actor(npc)
            and camp_id(npc.get("camp")) == player_camp
            and npc.get("sub_type") == 23
            and float(npc.get("hp", 0) or 0) > 0
            and not self._is_unseen_actor(npc)
        ]
        if base_towers:
            hero_pos = self._actor_pos(hero)
            return min(base_towers, key=lambda tower: math.dist(hero_pos, self._actor_pos(tower)))
        return self._nearest_our_tower(observation, hero)

    def _safe_enemy_tower_target(self, observation):
        hero = self._main_hero_state(observation)
        if not hero:
            return None
        frame_state = observation.get("frame_state", {})
        player_camp = camp_id(observation.get("camp", observation.get("player_camp", self.hero_camp)))
        enemy_tower = self._nearest_enemy_tower(observation, hero)
        if enemy_tower is None:
            return None

        tower_target = enemy_tower.get("attack_target", 0)
        our_soldier_ids = {
            npc.get("runtime_id", npc.get("player_id"))
            for npc in frame_state.get("npc_states", [])
            if self._is_soldier_actor(npc) and camp_id(npc.get("camp")) == player_camp and float(npc.get("hp", 0) or 0) > 0
        }
        tower_tanking_minion = tower_target in our_soldier_ids and tower_target != 0
        if not tower_tanking_minion:
            return None

        hero_can_hit_tower = math.dist(self._actor_pos(hero), self._actor_pos(enemy_tower)) <= float(hero.get("attack_range", 0) or 0) + 1000.0
        if not hero_can_hit_tower:
            return None

        for enemy in frame_state.get("hero_states", []):
            if camp_id(enemy.get("camp")) == player_camp:
                continue
            if float(enemy.get("hp", 0) or 0) <= 0 or float(enemy.get("revive_time", 0) or 0) > 0:
                continue
            if self._is_unseen_actor(enemy):
                continue
            enemy_range = float(enemy.get("attack_range", 0) or 0) + 2000.0
            if math.dist(self._actor_pos(hero), self._actor_pos(enemy)) <= enemy_range:
                return None
        return enemy_tower

    def _delta_action_16x16(self, center, target):
        delta = np.array(target, dtype=np.float32) - np.array(center, dtype=np.float32)
        max_abs = float(np.max(np.abs(delta)))
        if max_abs <= 1e-6:
            return 8, 8
        grid = np.ceil(delta / max_abs * 7).astype(np.int32) + np.array([8, 8], dtype=np.int32)
        grid = np.clip(grid, 0, 15)
        return int(grid[0]), int(grid[1])

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

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
