#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""2024 baseline reward ported to the 2026 flat observation schema."""

import math

from agent_ppo.conf.conf import GameConfig


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight


def init_calc_frame_map():
    return {key: RewardStruct(weight) for key, weight in GameConfig.REWARD_WEIGHT_DICT.items()}


def get_any(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict) and key in d:
            return d[key]
    return default


def runtime_id(actor):
    return get_any(actor or {}, "runtime_id", "player_id", default=0)


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


def actor_pos(actor):
    loc = actor.get("location", {}) if isinstance(actor, dict) else {}
    if not loc and isinstance(actor, dict):
        loc = ((actor.get("actor_state", {}) or {}).get("location", {}) or {})
    if isinstance(loc, dict):
        return [float(loc.get("x", 0) or 0), float(loc.get("z", 0) or 0)]
    if isinstance(loc, (list, tuple)) and len(loc) >= 3:
        return [float(loc[0] or 0), float(loc[2] or 0)]
    return [0.0, 0.0]


def hp_ratio(actor):
    max_hp = float(get_any(actor or {}, "max_hp", default=0) or 0)
    if max_hp <= 0:
        return 0.0
    return float(get_any(actor or {}, "hp", default=0) or 0) / max_hp


def is_tower(actor):
    cfg = int(get_any(actor or {}, "config_id", "configId", default=0) or 0)
    return get_any(actor or {}, "sub_type", default=None) in (21, "ACTOR_SUB_TOWER") or cfg in (1111, 1112)


def is_soldier(actor):
    cfg = int(get_any(actor or {}, "config_id", "configId", default=0) or 0)
    return get_any(actor or {}, "sub_type", default=None) in (11, "ACTOR_SUB_SOLDIER") or cfg in (6800, 6801, 6802, 6803, 6804, 6805)


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_each_level_max_exp = {}
        self.last_hp_by_camp = {}

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp = {
            1: 160,
            2: 298,
            3: 446,
            4: 524,
            5: 613,
            6: 713,
            7: 825,
            8: 950,
            9: 1088,
            10: 1240,
            11: 1406,
            12: 1585,
            13: 1778,
            14: 1984,
        }

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        frame_no = int(get_any(frame_data, "frame_no", "frameNo", default=0) or 0)
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
        return self.m_reward_value

    def set_cur_calc_frame_vec(self, calc_frame_map, frame_data, camp):
        main_hero, enemy_hero = None, None
        for hero in frame_data.get("hero_states", []) or []:
            if camp_id(hero.get("camp")) == camp_id(camp):
                main_hero = hero
            else:
                enemy_hero = hero

        main_tower, enemy_tower = None, None
        for npc in frame_data.get("npc_states", []) or []:
            if not is_tower(npc):
                continue
            if camp_id(npc.get("camp")) == camp_id(camp):
                main_tower = main_tower or npc
            else:
                enemy_tower = enemy_tower or npc

        for reward_name, reward_struct in calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            if reward_name == "money":
                reward_struct.cur_frame_value = float(get_any(main_hero or {}, "money_cnt", "moneyCnt", "money", default=0) or 0)
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(max(hp_ratio(main_hero), 0.0))) if main_hero else 0.0
            elif reward_name == "ep_rate":
                max_ep = float(get_any(main_hero or {}, "max_ep", default=0) or 0)
                hp = float(get_any(main_hero or {}, "hp", default=0) or 0)
                reward_struct.cur_frame_value = 0.0 if max_ep <= 0 or hp <= 0 else float(get_any(main_hero, "ep", default=0) or 0) / max_ep
            elif reward_name == "kill":
                reward_struct.cur_frame_value = float(get_any(main_hero or {}, "kill_cnt", "killCnt", default=0) or 0)
            elif reward_name == "death":
                reward_struct.cur_frame_value = float(get_any(main_hero or {}, "dead_cnt", "deadCnt", default=0) or 0)
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = hp_ratio(main_tower)
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = self.calculate_last_hit(frame_data, main_hero, enemy_hero)
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "low_hp_recovery":
                reward_struct.cur_frame_value = self.calculate_low_hp_recovery(camp, main_hero, main_tower, enemy_hero)
            elif reward_name == "low_hp_danger":
                reward_struct.cur_frame_value = self.calculate_low_hp_danger(main_hero, enemy_hero, enemy_tower)

    def calculate_last_hit(self, frame_data, main_hero, enemy_hero):
        value = 0.0
        frame_action = frame_data.get("frame_action", {}) or {}
        for dead_action in frame_action.get("dead_action", []) or []:
            killer = dead_action.get("killer", {}) or {}
            death = dead_action.get("death", {}) or {}
            if not is_soldier(death):
                continue
            if runtime_id(killer) == runtime_id(main_hero):
                value += 1.0
            elif runtime_id(killer) == runtime_id(enemy_hero):
                value -= 1.0
        return value

    def calculate_exp_sum(self, hero):
        if not hero:
            return 0.0
        level = int(get_any(hero, "level", default=1) or 1)
        exp_sum = sum(self.m_each_level_max_exp.get(i, 0) for i in range(1, level))
        return exp_sum + float(get_any(hero, "exp", default=0) or 0)

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        if not main_hero or not main_tower or not enemy_tower:
            return 0.0
        dist_hero2emy = math.dist(actor_pos(main_hero), actor_pos(enemy_tower))
        dist_main2emy = max(math.dist(actor_pos(main_tower), actor_pos(enemy_tower)), 1.0)
        if hp_ratio(main_hero) > 0.99 and dist_hero2emy > dist_main2emy:
            return (dist_main2emy - dist_hero2emy) / dist_main2emy
        return 0.0

    def calculate_low_hp_recovery(self, camp, main_hero, main_tower, enemy_hero):
        if not main_hero:
            return 0.0
        camp = camp_id(camp)
        hp = hp_ratio(main_hero)
        last_hp = self.last_hp_by_camp.get(camp)
        self.last_hp_by_camp[camp] = hp
        if last_hp is None or last_hp > GameConfig.LOW_HP_LEARN_THRESHOLD:
            return 0.0
        gain = hp - last_hp
        if gain < GameConfig.LOW_HP_RECOVERY_GAIN:
            return 0.0

        safe_bonus = 0.0
        if enemy_hero:
            safe_bonus += float(math.dist(actor_pos(main_hero), actor_pos(enemy_hero)) >= GameConfig.LOW_HP_SAFE_ENEMY_DISTANCE)
        if main_tower:
            safe_bonus += float(math.dist(actor_pos(main_hero), actor_pos(main_tower)) <= GameConfig.LOW_HP_SAFE_ENEMY_DISTANCE)
        return min(gain / 0.25, 1.0) * (1.0 + 0.25 * safe_bonus)

    def calculate_low_hp_danger(self, main_hero, enemy_hero, enemy_tower):
        if not main_hero:
            return 0.0
        hp = hp_ratio(main_hero)
        if hp <= 0.0 or hp > GameConfig.LOW_HP_LEARN_THRESHOLD:
            return 0.0

        hp_factor = (GameConfig.LOW_HP_LEARN_THRESHOLD - hp) / max(GameConfig.LOW_HP_LEARN_THRESHOLD, 1e-6)
        danger = 0.0
        if enemy_hero:
            enemy_dist = math.dist(actor_pos(main_hero), actor_pos(enemy_hero))
            danger += max(0.0, 1.0 - enemy_dist / GameConfig.LOW_HP_DANGER_ENEMY_DISTANCE)
        if enemy_tower:
            tower_dist = math.dist(actor_pos(main_hero), actor_pos(enemy_tower))
            danger += 0.75 * max(0.0, 1.0 - tower_dist / GameConfig.LOW_HP_DANGER_TOWER_DISTANCE)
        return min(danger, 1.0) * hp_factor

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data.get("hero_states", []) or []:
            if runtime_id(hero) == self.main_hero_player_id:
                main_camp = camp_id(hero.get("camp"))
                self.main_hero_camp = main_camp
            else:
                enemy_camp = camp_id(hero.get("camp"))
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum = 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                main = self.m_main_calc_frame_map[reward_name]
                enemy = self.m_enemy_calc_frame_map[reward_name]
                reward_struct.cur_frame_value = main.cur_frame_value - enemy.cur_frame_value
                reward_struct.last_frame_value = main.last_frame_value - enemy.last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "ep_rate":
                main = self.m_main_calc_frame_map[reward_name]
                reward_struct.value = main.cur_frame_value - main.last_frame_value if main.last_frame_value > 0 else 0.0
            elif reward_name == "exp":
                main_hero = next((h for h in frame_data.get("hero_states", []) or [] if runtime_id(h) == self.main_hero_player_id), None)
                if main_hero and int(get_any(main_hero, "level", default=1) or 1) >= 15:
                    reward_struct.value = 0.0
                else:
                    main = self.m_main_calc_frame_map[reward_name]
                    enemy = self.m_enemy_calc_frame_map[reward_name]
                    reward_struct.value = (main.cur_frame_value - enemy.cur_frame_value) - (main.last_frame_value - enemy.last_frame_value)
            elif reward_name in ("forward", "last_hit", "low_hp_recovery"):
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "low_hp_danger":
                main = self.m_main_calc_frame_map[reward_name]
                enemy = self.m_enemy_calc_frame_map[reward_name]
                reward_struct.value = main.cur_frame_value - enemy.cur_frame_value
            else:
                main = self.m_main_calc_frame_map[reward_name]
                enemy = self.m_enemy_calc_frame_map[reward_name]
                reward_struct.value = (main.cur_frame_value - enemy.cur_frame_value) - (main.last_frame_value - enemy.last_frame_value)

            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum
