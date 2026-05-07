#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
2025 top1 reward design adapted to the 2026 flat frame_state protocol.
"""

import math
from agent_ppo.conf.conf import GameConfig


def get_any(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict) and key in d:
            return d[key]
    return default


def camp_id(camp):
    if isinstance(camp, str):
        if camp in ("0", "1", "2"):
            value = int(camp)
            return value + 1 if value in (0, 1) else value
        if camp[-1:].isdigit():
            return int(camp[-1])
    if isinstance(camp, int):
        return camp + 1 if camp in (0, 1) else camp
    return camp


def is_tower(npc):
    st = npc.get("sub_type")
    at = npc.get("actor_type")
    return at == 2 and st in (21, 23, 24, "ACTOR_SUB_TOWER")


def is_soldier_actor(actor):
    st = actor.get("sub_type") if isinstance(actor, dict) else None
    at = actor.get("actor_type") if isinstance(actor, dict) else None
    return at == 1 or st in (1, 11, "ACTOR_SUB_SOLDIER")


def pos(actor):
    p = actor.get("location", {}) if isinstance(actor, dict) else {}
    if isinstance(p, dict):
        return (p.get("x", 0), p.get("z", 0))
    if isinstance(p, (list, tuple)) and len(p) >= 3:
        return (p[0], p[2])
    return (0, 0)


def actor_id(actor):
    return get_any(actor or {}, "runtime_id", "player_id", default=0)


def hp_ratio(actor):
    return float(get_any(actor or {}, "hp", default=0) or 0) / max(float(get_any(actor or {}, "max_hp", default=1) or 1), 1.0)


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight


def init_calc_frame_map():
    return {key: RewardStruct(weight) for key, weight in GameConfig.REWARD_WEIGHT_DICT.items()}


class GameRewardManager:
    def __init__(self, main_hero_runtime_id, main_hero_camp=None):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = camp_id(main_hero_camp) if main_hero_camp is not None else -1
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_each_level_max_exp = {}
        self.last_main_pos = None
        self.init_max_exp_of_each_hero()

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
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        return self.m_reward_value

    def set_cur_calc_frame_vec(self, calc_frame_map, frame_data, camp):
        main_hero, enemy_hero = None, None
        for hero in frame_data.get("hero_states", []):
            if camp_id(hero.get("camp")) == camp:
                main_hero = hero
            else:
                enemy_hero = hero

        main_tower, enemy_tower = None, None
        for npc in frame_data.get("npc_states", []):
            if not is_tower(npc):
                continue
            if camp_id(npc.get("camp")) == camp:
                main_tower = npc
            else:
                enemy_tower = npc

        if not main_hero or not main_tower or not enemy_tower:
            return

        for reward_name, reward_struct in calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            if reward_name == "money":
                reward_struct.cur_frame_value = get_any(main_hero, "money_cnt", "moneyCnt", "money", default=0) or 0
            elif reward_name == "hp_point":
                cur_hp = max(float(get_any(main_hero, "hp", default=0) or 0), 0.0)
                max_hp = max(float(get_any(main_hero, "max_hp", default=1) or 1), 1.0)
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(cur_hp / max_hp))
            elif reward_name == "ep_rate":
                max_ep = float(get_any(main_hero, "max_ep", default=0) or 0)
                if max_ep <= 0 or (get_any(main_hero, "hp", default=0) or 0) <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = float(get_any(main_hero, "ep", default=0) or 0) / max_ep
            elif reward_name == "kill":
                reward_struct.cur_frame_value = get_any(main_hero, "kill_cnt", "killCnt", default=0) or 0
            elif reward_name == "death":
                reward_struct.cur_frame_value = get_any(main_hero, "dead_cnt", "deadCnt", default=0) or 0
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = float(get_any(main_tower, "hp", default=0) or 0) / max(float(get_any(main_tower, "max_hp", default=1) or 1), 1.0)
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = self.calculate_last_hit(frame_data, main_hero, enemy_hero)
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "danger":
                reward_struct.cur_frame_value = self.calculate_danger(main_hero, enemy_tower)
            elif reward_name == "idle":
                reward_struct.cur_frame_value = self.calculate_idle(main_hero)
            elif reward_name == "tower_attack":
                reward_struct.cur_frame_value = self.calculate_tower_attack(main_hero, enemy_tower)
            elif reward_name == "safe_push":
                reward_struct.cur_frame_value = self.calculate_safe_push(frame_data, main_hero, enemy_hero, enemy_tower)

    def calculate_last_hit(self, frame_data, main_hero, enemy_hero):
        value = 0.0
        main_id = get_any(main_hero, "runtime_id", "player_id", default=-1)
        enemy_id = get_any(enemy_hero or {}, "runtime_id", "player_id", default=-2)
        frame_action = frame_data.get("frame_action", {}) or {}
        for dead_action in frame_action.get("dead_action", []) or []:
            death = dead_action.get("death", {}) or {}
            killer = dead_action.get("killer", {}) or {}
            if not is_soldier_actor(death):
                continue
            killer_id = get_any(killer, "runtime_id", default=-3)
            if killer_id == main_id:
                value += 1.0
            elif killer_id == enemy_id:
                value -= 1.0
        return value

    def calculate_exp_sum(self, hero):
        exp_sum = 0.0
        level = int(get_any(hero, "level", default=1) or 1)
        for i in range(1, level):
            exp_sum += self.m_each_level_max_exp.get(i, 0)
        exp_sum += get_any(hero, "exp", default=0) or 0
        return exp_sum

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        dist_hero2emy = math.dist(pos(main_hero), pos(enemy_tower))
        dist_main2emy = max(math.dist(pos(main_tower), pos(enemy_tower)), 1.0)
        hp_rate = float(get_any(main_hero, "hp", default=0) or 0) / max(float(get_any(main_hero, "max_hp", default=1) or 1), 1.0)
        if hp_rate > 0.45:
            return (dist_main2emy - dist_hero2emy) / dist_main2emy
        return 0.0

    def calculate_danger(self, main_hero, enemy_tower):
        if not main_hero or not enemy_tower:
            return 0.0
        hp_rate = float(get_any(main_hero, "hp", default=0) or 0) / max(float(get_any(main_hero, "max_hp", default=1) or 1), 1.0)
        tower_range = float(get_any(enemy_tower, "attack_range", default=0) or 0) + 2000.0
        in_tower_range = math.dist(pos(main_hero), pos(enemy_tower)) <= tower_range
        tower_targeting = get_any(enemy_tower, "attack_target", default=0) == get_any(main_hero, "runtime_id", "player_id", default=-1)
        return float((hp_rate < 0.55 and in_tower_range) or tower_targeting)

    def calculate_tower_attack(self, main_hero, enemy_tower):
        if not main_hero or not enemy_tower:
            return 0.0
        enemy_tower_id = actor_id(enemy_tower)
        if get_any(main_hero, "attack_target", default=0) == enemy_tower_id:
            return 1.0
        for hit in get_any(main_hero, "hit_target_info", default=[]) or []:
            if get_any(hit, "hit_target", "runtime_id", default=0) == enemy_tower_id:
                return 1.0
        return 0.0

    def calculate_safe_push(self, frame_data, main_hero, enemy_hero, enemy_tower):
        if not main_hero or not enemy_tower:
            return 0.0
        main_camp = camp_id(main_hero.get("camp"))
        target = get_any(enemy_tower, "attack_target", default=0)
        soldier_ids = {
            actor_id(npc)
            for npc in frame_data.get("npc_states", [])
            if is_soldier_actor(npc) and camp_id(npc.get("camp")) == main_camp
        }
        tower_tanking_minion = target in soldier_ids and target != 0
        if not tower_tanking_minion:
            return 0.0

        enemy_threat = False
        if enemy_hero and hp_ratio(enemy_hero) > 0 and get_any(enemy_hero, "revive_time", default=0) <= 0:
            enemy_range = float(get_any(enemy_hero, "attack_range", default=0) or 0) + 2000.0
            enemy_threat = math.dist(pos(main_hero), pos(enemy_hero)) <= enemy_range

        hero_can_hit_tower = math.dist(pos(main_hero), pos(enemy_tower)) <= float(get_any(main_hero, "attack_range", default=0) or 0) + 1000.0
        return float(hero_can_hit_tower and not enemy_threat)

    def calculate_idle(self, main_hero):
        cur_pos = pos(main_hero)
        if self.last_main_pos is None:
            self.last_main_pos = cur_pos
            return 0.0
        moved = math.dist(cur_pos, self.last_main_pos)
        self.last_main_pos = cur_pos
        hp_rate = float(get_any(main_hero, "hp", default=0) or 0) / max(float(get_any(main_hero, "max_hp", default=1) or 1), 1.0)
        return float(hp_rate > 0.5 and moved < 120.0)

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data.get("hero_states", []):
            if get_any(hero, "runtime_id", "player_id", default=None) == self.main_hero_player_id:
                main_camp = camp_id(hero.get("camp"))
                self.main_hero_camp = main_camp
            else:
                enemy_camp = camp_id(hero.get("camp"))
        if main_camp == -1 and self.main_hero_camp != -1:
            main_camp = self.main_hero_camp
            for hero in frame_data.get("hero_states", []):
                hero_camp = camp_id(hero.get("camp"))
                if hero_camp != main_camp:
                    enemy_camp = hero_camp
                    break
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        frame_no = get_any(frame_data, "frame_no", "frameNo", default=0) or 0
        reward_sum = 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                main_last = self.m_main_calc_frame_map[reward_name].last_frame_value
                enemy_last = self.m_enemy_calc_frame_map[reward_name].last_frame_value
                if main_last == 0.0 and enemy_last == 0.0:
                    reward_struct.value = 0.0
                else:
                    reward_struct.value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                        - main_last
                        + enemy_last
                    )
            elif reward_name == "ep_rate":
                reward_struct.value = 0.0
                last_ep = self.m_main_calc_frame_map[reward_name].last_frame_value
                if last_ep > 0:
                    reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value - last_ep
            elif reward_name == "exp":
                main_hero = None
                for hero in frame_data.get("hero_states", []):
                    if (
                        get_any(hero, "runtime_id", "player_id", default=None) == self.main_hero_player_id
                        or camp_id(hero.get("camp")) == self.main_hero_camp
                    ):
                        main_hero = hero
                        break
                if main_hero and (get_any(main_hero, "level", default=1) or 1) >= 15:
                    reward_struct.value = 0.0
                else:
                    reward_struct.value = self.zero_sum_delta(reward_name)
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                if GameConfig.REMOVE_FORWARD_AFTER is not None and frame_no > GameConfig.REMOVE_FORWARD_AFTER:
                    reward_struct.value = 0.0
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name in ("danger", "idle", "tower_attack", "safe_push"):
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            else:
                reward_struct.value = self.zero_sum_delta(reward_name)

            time_scale = 1.0
            if self.time_scale_arg > 0 and reward_name not in GameConfig.REWARD_WITHOUT_TIME_SCALE:
                time_scale = math.pow(0.6, float(frame_no) / self.time_scale_arg)
            reward_dict[reward_name] = reward_struct.value
            reward_dict[f"{reward_name}_origin"] = reward_struct.value
            reward_dict[f"{reward_name}_weight"] = reward_struct.value * reward_struct.weight * time_scale
            reward_sum += reward_dict[f"{reward_name}_weight"]
        reward_dict["reward_sum"] = reward_sum

    def zero_sum_delta(self, reward_name):
        cur = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
        last = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
        return cur - last
