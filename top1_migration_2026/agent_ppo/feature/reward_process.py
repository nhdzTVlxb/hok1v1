#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
2025 top1 reward design adapted to the 2026 flat frame_state protocol.
"""

import math
from agent_ppo.conf.conf import GameConfig

ACTOR_TYPE_MONSTER = 1
ACTOR_TYPE_ORGAN = 2
SUB_TYPE_NEUTRAL_MONSTER = 0
SUB_TYPE_LANE_SOLDIER = 11
SUB_TYPE_TOWER = 21
SUB_TYPE_SPRING_TOWER = 23
SUB_TYPE_CRYSTAL = 24
RIVER_SPIRIT_CONFIG_ID = 6827
LANE_SOLDIER_CONFIG_IDS = {6800, 6801, 6802, 6803, 6804, 6805}
TOWER_CONFIG_IDS = {1111, 1112}
CRYSTAL_CONFIG_IDS = {1113, 1114}
SPRING_TOWER_CONFIG_IDS = {44, 46}
CAKE_LOCATIONS_BY_CAMP = {
    1: {"main": (-15220.0, -15120.0), "enemy": (15340.0, 15100.0)},
    2: {"main": (15340.0, 15100.0), "enemy": (-15220.0, -15120.0)},
}


def get_any(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict) and key in d:
            return d[key]
    return default


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


def is_tower(npc):
    st = npc.get("sub_type")
    at = npc.get("actor_type")
    cfg = int(get_any(npc or {}, "config_id", "configId", default=0) or 0)
    return at == ACTOR_TYPE_ORGAN and (
        st in (SUB_TYPE_TOWER, SUB_TYPE_SPRING_TOWER, SUB_TYPE_CRYSTAL, "ACTOR_SUB_TOWER")
        or cfg in TOWER_CONFIG_IDS
        or cfg in CRYSTAL_CONFIG_IDS
        or cfg in SPRING_TOWER_CONFIG_IDS
    )


def is_soldier_actor(actor):
    st = actor.get("sub_type") if isinstance(actor, dict) else None
    at = actor.get("actor_type") if isinstance(actor, dict) else None
    cfg = int(get_any(actor or {}, "config_id", "configId", default=0) or 0)
    if is_neutral_camp(get_any(actor or {}, "camp", default=None)):
        return False
    return at == ACTOR_TYPE_MONSTER and (st in (SUB_TYPE_LANE_SOLDIER, "ACTOR_SUB_SOLDIER") or cfg in LANE_SOLDIER_CONFIG_IDS)


def is_monster_actor(actor):
    at = get_any(actor or {}, "actor_type", default=None)
    st = get_any(actor or {}, "sub_type", default=None)
    cfg = int(get_any(actor or {}, "config_id", "configId", default=0) or 0)
    return (
        cfg == RIVER_SPIRIT_CONFIG_ID
        or at in (3, "ACTOR_TYPE_MONSTER")
        or st in ("ACTOR_SUB_MONSTER", "ACTOR_SUB_NEUTRAL_MONSTER")
        or (is_neutral_camp(get_any(actor or {}, "camp", default=None)) and at == ACTOR_TYPE_MONSTER and st == SUB_TYPE_NEUTRAL_MONSTER)
    )


def actor_type_priority(actor):
    if not actor:
        return 0.1
    if is_tower(actor):
        return 2.2
    if get_any(actor, "actor_type", default=None) == 0:
        return 1.5
    if is_monster_actor(actor):
        return 1.0
    if is_soldier_actor(actor):
        return 0.45
    return 0.5


def skill_hit_priority(actor):
    if not actor:
        return 0.0
    if get_any(actor, "actor_type", default=None) == 0:
        return 2.0
    if is_soldier_actor(actor):
        return 0.75
    if is_monster_actor(actor):
        return 0.45
    if is_tower(actor):
        return 0.25
    return 0.35


def pos(actor):
    p = actor.get("location", {}) if isinstance(actor, dict) else {}
    if not p and isinstance(actor, dict):
        p = ((actor.get("collider", {}) or {}).get("location", {}) or {})
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
        self.last_enemy_tower_hp_by_camp = {}
        self.last_finish_push_tower_hp_by_camp = {}
        self.last_phy_atk_by_camp = {}
        self.last_atk_spd_by_camp = {}
        self.last_rune_state_by_camp = {}
        self.last_recall_used_by_camp = {}
        self.last_enemy_hero_hit_frame = {}
        self.last_be_hurt_by_hero = {}
        self.last_skill_used_by_camp = {}
        self.last_stutter_pos_by_camp = {}
        self.monster_kills_by_camp = {}
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

        main_towers, enemy_towers = [], []
        for npc in frame_data.get("npc_states", []):
            if not is_tower(npc):
                continue
            if camp_id(npc.get("camp")) == camp:
                main_towers.append(npc)
            else:
                enemy_towers.append(npc)

        main_tower = min(main_towers, key=lambda tower: math.dist(pos(main_hero), pos(tower)), default=None)
        enemy_tower = min(enemy_towers, key=lambda tower: math.dist(pos(main_hero), pos(tower)), default=None)

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
            elif reward_name == "attack_hit":
                reward_struct.cur_frame_value = self.calculate_attack_hit(frame_data, main_hero)
            elif reward_name == "skill_hit":
                reward_struct.cur_frame_value = self.calculate_skill_hit(frame_data, main_hero)
            elif reward_name == "skill_misuse":
                reward_struct.cur_frame_value = self.calculate_skill_misuse(frame_data, main_hero, enemy_hero)
            elif reward_name == "monster_last_hit":
                reward_struct.cur_frame_value = self.calculate_monster_last_hit(frame_data, main_hero, enemy_hero)
            elif reward_name == "hero_lane_advantage":
                reward_struct.cur_frame_value = self.calculate_hero_lane_advantage(frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "monster_contest":
                reward_struct.cur_frame_value = self.calculate_monster_contest(frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "tower_chase_risk":
                reward_struct.cur_frame_value = self.calculate_tower_chase_risk(frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "attack_power":
                reward_struct.cur_frame_value = self.calculate_attack_power(camp, main_hero)
            elif reward_name == "attack_speed":
                reward_struct.cur_frame_value = self.calculate_attack_speed(camp, main_hero)
            elif reward_name == "tower_attack":
                reward_struct.cur_frame_value = self.calculate_tower_attack(main_hero, enemy_tower)
            elif reward_name == "safe_push":
                reward_struct.cur_frame_value = self.calculate_safe_push(frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "safe_tower_damage":
                reward_struct.cur_frame_value = self.calculate_safe_tower_damage(camp, frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "finish_push":
                reward_struct.cur_frame_value = self.calculate_finish_push(camp, frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "tower_hp_gap":
                reward_struct.cur_frame_value = self.calculate_tower_hp_gap(main_tower, enemy_tower)
            elif reward_name == "defend_tower_clear":
                reward_struct.cur_frame_value = self.calculate_defend_tower_clear(frame_data, main_hero, main_tower)
            elif reward_name == "post_fight_recall":
                reward_struct.cur_frame_value = self.calculate_post_fight_recall(frame_data, main_hero, enemy_hero, main_tower)
            elif reward_name == "recall_use":
                reward_struct.cur_frame_value = self.calculate_recall_use(camp, frame_data, main_hero, enemy_hero, main_tower)
            elif reward_name == "rune_approach":
                reward_struct.cur_frame_value = self.calculate_rune_approach(frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "rune_pickup":
                reward_struct.cur_frame_value = self.calculate_rune_pickup(camp, frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "enemy_rune_after_kill":
                reward_struct.cur_frame_value = self.calculate_enemy_rune_after_kill(frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "enemy_wave_overextend":
                reward_struct.cur_frame_value = self.calculate_enemy_wave_overextend(frame_data, main_hero, enemy_hero)
            elif reward_name == "lane_anchor":
                reward_struct.cur_frame_value = self.calculate_lane_anchor(frame_data, main_hero)
            elif reward_name == "grass_ambush":
                reward_struct.cur_frame_value = self.calculate_grass_ambush(frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "grass_engage":
                reward_struct.cur_frame_value = self.calculate_grass_engage(frame_data, main_hero, enemy_hero, enemy_tower)
            elif reward_name == "stutter_step":
                reward_struct.cur_frame_value = self.calculate_stutter_step(camp, frame_data, main_hero)
            elif reward_name == "retaliate_hit":
                reward_struct.cur_frame_value = self.calculate_retaliate_hit(camp, frame_data, main_hero, enemy_hero)
            elif reward_name == "fight_risk":
                reward_struct.cur_frame_value = self.calculate_fight_risk(frame_data, main_hero, enemy_hero, main_tower, enemy_tower)

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

    def calculate_attack_hit(self, frame_data, main_hero):
        if not main_hero:
            return 0.0
        actor_by_id = {}
        for item in (frame_data.get("hero_states", []) or []) + (frame_data.get("npc_states", []) or []):
            actor_by_id[actor_id(item)] = item

        hit_infos = get_any(main_hero, "hit_target_info", default=[]) or []
        if hit_infos:
            best = 0.0
            for hit in hit_infos:
                target = actor_by_id.get(get_any(hit, "hit_target", "runtime_id", default=0))
                slot_type = int(get_any(hit, "slot_type", default=0) or 0)
                if slot_type == 0:
                    best = max(best, actor_type_priority(target))
            level_scale = 0.5 + 0.5 * min(float(get_any(main_hero, "level", default=1) or 1), 15.0) / 15.0
            return best * level_scale

        target_id = get_any(main_hero, "attack_target", default=0)
        if target_id:
            return 0.08 * actor_type_priority(actor_by_id.get(target_id))
        return 0.0

    def calculate_skill_hit(self, frame_data, main_hero):
        if not main_hero:
            return 0.0
        actor_by_id = {}
        for item in (frame_data.get("hero_states", []) or []) + (frame_data.get("npc_states", []) or []):
            actor_by_id[actor_id(item)] = item

        value = 0.0
        for hit in get_any(main_hero, "hit_target_info", default=[]) or []:
            slot_type = int(get_any(hit, "slot_type", default=-1) or -1)
            if slot_type not in (1, 2, 3):
                continue
            target = actor_by_id.get(get_any(hit, "hit_target", "runtime_id", default=0))
            hit_value = skill_hit_priority(target)
            if slot_type == 3 and get_any(target or {}, "actor_type", default=None) == 0:
                hit_value *= 1.6
            elif slot_type in (2, 3) and get_any(target or {}, "actor_type", default=None) != 0:
                hit_value *= 0.3
            value += hit_value
        return min(value, 5.0)

    def calculate_skill_misuse(self, frame_data, main_hero, enemy_hero):
        if not main_hero:
            return 0.0
        camp = camp_id(main_hero.get("camp"))
        used_now = {
            slot_type: self.slot_used_times(main_hero, slot_type)
            for slot_type in (1, 2, 3)
        }
        last_used = self.last_skill_used_by_camp.get(camp)
        self.last_skill_used_by_camp[camp] = used_now
        if last_used is None:
            return 0.0

        used_slots = [slot_type for slot_type in used_now if used_now[slot_type] > last_used.get(slot_type, 0)]
        used_skill = bool(used_slots)
        if not used_skill:
            return 0.0
        if int(get_any(main_hero, "config_id", "configId", default=0) or 0) == 133 and used_slots == [2]:
            return 0.0

        hit_infos = get_any(main_hero, "hit_target_info", default=[]) or []
        if any(int(get_any(hit, "slot_type", default=-1) or -1) in (1, 2, 3) for hit in hit_infos):
            return 0.0

        if not enemy_hero or hp_ratio(enemy_hero) <= 0:
            return 0.4
        target = get_any(main_hero, "attack_target", default=0)
        in_combat = target == actor_id(enemy_hero) or math.dist(pos(main_hero), pos(enemy_hero)) <= float(get_any(main_hero, "attack_range", default=0) or 0) + 2000.0
        if in_combat:
            return 0.3
        return 0.8

    def calculate_monster_last_hit(self, frame_data, main_hero, enemy_hero):
        main_camp = camp_id(main_hero.get("camp")) if main_hero else -1
        main_id = actor_id(main_hero)
        enemy_id = actor_id(enemy_hero)
        value = 0.0
        frame_action = frame_data.get("frame_action", {}) or {}
        for dead_action in frame_action.get("dead_action", []) or []:
            death = dead_action.get("death", {}) or {}
            if not is_monster_actor(death):
                continue
            killer_id = get_any(dead_action.get("killer", {}) or {}, "runtime_id", "player_id", default=0)
            if killer_id == main_id:
                kills = self.monster_kills_by_camp.get(main_camp, 0)
                value += self.monster_value_scale(main_hero, kills)
                self.monster_kills_by_camp[main_camp] = kills + 1
            elif enemy_id and killer_id == enemy_id:
                value -= 0.7
        return value

    def monster_value_scale(self, main_hero, kills):
        money = float(get_any(main_hero, "money_cnt", "money", default=0) or 0)
        if money >= getattr(GameConfig, "ENDGAME_MONEY", 6000):
            money_scale = 0.20
        elif money >= getattr(GameConfig, "TOWER_FULL_PUSH_MONEY", 6000):
            money_scale = 0.35
        elif money >= getattr(GameConfig, "TOWER_EARLY_PUSH_MONEY", 5500):
            money_scale = 0.60
        else:
            money_scale = 1.0
        decay_kills = max(float(getattr(GameConfig, "MONSTER_REWARD_DECAY_KILLS", 3) or 3), 1.0)
        count_scale = max(getattr(GameConfig, "MONSTER_REWARD_MIN_SCALE", 0.10), 1.0 / (1.0 + kills / decay_kills))
        return money_scale * count_scale

    def lane_counts(self, frame_data, camp):
        our, enemy = 0, 0
        for npc in frame_data.get("npc_states", []) or []:
            if not is_soldier_actor(npc) or float(get_any(npc, "hp", default=0) or 0) <= 0:
                continue
            if camp_id(npc.get("camp")) == camp:
                our += 1
            else:
                enemy += 1
        return our, enemy

    def calculate_hero_lane_advantage(self, frame_data, main_hero, enemy_hero, enemy_tower):
        if not main_hero or not enemy_hero or hp_ratio(enemy_hero) <= 0:
            return 0.0
        main_camp = camp_id(main_hero.get("camp"))
        our_soldiers, enemy_soldiers = self.lane_counts(frame_data, main_camp)
        if enemy_soldiers > 0:
            return 0.0
        hp_gap = abs(hp_ratio(main_hero) - hp_ratio(enemy_hero))
        if hp_gap > 0.15:
            return 0.0
        if self.in_enemy_tower_range(main_hero, enemy_tower):
            return 0.0
        enemy_targeting_lane = get_any(enemy_hero, "attack_target", default=0) not in (0, actor_id(main_hero))
        attacking_enemy = get_any(main_hero, "attack_target", default=0) == actor_id(enemy_hero)
        hit_enemy = any(
            get_any(hit, "hit_target", "runtime_id", default=0) == actor_id(enemy_hero)
            for hit in get_any(main_hero, "hit_target_info", default=[]) or []
        )
        return float((enemy_targeting_lane or our_soldiers > 0) and (attacking_enemy or hit_enemy))

    def calculate_monster_contest(self, frame_data, main_hero, enemy_hero, enemy_tower):
        if not main_hero:
            return 0.0
        main_camp = camp_id(main_hero.get("camp"))
        _, enemy_soldiers = self.lane_counts(frame_data, main_camp)
        if enemy_soldiers > 0:
            return 0.0
        if enemy_hero and hp_ratio(enemy_hero) > 0 and abs(hp_ratio(main_hero) - hp_ratio(enemy_hero)) <= 0.15:
            if not self.in_enemy_tower_range(main_hero, enemy_tower):
                return 0.0
        monsters = [
            npc
            for npc in frame_data.get("npc_states", []) or []
            if is_monster_actor(npc) and float(get_any(npc, "hp", default=0) or 0) > 0
        ]
        if not monsters:
            return 0.0
        nearest = min(monsters, key=lambda m: math.dist(pos(main_hero), pos(m)))
        near = math.dist(pos(main_hero), pos(nearest)) <= float(get_any(main_hero, "attack_range", default=0) or 0) + 4000.0
        hit_monster = any(
            get_any(hit, "hit_target", "runtime_id", default=0) == actor_id(nearest)
            for hit in get_any(main_hero, "hit_target_info", default=[]) or []
        )
        money = float(get_any(main_hero, "money_cnt", "money", default=0) or 0)
        if money >= getattr(GameConfig, "ENDGAME_MONEY", 6000):
            scale = 0.12
        elif money < getattr(GameConfig, "TOWER_EARLY_PUSH_MONEY", 5500):
            scale = 1.0
        elif money < getattr(GameConfig, "TOWER_FULL_PUSH_MONEY", 6000):
            scale = 0.55
        else:
            scale = 0.25
        return float(near or hit_monster) * scale

    def calculate_enemy_wave_overextend(self, frame_data, main_hero, enemy_hero):
        if not main_hero:
            return 0.0
        main_camp = camp_id(main_hero.get("camp"))
        hero_pos = pos(main_hero)
        enemy_soldiers = [
            npc
            for npc in frame_data.get("npc_states", []) or []
            if is_soldier_actor(npc)
            and camp_id(npc.get("camp")) != main_camp
            and float(get_any(npc, "hp", default=0) or 0) > 0
            and math.dist(hero_pos, pos(npc)) <= 8000.0
        ]
        if len(enemy_soldiers) < 2:
            return 0.0
        allied_near = [
            npc
            for npc in frame_data.get("npc_states", []) or []
            if is_soldier_actor(npc)
            and camp_id(npc.get("camp")) == main_camp
            and float(get_any(npc, "hp", default=0) or 0) > 0
            and math.dist(hero_pos, pos(npc)) <= 9000.0
        ]
        if allied_near:
            return 0.0

        enemy_ids = {actor_id(npc) for npc in enemy_soldiers}
        attacking_enemy_wave = get_any(main_hero, "attack_target", default=0) in enemy_ids
        hitting_enemy_wave = any(
            get_any(hit, "hit_target", "runtime_id", default=0) in enemy_ids
            for hit in get_any(main_hero, "hit_target_info", default=[]) or []
        )
        if attacking_enemy_wave or hitting_enemy_wave:
            return 0.35

        if enemy_hero and hp_ratio(enemy_hero) > 0 and hp_ratio(main_hero) - hp_ratio(enemy_hero) > 0.10:
            return 0.4
        return 1.0

    def calculate_lane_anchor(self, frame_data, main_hero):
        if not main_hero:
            return 0.0
        main_camp = camp_id(main_hero.get("camp"))
        hero_pos = pos(main_hero)
        allied_near = [
            npc
            for npc in frame_data.get("npc_states", []) or []
            if is_soldier_actor(npc)
            and camp_id(npc.get("camp")) == main_camp
            and float(get_any(npc, "hp", default=0) or 0) > 0
            and math.dist(hero_pos, pos(npc)) <= 8000.0
        ]
        enemy_near = [
            npc
            for npc in frame_data.get("npc_states", []) or []
            if is_soldier_actor(npc)
            and camp_id(npc.get("camp")) != main_camp
            and float(get_any(npc, "hp", default=0) or 0) > 0
            and math.dist(hero_pos, pos(npc)) <= 12000.0
        ]
        return float(bool(allied_near and enemy_near))

    def in_enemy_tower_range(self, hero, enemy_tower):
        if not hero or not enemy_tower:
            return False
        tower_range = float(get_any(enemy_tower, "attack_range", default=0) or 0) + 1000.0
        return math.dist(pos(hero), pos(enemy_tower)) <= tower_range

    def calculate_tower_chase_risk(self, frame_data, main_hero, enemy_hero, enemy_tower):
        if not main_hero or not enemy_hero or not enemy_tower:
            return 0.0
        if not self.in_enemy_tower_range(main_hero, enemy_tower):
            return 0.0
        enemy_low = hp_ratio(enemy_hero) <= 0.18
        main_safe = hp_ratio(main_hero) >= 0.55 and get_any(enemy_tower, "attack_target", default=0) != actor_id(main_hero)
        if enemy_low and main_safe:
            return -0.2
        return 1.0

    def calculate_attack_power(self, camp, main_hero):
        phy_atk = float(get_any(main_hero, "phy_atk", default=0) or 0)
        last = self.last_phy_atk_by_camp.get(camp)
        self.last_phy_atk_by_camp[camp] = phy_atk
        if last is None:
            return 0.0
        return max(phy_atk - last, 0.0) / 100.0

    def calculate_attack_speed(self, camp, main_hero):
        atk_spd = float(get_any(main_hero, "atk_spd", default=0) or 0)
        last = self.last_atk_spd_by_camp.get(camp)
        self.last_atk_spd_by_camp[camp] = atk_spd
        if last is None:
            return 0.0
        return max(atk_spd - last, 0.0) / 1000.0

    def calculate_tower_attack(self, main_hero, enemy_tower):
        if not main_hero or not enemy_tower:
            return 0.0
        enemy_tower_id = actor_id(enemy_tower)
        if get_any(main_hero, "attack_target", default=0) == enemy_tower_id:
            return self.push_money_scale(main_hero)
        for hit in get_any(main_hero, "hit_target_info", default=[]) or []:
            if get_any(hit, "hit_target", "runtime_id", default=0) == enemy_tower_id:
                return self.push_money_scale(main_hero)
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

    def calculate_safe_tower_damage(self, camp, frame_data, main_hero, enemy_hero, enemy_tower):
        if not main_hero or not enemy_tower:
            return 0.0
        enemy_tower_hp = float(get_any(enemy_tower, "hp", default=0) or 0)
        last_hp = self.last_enemy_tower_hp_by_camp.get(camp)
        self.last_enemy_tower_hp_by_camp[camp] = enemy_tower_hp
        if last_hp is None:
            return 0.0

        damage = max(last_hp - enemy_tower_hp, 0.0)
        if damage <= 0:
            return 0.0

        enemy_tower_id = actor_id(enemy_tower)
        hero_hit_tower = get_any(main_hero, "attack_target", default=0) == enemy_tower_id
        hero_hit_tower = hero_hit_tower or any(
            get_any(hit, "hit_target", "runtime_id", default=0) == enemy_tower_id
            for hit in get_any(main_hero, "hit_target_info", default=[]) or []
        )
        if not hero_hit_tower:
            return 0.0
        if self.calculate_safe_push(frame_data, main_hero, enemy_hero, enemy_tower) <= 0:
            return 0.0
        return min(damage / 100.0, 5.0) * self.push_money_scale(main_hero)

    def push_money_scale(self, main_hero):
        if hp_ratio(main_hero) <= getattr(GameConfig, "CRITICAL_HOME_HP_THRESHOLD", 0.20):
            return 0.0
        return 1.0

    def calculate_finish_push(self, camp, frame_data, main_hero, enemy_hero, enemy_tower):
        if not main_hero or not enemy_tower:
            return 0.0
        if hp_ratio(main_hero) <= getattr(GameConfig, "CRITICAL_HOME_HP_THRESHOLD", 0.20):
            return 0.0
        enemy_dead = enemy_hero and (
            hp_ratio(enemy_hero) <= 0 or float(get_any(enemy_hero, "revive_time", default=0) or 0) > 0
        )
        if not enemy_dead:
            return 0.0
        if not self.wave_can_support_push(frame_data, main_hero, enemy_hero, enemy_tower):
            return 0.0

        enemy_tower_hp = float(get_any(enemy_tower, "hp", default=0) or 0)
        last_hp = self.last_finish_push_tower_hp_by_camp.get(camp)
        self.last_finish_push_tower_hp_by_camp[camp] = enemy_tower_hp
        if last_hp is None:
            return 0.0
        damage = max(last_hp - enemy_tower_hp, 0.0)
        if damage <= 0:
            return 0.0
        tower_dead_bonus = 8.0 if enemy_tower_hp <= 0 else 0.0
        return min(damage / 80.0, 8.0) + tower_dead_bonus

    def calculate_tower_hp_gap(self, main_tower, enemy_tower):
        if not main_tower or not enemy_tower:
            return 0.0
        our_hp = hp_ratio(main_tower)
        enemy_hp = hp_ratio(enemy_tower)
        return max(min(our_hp - enemy_hp, 1.0), -1.0)

    def calculate_defend_tower_clear(self, frame_data, main_hero, main_tower):
        if not main_hero or not main_tower:
            return 0.0
        main_camp = camp_id(main_hero.get("camp"))
        tower_pos = pos(main_tower)
        enemy_soldiers = [
            npc
            for npc in frame_data.get("npc_states", []) or []
            if is_soldier_actor(npc)
            and camp_id(npc.get("camp")) != main_camp
            and float(get_any(npc, "hp", default=0) or 0) > 0
            and math.dist(pos(npc), tower_pos) <= 15000.0
        ]
        if not enemy_soldiers:
            return 0.0
        enemy_ids = {actor_id(npc) for npc in enemy_soldiers}
        attacking = get_any(main_hero, "attack_target", default=0) in enemy_ids
        hitting = any(
            get_any(hit, "hit_target", "runtime_id", default=0) in enemy_ids
            for hit in get_any(main_hero, "hit_target_info", default=[]) or []
        )
        if attacking or hitting:
            return 1.0 + min(len(enemy_soldiers), 4) * 0.15
        return -0.5 if hp_ratio(main_tower) <= 0.65 else -0.2

    def calculate_post_fight_recall(self, frame_data, main_hero, enemy_hero, main_tower):
        if not main_hero or not main_tower:
            return 0.0
        enemy_dead_or_gone = not enemy_hero or hp_ratio(enemy_hero) <= 0 or float(get_any(enemy_hero, "revive_time", default=0) or 0) > 0
        if not enemy_dead_or_gone:
            return 0.0
        hp_rate = hp_ratio(main_hero)
        level = int(get_any(main_hero, "level", default=1) or 1)
        if level >= getattr(GameConfig, "POST_FIGHT_PUSH_LEVEL", 9):
            should_return = hp_rate < getattr(GameConfig, "POST_FIGHT_RECALL_HP_THRESHOLD_LATE", 0.30)
        else:
            should_return = hp_rate < getattr(GameConfig, "POST_FIGHT_RECALL_HP_THRESHOLD_EARLY", 0.50)
        if not should_return:
            return 0.0
        dist_to_our_tower = math.dist(pos(main_hero), pos(main_tower))
        return 1.0 if dist_to_our_tower <= getattr(GameConfig, "RECALL_NEAR_TOWER_RADIUS", 9500.0) else 0.0

    def calculate_recall_use(self, camp, frame_data, main_hero, enemy_hero, main_tower):
        if not main_hero or not main_tower:
            return 0.0
        used = self.slot_used_times(main_hero, 7)
        last = self.last_recall_used_by_camp.get(camp)
        self.last_recall_used_by_camp[camp] = used
        if last is None or used <= last:
            return 0.0
        hp_rate = hp_ratio(main_hero)
        near_tower = math.dist(pos(main_hero), pos(main_tower)) <= getattr(GameConfig, "RECALL_NEAR_TOWER_RADIUS", 9500.0)
        enemy_threat = enemy_hero and hp_ratio(enemy_hero) > 0 and math.dist(pos(main_hero), pos(enemy_hero)) <= 12000.0
        enemy_wave_near = self.enemy_soldier_near(frame_data, camp, main_tower, 12000.0)
        if near_tower and hp_rate <= 0.50 and not enemy_threat and not enemy_wave_near:
            return 1.0
        if hp_rate <= 0.35 and not enemy_threat:
            return 0.2
        return -0.15

    def calculate_rune_approach(self, frame_data, main_hero, enemy_hero=None, enemy_tower=None):
        if not main_hero:
            return 0.0
        nearest_dist = self.nearest_allowed_cake_distance(frame_data, main_hero, enemy_hero, enemy_tower)
        if nearest_dist is None or nearest_dist > getattr(GameConfig, "RUNE_SEARCH_RADIUS", 25000.0):
            return 0.0
        return 1.0 - nearest_dist / max(getattr(GameConfig, "RUNE_SEARCH_RADIUS", 25000.0), 1.0)

    def calculate_rune_pickup(self, camp, frame_data, main_hero, enemy_hero=None, enemy_tower=None):
        if not main_hero:
            return 0.0
        hp_rate = hp_ratio(main_hero)
        allowed_cakes = self.allowed_cakes(frame_data, main_hero, enemy_hero, enemy_tower)
        nearest_dist = self.nearest_allowed_cake_distance(frame_data, main_hero, enemy_hero, enemy_tower)
        nearest_dist = nearest_dist if nearest_dist is not None else 999999.0
        last = self.last_rune_state_by_camp.get(camp)
        self.last_rune_state_by_camp[camp] = (hp_rate, len(allowed_cakes), nearest_dist)
        if last is None:
            return 0.0
        last_hp, last_count, last_dist = last
        near_last_rune = last_dist <= getattr(GameConfig, "RUNE_PICKUP_RADIUS", 6000.0)
        hp_recovered = hp_rate - last_hp >= 0.03
        cake_consumed = len(allowed_cakes) < last_count
        return 1.0 if near_last_rune and (hp_recovered or cake_consumed) else 0.0

    def calculate_enemy_rune_after_kill(self, frame_data, main_hero, enemy_hero=None, enemy_tower=None):
        if not main_hero or not enemy_hero:
            return 0.0
        enemy_dead = hp_ratio(enemy_hero) <= 0 or float(get_any(enemy_hero, "revive_time", default=0) or 0) > 0
        if not enemy_dead:
            return 0.0
        cakes = self.enemy_side_cakes(frame_data, main_hero)
        if not cakes:
            return 0.0
        nearest_dist = min(math.dist(pos(main_hero), pos(cake)) for cake in cakes)
        radius = getattr(GameConfig, "ENEMY_RUNE_AFTER_KILL_RADIUS", 36000.0)
        if nearest_dist > radius:
            return 0.0
        if enemy_tower and self.calculate_safe_push(frame_data, main_hero, enemy_hero, enemy_tower) > 0:
            return 1.0 - nearest_dist / max(radius, 1.0)
        return 0.35 * (1.0 - nearest_dist / max(radius, 1.0))

    def nearest_cake_distance(self, frame_data, main_hero):
        cakes = frame_data.get("cakes", []) or []
        if not cakes:
            return None
        return min(math.dist(pos(main_hero), pos(cake)) for cake in cakes)

    def nearest_allowed_cake_distance(self, frame_data, main_hero, enemy_hero=None, enemy_tower=None):
        cakes = self.allowed_cakes(frame_data, main_hero, enemy_hero, enemy_tower)
        if not cakes:
            return None
        return min(math.dist(pos(main_hero), pos(cake)) for cake in cakes)

    def allowed_cakes(self, frame_data, main_hero, enemy_hero=None, enemy_tower=None):
        cakes = frame_data.get("cakes", []) or []
        if not cakes or not main_hero:
            return []
        main_camp = camp_id(main_hero.get("camp"))
        hp_rate = hp_ratio(main_hero)
        enemy_dead = enemy_hero and (hp_ratio(enemy_hero) <= 0 or float(get_any(enemy_hero, "revive_time", default=0) or 0) > 0)
        safe_enemy_cake = enemy_dead and enemy_tower and self.calculate_safe_push(frame_data, main_hero, enemy_hero, enemy_tower) > 0
        enemy_threat = enemy_hero and hp_ratio(enemy_hero) > 0 and math.dist(pos(main_hero), pos(enemy_hero)) <= 13000.0
        urgent_enemy_wave = self.enemy_soldier_near(frame_data, main_camp, main_hero, 7000.0)
        self_cake_ok = hp_rate < getattr(GameConfig, "RUNE_LOW_HP_THRESHOLD", 0.80) and not enemy_threat and not urgent_enemy_wave

        allowed = []
        for cake in cakes:
            cake_side = self.cake_side(cake)
            if cake_side == main_camp and self_cake_ok:
                allowed.append(cake)
            elif cake_side not in (0, main_camp) and safe_enemy_cake:
                allowed.append(cake)
        return allowed

    def enemy_side_cakes(self, frame_data, main_hero):
        cakes = frame_data.get("cakes", []) or []
        if not cakes or not main_hero:
            return []
        main_camp = camp_id(main_hero.get("camp"))
        return [cake for cake in cakes if self.cake_side(cake) not in (0, main_camp)]

    def cake_side(self, cake):
        x, z = pos(cake)
        if x == 0 and z == 0:
            return 0
        blue_main = CAKE_LOCATIONS_BY_CAMP[1]["main"]
        red_main = CAKE_LOCATIONS_BY_CAMP[2]["main"]
        return 1 if math.dist((x, z), blue_main) <= math.dist((x, z), red_main) else 2

    def map_objectives_clear(self, frame_data, main_hero):
        main_camp = camp_id(main_hero.get("camp"))
        hero_pos = pos(main_hero)
        for npc in frame_data.get("npc_states", []) or []:
            if float(get_any(npc, "hp", default=0) or 0) <= 0:
                continue
            if is_soldier_actor(npc) and camp_id(npc.get("camp")) != main_camp:
                return False
            if is_monster_actor(npc) and math.dist(hero_pos, pos(npc)) <= 22000.0:
                return False
        return True

    def slot_used_times(self, hero, slot_type):
        for slot in (get_any(hero, "skill_state", default={}) or {}).get("slot_states", []) or []:
            if int(get_any(slot, "slot_type", default=-1) or -1) == slot_type:
                return int(get_any(slot, "usedTimes", "used_times", default=0) or 0)
        return 0

    def enemy_soldier_near(self, frame_data, camp, target, radius):
        for npc in frame_data.get("npc_states", []) or []:
            if (
                is_soldier_actor(npc)
                and camp_id(npc.get("camp")) != camp
                and float(get_any(npc, "hp", default=0) or 0) > 0
                and math.dist(pos(npc), pos(target)) <= radius
            ):
                return True
        return False

    def wave_can_support_push(self, frame_data, main_hero, enemy_hero, enemy_tower):
        main_camp = camp_id(main_hero.get("camp"))
        our_soldiers = [
            npc
            for npc in frame_data.get("npc_states", []) or []
            if is_soldier_actor(npc)
            and camp_id(npc.get("camp")) == main_camp
            and float(get_any(npc, "hp", default=0) or 0) > 0
        ]
        if len(our_soldiers) < 2:
            return False
        tower_target = get_any(enemy_tower, "attack_target", default=0)
        tower_tanking = tower_target in {actor_id(npc) for npc in our_soldiers} and tower_target != 0
        near_tower = [npc for npc in our_soldiers if math.dist(pos(npc), pos(enemy_tower)) <= 12000.0]
        if not tower_tanking and len(near_tower) < 2:
            return False
        revive_time = float(get_any(enemy_hero or {}, "revive_time", default=0) or 0)
        return revive_time <= 0 or revive_time >= 60 or len(near_tower) >= 3

    def calculate_grass_ambush(self, frame_data, main_hero, enemy_hero, enemy_tower):
        if not main_hero or not bool(get_any(main_hero, "is_in_grass", default=False)):
            return 0.0
        frame_no = int(get_any(frame_data, "frame_no", "frameNo", default=0) or 0)
        if frame_no > getattr(GameConfig, "GRASS_AMBUSH_FRAMES", 1800):
            return 0.0
        if hp_ratio(main_hero) < getattr(GameConfig, "GRASS_AMBUSH_MIN_HP", 0.45):
            return 0.0

        main_camp = camp_id(main_hero.get("camp"))
        # Do not reward hiding in grass when there is immediate farm or a safe tower window.
        for npc in frame_data.get("npc_states", []) or []:
            if float(get_any(npc, "hp", default=0) or 0) <= 0:
                continue
            if is_soldier_actor(npc) and camp_id(npc.get("camp")) != main_camp and math.dist(pos(main_hero), pos(npc)) <= 9000.0:
                return 0.0
            if is_monster_actor(npc) and math.dist(pos(main_hero), pos(npc)) <= 9000.0:
                return 0.0
        if enemy_tower and self.calculate_safe_push(frame_data, main_hero, enemy_hero, enemy_tower) > 0:
            return 0.0
        if enemy_hero and hp_ratio(enemy_hero) > 0 and not self.in_enemy_tower_range(main_hero, enemy_tower):
            hp_gap = abs(hp_ratio(main_hero) - hp_ratio(enemy_hero))
            if hp_gap > getattr(GameConfig, "GRASS_AMBUSH_MAX_HP_GAP", 0.18):
                return 0.0
            if math.dist(pos(main_hero), pos(enemy_hero)) <= float(get_any(main_hero, "attack_range", default=0) or 0) + 9000.0:
                return 1.0
        return 0.0

    def calculate_grass_engage(self, frame_data, main_hero, enemy_hero, enemy_tower):
        if not main_hero or not enemy_hero or hp_ratio(enemy_hero) <= 0:
            return 0.0
        if not bool(get_any(main_hero, "is_in_grass", default=False)):
            return 0.0
        if self.in_enemy_tower_range(main_hero, enemy_tower):
            return 0.0
        enemy_id = actor_id(enemy_hero)
        dist = math.dist(pos(main_hero), pos(enemy_hero))
        in_range = dist <= float(get_any(main_hero, "attack_range", default=0) or 0) + 3500.0
        hit_enemy = any(
            get_any(hit, "hit_target", "runtime_id", default=0) == enemy_id
            for hit in get_any(main_hero, "hit_target_info", default=[]) or []
        )
        targeting_enemy = get_any(main_hero, "attack_target", default=0) == enemy_id
        return 1.0 if in_range and (hit_enemy or targeting_enemy) else 0.0

    def calculate_stutter_step(self, camp, frame_data, main_hero):
        if not main_hero:
            return 0.0
        current_pos = pos(main_hero)
        last_pos = self.last_stutter_pos_by_camp.get(camp)
        self.last_stutter_pos_by_camp[camp] = current_pos
        if last_pos is None:
            return 0.0

        hit_infos = get_any(main_hero, "hit_target_info", default=[]) or []
        moved = math.dist(current_pos, last_pos)
        if hit_infos and 250.0 <= moved <= 4500.0:
            return 1.0
        if hit_infos and moved < 120.0:
            return -0.5
        return 0.0

    def calculate_retaliate_hit(self, camp, frame_data, main_hero, enemy_hero):
        if not main_hero or not enemy_hero or hp_ratio(enemy_hero) <= 0:
            return 0.0
        hero_id = actor_id(main_hero)
        enemy_id = actor_id(enemy_hero)
        frame_no = int(get_any(frame_data, "frame_no", "frameNo", default=0) or 0)
        be_hurt = float(get_any(main_hero, "total_be_hurt_by_hero", default=0) or 0)
        last = self.last_be_hurt_by_hero.get(camp)
        if last is None:
            self.last_be_hurt_by_hero[camp] = (be_hurt, frame_no)
            return 0.0
        last_hurt, last_frame = last
        recently_hurt = be_hurt > last_hurt or frame_no - last_frame <= 45
        self.last_be_hurt_by_hero[camp] = (be_hurt, frame_no if be_hurt > last_hurt else last_frame)
        if not recently_hurt:
            return 0.0
        hit_enemy = any(
            get_any(hit, "hit_target", "runtime_id", default=0) == enemy_id
            for hit in get_any(main_hero, "hit_target_info", default=[]) or []
        )
        enemy_target_us = get_any(enemy_hero, "attack_target", default=0) == hero_id
        return 1.0 if hit_enemy or get_any(main_hero, "attack_target", default=0) == enemy_id else (-0.25 if enemy_target_us else 0.0)

    def calculate_fight_risk(self, frame_data, main_hero, enemy_hero, main_tower, enemy_tower):
        if not main_hero or not enemy_hero or hp_ratio(enemy_hero) <= 0:
            return 0.0
        hero_pos = pos(main_hero)
        enemy_pos = pos(enemy_hero)
        dist = math.dist(hero_pos, enemy_pos)
        hero_range = float(get_any(main_hero, "attack_range", default=0) or 0)
        enemy_range = float(get_any(enemy_hero, "attack_range", default=0) or 0)
        threatening = (
            dist <= max(hero_range, enemy_range) + 3500.0
            or get_any(main_hero, "attack_target", default=0) == actor_id(enemy_hero)
            or get_any(enemy_hero, "attack_target", default=0) == actor_id(main_hero)
            or any(get_any(hit, "hit_target", "runtime_id", default=0) == actor_id(enemy_hero) for hit in get_any(main_hero, "hit_target_info", default=[]) or [])
        )
        if not threatening:
            return 0.0
        hp_gap = hp_ratio(main_hero) - hp_ratio(enemy_hero)
        money_gap = (
            float(get_any(main_hero, "money_cnt", "money", default=0) or 0)
            - float(get_any(enemy_hero, "money_cnt", "money", default=0) or 0)
        )
        level_gap = int(get_any(main_hero, "level", default=1) or 1) - int(get_any(enemy_hero, "level", default=1) or 1)
        bad_hp = hp_gap <= -0.18 or (hp_ratio(main_hero) <= 0.35 and hp_ratio(enemy_hero) > 0.25)
        bad_econ = money_gap < -1200 or level_gap <= -2
        tower_bad = enemy_tower is not None and self.in_enemy_tower_range(main_hero, enemy_tower)
        own_tower_safe = main_tower is not None and math.dist(hero_pos, pos(main_tower)) <= 13000.0
        if (bad_hp or bad_econ or tower_bad) and not own_tower_safe:
            return 1.0
        if bad_hp or tower_bad:
            return 0.5
        return 0.0

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
            elif reward_name == "tower_hp_gap":
                cur = self.m_main_calc_frame_map[reward_name].cur_frame_value
                last = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = 0.0 if last == 0.0 else cur - last
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name in (
                "danger",
                "idle",
                "attack_hit",
                "skill_hit",
                "skill_misuse",
                "monster_last_hit",
                "hero_lane_advantage",
                "monster_contest",
                "tower_chase_risk",
                "attack_power",
                "attack_speed",
                "tower_attack",
                "safe_push",
                "safe_tower_damage",
                "finish_push",
                "defend_tower_clear",
                "post_fight_recall",
                "recall_use",
                "rune_approach",
                "rune_pickup",
                "enemy_rune_after_kill",
                "enemy_wave_overextend",
                "lane_anchor",
                "grass_ambush",
                "grass_engage",
                "stutter_step",
                "retaliate_hit",
                "fight_risk",
            ):
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
