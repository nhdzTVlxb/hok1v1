#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Per-episode behavior diagnostics built from raw 2026 frame_state."""

import math


UNSEEN_PADDING = 100000
TOWER_DANGER_MARGIN = 2000.0
BASE_RETURN_RADIUS = 14000.0


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


def actor_id(actor):
    return get_any(actor or {}, "runtime_id", "player_id", default=0)


def actor_pos(actor):
    loc = actor.get("location", {}) if isinstance(actor, dict) else {}
    if not loc and isinstance(actor, dict):
        loc = ((actor.get("collider", {}) or {}).get("location", {}) or {})
    if isinstance(loc, dict):
        return [float(loc.get("x", 0) or 0), float(loc.get("z", 0) or 0)]
    if isinstance(loc, (list, tuple)) and len(loc) >= 3:
        return [float(loc[0] or 0), float(loc[2] or 0)]
    return [0.0, 0.0]


def is_unseen(actor):
    x, z = actor_pos(actor)
    return abs(x) >= UNSEEN_PADDING or abs(z) >= UNSEEN_PADDING


def distance(a, b):
    return math.dist(actor_pos(a), actor_pos(b))


def hp_ratio(actor):
    return float(get_any(actor or {}, "hp", default=0) or 0) / max(float(get_any(actor or {}, "max_hp", default=1) or 1), 1.0)


def is_tower(actor):
    return get_any(actor, "actor_type", default=None) == 2 and get_any(actor, "sub_type", default=None) in (
        21,
        23,
        24,
        "ACTOR_SUB_TOWER",
    )


def is_soldier(actor):
    if is_neutral_camp(get_any(actor or {}, "camp", default=None)):
        return False
    return get_any(actor, "actor_type", default=None) == 1 and get_any(actor, "sub_type", default=None) in (
        1,
        11,
        "ACTOR_SUB_SOLDIER",
    )


def is_monster(actor):
    actor_type = get_any(actor or {}, "actor_type", default=None)
    sub_type = get_any(actor or {}, "sub_type", default=None)
    return (
        actor_type in (3, "ACTOR_TYPE_MONSTER")
        or sub_type in ("ACTOR_SUB_MONSTER", "ACTOR_SUB_NEUTRAL_MONSTER")
        or (is_neutral_camp(get_any(actor or {}, "camp", default=None)) and actor_type == 1 and sub_type == 0)
    )


def is_visible_to(actor, viewer_camp):
    if not actor or is_unseen(actor):
        return False
    visible = actor.get("camp_visible")
    idx = camp_id(viewer_camp) - 1
    if isinstance(visible, (list, tuple)) and 0 <= idx < len(visible):
        return bool(visible[idx])
    return True


def split_state(observation):
    frame_state = observation.get("frame_state", {})
    main_camp = camp_id(observation.get("camp", observation.get("player_camp", 1)))
    player_id = observation.get("player_id")
    heroes = frame_state.get("hero_states", []) or []
    npcs = frame_state.get("npc_states", []) or []

    hero, enemy_hero = None, None
    for item in heroes:
        if actor_id(item) == player_id:
            hero = item
        elif camp_id(item.get("camp")) == main_camp:
            hero = item
        else:
            enemy_hero = item

    our_tower, enemy_tower, our_base = None, None, None
    for npc in npcs:
        if not is_tower(npc):
            continue
        if camp_id(npc.get("camp")) == main_camp:
            if get_any(npc, "sub_type", default=None) == 23 and not is_unseen(npc):
                if our_base is None or (hero and distance(hero, npc) < distance(hero, our_base)):
                    our_base = npc
            if our_tower is None or (hero and distance(hero, npc) < distance(hero, our_tower)):
                our_tower = npc
        else:
            if enemy_tower is None or (hero and distance(hero, npc) < distance(hero, enemy_tower)):
                enemy_tower = npc

    our_soldiers = [
        npc
        for npc in npcs
        if is_soldier(npc) and camp_id(npc.get("camp")) == main_camp and float(get_any(npc, "hp", default=0) or 0) > 0
    ]
    return frame_state, main_camp, hero, enemy_hero, our_tower, enemy_tower, our_soldiers, our_base


def hits_target(hero, target_id):
    for hit in get_any(hero or {}, "hit_target_info", default=[]) or []:
        if get_any(hit, "hit_target", "runtime_id", default=0) == target_id:
            return True
    return False


def any_hit(hero):
    return bool(get_any(hero or {}, "hit_target_info", default=[]) or [])


def safe_push_state(hero, enemy_hero, enemy_tower, our_soldiers):
    if not hero or not enemy_tower:
        return False
    tower_target = get_any(enemy_tower, "attack_target", default=0)
    soldier_ids = {actor_id(s) for s in our_soldiers}
    tower_tanking_minion = tower_target in soldier_ids and tower_target != 0
    if not tower_tanking_minion:
        return False

    if enemy_hero and hp_ratio(enemy_hero) > 0 and get_any(enemy_hero, "revive_time", default=0) <= 0:
        if is_visible_to(enemy_hero, camp_id(hero.get("camp"))):
            enemy_range = float(get_any(enemy_hero, "attack_range", default=0) or 0) + 2000.0
            if distance(hero, enemy_hero) <= enemy_range:
                return False

    return distance(hero, enemy_tower) <= float(get_any(hero, "attack_range", default=0) or 0) + 1000.0


class CombatDiagnostics:
    def __init__(self, agent_num):
        self.agent_num = agent_num
        self.reset()

    def reset(self):
        self.stats = [self._empty_stats() for _ in range(self.agent_num)]

    def _empty_stats(self):
        return {
            "frames": 0,
            "tower_damage": 0.0,
            "hero_tower_damage": 0.0,
            "safe_push_frames": 0,
            "safe_push_attack_frames": 0,
            "tower_hit_frames": 0,
            "hit_frames": 0,
            "normal_attack_actions": 0,
            "attack_interval_sum": 0.0,
            "attack_interval_count": 0,
            "phy_atk_sum": 0.0,
            "atk_spd_sum": 0.0,
            "phy_atk_gain": 0.0,
            "atk_spd_gain": 0.0,
            "tower_target_legal_frames": 0,
            "tower_target_action_frames": 0,
            "tower_targeted_me_frames": 0,
            "skill1_actions": 0,
            "skill2_actions": 0,
            "skill3_actions": 0,
            "skill_hit_frames": 0,
            "skill_hero_hit_frames": 0,
            "monster_hit_frames": 0,
            "recall_actions": 0,
            "rune_pickups": 0,
            "monster_kills": 0,
            "base_returns": 0,
            "bad_fight_frames": 0,
            "has_left_base": False,
            "was_in_base": False,
            "last_dead_cnt": None,
            "last_enemy_tower_hp": None,
            "last_attack_frame": None,
            "last_phy_atk": None,
            "last_atk_spd": None,
            "last_cake_count": None,
            "last_nearest_cake_dist": None,
        }

    def observe(self, agent_idx, observation, action):
        if agent_idx >= self.agent_num:
            return
        stat = self.stats[agent_idx]
        frame_state, _, hero, enemy_hero, _, enemy_tower, our_soldiers, our_base = split_state(observation)
        if not hero:
            return

        stat["frames"] += 1
        frame_no = float(get_any(frame_state, "frame_no", default=0) or 0)
        phy_atk = float(get_any(hero, "phy_atk", default=0) or 0)
        atk_spd = float(get_any(hero, "atk_spd", default=0) or 0)
        stat["phy_atk_sum"] += phy_atk
        stat["atk_spd_sum"] += atk_spd
        if stat["last_phy_atk"] is not None:
            stat["phy_atk_gain"] += max(phy_atk - stat["last_phy_atk"], 0.0)
        if stat["last_atk_spd"] is not None:
            stat["atk_spd_gain"] += max(atk_spd - stat["last_atk_spd"], 0.0)
        stat["last_phy_atk"] = phy_atk
        stat["last_atk_spd"] = atk_spd

        dead_cnt = int(get_any(hero, "dead_cnt", "deadCnt", default=0) or 0)
        revive_time = float(get_any(hero, "revive_time", default=0) or 0)
        alive = hp_ratio(hero) > 0 and revive_time <= 0
        death_changed = stat["last_dead_cnt"] is not None and dead_cnt > stat["last_dead_cnt"]
        if our_base is not None:
            in_base = distance(hero, our_base) <= BASE_RETURN_RADIUS
            if alive and not in_base:
                stat["has_left_base"] = True
            if alive and stat["has_left_base"] and in_base and not stat["was_in_base"] and not death_changed:
                stat["base_returns"] += 1
            stat["was_in_base"] = in_base
        stat["last_dead_cnt"] = dead_cnt

        if action and action[0] == 3:
            stat["normal_attack_actions"] += 1
            if len(action) > 5 and action[5] == 7:
                stat["tower_target_action_frames"] += 1
        if action and action[0] in (4, 5, 6):
            if action[0] == 4:
                stat["skill1_actions"] += 1
            elif action[0] == 5:
                stat["skill2_actions"] += 1
            elif action[0] == 6:
                stat["skill3_actions"] += 1
        if action and action[0] == 9:
            stat["recall_actions"] += 1

        cakes = frame_state.get("cakes", []) or []
        nearest_cake_dist = min((distance(hero, cake) for cake in cakes), default=None)
        if stat["last_cake_count"] is not None and len(cakes) < stat["last_cake_count"]:
            if stat["last_nearest_cake_dist"] is not None and stat["last_nearest_cake_dist"] <= 6000.0:
                stat["rune_pickups"] += 1
        stat["last_cake_count"] = len(cakes)
        stat["last_nearest_cake_dist"] = nearest_cake_dist

        frame_action = frame_state.get("frame_action", {}) or {}
        hero_id = actor_id(hero)
        for dead_action in frame_action.get("dead_action", []) or []:
            death = dead_action.get("death", {}) or {}
            killer = dead_action.get("killer", {}) or {}
            if is_monster(death) and actor_id(killer) == hero_id:
                stat["monster_kills"] += 1

        if enemy_tower:
            tower_id = actor_id(enemy_tower)
            tower_hp = float(get_any(enemy_tower, "hp", default=0) or 0)
            if stat["last_enemy_tower_hp"] is not None:
                tower_damage = max(stat["last_enemy_tower_hp"] - tower_hp, 0.0)
                stat["tower_damage"] += tower_damage
                if tower_damage > 0 and (
                    hits_target(hero, tower_id) or get_any(hero, "attack_target", default=0) == tower_id
                ):
                    stat["hero_tower_damage"] += tower_damage
            stat["last_enemy_tower_hp"] = tower_hp

            if hits_target(hero, tower_id) or get_any(hero, "attack_target", default=0) == tower_id:
                stat["tower_hit_frames"] += 1
            if get_any(enemy_tower, "attack_target", default=0) == actor_id(hero):
                stat["tower_targeted_me_frames"] += 1

        if enemy_hero and hp_ratio(enemy_hero) > 0 and is_visible_to(enemy_hero, camp_id(hero.get("camp"))):
            hero_id = actor_id(hero)
            enemy_id = actor_id(enemy_hero)
            hero_hp = hp_ratio(hero)
            enemy_hp = hp_ratio(enemy_hero)
            dist = distance(hero, enemy_hero)
            threat_range = max(
                float(get_any(hero, "attack_range", default=0) or 0),
                float(get_any(enemy_hero, "attack_range", default=0) or 0),
            ) + 4500.0
            fighting = (
                dist <= threat_range
                or get_any(hero, "attack_target", default=0) == enemy_id
                or get_any(enemy_hero, "attack_target", default=0) == hero_id
                or hits_target(enemy_hero, hero_id)
            )
            if fighting and (hero_hp + 0.18 <= enemy_hp or (hero_hp <= 0.35 and enemy_hp > 0.25)):
                stat["bad_fight_frames"] += 1

        safe_push = safe_push_state(hero, enemy_hero, enemy_tower, our_soldiers)
        if safe_push:
            stat["safe_push_frames"] += 1
            if action and action[0] == 3 and len(action) > 5 and action[5] == 7:
                stat["safe_push_attack_frames"] += 1

        if any_hit(hero):
            stat["hit_frames"] += 1
            actor_by_id = {}
            for item in (frame_state.get("hero_states", []) or []) + (frame_state.get("npc_states", []) or []):
                actor_by_id[actor_id(item)] = item
            skill_hit = False
            skill_hero_hit = False
            monster_hit = False
            for hit in get_any(hero or {}, "hit_target_info", default=[]) or []:
                slot_type = int(get_any(hit, "slot_type", default=-1) or -1)
                target = actor_by_id.get(get_any(hit, "hit_target", "runtime_id", default=0))
                if slot_type in (1, 2, 3):
                    skill_hit = True
                    if get_any(target or {}, "actor_type", default=None) == 0:
                        skill_hero_hit = True
                if is_monster(target):
                    monster_hit = True
            stat["skill_hit_frames"] += int(skill_hit)
            stat["skill_hero_hit_frames"] += int(skill_hero_hit)
            stat["monster_hit_frames"] += int(monster_hit)
            if stat["last_attack_frame"] is not None and frame_no > stat["last_attack_frame"]:
                stat["attack_interval_sum"] += frame_no - stat["last_attack_frame"]
                stat["attack_interval_count"] += 1
            stat["last_attack_frame"] = frame_no

        legal = observation.get("legal_action", []) or []
        if len(legal) == 184:
            target_start = 12 + 16 * 4
            tower_target_idx = target_start + 3 * 9 + 7
            if 0 <= tower_target_idx < len(legal) and legal[tower_target_idx] > 0:
                stat["tower_target_legal_frames"] += 1

    def episode_metrics(self, agent_idx):
        stat = self.stats[agent_idx]
        frames = max(stat["frames"], 1)
        attack_intervals = max(stat["attack_interval_count"], 1)
        return {
            "tower_damage": stat["tower_damage"],
            "hero_tower_damage": stat["hero_tower_damage"],
            "safe_push_frames": stat["safe_push_frames"],
            "safe_push_attack_frames": stat["safe_push_attack_frames"],
            "tower_hit_frames": stat["tower_hit_frames"],
            "hit_frames": stat["hit_frames"],
            "normal_attack_per_1000": stat["normal_attack_actions"] * 1000.0 / frames,
            "hit_per_1000": stat["hit_frames"] * 1000.0 / frames,
            "avg_attack_interval": stat["attack_interval_sum"] / attack_intervals,
            "avg_phy_atk": stat["phy_atk_sum"] / frames,
            "avg_atk_spd": stat["atk_spd_sum"] / frames,
            "phy_atk_gain": stat["phy_atk_gain"],
            "atk_spd_gain": stat["atk_spd_gain"],
            "tower_target_legal_frames": stat["tower_target_legal_frames"],
            "tower_target_action_frames": stat["tower_target_action_frames"],
            "tower_targeted_me_frames": stat["tower_targeted_me_frames"],
            "skill1_actions": stat["skill1_actions"],
            "skill2_actions": stat["skill2_actions"],
            "skill3_actions": stat["skill3_actions"],
            "skill_hit_frames": stat["skill_hit_frames"],
            "skill_hero_hit_frames": stat["skill_hero_hit_frames"],
            "monster_hit_frames": stat["monster_hit_frames"],
            "recall_actions": stat["recall_actions"],
            "rune_pickups": stat["rune_pickups"],
            "monster_kills": stat["monster_kills"],
            "base_returns": stat["base_returns"],
            "bad_fight_frames": stat["bad_fight_frames"],
        }
