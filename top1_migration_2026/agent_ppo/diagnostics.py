#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Per-episode behavior diagnostics built from raw 2026 frame_state."""

import math


UNSEEN_PADDING = 100000
TOWER_DANGER_MARGIN = 2000.0


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


def actor_id(actor):
    return get_any(actor or {}, "runtime_id", "player_id", default=0)


def actor_pos(actor):
    loc = actor.get("location", {}) if isinstance(actor, dict) else {}
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
    return get_any(actor, "actor_type", default=None) == 1 or get_any(actor, "sub_type", default=None) in (
        1,
        11,
        "ACTOR_SUB_SOLDIER",
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

    our_tower, enemy_tower = None, None
    for npc in npcs:
        if not is_tower(npc):
            continue
        if camp_id(npc.get("camp")) == main_camp:
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
    return frame_state, main_camp, hero, enemy_hero, our_tower, enemy_tower, our_soldiers


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
            "last_enemy_tower_hp": None,
            "last_attack_frame": None,
            "last_phy_atk": None,
            "last_atk_spd": None,
        }

    def observe(self, agent_idx, observation, action):
        if agent_idx >= self.agent_num:
            return
        stat = self.stats[agent_idx]
        frame_state, _, hero, enemy_hero, _, enemy_tower, our_soldiers = split_state(observation)
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

        if action and action[0] == 3:
            stat["normal_attack_actions"] += 1
            if len(action) > 5 and action[5] == 7:
                stat["tower_target_action_frames"] += 1

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

        safe_push = safe_push_state(hero, enemy_hero, enemy_tower, our_soldiers)
        if safe_push:
            stat["safe_push_frames"] += 1
            if action and action[0] == 3 and len(action) > 5 and action[5] == 7:
                stat["safe_push_attack_frames"] += 1

        if any_hit(hero):
            stat["hit_frames"] += 1
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
        }
