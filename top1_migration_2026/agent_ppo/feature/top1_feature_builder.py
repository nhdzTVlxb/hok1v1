#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Small fixed-length observation builder for the 2026 flat frame_state protocol.

The first production feature version intentionally stays compact and readable:
raw obs dict -> 128 float32 values.  This is easier to debug than the old
2025-style large entity vector while we are still validating behavior.
"""

import math
import numpy as np

from agent_ppo.conf.conf import DimConfig


MAP_SCALE = 100000.0
UNSEEN_PADDING = 100000
NEARBY_RANGE = 20000.0
TOWER_DANGER_MARGIN = 2000.0


def clip(value, low=-1.0, high=1.0):
    return max(low, min(high, float(value)))


def ratio(value, denom):
    denom = float(denom or 0)
    if denom <= 0:
        return 0.0
    return clip(float(value or 0) / denom, 0.0, 1.0)


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


def get_any(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict) and key in d:
            return d[key]
    return default


def runtime_id(actor):
    return get_any(actor or {}, "runtime_id", "player_id", default=0)


def actor_pos(actor, mirror=False):
    loc = actor.get("location", {}) if isinstance(actor, dict) else {}
    if isinstance(loc, dict):
        x, z = float(loc.get("x", 0) or 0), float(loc.get("z", 0) or 0)
    if isinstance(loc, (list, tuple)) and len(loc) >= 3:
        x, z = float(loc[0] or 0), float(loc[2] or 0)
    elif not isinstance(loc, dict):
        x, z = 0.0, 0.0
    if mirror and abs(x) < UNSEEN_PADDING and abs(z) < UNSEEN_PADDING:
        x, z = -x, -z
    return [x, z]


def norm_pos(actor, mirror=False):
    x, z = actor_pos(actor, mirror)
    if abs(x) >= UNSEEN_PADDING or abs(z) >= UNSEEN_PADDING:
        return [0.0, 0.0]
    return [clip(x / MAP_SCALE), clip(z / MAP_SCALE)]


def rel_pos(src, dst, mirror=False):
    sx, sz = actor_pos(src, mirror)
    dx, dz = actor_pos(dst, mirror)
    if abs(dx) >= UNSEEN_PADDING or abs(dz) >= UNSEEN_PADDING:
        return [0.0, 0.0, 0.0]
    rx, rz = dx - sx, dz - sz
    return [clip(rx / MAP_SCALE), clip(rz / MAP_SCALE), clip(math.hypot(rx, rz) / MAP_SCALE, 0.0, 1.0)]


def is_visible_to(actor, viewer_camp):
    if not actor:
        return False
    x, z = actor_pos(actor)
    if abs(x) >= UNSEEN_PADDING or abs(z) >= UNSEEN_PADDING:
        return False
    visible = actor.get("camp_visible")
    idx = camp_id(viewer_camp) - 1
    if isinstance(visible, (list, tuple)) and 0 <= idx < len(visible):
        return bool(visible[idx])
    return True


def is_soldier(actor):
    actor_type = get_any(actor, "actor_type", default=None)
    sub_type = get_any(actor, "sub_type", default=None)
    return actor_type == 1 or sub_type in (1, 11, "ACTOR_SUB_SOLDIER")


def is_tower(actor):
    actor_type = get_any(actor, "actor_type", default=None)
    sub_type = get_any(actor, "sub_type", default=None)
    return actor_type == 2 and sub_type in (21, 23, 24, "ACTOR_SUB_TOWER")


def hp_ratio(actor):
    return ratio(get_any(actor or {}, "hp", default=0), get_any(actor or {}, "max_hp", default=1))


def distance(a, b):
    return math.dist(actor_pos(a), actor_pos(b))


class Top1FeatureBuilder:
    FEATURE_DIM = DimConfig.DIM_OF_FEATURE[0]

    def __init__(self, logger=None):
        self.logger = logger
        self.mirror = False

    def reset(self):
        pass

    def build_observation(self, observation):
        frame_state = observation.get("frame_state", {})
        main_camp = camp_id(observation.get("camp", observation.get("player_camp", 1)))
        self.mirror = main_camp == 2
        player_id = observation.get("player_id", 0)

        heroes = frame_state.get("hero_states", []) or []
        npcs = frame_state.get("npc_states", []) or []
        our_hero, enemy_hero = self._split_heroes(heroes, main_camp, player_id)
        our_tower, enemy_tower = self._split_towers(npcs, main_camp, our_hero)
        our_soldiers, enemy_soldiers = self._split_soldiers(npcs, main_camp)

        feature = []
        feature += self._hero_features(our_hero, our_hero, main_camp, enemy_tower, always_visible=True)
        feature += self._enemy_hero_features(our_hero, enemy_hero, main_camp)
        feature += self._skill_features(our_hero)
        feature += self._tower_features(our_hero, our_tower, enemy_tower)
        feature += self._soldier_features(our_hero, our_soldiers, enemy_soldiers, main_camp)
        feature += self._global_features(frame_state, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers)
        feature += self._legal_features(observation)

        if len(feature) < self.FEATURE_DIM:
            feature += [0.0] * (self.FEATURE_DIM - len(feature))
        elif len(feature) > self.FEATURE_DIM:
            feature = feature[: self.FEATURE_DIM]
        return np.array(feature, dtype=np.float32)

    def _split_heroes(self, heroes, main_camp, player_id):
        our_hero, enemy_hero = None, None
        for hero in heroes:
            if runtime_id(hero) == player_id:
                our_hero = hero
                continue
            if camp_id(hero.get("camp")) == main_camp:
                our_hero = hero
            else:
                enemy_hero = hero
        return our_hero, enemy_hero

    def _split_towers(self, npcs, main_camp, our_hero):
        towers = [npc for npc in npcs if is_tower(npc)]
        our = [t for t in towers if camp_id(t.get("camp")) == main_camp]
        enemy = [t for t in towers if camp_id(t.get("camp")) != main_camp]
        if our_hero:
            our.sort(key=lambda t: distance(our_hero, t))
            enemy.sort(key=lambda t: distance(our_hero, t))
        return (our[0] if our else None), (enemy[0] if enemy else None)

    def _split_soldiers(self, npcs, main_camp):
        soldiers = [npc for npc in npcs if is_soldier(npc) and hp_ratio(npc) > 0]
        our = [s for s in soldiers if camp_id(s.get("camp")) == main_camp]
        enemy = [s for s in soldiers if camp_id(s.get("camp")) != main_camp]
        return our, enemy

    def _hero_features(self, hero, our_hero, main_camp, enemy_tower, always_visible=False):
        if not hero:
            return [0.0] * 24
        visible = always_visible or is_visible_to(hero, main_camp)
        x, z = norm_pos(hero, self.mirror) if visible else [0.0, 0.0]
        forward = hero.get("forward", {}) or {}
        fx = clip(float(forward.get("x", 0) or 0) / 1000.0)
        fz = clip(float(forward.get("z", 0) or 0) / 1000.0)
        tower_danger = self._in_tower_range(hero, enemy_tower) if enemy_tower else 0.0
        attacked_by_tower = float(enemy_tower is not None and runtime_id(hero) == get_any(enemy_tower, "attack_target", default=-1))
        rel = [0.0, 0.0, 0.0] if our_hero is hero else rel_pos(our_hero, hero, self.mirror)
        return [
            float(visible),
            hp_ratio(hero),
            ratio(get_any(hero, "ep", default=0), get_any(hero, "max_ep", default=1)),
            ratio(get_any(hero, "level", default=1), 15),
            ratio(get_any(hero, "money", default=0), 2000),
            ratio(get_any(hero, "money_cnt", default=0), 12000),
            ratio(get_any(hero, "kill_cnt", default=0), 10),
            ratio(get_any(hero, "dead_cnt", default=0), 10),
            ratio(get_any(hero, "assist_cnt", default=0), 10),
            x,
            z,
            fx,
            fz,
            float(bool(get_any(hero, "is_in_grass", default=False))),
            float(get_any(hero, "attack_target", default=0) not in (0, None)),
            ratio(get_any(hero, "attack_range", default=0), 15000),
            ratio(get_any(hero, "total_hurt_to_hero", default=0), 50000),
            ratio(get_any(hero, "total_be_hurt_by_hero", default=0), 50000),
            tower_danger,
            attacked_by_tower,
            rel[0],
            rel[1],
            rel[2],
            float(get_any(hero, "revive_time", default=0) > 0),
            ratio(get_any(hero, "phy_atk", default=0), 500),
        ][:24]

    def _enemy_hero_features(self, our_hero, enemy_hero, main_camp):
        if not enemy_hero:
            return [0.0] * 18
        visible = is_visible_to(enemy_hero, main_camp)
        rel = rel_pos(our_hero, enemy_hero, self.mirror) if our_hero and visible else [0.0, 0.0, 0.0]
        pos = norm_pos(enemy_hero, self.mirror) if visible else [0.0, 0.0]
        in_attack_range = 0.0
        if our_hero and visible:
            in_attack_range = float(distance(our_hero, enemy_hero) <= float(get_any(our_hero, "attack_range", default=0) or 0))
        return [
            float(visible),
            hp_ratio(enemy_hero) if visible else 0.0,
            ratio(get_any(enemy_hero, "ep", default=0), get_any(enemy_hero, "max_ep", default=1)) if visible else 0.0,
            ratio(get_any(enemy_hero, "level", default=1), 15) if visible else 0.0,
            ratio(get_any(enemy_hero, "money_cnt", default=0), 12000) if visible else 0.0,
            ratio(get_any(enemy_hero, "kill_cnt", default=0), 10) if visible else 0.0,
            ratio(get_any(enemy_hero, "dead_cnt", default=0), 10) if visible else 0.0,
            pos[0],
            pos[1],
            rel[0],
            rel[1],
            rel[2],
            in_attack_range,
            float(get_any(enemy_hero, "attack_target", default=0) == runtime_id(our_hero or {})) if visible else 0.0,
            ratio(get_any(enemy_hero, "attack_range", default=0), 15000) if visible else 0.0,
            ratio(get_any(enemy_hero, "total_hurt_to_hero", default=0), 50000) if visible else 0.0,
            ratio(get_any(enemy_hero, "total_be_hurt_by_hero", default=0), 50000) if visible else 0.0,
            float(bool(get_any(enemy_hero, "is_in_grass", default=False))) if visible else 0.0,
        ]

    def _skill_features(self, hero):
        slots = ((hero or {}).get("skill_state", {}) or {}).get("slot_states", []) or []
        slots = sorted(slots, key=lambda s: int(get_any(s, "slot_type", default=99) or 99))[:7]
        while len(slots) < 7:
            slots.append({})
        out = []
        for slot in slots:
            cd = float(get_any(slot, "cooldown", default=0) or 0)
            cd_max = max(float(get_any(slot, "cooldown_max", default=1) or 1), 1.0)
            out += [
                ratio(get_any(slot, "level", default=0), 6),
                float(bool(get_any(slot, "usable", default=False))),
                clip(cd / cd_max, 0.0, 1.0),
            ]
        return out[:21]

    def _tower_features(self, our_hero, our_tower, enemy_tower):
        out = []
        for tower in [our_tower, enemy_tower]:
            if tower and our_hero:
                rel = rel_pos(our_hero, tower, self.mirror)
                out += [
                    hp_ratio(tower),
                    rel[0],
                    rel[1],
                    rel[2],
                    self._in_tower_range(our_hero, tower),
                    float(get_any(tower, "attack_target", default=0) == runtime_id(our_hero)),
                    ratio(get_any(tower, "attack_range", default=0), 15000),
                    float(camp_id(tower.get("camp")) == camp_id(our_hero.get("camp"))),
                ]
            else:
                out += [0.0] * 8
        return out

    def _soldier_features(self, our_hero, our_soldiers, enemy_soldiers, main_camp):
        if not our_hero:
            return [0.0] * 24

        def pack_group(soldiers):
            visible = [s for s in soldiers if is_visible_to(s, main_camp)]
            nearby = [s for s in visible if distance(our_hero, s) <= NEARBY_RANGE]
            nearest = min(visible, key=lambda s: distance(our_hero, s), default=None)
            low_hp = [s for s in visible if hp_ratio(s) <= 0.35]
            rel = rel_pos(our_hero, nearest, self.mirror) if nearest else [0.0, 0.0, 0.0]
            return [
                ratio(len(visible), 8),
                ratio(len(nearby), 8),
                hp_ratio(nearest) if nearest else 0.0,
                rel[0],
                rel[1],
                rel[2],
                float(bool(low_hp)),
                ratio(min([hp_ratio(s) for s in low_hp], default=0.0), 1.0),
                ratio(sum(get_any(s, "kill_income", default=0) or 0 for s in visible), 500),
                ratio(sum(get_any(s, "hp", default=0) or 0 for s in visible), 10000),
                ratio(sum(get_any(s, "max_hp", default=0) or 0 for s in visible), 10000),
                float(nearest is not None and get_any(nearest, "attack_target", default=0) == runtime_id(our_hero)),
            ]

        return pack_group(our_soldiers) + pack_group(enemy_soldiers)

    def _global_features(self, frame_state, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers):
        hero_x, hero_z = norm_pos(our_hero or {}, self.mirror)
        center_dist = clip(math.hypot(hero_x, hero_z), 0.0, 1.0)
        enemy_visible = float(enemy_hero is not None and is_visible_to(enemy_hero, camp_id((our_hero or {}).get("camp", 1))))
        safe_push, tower_tanking_minion, enemy_threat = self._safe_push_state(our_hero, enemy_hero, enemy_tower, our_soldiers)
        return [
            ratio(get_any(frame_state, "frame_no", default=0), 20000),
            hero_x,
            hero_z,
            center_dist,
            enemy_visible,
            hp_ratio(our_tower),
            hp_ratio(enemy_tower),
            float(bool(frame_state.get("map_state", True))),
            float(our_hero is not None and hp_ratio(our_hero) <= 0.35),
            safe_push,
            tower_tanking_minion,
            enemy_threat,
        ]

    def _legal_features(self, observation):
        legal = np.array(observation.get("legal_action", []), dtype=np.float32)
        if legal.size < 85:
            return [0.0] * 15
        top = legal[:12].tolist()
        move_x = legal[12:28]
        move_z = legal[28:44]
        return top + [ratio(np.sum(move_x), len(move_x)), ratio(np.sum(move_z), len(move_z)), ratio(np.sum(legal), max(len(legal), 1))]

    def _in_tower_range(self, hero, tower):
        if not hero or not tower:
            return 0.0
        attack_range = float(get_any(tower, "attack_range", default=0) or 0) + TOWER_DANGER_MARGIN
        return float(distance(hero, tower) <= attack_range)

    def _safe_push_state(self, our_hero, enemy_hero, enemy_tower, our_soldiers):
        if not our_hero or not enemy_tower:
            return 0.0, 0.0, 1.0

        target = get_any(enemy_tower, "attack_target", default=0)
        soldier_ids = {runtime_id(s) for s in our_soldiers}
        tower_tanking_minion = float(target in soldier_ids and target != 0)

        enemy_threat = 0.0
        if enemy_hero and hp_ratio(enemy_hero) > 0 and get_any(enemy_hero, "revive_time", default=0) <= 0:
            if is_visible_to(enemy_hero, camp_id(our_hero.get("camp"))):
                enemy_range = float(get_any(enemy_hero, "attack_range", default=0) or 0) + 2000.0
                enemy_threat = float(distance(our_hero, enemy_hero) <= enemy_range)

        hero_can_hit_tower = float(distance(our_hero, enemy_tower) <= float(get_any(our_hero, "attack_range", default=0) or 0) + 1000.0)
        safe_push = float(tower_tanking_minion and not enemy_threat and hero_can_hit_tower)
        return safe_push, tower_tanking_minion, enemy_threat
