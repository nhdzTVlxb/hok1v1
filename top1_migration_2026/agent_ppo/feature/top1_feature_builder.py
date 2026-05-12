#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Small fixed-length observation builder for the 2026 flat frame_state protocol.

The first production feature version intentionally stays compact and readable:
raw obs dict -> a fixed float32 vector.  It keeps the compact summary features,
then adds a small 2025-style entity tail for lane units and projectiles.
"""

import math
import numpy as np

from agent_ppo.conf.conf import DimConfig


MAP_SCALE = 100000.0
UNSEEN_PADDING = 100000
NEARBY_RANGE = 20000.0
TOWER_DANGER_MARGIN = 2000.0
UNIT_SLOT_COUNT = 7
UNIT_SLOT_DIM = 8
PROJECTILE_SLOT_COUNT = 10
PROJECTILE_SLOT_DIM = 7


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


def is_neutral_camp(camp):
    return camp == 0 or camp == "0"


def get_any(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict) and key in d:
            return d[key]
    return default


def runtime_id(actor):
    return get_any(actor or {}, "runtime_id", "player_id", default=0)


def actor_pos(actor, mirror=False):
    loc = actor.get("location", {}) if isinstance(actor, dict) else {}
    if not loc and isinstance(actor, dict):
        loc = ((actor.get("collider", {}) or {}).get("location", {}) or {})
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
    if is_neutral_camp(get_any(actor, "camp", default=None)):
        return False
    return actor_type == 1 and sub_type in (1, 11, "ACTOR_SUB_SOLDIER")


def is_tower(actor):
    actor_type = get_any(actor, "actor_type", default=None)
    sub_type = get_any(actor, "sub_type", default=None)
    return actor_type == 2 and sub_type in (21, 23, 24, "ACTOR_SUB_TOWER")


def is_monster(actor):
    actor_type = get_any(actor, "actor_type", default=None)
    sub_type = get_any(actor, "sub_type", default=None)
    return (
        actor_type in (3, "ACTOR_TYPE_MONSTER")
        or sub_type in ("ACTOR_SUB_MONSTER", "ACTOR_SUB_NEUTRAL_MONSTER")
        or (is_neutral_camp(get_any(actor, "camp", default=None)) and actor_type == 1 and sub_type == 0)
    )


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
        monsters = [npc for npc in npcs if is_monster(npc) and hp_ratio(npc) > 0]
        cakes = frame_state.get("cakes", []) or []
        bullets = frame_state.get("bullets", []) or []

        feature = []
        feature += self._hero_features(our_hero, our_hero, main_camp, enemy_tower, always_visible=True)
        feature += self._enemy_hero_features(our_hero, enemy_hero, main_camp)
        feature += self._skill_features(our_hero)
        feature += self._tower_features(our_hero, our_tower, enemy_tower)
        feature += self._soldier_features(our_hero, our_soldiers, enemy_soldiers, main_camp)
        feature += self._monster_features(our_hero, monsters)
        feature += self._cake_features(our_hero, cakes)
        feature += self._unit_slot_features(our_hero, our_soldiers, enemy_soldiers, monsters, main_camp)
        feature += self._projectile_features(our_hero, bullets, main_camp)
        feature += self._combat_risk_features(our_hero, enemy_hero, our_tower, enemy_tower, main_camp)
        feature += self._objective_features(frame_state, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, monsters, cakes, main_camp)
        feature += self._global_features(frame_state, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers)
        feature += self._status_macro_features(frame_state, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, cakes, bullets, main_camp)
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

    def _monster_features(self, our_hero, monsters):
        if not our_hero:
            return [0.0] * 12
        visible = [m for m in monsters if not (abs(actor_pos(m)[0]) >= UNSEEN_PADDING or abs(actor_pos(m)[1]) >= UNSEEN_PADDING)]
        nearest = min(visible, key=lambda m: distance(our_hero, m), default=None)
        low_hp = [m for m in visible if hp_ratio(m) <= 0.35]
        rel = rel_pos(our_hero, nearest, self.mirror) if nearest else [0.0, 0.0, 0.0]
        return [
            ratio(len(visible), 4),
            hp_ratio(nearest) if nearest else 0.0,
            rel[0],
            rel[1],
            rel[2],
            float(bool(low_hp)),
            ratio(min([hp_ratio(m) for m in low_hp], default=0.0), 1.0),
            float(nearest is not None and get_any(nearest, "attack_target", default=0) == runtime_id(our_hero)),
            ratio(get_any(nearest or {}, "kill_income", default=0), 300),
            ratio(get_any(nearest or {}, "hp", default=0), 10000),
            ratio(get_any(nearest or {}, "max_hp", default=0), 10000),
            ratio(distance(our_hero, nearest), MAP_SCALE) if nearest else 0.0,
        ]

    def _cake_features(self, our_hero, cakes):
        if not our_hero:
            return [0.0] * 8
        visible = [cake for cake in cakes if not (abs(actor_pos(cake)[0]) >= UNSEEN_PADDING or abs(actor_pos(cake)[1]) >= UNSEEN_PADDING)]
        nearest = min(visible, key=lambda cake: distance(our_hero, cake), default=None)
        rel = rel_pos(our_hero, nearest, self.mirror) if nearest else [0.0, 0.0, 0.0]
        dist = distance(our_hero, nearest) if nearest else 0.0
        hp_low = float(hp_ratio(our_hero) <= 0.85)
        return [
            ratio(len(visible), 2),
            float(nearest is not None),
            rel[0],
            rel[1],
            rel[2],
            ratio(dist, MAP_SCALE) if nearest else 0.0,
            hp_low,
            float(nearest is not None and dist <= 6000.0),
        ]

    def _unit_slot_features(self, our_hero, our_soldiers, enemy_soldiers, monsters, main_camp):
        if not our_hero:
            return [0.0] * (UNIT_SLOT_COUNT * UNIT_SLOT_DIM)

        def nearest_group(units, count):
            visible = [u for u in units if is_visible_to(u, main_camp)]
            visible.sort(key=lambda u: distance(our_hero, u))
            return visible[:count]

        slots = []
        slots += [(unit, 1.0) for unit in nearest_group(our_soldiers, 3)]
        slots += [(unit, -1.0) for unit in nearest_group(enemy_soldiers, 3)]
        monster_visible = [
            m for m in monsters if not (abs(actor_pos(m)[0]) >= UNSEEN_PADDING or abs(actor_pos(m)[1]) >= UNSEEN_PADDING)
        ]
        monster_visible.sort(key=lambda m: distance(our_hero, m))
        slots += [(unit, 0.5) for unit in monster_visible[:1]]

        out = []
        for unit, category in slots[:UNIT_SLOT_COUNT]:
            rel = rel_pos(our_hero, unit, self.mirror)
            out += [
                1.0,
                category,
                hp_ratio(unit),
                rel[0],
                rel[1],
                rel[2],
                float(hp_ratio(unit) <= 0.35),
                float(get_any(unit, "attack_target", default=0) == runtime_id(our_hero)),
            ]
        while len(out) < UNIT_SLOT_COUNT * UNIT_SLOT_DIM:
            out += [0.0] * UNIT_SLOT_DIM
        return out[: UNIT_SLOT_COUNT * UNIT_SLOT_DIM]

    def _projectile_features(self, our_hero, bullets, main_camp):
        if not our_hero:
            return [0.0] * (PROJECTILE_SLOT_COUNT * PROJECTILE_SLOT_DIM)

        visible = [
            bullet
            for bullet in bullets
            if not (abs(actor_pos(bullet)[0]) >= UNSEEN_PADDING or abs(actor_pos(bullet)[1]) >= UNSEEN_PADDING)
        ]
        visible.sort(key=lambda bullet: distance(our_hero, bullet))

        out = []
        for bullet in visible[:PROJECTILE_SLOT_COUNT]:
            rel = rel_pos(our_hero, bullet, self.mirror)
            bullet_camp = camp_id(get_any(bullet, "camp", default=0))
            out += [
                1.0,
                float(bullet_camp != main_camp and not is_neutral_camp(bullet_camp)),
                float(bullet_camp == main_camp),
                ratio(get_any(bullet, "slot_type", default=0), 7),
                rel[0],
                rel[1],
                rel[2],
            ]
        while len(out) < PROJECTILE_SLOT_COUNT * PROJECTILE_SLOT_DIM:
            out += [0.0] * PROJECTILE_SLOT_DIM
        return out[: PROJECTILE_SLOT_COUNT * PROJECTILE_SLOT_DIM]

    def _combat_risk_features(self, our_hero, enemy_hero, our_tower, enemy_tower, main_camp):
        if not our_hero:
            return [0.0] * 20
        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0
        hero_hp = hp_ratio(our_hero)
        enemy_hp = hp_ratio(enemy_hero) if enemy_visible else 0.0
        hp_gap = clip(hero_hp - enemy_hp)
        money_gap = clip(
            (float(get_any(our_hero, "money_cnt", "money", default=0) or 0)
            - float(get_any(enemy_hero or {}, "money_cnt", "money", default=0) or 0)) / 10000.0
        ) if enemy_visible else 0.0
        level_gap = clip(
            (float(get_any(our_hero, "level", default=1) or 1)
            - float(get_any(enemy_hero or {}, "level", default=1) or 1)) / 15.0
        ) if enemy_visible else 0.0
        dist_enemy = distance(our_hero, enemy_hero) if enemy_visible else MAP_SCALE
        our_range = float(get_any(our_hero, "attack_range", default=0) or 0)
        enemy_range = float(get_any(enemy_hero or {}, "attack_range", default=0) or 0)
        in_our_range = float(enemy_visible and dist_enemy <= our_range + 1500.0)
        in_enemy_range = float(enemy_visible and dist_enemy <= enemy_range + 2000.0)
        enemy_target_us = float(enemy_visible and get_any(enemy_hero, "attack_target", default=0) == runtime_id(our_hero))
        us_target_enemy = float(enemy_visible and get_any(our_hero, "attack_target", default=0) == runtime_id(enemy_hero))
        near_our_tower = float(our_tower is not None and distance(our_hero, our_tower) <= 13000.0)
        near_enemy_tower = float(enemy_tower is not None and distance(our_hero, enemy_tower) <= float(get_any(enemy_tower, "attack_range", default=0) or 0) + 2500.0)
        enemy_tower_target_us = float(enemy_tower is not None and get_any(enemy_tower, "attack_target", default=0) == runtime_id(our_hero))
        our_tower_target_enemy = float(our_tower is not None and enemy_visible and get_any(our_tower, "attack_target", default=0) == runtime_id(enemy_hero))
        summon_ready = self._slot_ready(our_hero, 6)
        recover_ready = self._slot_ready(our_hero, 5)
        enemy_skill_ready = self._ready_skill_count(enemy_hero) / 4.0 if enemy_visible else 0.0
        bad_fight = float(enemy_visible and (hp_gap <= -0.18 or (hero_hp <= 0.35 and enemy_hp > 0.25)))
        no_summoner_caution = float(enemy_visible and not summon_ready and hp_gap < 0.10)
        return [
            hp_gap,
            money_gap,
            level_gap,
            float(enemy_visible),
            ratio(dist_enemy, MAP_SCALE) if enemy_visible else 1.0,
            in_our_range,
            in_enemy_range,
            enemy_target_us,
            us_target_enemy,
            near_our_tower,
            near_enemy_tower,
            enemy_tower_target_us,
            our_tower_target_enemy,
            float(summon_ready),
            float(recover_ready),
            enemy_skill_ready,
            bad_fight,
            no_summoner_caution,
            float(hero_hp <= 0.50 and near_our_tower),
            float(hero_hp >= 0.60 and money_gap >= 0.0),
        ]

    def _objective_features(self, frame_state, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, monsters, cakes, main_camp):
        if not our_hero:
            return [0.0] * 20
        money = float(get_any(our_hero, "money_cnt", "money", default=0) or 0)
        visible_monsters = [m for m in monsters if not (abs(actor_pos(m)[0]) >= UNSEEN_PADDING or abs(actor_pos(m)[1]) >= UNSEEN_PADDING)]
        nearest_monster = min(visible_monsters, key=lambda m: distance(our_hero, m), default=None)
        low_monster = min([hp_ratio(m) for m in visible_monsters], default=0.0)
        visible_cakes = [cake for cake in cakes if not (abs(actor_pos(cake)[0]) >= UNSEEN_PADDING or abs(actor_pos(cake)[1]) >= UNSEEN_PADDING)]
        nearest_cake = min(visible_cakes, key=lambda cake: distance(our_hero, cake), default=None)
        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0
        enemy_threat_close = float(enemy_visible and distance(our_hero, enemy_hero) <= float(get_any(enemy_hero, "attack_range", default=0) or 0) + 3500.0)
        safe_push, tower_tanking_minion, enemy_threat = self._safe_push_state(our_hero, enemy_hero, enemy_tower, our_soldiers)
        our_near = [s for s in our_soldiers if distance(our_hero, s) <= NEARBY_RANGE]
        enemy_near = [s for s in enemy_soldiers if distance(our_hero, s) <= NEARBY_RANGE]
        return [
            ratio(money, 12000),
            float(money < 5500),
            float(money >= 6000),
            ratio(len(visible_monsters), 4),
            hp_ratio(nearest_monster) if nearest_monster else 0.0,
            ratio(distance(our_hero, nearest_monster), MAP_SCALE) if nearest_monster else 1.0,
            float(nearest_monster is not None and low_monster <= 0.35),
            float(nearest_monster is not None and get_any(nearest_monster, "attack_target", default=0) == runtime_id(our_hero)),
            ratio(len(visible_cakes), 2),
            ratio(distance(our_hero, nearest_cake), MAP_SCALE) if nearest_cake else 1.0,
            float(hp_ratio(our_hero) < 0.80 and nearest_cake is not None and not enemy_threat_close),
            safe_push,
            tower_tanking_minion,
            enemy_threat,
            ratio(len(our_near), 6),
            ratio(len(enemy_near), 6),
            float(len(enemy_near) == 0),
            float(len(our_near) > 0 and len(enemy_near) > 0),
            float(enemy_tower is not None and get_any(enemy_tower, "attack_target", default=0) == runtime_id(our_hero)),
            float(our_tower is not None and distance(our_hero, our_tower) <= 13000.0),
        ]

    def _slot_ready(self, hero, slot_type):
        for slot in ((hero or {}).get("skill_state", {}) or {}).get("slot_states", []) or []:
            if int(get_any(slot, "slot_type", default=-1) or -1) == slot_type:
                cd = float(get_any(slot, "cooldown", default=0) or 0)
                return cd <= 0 and bool(get_any(slot, "usable", default=True))
        return False

    def _ready_skill_count(self, hero):
        count = 0
        for slot in ((hero or {}).get("skill_state", {}) or {}).get("slot_states", []) or []:
            slot_type = int(get_any(slot, "slot_type", default=-1) or -1)
            if slot_type in (1, 2, 3, 6) and float(get_any(slot, "cooldown", default=0) or 0) <= 0:
                count += 1
        return count

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

    def _status_macro_features(self, frame_state, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, cakes, bullets, main_camp):
        if not our_hero:
            return [0.0] * 64

        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0
        hero_id = runtime_id(our_hero)
        enemy_id = runtime_id(enemy_hero or {})
        our_tower_id = runtime_id(our_tower or {})
        enemy_tower_id = runtime_id(enemy_tower or {})
        our_tower_target = get_any(our_tower or {}, "attack_target", default=0)
        enemy_tower_target = get_any(enemy_tower or {}, "attack_target", default=0)
        our_soldier_ids = {runtime_id(s) for s in our_soldiers}
        enemy_soldier_ids = {runtime_id(s) for s in enemy_soldiers}

        visible_cakes = [cake for cake in cakes if not (abs(actor_pos(cake)[0]) >= UNSEEN_PADDING or abs(actor_pos(cake)[1]) >= UNSEEN_PADDING)]
        own_cakes = self._side_cakes(visible_cakes, our_tower, enemy_tower, own_side=True)
        enemy_cakes = self._side_cakes(visible_cakes, our_tower, enemy_tower, own_side=False)
        nearest_own_cake = min(own_cakes, key=lambda cake: distance(our_hero, cake), default=None)
        nearest_enemy_cake = min(enemy_cakes, key=lambda cake: distance(our_hero, cake), default=None)

        enemy_bullets = [b for b in bullets if self._bullet_is_enemy(b, main_camp)]
        enemy_bullets.sort(key=lambda b: distance(our_hero, b))
        nearest_enemy_bullet = enemy_bullets[0] if enemy_bullets else None
        tower_bullets = [
            b for b in enemy_bullets
            if get_any(b, "source_actor", default=0) in (enemy_tower_id, our_tower_id)
            or get_any(b, "slot_type", default=-1) == 16
        ]
        nearest_tower_bullet = tower_bullets[0] if tower_bullets else None

        hero_buff = self._buff_summary(our_hero)
        enemy_buff = self._buff_summary(enemy_hero) if enemy_visible else [0.0] * 8
        hero_skill = self._skill_summary(our_hero)
        enemy_skill = self._skill_summary(enemy_hero) if enemy_visible else [0.0] * 8
        abilities = self._ability_summary(our_hero)
        enemy_abilities = self._ability_summary(enemy_hero) if enemy_visible else [0.0] * 5
        command = self._command_summary(our_hero)

        out = [
            ratio(get_any(our_hero, "behav_mode", default=0), 32),
            ratio(get_any(enemy_hero or {}, "behav_mode", default=0), 32) if enemy_visible else 0.0,
            float(our_tower_target == enemy_id and enemy_id != 0),
            float(our_tower_target in enemy_soldier_ids and our_tower_target != 0),
            float(enemy_tower_target == hero_id and hero_id != 0),
            float(enemy_tower_target in our_soldier_ids and enemy_tower_target != 0),
            ratio(get_any(our_tower or {}, "attack_range", default=0), 15000),
            ratio(get_any(enemy_tower or {}, "attack_range", default=0), 15000),
            self._in_tower_range(our_hero, our_tower) if our_tower else 0.0,
            self._in_tower_range(our_hero, enemy_tower) if enemy_tower else 0.0,
            ratio(distance(our_hero, nearest_own_cake), MAP_SCALE) if nearest_own_cake else 1.0,
            ratio(distance(our_hero, nearest_enemy_cake), MAP_SCALE) if nearest_enemy_cake else 1.0,
            float(nearest_own_cake is not None),
            float(nearest_enemy_cake is not None),
            ratio(len(own_cakes), 2),
            ratio(len(enemy_cakes), 2),
            ratio(distance(our_hero, nearest_enemy_bullet), MAP_SCALE) if nearest_enemy_bullet else 1.0,
            ratio(get_any(nearest_enemy_bullet or {}, "slot_type", default=0), 16),
            float(nearest_enemy_bullet is not None and distance(our_hero, nearest_enemy_bullet) <= 12000.0),
            ratio(distance(our_hero, nearest_tower_bullet), MAP_SCALE) if nearest_tower_bullet else 1.0,
            float(nearest_tower_bullet is not None and distance(our_hero, nearest_tower_bullet) <= 12000.0),
            ratio(len(enemy_bullets), 10),
        ]
        out += hero_buff + enemy_buff + hero_skill + enemy_skill + abilities + enemy_abilities + command
        if len(out) < 64:
            out += [0.0] * (64 - len(out))
        return out[:64]

    def _side_cakes(self, cakes, our_tower, enemy_tower, own_side):
        if not cakes:
            return []
        if not our_tower or not enemy_tower:
            return cakes if own_side else []
        out = []
        for cake in cakes:
            own_dist = distance(cake, our_tower)
            enemy_dist = distance(cake, enemy_tower)
            if (own_dist <= enemy_dist) == own_side:
                out.append(cake)
        return out

    def _bullet_is_enemy(self, bullet, main_camp):
        bullet_camp = camp_id(get_any(bullet, "camp", default=0))
        return bullet_camp != main_camp and not is_neutral_camp(bullet_camp)

    def _buff_summary(self, actor):
        buff_state = (actor or {}).get("buff_state", {}) or {}
        skills = buff_state.get("buff_skills", []) or []
        marks = buff_state.get("buff_marks", []) or []
        ids = [float(get_any(item, "configId", "config_id", default=0) or 0) for item in skills[:4]]
        while len(ids) < 4:
            ids.append(0.0)
        return [
            ratio(len(skills), 12),
            ratio(len(marks), 12),
            ratio(sum(get_any(item, "times", default=0) or 0 for item in skills), 20),
            ratio(max([get_any(item, "startTime", default=0) or 0 for item in skills], default=0), 120000),
            ratio(ids[0] % 100000, 100000),
            ratio(ids[1] % 100000, 100000),
            ratio(ids[2] % 100000, 100000),
            ratio(ids[3] % 100000, 100000),
        ]

    def _skill_summary(self, hero):
        slots = ((hero or {}).get("skill_state", {}) or {}).get("slot_states", []) or []
        ready = 0
        usable = 0
        used = 0
        hit_hero = 0
        cd_sum = 0.0
        skill_levels = 0
        for slot in slots:
            slot_type = int(get_any(slot, "slot_type", default=-1) or -1)
            if slot_type not in (1, 2, 3, 5, 6):
                continue
            cooldown = float(get_any(slot, "cooldown", default=0) or 0)
            cooldown_max = max(float(get_any(slot, "cooldown_max", default=1) or 1), 1.0)
            ready += int(cooldown <= 0 and bool(get_any(slot, "usable", default=True)))
            usable += int(bool(get_any(slot, "usable", default=False)))
            used += int(get_any(slot, "usedTimes", "used_times", default=0) or 0)
            hit_hero += int(get_any(slot, "hitHeroTimes", "hit_hero_times", default=0) or 0)
            cd_sum += clip(cooldown / cooldown_max, 0.0, 1.0)
            skill_levels += int(get_any(slot, "level", default=0) or 0)
        return [
            ratio(ready, 5),
            ratio(usable, 5),
            ratio(used, 100),
            ratio(hit_hero, 20),
            ratio(cd_sum, 5),
            ratio(skill_levels, 20),
            float(self._slot_ready(hero, 2)),
            float(self._slot_ready(hero, 3)),
        ]

    def _ability_summary(self, hero):
        abilities = get_any(hero or {}, "abilities", default=[]) or []
        if not isinstance(abilities, (list, tuple)) or not abilities:
            return [0.0] * 5
        true_count = sum(bool(v) for v in abilities)
        first_ten = abilities[:10]
        return [
            ratio(true_count, len(abilities)),
            float(any(first_ten)),
            float(all(not bool(v) for v in first_ten)),
            float(bool(abilities[1])) if len(abilities) > 1 else 0.0,
            float(bool(abilities[5])) if len(abilities) > 5 else 0.0,
        ]

    def _command_summary(self, hero):
        command = ((hero or {}).get("real_cmd", []) or [{}])[0] or {}
        command_type = int(get_any(command, "command_type", default=0) or 0)
        charge = command.get("charge_skill", {}) or {}
        attack_common = command.get("attack_common", {}) or {}
        return [
            ratio(command_type, 16),
            ratio(get_any(charge, "state", default=0), 4),
            float(get_any(attack_common, "actorID", default=0) not in (0, None)),
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
