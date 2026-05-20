#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Small fixed-length observation builder for the 2026 flat frame_state protocol.

The first production feature version intentionally stays compact and readable:
raw obs dict -> a fixed float32 vector.  It keeps the compact summary features,
then adds a structured entity tail for lane units and target selection.
"""

import math
import numpy as np

from agent_ppo.conf.conf import Config, DimConfig


MAP_SCALE = 100000.0
UNSEEN_PADDING = 100000
NEARBY_RANGE = 20000.0
TOWER_DANGER_MARGIN = 2000.0
UNIT_SLOT_COUNT = 7
UNIT_SLOT_DIM = 8
PROJECTILE_SLOT_COUNT = 10
PROJECTILE_SLOT_DIM = 7
ACTOR_TYPE_HERO = 0
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
HEAL_SKILL_ID = 80102
BERSERK_SKILL_ID = 80110
# Buff ids below are only from the current 2026 debug dumps in this workspace
# (eyes_debug_buff.json and debug_buff_summary2.json), not old 2025 mappings.
LUBAN_ENHANCED_ATTACK_BUFF_ID = 112045
LUBAN_OBSERVED_EXTRA_BUFF_IDS = (112044, 112046, 112047, 112048)
DIRENJIE_OBSERVED_BUFF_IDS = (133950, 133951, 133260)
RECOVER_BUFF_IDS = (10000, 10010)
SPRING_RECOVER_BUFF_IDS = (90015,)
MOVE_SPEED_BUFF_IDS = (11001, 11002)
CLEANSE_BUFF_IDS = (11010,)
COMMON_OBSERVED_BUFF_IDS = (
    90015, 10000, 10010, 10014, 11001, 11002, 11010, 90025, 911290, 914110, 914210, 914211,
    50000, 500009, 801100, 90019, 90110, 911260, 911261, 912260, 912262, 912263, 912330, 914230,
    914232, 919900,
)
LUBAN_OBSERVED_BUFF_IDS = (
    112000, 112001, 112010, 112015, 112020, 112025, 112030, 112035, 112040, 112041, 112042,
    112043, 112044, 112045, 112046, 112047, 112048, 112100, 112190, 112191, 112192, 112200,
    112201, 112210, 112300, 112301, 112320, 112890, 112910, 112990, 112991,
)
DIRENJIE_OBSERVED_ALL_BUFF_IDS = (
    133000, 133001, 133010, 133011, 133020, 133090, 133100, 133200, 133250, 133260, 133300,
    133310, 133950, 133951, 131956, 167600, 167602,
)
KNOWN_BUFF_IDS = set(COMMON_OBSERVED_BUFF_IDS + LUBAN_OBSERVED_BUFF_IDS + DIRENJIE_OBSERVED_ALL_BUFF_IDS)
ENDGAME_MONEY = 6000.0
GRASS_POINTS = [
    (-5200.0, -9000.0),
    (-2600.0, -9800.0),
    (2600.0, -9200.0),
    (7200.0, -5200.0),
    (9800.0, -1200.0),
    (9000.0, 3200.0),
    (4500.0, 8200.0),
    (-1200.0, 9800.0),
    (-7200.0, 5200.0),
    (-9800.0, 1200.0),
]


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


def config_id(actor):
    return int_value(get_any(actor or {}, "config_id", "configId", default=0))


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


def feature_point(point, mirror=False):
    if point is None:
        return [0.0, 0.0]
    x, z = float(point[0] or 0), float(point[1] or 0)
    if mirror:
        x, z = -x, -z
    return [x, z]


def rel_pos(src, dst, mirror=False):
    sx, sz = actor_pos(src, mirror)
    dx, dz = actor_pos(dst, mirror)
    if abs(dx) >= UNSEEN_PADDING or abs(dz) >= UNSEEN_PADDING:
        return [0.0, 0.0, 0.0]
    rx, rz = dx - sx, dz - sz
    return [clip(rx / MAP_SCALE), clip(rz / MAP_SCALE), clip(math.hypot(rx, rz) / MAP_SCALE, 0.0, 1.0)]


def rel_point(src, point, mirror=False):
    sx, sz = actor_pos(src, mirror)
    dx, dz = feature_point(point, mirror)
    rx, rz = dx - sx, dz - sz
    return [clip(rx / MAP_SCALE), clip(rz / MAP_SCALE), clip(math.hypot(rx, rz) / MAP_SCALE, 0.0, 1.0)]


def slot_type_value(slot_type):
    if isinstance(slot_type, str):
        if slot_type.rsplit("_", 1)[-1].isdigit():
            return int(slot_type.rsplit("_", 1)[-1])
        return 0
    try:
        return int(slot_type or 0)
    except (TypeError, ValueError):
        return 0


def int_value(value, default=0):
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


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
    cfg = config_id(actor)
    if is_neutral_camp(get_any(actor, "camp", default=None)):
        return False
    return actor_type == ACTOR_TYPE_MONSTER and (sub_type in (SUB_TYPE_LANE_SOLDIER, "ACTOR_SUB_SOLDIER") or cfg in LANE_SOLDIER_CONFIG_IDS)


def is_tower(actor):
    actor_type = get_any(actor, "actor_type", default=None)
    sub_type = get_any(actor, "sub_type", default=None)
    cfg = config_id(actor)
    return actor_type == ACTOR_TYPE_ORGAN and (
        sub_type in (SUB_TYPE_TOWER, SUB_TYPE_SPRING_TOWER, SUB_TYPE_CRYSTAL, "ACTOR_SUB_TOWER")
        or cfg in TOWER_CONFIG_IDS
        or cfg in CRYSTAL_CONFIG_IDS
        or cfg in SPRING_TOWER_CONFIG_IDS
    )


def is_monster(actor):
    actor_type = get_any(actor, "actor_type", default=None)
    sub_type = get_any(actor, "sub_type", default=None)
    cfg = config_id(actor)
    return (
        cfg == RIVER_SPIRIT_CONFIG_ID
        or actor_type in (3, "ACTOR_TYPE_MONSTER")
        or sub_type in ("ACTOR_SUB_MONSTER", "ACTOR_SUB_NEUTRAL_MONSTER")
        or (is_neutral_camp(get_any(actor, "camp", default=None)) and actor_type == ACTOR_TYPE_MONSTER and sub_type == SUB_TYPE_NEUTRAL_MONSTER)
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
        self.last_enemy_visible_pos = None
        self.last_enemy_seen_frame = 0
        self.last_total_hurt_to_hero = 0.0
        self.last_total_be_hurt_by_hero = 0.0
        self.recent_trade_score = 0.0

    def reset(self):
        self.last_enemy_visible_pos = None
        self.last_enemy_seen_frame = 0
        self.last_total_hurt_to_hero = 0.0
        self.last_total_be_hurt_by_hero = 0.0
        self.recent_trade_score = 0.0

    def build_observation(self, observation):
        frame_state = observation.get("frame_state", {})
        frame_no = int_value(get_any(frame_state, "frame_no", default=0))
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
        feature += self._strategic_macro_features(frame_state, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, monsters, cakes, bullets, main_camp)
        feature += self._legal_features(observation)

        feature += self._ability_control_features(our_hero, enemy_hero, main_camp)
        feature += self._lane_entity_features(our_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, main_camp)
        feature += self._tower_wave_features(our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, cakes, main_camp)
        feature += self._projectile_detail_features(our_hero, bullets, enemy_hero, main_camp)
        feature += self._position_bucket_features(our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers)
        feature += self._buff_mark_detail_features(our_hero, enemy_hero, main_camp)
        feature += self._economy_phase_features(our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, monsters, cakes, main_camp)

        target_features = self._target_entity_features(our_hero, enemy_hero, enemy_tower, enemy_soldiers, monsters, main_camp)
        main_dim = self.FEATURE_DIM - Config.TARGET_FEATURE_DIM
        feature = self._fit(feature, main_dim) + self._fit(target_features, Config.TARGET_FEATURE_DIM)
        self._update_enemy_memory(enemy_hero, main_camp, frame_no)
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
        our.sort(key=runtime_id)
        enemy.sort(key=runtime_id)
        return our, enemy

    def _hero_features(self, hero, our_hero, main_camp, enemy_tower, always_visible=False):
        if not hero:
            return [0.0] * 24
        visible = always_visible or is_visible_to(hero, main_camp)
        hero_cfg = config_id(hero)
        x, z = norm_pos(hero, self.mirror) if visible else [0.0, 0.0]
        forward = hero.get("forward", {}) or {}
        fx = clip(float(forward.get("x", 0) or 0) / 1000.0)
        fz = clip(float(forward.get("z", 0) or 0) / 1000.0)
        tower_danger = self._in_tower_range(hero, enemy_tower) if enemy_tower else 0.0
        attacked_by_tower = float(enemy_tower is not None and runtime_id(hero) == get_any(enemy_tower, "attack_target", default=-1))
        rel = [0.0, 0.0, 0.0] if our_hero is hero else rel_pos(our_hero, hero, self.mirror)
        return [
            float(hero_cfg == 112),
            float(hero_cfg == 133),
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
            visible.sort(key=runtime_id)
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
                ratio(slot_type_value(get_any(bullet, "slot_type", default=0)), 7),
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

    def _strategic_macro_features(self, frame_state, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, monsters, cakes, bullets, main_camp):
        if not our_hero:
            return [0.0] * 128

        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0
        hero_hp = hp_ratio(our_hero)
        enemy_hp = hp_ratio(enemy_hero) if enemy_visible else 0.0
        money = float(get_any(our_hero, "money_cnt", "money", default=0) or 0)
        enemy_money = float(get_any(enemy_hero or {}, "money_cnt", "money", default=0) or 0) if enemy_visible else 0.0
        level = float(get_any(our_hero, "level", default=1) or 1)
        enemy_level = float(get_any(enemy_hero or {}, "level", default=1) or 1) if enemy_visible else 1.0
        hp_gap = hero_hp - enemy_hp
        money_gap = money - enemy_money
        level_gap = level - enemy_level
        wave_pressure = self._wave_pressure_score(our_hero, our_soldiers, enemy_soldiers)
        fight_score = self._fight_score(our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, main_camp)
        trade_delta = self._trade_delta_feature(our_hero)

        safe_push, tower_tanking_minion, enemy_threat = self._safe_push_state(our_hero, enemy_hero, enemy_tower, our_soldiers)
        enemy_dead = float(enemy_hero is not None and (hp_ratio(enemy_hero) <= 0 or float(get_any(enemy_hero, "revive_time", default=0) or 0) > 0))
        enemy_revive = ratio(get_any(enemy_hero or {}, "revive_time", default=0), 600)
        near_enemy_tower_minions = [
            s for s in our_soldiers
            if enemy_tower is not None and distance(s, enemy_tower) <= 15000.0
        ]
        enemy_wave_near_our_tower = [
            s for s in enemy_soldiers
            if our_tower is not None and distance(s, our_tower) <= 15000.0
        ]
        finish_window = float(
            hero_hp >= 0.26
            and (safe_push or (enemy_dead and len(near_enemy_tower_minions) >= 2))
        )

        summon_id = self._summoner_skill_id(our_hero)
        summon_ready = self._slot_ready(our_hero, 6)
        heal_ready = float(summon_id == HEAL_SKILL_ID and summon_ready)
        berserk_ready = float(summon_id == BERSERK_SKILL_ID and summon_ready)
        skill1_ready = float(self._slot_ready(our_hero, 1))
        skill2_ready = float(self._slot_ready(our_hero, 2))
        skill3_ready = float(self._slot_ready(our_hero, 3))
        enemy_skill1_ready = float(enemy_visible and self._slot_ready(enemy_hero, 1))
        enemy_skill2_ready = float(enemy_visible and self._slot_ready(enemy_hero, 2))
        enemy_skill3_ready = float(enemy_visible and self._slot_ready(enemy_hero, 3))
        luban_enhanced_attack = float(self._has_buff(our_hero, LUBAN_ENHANCED_ATTACK_BUFF_ID))
        enemy_luban_enhanced_attack = float(enemy_visible and self._has_buff(enemy_hero, LUBAN_ENHANCED_ATTACK_BUFF_ID))

        grass_dist = self._nearest_grass_distance(our_hero)
        enemy_dist = distance(our_hero, enemy_hero) if enemy_visible else MAP_SCALE
        grass_ambush_window = float(
            enemy_visible
            and hero_hp >= 0.38
            and abs(hp_gap) <= 0.25
            and grass_dist <= 18000.0
            and enemy_dist <= float(get_any(our_hero, "attack_range", default=0) or 0) + 9500.0
            and not enemy_wave_near_our_tower
        )

        visible_monsters = [m for m in monsters if not (abs(actor_pos(m)[0]) >= UNSEEN_PADDING or abs(actor_pos(m)[1]) >= UNSEEN_PADDING)]
        nearest_monster = min(visible_monsters, key=lambda m: distance(our_hero, m), default=None)
        monster_value = self._monster_value_scale(money, len(visible_monsters))
        visible_cakes = [cake for cake in cakes if not (abs(actor_pos(cake)[0]) >= UNSEEN_PADDING or abs(actor_pos(cake)[1]) >= UNSEEN_PADDING)]
        nearest_cake = min(visible_cakes, key=lambda cake: distance(our_hero, cake), default=None)

        enemy_bullets = [b for b in bullets if self._bullet_is_enemy(b, main_camp)]
        nearest_enemy_bullet = min(enemy_bullets, key=lambda b: distance(our_hero, b), default=None)
        enemy_luban_ult_danger = float(
            enemy_visible
            and int(get_any(enemy_hero, "config_id", "configId", default=0) or 0) == 112
            and enemy_skill3_ready
            and enemy_dist <= 10000.0
        )
        heal_value_window = float(heal_ready and hero_hp <= 0.70)

        out = [
            clip(hp_gap),
            clip(money_gap / 3000.0),
            clip(level_gap / 3.0),
            clip(fight_score),
            float(fight_score >= 0.18),
            float(fight_score >= 0.45),
            float(fight_score <= -0.25),
            hero_hp,
            enemy_hp,
            ratio(money, 12000),
            ratio(enemy_money, 12000),
            float(money >= ENDGAME_MONEY),
            float(money_gap >= 1200.0),
            float(money_gap >= 2200.0),
            ratio(level, 15),
            ratio(enemy_level, 15),
            clip(wave_pressure),
            ratio(len(our_soldiers), 8),
            ratio(len(enemy_soldiers), 8),
            ratio(len([s for s in our_soldiers if distance(our_hero, s) <= 13000.0]), 6),
            ratio(len([s for s in enemy_soldiers if distance(our_hero, s) <= 13000.0]), 6),
            ratio(len(near_enemy_tower_minions), 6),
            ratio(len(enemy_wave_near_our_tower), 6),
            float(bool(enemy_wave_near_our_tower)),
            finish_window,
            safe_push,
            tower_tanking_minion,
            enemy_threat,
            hp_ratio(enemy_tower),
            hp_ratio(our_tower),
            enemy_dead,
            enemy_revive,
            float(enemy_visible),
            ratio(enemy_dist, MAP_SCALE),
            float(enemy_visible and enemy_dist <= float(get_any(our_hero, "attack_range", default=0) or 0) + 3500.0),
            float(enemy_visible and enemy_dist <= float(get_any(enemy_hero, "attack_range", default=0) or 0) + 3500.0),
            heal_ready,
            berserk_ready,
            float(summon_id == HEAL_SKILL_ID),
            float(summon_id == BERSERK_SKILL_ID),
            skill1_ready,
            skill2_ready,
            skill3_ready,
            enemy_skill1_ready,
            enemy_skill2_ready,
            enemy_skill3_ready,
            luban_enhanced_attack,
            enemy_luban_enhanced_attack,
            enemy_luban_ult_danger,
            heal_value_window,
            float(bool(get_any(our_hero, "is_in_grass", default=False))),
            ratio(grass_dist, MAP_SCALE),
            grass_ambush_window,
            ratio(len(visible_monsters), 4),
            monster_value,
            hp_ratio(nearest_monster) if nearest_monster else 0.0,
            ratio(distance(our_hero, nearest_monster), MAP_SCALE) if nearest_monster else 1.0,
            float(nearest_monster is not None and monster_value >= 0.5),
            ratio(len(visible_cakes), 2),
            ratio(distance(our_hero, nearest_cake), MAP_SCALE) if nearest_cake else 1.0,
            float(nearest_cake is not None and hero_hp <= 0.62 and not finish_window),
            ratio(distance(our_hero, nearest_enemy_bullet), MAP_SCALE) if nearest_enemy_bullet else 1.0,
            float(nearest_enemy_bullet is not None and distance(our_hero, nearest_enemy_bullet) <= 10000.0),
            float(enemy_tower is not None and get_any(enemy_tower, "attack_target", default=0) == runtime_id(our_hero)),
            float(our_tower is not None and enemy_visible and get_any(our_tower, "attack_target", default=0) == runtime_id(enemy_hero)),
            float(hero_hp <= 0.20),
            float(hero_hp <= 0.32 and not finish_window),
            float(hero_hp >= 0.45 and fight_score >= 0.18),
            trade_delta,
            self.recent_trade_score,
            float(self.recent_trade_score <= -0.18),
        ]
        out += self._enemy_memory_features(our_hero, enemy_hero, main_camp, int_value(get_any(frame_state, "frame_no", default=0)))
        if len(out) < 128:
            out += [0.0] * (128 - len(out))
        return out[:128]

    def _fight_score(self, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, main_camp):
        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0
        hero_hp = hp_ratio(our_hero)
        enemy_hp = hp_ratio(enemy_hero) if enemy_visible else 0.0
        money_gap = float(get_any(our_hero, "money_cnt", "money", default=0) or 0)
        level_gap = float(get_any(our_hero, "level", default=1) or 1)
        if enemy_visible:
            money_gap -= float(get_any(enemy_hero, "money_cnt", "money", default=0) or 0)
            level_gap -= float(get_any(enemy_hero, "level", default=1) or 1)
        score = (hero_hp - enemy_hp) * 0.9
        score += clip(money_gap / 2500.0) * 0.55
        score += clip(level_gap / 3.0) * 0.22
        score += self._wave_pressure_score(our_hero, our_soldiers, enemy_soldiers) * 0.18
        score += float(self._slot_ready(our_hero, 3)) * 0.10
        score += float(self._slot_ready(our_hero, 1)) * 0.05
        summon_id = self._summoner_skill_id(our_hero)
        if self._slot_ready(our_hero, 6):
            score += 0.16 if summon_id == BERSERK_SKILL_ID else 0.08 if summon_id == HEAL_SKILL_ID and hero_hp <= 0.70 else 0.0
        if enemy_tower is not None and self._in_tower_range(our_hero, enemy_tower):
            score -= 0.35
        if our_tower is not None and distance(our_hero, our_tower) <= 12000.0:
            score += 0.08
        if hero_hp < 0.25:
            score -= 0.35
        elif hero_hp >= 0.38 and money_gap > 1000.0:
            score += 0.15
        return score

    def _trade_delta_feature(self, hero):
        if not hero:
            return 0.0
        hurt_to = float(get_any(hero, "total_hurt_to_hero", default=0) or 0)
        hurt_by = float(get_any(hero, "total_be_hurt_by_hero", default=0) or 0)
        delta = ((hurt_to - self.last_total_hurt_to_hero) - (hurt_by - self.last_total_be_hurt_by_hero)) / 1500.0
        self.last_total_hurt_to_hero = hurt_to
        self.last_total_be_hurt_by_hero = hurt_by
        delta = clip(delta)
        self.recent_trade_score = clip(self.recent_trade_score * 0.82 + delta)
        return delta

    def _wave_pressure_score(self, our_hero, our_soldiers, enemy_soldiers):
        hero_pos = actor_pos(our_hero)
        our_near = [s for s in our_soldiers if math.dist(hero_pos, actor_pos(s)) <= 13000.0]
        enemy_near = [s for s in enemy_soldiers if math.dist(hero_pos, actor_pos(s)) <= 13000.0]
        return clip((len(our_near) - len(enemy_near)) / 4.0)

    def _update_enemy_memory(self, enemy_hero, main_camp, frame_no):
        if enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0:
            self.last_enemy_visible_pos = actor_pos(enemy_hero)
            self.last_enemy_seen_frame = int(frame_no or 0)

    def _enemy_memory_features(self, our_hero, enemy_hero, main_camp, frame_no):
        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0
        if enemy_visible:
            pos = actor_pos(enemy_hero)
            stale_frames = 0
        else:
            pos = self.last_enemy_visible_pos
            stale_frames = max(0, int(frame_no or 0) - int(self.last_enemy_seen_frame or 0))
        if not our_hero or pos is None:
            return [float(enemy_visible), 0.0, 0.0, 1.0, 1.0, 0.0]
        pseudo_actor = {"location": {"x": pos[0], "z": pos[1]}}
        rel = rel_pos(our_hero, pseudo_actor, self.mirror)
        return [
            float(enemy_visible),
            rel[0],
            rel[1],
            rel[2],
            ratio(stale_frames, 900),
            float((not enemy_visible) and stale_frames <= 180),
        ]

    def _summoner_skill_id(self, hero):
        for slot in ((hero or {}).get("skill_state", {}) or {}).get("slot_states", []) or []:
            if int(get_any(slot, "slot_type", default=-1) or -1) == 6:
                return int(get_any(slot, "configId", "config_id", default=0) or 0)
        return 0

    def _nearest_grass_distance(self, hero):
        if not hero:
            return MAP_SCALE
        hero_pos = actor_pos(hero, self.mirror)
        return min(math.dist(hero_pos, feature_point(point, self.mirror)) for point in GRASS_POINTS)

    def _monster_value_scale(self, money, visible_monster_count):
        if money >= ENDGAME_MONEY:
            money_scale = 0.20
        elif money >= 6000.0:
            money_scale = 0.35
        elif money >= 5500.0:
            money_scale = 0.60
        else:
            money_scale = 1.0
        count_scale = max(0.10, 1.0 / (1.0 + max(visible_monster_count - 1, 0) / 3.0))
        return money_scale * count_scale

    def _fit(self, values, size):
        values = list(values or [])
        if len(values) < size:
            values += [0.0] * (size - len(values))
        return values[:size]

    def _actor_abilities(self, actor):
        abilities = get_any(actor or {}, "abilities", default=None)
        if abilities is None and isinstance(actor, dict):
            abilities = get_any(actor.get("actor_state", {}) or {}, "abilities", default=None)
        return abilities if isinstance(abilities, (list, tuple)) else []

    def _ability_flags(self, actor):
        abilities = self._actor_abilities(actor)

        def flag(index):
            return float(index < len(abilities) and bool(abilities[index]))

        no_control = flag(0)
        no_move = flag(1)
        no_skill = flag(2)
        immune_negative = flag(3)
        immune_control = flag(4)
        no_rotate = flag(5)
        blind = flag(7)
        no_recover_energy = flag(9)
        freeze = flag(10)
        abort_move = flag(14)
        forbid_select = flag(15)
        renewal = flag(16)
        sprint = flag(17)
        no_move_rotate_ok = flag(18)
        repressed = flag(21)
        immune_slow = flag(22)
        hard_control = float(bool(no_control or no_move or no_skill or freeze or abort_move or repressed))
        can_control = 1.0 - max(no_control, freeze, repressed)
        can_move = 1.0 - max(no_move, freeze, abort_move, repressed)
        can_skill = 1.0 - max(no_skill, freeze, repressed)
        can_attack_damage = 1.0 - max(blind, freeze, repressed)
        return [
            no_control,
            no_move,
            no_skill,
            immune_negative,
            immune_control,
            no_rotate,
            blind,
            no_recover_energy,
            freeze,
            abort_move,
            forbid_select,
            renewal,
            sprint,
            no_move_rotate_ok,
            repressed,
            immune_slow,
            hard_control,
            can_control,
            can_move,
            can_skill,
            can_attack_damage,
            ratio(sum(bool(v) for v in abilities), max(len(abilities), 1)),
        ]

    def _ability_control_features(self, our_hero, enemy_hero, main_camp):
        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0
        out = self._ability_flags(our_hero)
        out += self._ability_flags(enemy_hero) if enemy_visible else [0.0] * 22
        out += [
            float(out[18] > 0.5 and out[19] > 0.5),
            float(out[16] > 0.5),
            float(enemy_visible),
            float(enemy_visible and len(self._actor_abilities(enemy_hero)) > 0),
        ]
        return self._fit(out, 64)

    def _unit_entity_slot(self, our_hero, unit, tower, main_camp, category):
        if not our_hero or not unit:
            return [0.0] * 32
        rel = rel_pos(our_hero, unit, self.mirror)
        dist = distance(our_hero, unit)
        tower_dist = distance(unit, tower) if tower else MAP_SCALE
        attack_target = get_any(unit, "attack_target", default=0)
        unit_id = runtime_id(unit)
        tower_target = get_any(tower or {}, "attack_target", default=0)
        config_id = int(get_any(unit, "config_id", "configId", default=0) or 0)
        income = float(get_any(unit, "kill_income", default=0) or 0)
        hp = hp_ratio(unit)
        return self._fit(
            [
                1.0,
                category,
                hp,
                float(hp <= 0.25),
                float(hp <= 0.40),
                rel[0],
                rel[1],
                rel[2],
                ratio(dist, MAP_SCALE),
                ratio(tower_dist, MAP_SCALE),
                float(tower is not None and tower_dist <= float(get_any(tower, "attack_range", default=0) or 0) + 1000.0),
                float(tower_target == unit_id and unit_id != 0),
                float(attack_target == runtime_id(our_hero)),
                float(camp_id(get_any(unit, "camp", default=0)) == main_camp),
                ratio(get_any(unit, "hp", default=0), 10000),
                ratio(get_any(unit, "max_hp", default=0), 10000),
                ratio(income, 400),
                float(config_id in (6802, 6805)),
                float(config_id in (6800, 6803)),
                float(config_id in (6801, 6804)),
                ratio(config_id % 10000, 10000),
                ratio(get_any(unit, "attack_range", default=0), 15000),
                ratio(get_any(unit, "behav_mode", default=0), 32),
            ],
            32,
        )

    def _lane_entity_features(self, our_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, main_camp):
        if not our_hero:
            return [0.0] * 256
        our = [s for s in our_soldiers if is_visible_to(s, main_camp)]
        enemy = [s for s in enemy_soldiers if is_visible_to(s, main_camp)]
        our.sort(key=runtime_id)
        enemy.sort(key=runtime_id)
        out = []
        for soldier in our[:4]:
            out += self._unit_entity_slot(our_hero, soldier, enemy_tower, main_camp, 1.0)
        while len(out) < 4 * 32:
            out += [0.0] * 32
        for soldier in enemy[:4]:
            out += self._unit_entity_slot(our_hero, soldier, our_tower, main_camp, -1.0)
        return self._fit(out, 256)

    def _tower_wave_features(self, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, cakes, main_camp):
        if not our_hero:
            return [0.0] * 96
        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0
        enemy_dead = float(enemy_hero is not None and (hp_ratio(enemy_hero) <= 0 or float(get_any(enemy_hero, "revive_time", default=0) or 0) > 0))
        our_near_enemy_tower = [s for s in our_soldiers if enemy_tower is not None and distance(s, enemy_tower) <= 16000.0]
        enemy_near_our_tower = [s for s in enemy_soldiers if our_tower is not None and distance(s, our_tower) <= 16000.0]
        our_near_hero = [s for s in our_soldiers if distance(our_hero, s) <= 16000.0]
        enemy_near_hero = [s for s in enemy_soldiers if distance(our_hero, s) <= 16000.0]
        safe_push, tower_tanking_minion, enemy_threat = self._safe_push_state(our_hero, enemy_hero, enemy_tower, our_soldiers)
        visible_cakes = [cake for cake in cakes if not (abs(actor_pos(cake)[0]) >= UNSEEN_PADDING or abs(actor_pos(cake)[1]) >= UNSEEN_PADDING)]
        out = [
            ratio(len(our_near_enemy_tower), 6),
            ratio(sum(get_any(s, "hp", default=0) or 0 for s in our_near_enemy_tower), 18000),
            ratio(len(enemy_near_our_tower), 6),
            ratio(sum(get_any(s, "hp", default=0) or 0 for s in enemy_near_our_tower), 18000),
            ratio(len(our_near_hero), 6),
            ratio(sum(get_any(s, "hp", default=0) or 0 for s in our_near_hero), 18000),
            ratio(len(enemy_near_hero), 6),
            ratio(sum(get_any(s, "hp", default=0) or 0 for s in enemy_near_hero), 18000),
            safe_push,
            tower_tanking_minion,
            enemy_threat,
            enemy_dead,
            ratio(get_any(enemy_hero or {}, "revive_time", default=0), 600),
            hp_ratio(our_tower),
            hp_ratio(enemy_tower),
            float(our_tower is not None and get_any(our_tower, "attack_target", default=0) == runtime_id(our_hero)),
            float(enemy_tower is not None and get_any(enemy_tower, "attack_target", default=0) == runtime_id(our_hero)),
            float(bool(enemy_near_our_tower) and hp_ratio(our_tower) <= 0.55),
            float(bool(our_near_enemy_tower) and hp_ratio(enemy_tower) <= 0.55),
            float(enemy_visible),
            ratio(len(visible_cakes), 2),
        ]
        out += self._tower_target_profile(our_tower, our_hero, enemy_hero, our_soldiers, enemy_soldiers)
        out += self._tower_target_profile(enemy_tower, our_hero, enemy_hero, our_soldiers, enemy_soldiers)
        out += self._soldier_type_counts(our_soldiers)
        out += self._soldier_type_counts(enemy_soldiers)
        out += self._wave_midpoint_features(our_hero, our_soldiers, enemy_soldiers)
        return self._fit(out, 96)

    def _tower_target_profile(self, tower, our_hero, enemy_hero, our_soldiers, enemy_soldiers):
        target_id = get_any(tower or {}, "attack_target", default=0)
        if not tower or not target_id:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        target = None
        target_type = 3
        if our_hero is not None and target_id == runtime_id(our_hero):
            target = our_hero
            target_type = 1
        elif enemy_hero is not None and target_id == runtime_id(enemy_hero):
            target = enemy_hero
            target_type = 1
        else:
            soldiers = list(our_soldiers or []) + list(enemy_soldiers or [])
            for soldier in soldiers:
                if target_id == runtime_id(soldier):
                    target = soldier
                    target_type = 2
                    break

        return [
            0.0,
            float(target_type == 1),
            float(target_type == 2),
            float(target_type == 3),
            hp_ratio(target) if target else 0.0,
            float(target is not None and hp_ratio(target) <= 0.35),
            ratio(distance(tower, target), MAP_SCALE) if target else 0.0,
            ratio(get_any(tower, "attack_range", default=0), 15000),
        ]

    def _soldier_type_counts(self, soldiers):
        melee = 0
        ranged = 0
        cannon = 0
        unknown = 0
        for soldier in soldiers or []:
            cfg = config_id(soldier)
            if cfg in (6800, 6803):
                melee += 1
            elif cfg in (6801, 6804):
                ranged += 1
            elif cfg in (6802, 6805):
                cannon += 1
            else:
                unknown += 1
        return [ratio(melee, 6), ratio(ranged, 6), ratio(cannon, 3), ratio(unknown, 6)]

    def _wave_midpoint_features(self, our_hero, our_soldiers, enemy_soldiers):
        if not our_hero:
            return [0.0] * 8

        def center(units):
            if not units:
                return None
            xs, zs = zip(*(actor_pos(unit) for unit in units))
            return {"location": {"x": sum(xs) / len(xs), "z": sum(zs) / len(zs)}}

        our_center = center(our_soldiers)
        enemy_center = center(enemy_soldiers)
        our_rel = rel_pos(our_hero, our_center, self.mirror) if our_center else [0.0, 0.0, 1.0]
        enemy_rel = rel_pos(our_hero, enemy_center, self.mirror) if enemy_center else [0.0, 0.0, 1.0]
        return [
            our_rel[0],
            our_rel[1],
            our_rel[2],
            enemy_rel[0],
            enemy_rel[1],
            enemy_rel[2],
            clip((len(our_soldiers or []) - len(enemy_soldiers or [])) / 6.0),
            clip(
                (
                    sum(hp_ratio(s) for s in (our_soldiers or []))
                    - sum(hp_ratio(s) for s in (enemy_soldiers or []))
                )
                / 6.0
            ),
        ]

    def _projectile_detail_slot(self, our_hero, bullet, enemy_hero):
        if not our_hero or not bullet:
            return [0.0] * 16
        rel = rel_pos(our_hero, bullet, self.mirror)
        source = int_value(get_any(bullet, "source_actor", default=0))
        skill_id = int_value(get_any(bullet, "skill_id", "skillId", default=0))
        slot_type = slot_type_value(get_any(bullet, "slot_type", default=0))
        enemy_id = runtime_id(enemy_hero or {})
        return self._fit(
            [
                1.0,
                rel[0],
                rel[1],
                rel[2],
                ratio(get_any(bullet, "speed", default=0), 30000),
                ratio(slot_type, 20),
                ratio(skill_id % 100000, 100000),
                float(source == enemy_id and enemy_id != 0),
                float(slot_type == 3),
                float(slot_type == 16),
                float("112" in str(skill_id)),
            ],
            16,
        )

    def _projectile_detail_features(self, our_hero, bullets, enemy_hero, main_camp):
        enemy_bullets = [b for b in bullets if self._bullet_is_enemy(b, main_camp)]
        if our_hero:
            enemy_bullets.sort(key=lambda b: distance(our_hero, b))
        out = []
        for bullet in enemy_bullets[:10]:
            out += self._projectile_detail_slot(our_hero, bullet, enemy_hero)
        return self._fit(out, 160)

    def _bucket_pair(self, value, bins):
        value = float(value)
        out = [0.0] * bins
        idx = int(clip(value, 0.0, 0.999999) * bins)
        out[min(max(idx, 0), bins - 1)] = 1.0
        return out

    def _position_bucket_features(self, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers):
        if not our_hero:
            return [0.0] * 64
        targets = [enemy_hero, our_tower, enemy_tower]
        nearest_enemy_soldier = min(enemy_soldiers, key=lambda s: distance(our_hero, s), default=None)
        nearest_our_soldier = min(our_soldiers, key=lambda s: distance(our_hero, s), default=None)
        targets += [nearest_enemy_soldier, nearest_our_soldier]
        out = []
        for target in targets:
            if target is None:
                out += [0.0] * 12
                continue
            rel = rel_pos(our_hero, target, self.mirror)
            out += self._bucket_pair((rel[0] + 1.0) / 2.0, 4)
            out += self._bucket_pair((rel[1] + 1.0) / 2.0, 4)
            out += self._bucket_pair(rel[2], 4)
        return self._fit(out, 64)

    def _buff_mark_detail_features(self, our_hero, enemy_hero, main_camp):
        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0

        def pack(actor):
            buff_state = (actor or {}).get("buff_state", {}) or {}
            skills = buff_state.get("buff_skills", []) or []
            marks = buff_state.get("buff_marks", []) or []
            ids = [int(get_any(item, "configId", "config_id", default=0) or 0) for item in (skills + marks)[:8]]
            layers = [float(get_any(item, "layer", "count", "times", default=0) or 0) for item in marks[:4]]
            all_ids = [int_value(get_any(item, "configId", "config_id", default=0)) for item in skills + marks]
            out = [
                ratio(len(skills), 12),
                ratio(len(marks), 12),
                float(self._has_buff(actor, LUBAN_ENHANCED_ATTACK_BUFF_ID)),
                float(self._has_any_buff(actor, LUBAN_OBSERVED_EXTRA_BUFF_IDS)),
                float(self._has_any_buff(actor, DIRENJIE_OBSERVED_BUFF_IDS)),
                ratio(sum(1 for buff_id in all_ids if buff_id in KNOWN_BUFF_IDS), 12),
                float(self._has_any_buff(actor, SPRING_RECOVER_BUFF_IDS)),
                float(self._has_any_buff(actor, RECOVER_BUFF_IDS)),
                float(self._has_any_buff(actor, MOVE_SPEED_BUFF_IDS)),
                float(self._has_any_buff(actor, CLEANSE_BUFF_IDS)),
                ratio(sum(1 for buff_id in all_ids if buff_id in LUBAN_OBSERVED_BUFF_IDS), 8),
                ratio(sum(1 for buff_id in all_ids if buff_id in DIRENJIE_OBSERVED_ALL_BUFF_IDS), 8),
            ]
            out += [ratio(item % 100000, 100000) for item in ids]
            out += [ratio(v, 8) for v in layers]
            return self._fit(out, 32)

        return self._fit(pack(our_hero) + (pack(enemy_hero) if enemy_visible else [0.0] * 32), 64)

    def _economy_phase_features(self, our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, monsters, cakes, main_camp):
        if not our_hero:
            return [0.0] * 36
        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp) and hp_ratio(enemy_hero) > 0
        money = float(get_any(our_hero, "money_cnt", "money", default=0) or 0)
        enemy_money = float(get_any(enemy_hero or {}, "money_cnt", "money", default=0) or 0) if enemy_visible else 0.0
        level = float(get_any(our_hero, "level", default=1) or 1)
        enemy_level = float(get_any(enemy_hero or {}, "level", default=1) or 1) if enemy_visible else 1.0
        fight_score = self._fight_score(our_hero, enemy_hero, our_tower, enemy_tower, our_soldiers, enemy_soldiers, main_camp)
        visible_monsters = [m for m in monsters if not (abs(actor_pos(m)[0]) >= UNSEEN_PADDING or abs(actor_pos(m)[1]) >= UNSEEN_PADDING)]
        return self._fit(
            [
                ratio(money, 14000),
                ratio(enemy_money, 14000),
                clip((money - enemy_money) / 4000.0),
                ratio(level, 15),
                ratio(enemy_level, 15),
                clip((level - enemy_level) / 4.0),
                float(money >= 4000),
                float(money >= 6000),
                float(money >= 8000),
                float(money >= 10000),
                clip(fight_score),
                float(fight_score >= 0.0),
                float(fight_score >= 0.25),
                float(fight_score >= 0.45),
                float(fight_score <= -0.25),
                ratio(len(visible_monsters), 4),
                self._monster_value_scale(money, len(visible_monsters)),
                ratio(len(cakes), 2),
                hp_ratio(our_tower),
                hp_ratio(enemy_tower),
                ratio(len(our_soldiers), 8),
                ratio(len(enemy_soldiers), 8),
            ],
            36,
        )

    def _target_entity_slot(self, our_hero, actor, target_kind, main_camp, tower=None):
        if target_kind == 0:
            return [0.0] * 32
        if not our_hero or not actor:
            return [0.0] * 32
        visible = float(not (abs(actor_pos(actor)[0]) >= UNSEEN_PADDING or abs(actor_pos(actor)[1]) >= UNSEEN_PADDING))
        rel = rel_pos(our_hero, actor, self.mirror)
        actor_hp = hp_ratio(actor)
        dist = distance(our_hero, actor)
        attack_range = float(get_any(our_hero, "attack_range", default=0) or 0)
        tower_target = get_any(tower or {}, "attack_target", default=0)
        actor_id = runtime_id(actor)
        return self._fit(
            [
                1.0,
                target_kind / 8.0,
                visible,
                actor_hp,
                float(actor_hp <= 0.25),
                float(actor_hp <= 0.40),
                rel[0],
                rel[1],
                rel[2],
                ratio(dist, MAP_SCALE),
                float(dist <= attack_range + 1500.0),
                float(tower_target == actor_id and actor_id != 0),
                float(get_any(actor, "attack_target", default=0) == runtime_id(our_hero)),
                ratio(get_any(actor, "kill_income", default=0), 400),
                ratio(get_any(actor, "attack_range", default=0), 15000),
                ratio(get_any(actor, "config_id", "configId", default=0) % 100000, 100000),
                float(camp_id(get_any(actor, "camp", default=0)) == main_camp),
            ],
            32,
        )

    def _target_entity_features(self, our_hero, enemy_hero, enemy_tower, enemy_soldiers, monsters, main_camp):
        out = []
        out += self._target_entity_slot(our_hero, None, 0, main_camp)
        out += self._target_entity_slot(our_hero, enemy_hero, 1, main_camp)
        out += self._target_entity_slot(our_hero, our_hero, 2, main_camp)
        soldiers = [s for s in enemy_soldiers if is_visible_to(s, main_camp)]
        if our_hero:
            soldiers.sort(key=runtime_id)
        for index, soldier in enumerate(soldiers[:4]):
            out += self._target_entity_slot(our_hero, soldier, 3 + index, main_camp, tower=enemy_tower)
        while len(out) < 7 * 32:
            out += [0.0] * 32
        out += self._target_entity_slot(our_hero, enemy_tower, 7, main_camp)
        visible_monsters = [m for m in monsters if not (abs(actor_pos(m)[0]) >= UNSEEN_PADDING or abs(actor_pos(m)[1]) >= UNSEEN_PADDING)]
        if our_hero:
            visible_monsters.sort(key=lambda m: distance(our_hero, m))
        out += self._target_entity_slot(our_hero, visible_monsters[0] if visible_monsters else None, 8, main_camp)
        return self._fit(out, Config.TARGET_FEATURE_DIM)

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

    def _has_buff(self, actor, buff_id):
        buff_state = (actor or {}).get("buff_state", {}) or {}
        for buff in (buff_state.get("buff_skills", []) or []) + (buff_state.get("buff_marks", []) or []):
            if int_value(get_any(buff, "configId", "config_id", default=0)) == int(buff_id):
                return True
        return False

    def _has_any_buff(self, actor, buff_ids):
        return any(self._has_buff(actor, buff_id) for buff_id in buff_ids)

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
        abilities = self._actor_abilities(hero)
        if not isinstance(abilities, (list, tuple)) or not abilities:
            return [0.0] * 5
        no_move = float(len(abilities) > 1 and bool(abilities[1]))
        no_skill = float(len(abilities) > 2 and bool(abilities[2]))
        blind = float(len(abilities) > 7 and bool(abilities[7]))
        hard_control = float(any(index < len(abilities) and bool(abilities[index]) for index in (0, 1, 2, 10, 14, 21)))
        return [
            hard_control,
            1.0 - no_move,
            1.0 - no_skill,
            1.0 - blind,
            ratio(sum(bool(v) for v in abilities), len(abilities)),
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
