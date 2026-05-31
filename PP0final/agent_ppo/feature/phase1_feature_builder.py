#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Phase 1 observation builder for the 2026 flat frame_state schema."""

import math
import numpy as np

from agent_ppo.conf.conf import Config


MAP_SCALE = 100000.0
UNSEEN_PADDING = 100000
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
HERO_IDS = (112, 133)
HERO_BEHAV_MODES = (0, 1, 2, 4, 9, 23)
SOLDIER_BEHAV_MODES = (0, 1, 5, 6, 23)
BULLET_SLOT_TYPES = (0, 1, 2, 3, 5, 6, 7)
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
CAKE_POINTS = [(-15000.0, -15000.0), (15000.0, 15000.0)]


def clip(value, low=-1.0, high=1.0):
    return max(low, min(high, float(value)))


def ratio(value, denom):
    denom = float(denom or 0)
    if denom <= 0:
        return 0.0
    return clip(float(value or 0) / denom, 0.0, 1.0)


def int_value(value, default=0):
    try:
        return int(value or default)
    except (TypeError, ValueError):
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
        loc = ((actor.get("actor_state", {}) or {}).get("location", {}) or {})
    if not loc and isinstance(actor, dict):
        loc = ((actor.get("collider", {}) or {}).get("location", {}) or {})
    if isinstance(loc, dict):
        x, z = float(loc.get("x", 0) or 0), float(loc.get("z", 0) or 0)
    elif isinstance(loc, (list, tuple)) and len(loc) >= 3:
        x, z = float(loc[0] or 0), float(loc[2] or 0)
    else:
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
    if not src or not dst:
        return [0.0, 0.0, 0.0]
    sx, sz = actor_pos(src, mirror)
    dx, dz = actor_pos(dst, mirror)
    if abs(dx) >= UNSEEN_PADDING or abs(dz) >= UNSEEN_PADDING:
        return [0.0, 0.0, 0.0]
    rx, rz = dx - sx, dz - sz
    return [clip(rx / MAP_SCALE), clip(rz / MAP_SCALE), ratio(math.hypot(rx, rz), MAP_SCALE)]


def rel_point(src, point, mirror=False):
    if not src:
        return [0.0, 0.0, 0.0]
    sx, sz = actor_pos(src, mirror)
    dx, dz = point
    if mirror:
        dx, dz = -dx, -dz
    rx, rz = dx - sx, dz - sz
    return [clip(rx / MAP_SCALE), clip(rz / MAP_SCALE), ratio(math.hypot(rx, rz), MAP_SCALE)]


def distance(a, b):
    return math.dist(actor_pos(a), actor_pos(b))


def hp_ratio(actor):
    return ratio(get_any(actor or {}, "hp", default=0), get_any(actor or {}, "max_hp", default=1))


def ep_ratio(actor):
    return ratio(get_any(actor or {}, "ep", default=0), get_any(actor or {}, "max_ep", default=1))


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
    if is_neutral_camp(get_any(actor, "camp", default=None)):
        return False
    return get_any(actor, "actor_type", default=None) == ACTOR_TYPE_MONSTER and (
        get_any(actor, "sub_type", default=None) in (SUB_TYPE_LANE_SOLDIER, "ACTOR_SUB_SOLDIER")
        or config_id(actor) in LANE_SOLDIER_CONFIG_IDS
    )


def is_organ(actor):
    return get_any(actor, "actor_type", default=None) == ACTOR_TYPE_ORGAN and (
        get_any(actor, "sub_type", default=None)
        in (SUB_TYPE_TOWER, SUB_TYPE_SPRING_TOWER, SUB_TYPE_CRYSTAL, "ACTOR_SUB_TOWER")
        or config_id(actor) in TOWER_CONFIG_IDS
        or config_id(actor) in CRYSTAL_CONFIG_IDS
        or config_id(actor) in SPRING_TOWER_CONFIG_IDS
    )


def is_monster(actor):
    return (
        config_id(actor) == RIVER_SPIRIT_CONFIG_ID
        or get_any(actor, "actor_type", default=None) in (3, "ACTOR_TYPE_MONSTER")
        or get_any(actor, "sub_type", default=None) in ("ACTOR_SUB_MONSTER", "ACTOR_SUB_NEUTRAL_MONSTER")
        or (
            is_neutral_camp(get_any(actor, "camp", default=None))
            and get_any(actor, "actor_type", default=None) == ACTOR_TYPE_MONSTER
            and get_any(actor, "sub_type", default=None) == SUB_TYPE_NEUTRAL_MONSTER
        )
    )


class Phase1FeatureBuilder:
    FEATURE_DIM = Config.FEATURE_DIM

    def __init__(self, logger=None):
        self.logger = logger
        self.mirror = False
        self.last_values = {}

    def reset(self):
        self.last_values.clear()

    def build_observation(self, observation):
        frame_state = observation.get("frame_state", {})
        main_camp = camp_id(observation.get("camp", observation.get("player_camp", 1)))
        self.mirror = main_camp == 2
        player_id = observation.get("player_id", 0)

        heroes = frame_state.get("hero_states", []) or []
        npcs = frame_state.get("npc_states", []) or []
        bullets = frame_state.get("bullets", []) or []
        cakes = frame_state.get("cakes", []) or []
        frame_no = int_value(get_any(frame_state, "frame_no", "frameNo", default=0))

        our_hero, enemy_hero = self._split_heroes(heroes, main_camp, player_id)
        our_soldiers, enemy_soldiers = self._split_soldiers(npcs, main_camp)
        our_organs, enemy_organs = self._split_organs(npcs, main_camp)
        monsters = [npc for npc in npcs if is_monster(npc) and hp_ratio(npc) > 0]
        hero_ids = {runtime_id(hero) for hero in heroes}
        organ_ids = {runtime_id(organ) for organ in our_organs + enemy_organs}

        feature = []
        feature += self._hero_block(our_hero, our_hero, main_camp, hero_ids, organ_ids, own=True)
        feature += self._hero_block(enemy_hero, our_hero, main_camp, hero_ids, organ_ids, own=False)
        feature += self._soldier_section(
            our_hero, our_soldiers, enemy_soldiers, main_camp, our_organs, enemy_organs, hero_ids, organ_ids
        )
        feature += self._organ_section(our_hero, our_organs, enemy_organs, our_soldiers, enemy_soldiers, hero_ids, organ_ids)
        feature += self._river_crab_block(our_hero, monsters)
        feature += self._bullet_section(our_hero, bullets, main_camp)
        feature += self._environment_block(our_hero, enemy_hero, cakes, main_camp)
        feature += [ratio(frame_no, 20000)]

        if len(feature) != self.FEATURE_DIM:
            raise ValueError(f"Phase1 feature expected {self.FEATURE_DIM}, got {len(feature)}")
        return np.array(feature, dtype=np.float32)

    def _fit(self, values, size):
        values = list(values or [])
        if len(values) < size:
            values += [0.0] * (size - len(values))
        return values[:size]

    def _one_hot(self, value, choices):
        value = int_value(value, default=-999)
        out = [float(value == item) for item in choices]
        out.append(float(value not in choices))
        return out

    def _delta(self, actor, key, scale):
        rid = runtime_id(actor)
        cur = float(get_any(actor or {}, key, default=0) or 0)
        cache_key = (rid, key)
        last = self.last_values.get(cache_key, cur)
        self.last_values[cache_key] = cur
        return clip((cur - last) / float(scale or 1))

    def _split_heroes(self, heroes, main_camp, player_id):
        our_hero, enemy_hero = None, None
        for hero in heroes:
            if runtime_id(hero) == player_id:
                our_hero = hero
            elif camp_id(hero.get("camp")) == main_camp:
                our_hero = hero
            else:
                enemy_hero = hero
        return our_hero, enemy_hero

    def _split_soldiers(self, npcs, main_camp):
        soldiers = [npc for npc in npcs if is_soldier(npc) and hp_ratio(npc) > 0]
        our = [s for s in soldiers if camp_id(s.get("camp")) == main_camp]
        enemy = [s for s in soldiers if camp_id(s.get("camp")) != main_camp]
        our.sort(key=runtime_id)
        enemy.sort(key=runtime_id)
        return our, enemy

    def _split_organs(self, npcs, main_camp):
        organs = [npc for npc in npcs if is_organ(npc)]
        our = [o for o in organs if camp_id(o.get("camp")) == main_camp]
        enemy = [o for o in organs if camp_id(o.get("camp")) != main_camp]
        return self._sort_organs(our), self._sort_organs(enemy)

    def _sort_organs(self, organs):
        def priority(organ):
            sub_type = get_any(organ, "sub_type", default=0)
            cfg = config_id(organ)
            if sub_type == SUB_TYPE_TOWER or cfg in TOWER_CONFIG_IDS:
                return 0
            if sub_type == SUB_TYPE_CRYSTAL or cfg in CRYSTAL_CONFIG_IDS:
                return 1
            if sub_type == SUB_TYPE_SPRING_TOWER or cfg in SPRING_TOWER_CONFIG_IDS:
                return 2
            return 3

        return sorted(organs, key=lambda organ: (priority(organ), runtime_id(organ)))

    def _outer_tower(self, organs):
        for organ in organs or []:
            if get_any(organ, "sub_type", default=None) == SUB_TYPE_TOWER or config_id(organ) in TOWER_CONFIG_IDS:
                return organ
        return organs[0] if organs else None

    def _hit_categories(self, actor, hero_ids, organ_ids):
        hero_hit, organ_hit, other_hit = 0.0, 0.0, 0.0
        for hit in get_any(actor or {}, "hit_target_info", default=[]) or []:
            target = get_any(hit, "hit_target", default=0)
            if target in hero_ids:
                hero_hit = 1.0
            elif target in organ_ids:
                organ_hit = 1.0
            elif target:
                other_hit = 1.0
        return [hero_hit, organ_hit, other_hit]

    def _max_conti_hit_count(self, actor):
        counts = [
            float(get_any(hit, "conti_hit_count", default=0) or 0)
            for hit in (get_any(actor or {}, "hit_target_info", default=[]) or [])
        ]
        return max(counts, default=0.0)

    def _hero_block(self, hero, our_hero, main_camp, hero_ids, organ_ids, own):
        size = 113 if own else 117
        if not hero:
            return [0.0] * size
        visible = own or is_visible_to(hero, main_camp)
        if not visible:
            base = [float(config_id(hero) == 112), float(config_id(hero) == 133), 0.0] + [0.0] * 110
            return self._fit(base + [0.0, 0.0, 0.0, 0.0], size)

        slots = sorted(
            ((hero.get("skill_state", {}) or {}).get("slot_states", []) or []),
            key=lambda s: int_value(get_any(s, "slot_type", default=99)),
        )
        skill_vec = []
        for slot in slots[:7]:
            cd = float(get_any(slot, "cooldown", default=0) or 0)
            cd_max = max(float(get_any(slot, "cooldown_max", default=1) or 1), 1.0)
            skill_vec += [
                float(cd <= 0 and bool(get_any(slot, "usable", default=False))),
                clip(cd / cd_max, 0.0, 1.0),
                ratio(cd_max, 120000),
                ratio(get_any(slot, "cost", "ep_cost", default=0), 200),
                ratio(get_any(slot, "level", default=0), 15),
            ]
        skill_vec = self._fit(skill_vec, 35)

        buff_vec = self._buff_flags(hero)
        taken = [
            ratio(max(0.0, self._delta(hero, "total_be_hurt_by_hero", 500)), 1),
            ratio(max(0.0, self._delta(hero, "total_be_hurt_by_organ", 500)), 1),
            ratio(max(0.0, self._delta(hero, "total_be_hurt_by_monster", 500)), 1),
        ]
        hit_hero, hit_organ, hit_other = self._hit_categories(hero, hero_ids, organ_ids)
        dealt = [
            max(ratio(max(0.0, self._delta(hero, "total_hurt_to_hero", 500)), 1), hit_hero),
            max(ratio(max(0.0, self._delta(hero, "total_hurt_to_organ", 500)), 1), hit_organ),
            max(ratio(max(0.0, self._delta(hero, "total_hurt_to_monster", 500)), 1), hit_other),
        ]
        combat_keys = [
            "phy_atk", "phy_def", "mgc_atk", "mgc_def", "mov_spd", "atk_spd", "crit_rate", "crit_effe",
            "phy_vamp", "mgc_vamp", "cd_reduce", "ctrl_reduce", "phy_armor_hurt", "mgc_armor_hurt",
            "hp_recover", "ep_recover", "sight_area",
        ]
        combat = []
        for key in combat_keys:
            scale = 10000 if key in ("crit_rate", "crit_effe", "phy_vamp", "mgc_vamp", "cd_reduce", "ctrl_reduce") else 2000
            combat += [ratio(get_any(hero, key, default=0), scale), self._delta(hero, key, scale * 0.5)]
        x, z = norm_pos(hero, self.mirror)
        base = [
            float(config_id(hero) == 112),
            float(config_id(hero) == 133),
            float(camp_id(hero.get("camp")) == main_camp),
            x,
            z,
        ]
        base += self._one_hot(get_any(hero, "behav_mode", default=0), HERO_BEHAV_MODES)
        abilities = get_any(hero, "abilities", default=[]) or []
        base += [float(len(abilities) > 0 and bool(abilities[0]))]
        base += [
            hp_ratio(hero),
            self._delta(hero, "hp", 500),
            ep_ratio(hero),
            self._delta(hero, "ep", 200),
            ratio(get_any(hero, "level", default=1), 15),
            self._delta(hero, "level", 1),
            ratio(get_any(hero, "exp", default=0), 10000),
            self._delta(hero, "exp", 500),
            ratio(get_any(hero, "money", default=0), 5000),
            self._delta(hero, "money", 200),
            ratio(get_any(hero, "money_cnt", default=0), 20000),
            self._delta(hero, "money_cnt", 200),
            ratio(get_any(hero, "attack_range", default=0), 13000),
            self._delta(hero, "attack_range", 1000),
        ]
        base += combat + skill_vec + buff_vec + taken + dealt
        if not own:
            base += rel_pos(our_hero, hero, self.mirror) + [float(visible)]
        return self._fit(base, size)

    def _buff_flags(self, hero):
        buff_state = (hero or {}).get("buff_state", {}) or {}
        items = (buff_state.get("buff_skills", []) or []) + (buff_state.get("buff_marks", []) or [])
        ids = {int_value(get_any(item, "configId", "config_id", default=0)) for item in items}
        known = [112045, 133260, 10000, 80102, 11010]
        out = []
        for buff_id in known:
            out += [float(buff_id in ids), 0.0]
        out += [float(bool(buff_state.get("buff_marks", []) or []))]
        return out[:11]

    def _soldier_section(self, our_hero, our_soldiers, enemy_soldiers, main_camp, our_organs, enemy_organs, hero_ids, organ_ids):
        out = []
        enemy_tower = self._outer_tower(enemy_organs)
        own_tower = self._outer_tower(our_organs)
        for soldiers, category in ((our_soldiers, 1.0), (enemy_soldiers, -1.0)):
            group = []
            visible = [s for s in soldiers if is_visible_to(s, main_camp)]
            if our_hero:
                visible.sort(key=lambda s: distance(our_hero, s))
            for soldier in visible[:6]:
                tower = enemy_tower if category > 0 else own_tower
                group += self._soldier_block(our_hero, soldier, tower, hero_ids, organ_ids)
            out += self._fit(group, 6 * 24)
        return out[:288]

    def _soldier_block(self, our_hero, soldier, enemy_tower, hero_ids, organ_ids):
        rel = rel_pos(our_hero, soldier, self.mirror)
        x, z = norm_pos(soldier, self.mirror)
        cfg = config_id(soldier)
        if cfg in (6800, 6803):
            type_vec = [1.0, 0.0, 0.0, 0.0]
        elif cfg in (6801, 6804):
            type_vec = [0.0, 1.0, 0.0, 0.0]
        elif cfg in (6802, 6805):
            type_vec = [0.0, 0.0, 1.0, 0.0]
        else:
            type_vec = [0.0, 0.0, 0.0, 1.0]
        tower_range = float(get_any(enemy_tower or {}, "attack_range", default=0) or 0)
        in_tower = float(enemy_tower is not None and distance(soldier, enemy_tower) <= tower_range)
        target_by_tower = float(enemy_tower is not None and get_any(enemy_tower, "attack_target", default=0) == runtime_id(soldier))
        hit_hero, hit_organ, hit_other = self._hit_categories(soldier, hero_ids, organ_ids)
        return self._fit(
            [
                1.0, x, z, rel[0], rel[1], rel[2], hp_ratio(soldier), self._delta(soldier, "hp", 500),
            ]
            + self._one_hot(get_any(soldier, "behav_mode", default=0), SOLDIER_BEHAV_MODES)
            + type_vec
            + [
                ratio(get_any(soldier, "attack_range", default=0), 13000),
                in_tower,
                target_by_tower,
                max(ratio(max(0.0, self._delta(soldier, "total_hurt_to_hero", 500)), 1), hit_hero),
                max(ratio(max(0.0, self._delta(soldier, "total_hurt_to_organ", 500)), 1), hit_organ),
                max(ratio(max(0.0, self._delta(soldier, "total_hurt_to_monster", 500)), 1), hit_other),
            ],
            24,
        )

    def _organ_section(self, our_hero, our_organs, enemy_organs, our_soldiers, enemy_soldiers, hero_ids, organ_ids):
        out = []
        for organs, soldiers in ((our_organs, enemy_soldiers), (enemy_organs, our_soldiers)):
            group = []
            for organ in organs[:3]:
                group += self._organ_block(our_hero, organ, soldiers, hero_ids, organ_ids)
            out += self._fit(group, 3 * 17)
        return out[:102]

    def _organ_block(self, our_hero, organ, soldiers, hero_ids, organ_ids):
        x, z = norm_pos(organ, self.mirror)
        rel = rel_pos(our_hero, organ, self.mirror)
        target_id = get_any(organ, "attack_target", default=0)
        target_type = [1.0, 0.0, 0.0]
        target_hp = 0.0
        if our_hero and target_id == runtime_id(our_hero):
            target_type = [0.0, 1.0, 0.0]
            target_hp = hp_ratio(our_hero)
        else:
            for soldier in soldiers or []:
                if target_id == runtime_id(soldier):
                    target_type = [0.0, 0.0, 1.0]
                    target_hp = hp_ratio(soldier)
                    break
        hit_hero, hit_organ, hit_other = self._hit_categories(organ, hero_ids, organ_ids)
        return self._fit(
            [
                1.0,
                x,
                z,
                rel[0],
                rel[1],
                rel[2],
                hp_ratio(organ),
                self._delta(organ, "hp", 500),
                ratio(get_any(organ, "attack_range", default=0), 13000),
            ]
            + target_type
            + [
                target_hp,
                ratio(max(get_any(organ, "attack_count", "attack_seq", default=0) or 0, self._max_conti_hit_count(organ)), 30),
                max(ratio(max(0.0, self._delta(organ, "total_hurt_to_hero", 500)), 1), hit_hero),
                max(ratio(max(0.0, self._delta(organ, "total_hurt_to_organ", 500)), 1), hit_organ),
                max(ratio(max(0.0, self._delta(organ, "total_hurt_to_monster", 500)), 1), hit_other),
            ],
            17,
        )

    def _river_crab_block(self, our_hero, monsters):
        visible = [m for m in monsters if config_id(m) == RIVER_SPIRIT_CONFIG_ID or is_neutral_camp(m.get("camp"))]
        if our_hero:
            visible.sort(key=lambda m: distance(our_hero, m))
        monster = visible[0] if visible else None
        if not monster:
            return [0.0] * 14
        x, z = norm_pos(monster, self.mirror)
        rel = rel_pos(our_hero, monster, self.mirror)
        return self._fit(
            [1.0, x, z, rel[0], rel[1], rel[2], hp_ratio(monster), self._delta(monster, "hp", 500)]
            + self._one_hot(get_any(monster, "behav_mode", default=0), SOLDIER_BEHAV_MODES),
            14,
        )

    def _bullet_section(self, our_hero, bullets, main_camp):
        own = []
        enemy = []
        for bullet in bullets:
            bullet_camp = camp_id(get_any(bullet, "camp", default=0))
            if bullet_camp == main_camp:
                own.append(bullet)
            elif not is_neutral_camp(bullet_camp):
                enemy.append(bullet)
        if our_hero:
            own.sort(key=lambda b: distance(our_hero, b))
            enemy.sort(key=lambda b: distance(our_hero, b))
        out = []
        for group, is_own in ((own, 1.0), (enemy, 0.0)):
            packed = []
            for bullet in group[:16]:
                packed += self._bullet_block(our_hero, bullet, is_own)
            out += self._fit(packed, 16 * 15)
        return out[:480]

    def _bullet_block(self, our_hero, bullet, is_own):
        x, z = norm_pos(bullet, self.mirror)
        rel = rel_pos(our_hero, bullet, self.mirror)
        slot = get_any(bullet, "slot_type", default=0)
        return self._fit([1.0, x, z, rel[0], rel[1], rel[2]] + self._one_hot(slot, BULLET_SLOT_TYPES) + [is_own], 15)

    def _environment_block(self, our_hero, enemy_hero, cakes, main_camp):
        visible_cakes = [cake for cake in cakes if abs(actor_pos(cake)[0]) < UNSEEN_PADDING and abs(actor_pos(cake)[1]) < UNSEEN_PADDING]
        out = []
        for point in CAKE_POINTS:
            cake = min(visible_cakes, key=lambda c: math.dist(actor_pos(c), point), default=None)
            present = float(cake is not None and math.dist(actor_pos(cake), point) <= 12000.0)
            rel = rel_point(our_hero, point, self.mirror)
            out += [present, rel[0], rel[1], rel[2], 0.0]
        own_grass_dist = min([math.dist(actor_pos(our_hero, self.mirror), list(point)) for point in GRASS_POINTS], default=MAP_SCALE) if our_hero else MAP_SCALE
        enemy_visible = enemy_hero is not None and is_visible_to(enemy_hero, main_camp)
        enemy_grass_dist = min([math.dist(actor_pos(enemy_hero, self.mirror), list(point)) for point in GRASS_POINTS], default=MAP_SCALE) if enemy_visible else MAP_SCALE
        out += [
            float(bool(get_any(our_hero or {}, "is_in_grass", default=False))),
            float(enemy_visible and bool(get_any(enemy_hero or {}, "is_in_grass", default=False))),
            ratio(own_grass_dist, MAP_SCALE),
            ratio(enemy_grass_dist, MAP_SCALE) if enemy_visible else 0.0,
        ]
        return self._fit(out, 14)
