#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Top1-style handcrafted observation builder for the 2026 flat dict protocol.
"""

import math
import numpy as np
from agent_ppo.conf.conf import Args


UNSEEN_PADDING = 100000


def clip(x, mn, mx):
    return max(min(x, mx), mn)


def fix(x):
    return math.floor(x) if x > 0 else math.ceil(x)


def get_any(d, *keys, default=None):
    cur = d
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            return cur[key]
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


def sub_type_id(actor):
    return get_any(actor, "sub_type", default="")


def is_tower(actor):
    st = sub_type_id(actor)
    return st == 21 or st == "ACTOR_SUB_TOWER"


def is_soldier(actor):
    st = sub_type_id(actor)
    return st == 1 or st == "ACTOR_SUB_SOLDIER"


def is_monster(actor):
    if is_soldier(actor) or is_tower(actor):
        return False
    at = get_any(actor, "actor_type", default="")
    return at in (2, "ACTOR_TYPE_MONSTER", "ACTOR_MONSTER") or get_any(actor, "config_id", default=-1) == 6827


def pos2(actor_or_pos, mirror=False):
    pos = actor_or_pos
    if isinstance(actor_or_pos, dict) and "location" in actor_or_pos:
        pos = actor_or_pos["location"]
    if isinstance(pos, dict):
        x = get_any(pos, "x", default=UNSEEN_PADDING)
        z = get_any(pos, "z", default=UNSEEN_PADDING)
    elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
        x, z = pos[0], pos[2]
    else:
        x, z = UNSEEN_PADDING, UNSEEN_PADDING
    if mirror and x != UNSEEN_PADDING:
        x, z = -x, -z
    return [float(x), float(z)]


def hp(actor):
    return float(get_any(actor, "hp", default=0) or 0)


def max_hp(actor):
    return max(float(get_any(actor, "max_hp", default=1) or 1), 1.0)


def runtime_id(actor):
    return get_any(actor, "runtime_id", "player_id", default=0)


class Top1FeatureBuilder:
    def __init__(self, logger=None):
        self.logger = logger
        self.reset()

    def reset(self):
        self.last_money = [None, None]
        self.next_cake_frame = [1778, 1778]
        self.id2type = {}
        self.main_camp = 1
        self.mirror = False
        self.pos = [0.0, 0.0]

    def build_observation(self, observation):
        frame_state = observation["frame_state"]
        self.n_frame = get_any(frame_state, "frame_no", "frameNo", default=0)
        self.main_camp = camp_id(get_any(observation, "camp", "player_camp", default=1))
        self.mirror = self.main_camp == 2

        heroes = frame_state.get("hero_states", [])
        npcs = frame_state.get("npc_states", [])
        bullets = frame_state.get("bullets", []) or []
        cakes = frame_state.get("cakes", []) or []

        our_hero, enemy_hero = self.split_heroes(heroes)
        our_tower, enemy_tower = self.split_towers(npcs)
        our_soldiers, enemy_soldiers, monsters = self.split_npcs(npcs)

        self.id2type.clear()
        for actor, typ in [(our_hero, "hero"), (enemy_hero, "hero"), (our_tower, "organ"), (enemy_tower, "organ")]:
            if actor:
                self.id2type[runtime_id(actor)] = typ
        for soldier in our_soldiers + enemy_soldiers:
            self.id2type[runtime_id(soldier)] = "soldier"
        for monster in monsters:
            self.id2type[runtime_id(monster)] = "monster"

        self.pos = pos2(our_hero or {}, self.mirror)
        if self.last_money[0] is None:
            self.last_money = [self.money_total(our_hero), self.money_total(enemy_hero)]

        x_hero_our = self.process_hero(our_hero, False, enemy_tower)
        x_hero_enemy = self.process_hero(enemy_hero, True, our_tower)
        x_soldier_our = self.process_soldiers(our_soldiers, enemy_tower)
        x_soldier_enemy = self.process_soldiers(enemy_soldiers, our_tower)
        x_monster = self.process_monster(monsters)
        x_tower_our = self.process_tower(our_tower, cakes, False)
        x_tower_enemy = self.process_tower(enemy_tower, cakes, True)
        x_bullets = self.process_bullets(bullets)

        feature = np.array(
            x_hero_our
            + x_hero_enemy
            + x_soldier_our
            + x_soldier_enemy
            + x_monster
            + x_tower_our
            + x_tower_enemy
            + x_bullets,
            dtype=np.float32,
        )
        assert len(feature) == Args.DIM_ALL, f"feature dim mismatch: {len(feature)} != {Args.DIM_ALL}"
        return feature

    def split_heroes(self, heroes):
        our_hero, enemy_hero = None, None
        for hero in heroes:
            if camp_id(hero.get("camp")) == self.main_camp:
                our_hero = hero
            else:
                enemy_hero = hero
        return our_hero, enemy_hero

    def split_towers(self, npcs):
        our_tower, enemy_tower = None, None
        for npc in npcs:
            if not is_tower(npc):
                continue
            if camp_id(npc.get("camp")) == self.main_camp:
                our_tower = npc
            else:
                enemy_tower = npc
        return our_tower, enemy_tower

    def split_npcs(self, npcs):
        our_soldiers, enemy_soldiers, monsters = [], [], []
        for npc in npcs:
            if is_soldier(npc):
                if camp_id(npc.get("camp")) == self.main_camp:
                    our_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
            elif is_monster(npc):
                monsters.append(npc)
        return our_soldiers, enemy_soldiers, monsters

    def process_position(self, position):
        if position[0] == UNSEEN_PADDING:
            return [0.0] * Args.DIM_DISTANCE

        unit_size, max_size = Args.RELATIVE_DISTANCE_UNIT_SIZE, Args.RELATIVE_DISTANCE_MAX_SIZE
        max_idx = max_size / unit_size + 1
        max_idx = max_idx / 2
        rpos = [int(clip(fix((position[i] - self.pos[i]) / unit_size), -max_idx, max_idx) + max_idx) for i in range(2)]
        rpos_dim = int(2 * max_idx + 1)
        x_dis = clip(math.dist(position, self.pos) / (max_size / 2), 0, 1)
        x_rpos = [0.0] * (rpos_dim * 2 + 1)
        x_rpos[rpos[0]] = 1.0
        x_rpos[rpos[1] + rpos_dim] = 1.0
        x_rpos[-1] = x_dis

        unit_size = Args.WHOLE_DISTANCE_UNIT_SIZE
        max_size = Args.WHOLE_DISTANCE_MAX_SIZE
        wpos_dim = int(max_size / unit_size)
        wpos = [int(clip(math.floor((position[i] + max_size / 2) / unit_size), 0, wpos_dim - 1)) for i in range(2)]
        x_wratio = [clip((position[i] + max_size / 2) / unit_size - wpos[i], -1, 1) for i in range(2)]
        x_wpos = [0.0] * (wpos_dim * 2 + 2)
        x_wpos[wpos[0]] = 1.0
        x_wpos[wpos[1] + wpos_dim] = 1.0
        x_wpos[-2] = x_wratio[0]
        x_wpos[-1] = x_wratio[1]
        return x_rpos + x_wpos

    def process_unit(self, actor):
        if not actor:
            return [0.0] * Args.DIM_UNIT
        x_pos = self.process_position(pos2(actor, self.mirror))
        hp_dim = int(Args.HP_MAX_SIZE / Args.HP_UNIT_SIZE) + 2
        x_hp = [0.0] * (1 + hp_dim)
        x_hp[0] = hp(actor) / max_hp(actor)
        x_hp[1 + min(int(math.ceil(hp(actor) / Args.HP_UNIT_SIZE)), hp_dim - 1)] = 1.0

        x_mark = [0.0] * Args.DIM_MARK
        buff_state = actor.get("buff_state", {}) or {}
        marks = buff_state.get("buff_marks", []) or []
        mark_ids = [get_any(mark, "configId", "config_id", default=-1) for mark in marks]
        mark_layers = [get_any(mark, "layer", default=0) for mark in marks]
        now_feature_idx = 0
        used = 0
        for mark_id, max_layer in Args.MARK_ID_LAYERS.items():
            if mark_id in mark_ids:
                index = mark_ids.index(mark_id)
                layer = int(clip(mark_layers[index], 0, max_layer))
                x_mark[now_feature_idx + layer] = 1.0
                used += 1
            now_feature_idx += max_layer + 1
        if used < len(mark_ids):
            x_mark[-1] = 1.0
        return x_hp + x_mark + x_pos

    def skill_slots(self, hero):
        slots = get_any(hero or {}, "skill_state", default={}) or {}
        slots = slots.get("slot_states", []) if isinstance(slots, dict) else []
        while len(slots) < 7:
            slots.append({})
        return slots

    def process_skill(self, slot):
        cd_dim = int(Args.CD_MAX_SIZE / Args.CD_UNIT_SIZE) + 2
        x_cd = [0.0] * (2 + cd_dim)
        cd = float(get_any(slot, "cooldown", default=0) or 0)
        cd_max = max(float(get_any(slot, "cooldown_max", default=Args.CD_MAX_SIZE) or Args.CD_MAX_SIZE), 1.0)
        level = int(get_any(slot, "level", default=1) or 0)
        usable = bool(get_any(slot, "usable", default=True))
        x_cd[0] = clip(cd / cd_max, 0, 1)
        x_cd[1 + min(int(math.ceil(cd / Args.CD_UNIT_SIZE)), cd_dim - 1)] = 1.0
        if not usable and level == 0:
            x_cd[-1] = 1.0
        return x_cd

    def money_total(self, hero):
        return float(get_any(hero or {}, "money_cnt", "moneyCnt", "money", default=0) or 0)

    def process_money(self, money, is_enemy):
        idx = int(is_enemy)
        delta = money - self.last_money[idx]
        self.last_money[idx] = money
        money_dim = int(Args.MONEY_MAX_SIZE / Args.MONEY_UNIT_SIZE) + 1
        x_money = [0.0] * (2 + money_dim)
        if 0 < delta < Args.MONEY_UNIT_SIZE:
            x_money[-2] = 1.0
        else:
            x_money[max(min(int(math.floor(delta / Args.MONEY_UNIT_SIZE)), money_dim - 1), 0)] = 1.0
        x_money[-1] = min(money / 10000.0, 1.0)
        return x_money

    def process_hero(self, hero, is_enemy, opposed_tower):
        if not hero:
            return [0.0] * Args.DIM_HERO
        x_unit = self.process_unit(hero)
        config_id = int(get_any(hero, "config_id", default=Args.HERO_CONFIG_ID[0]) or Args.HERO_CONFIG_ID[0])
        x_hero_id = [Args.HERO_CONFIG_ID.index(config_id) * 2 - 1 if config_id in Args.HERO_CONFIG_ID else 0.0]

        behave = get_any(hero, "behav_mode", default="")
        x_behave = [0.0] * (len(Args.HERO_BEHAVE) + 1)
        idx = Args.HERO_BEHAVE.index(behave) if behave in Args.HERO_BEHAVE else len(Args.HERO_BEHAVE)
        x_behave[idx] = 1.0

        ep = float(get_any(hero, "ep", default=0) or 0)
        max_ep = max(float(get_any(hero, "max_ep", default=1) or 1), 1.0)
        ep_dim = int(Args.EP_MAX_SIZE / Args.EP_UNIT_SIZE) + 1
        x_ep = [0.0] * (1 + ep_dim)
        x_ep[0] = ep / max_ep
        x_ep[1 + min(int(math.floor(ep / Args.EP_UNIT_SIZE)), ep_dim - 1)] = 1.0

        slots = self.skill_slots(hero)
        # 1/2/3 skills plus chosen summoner and heal/recover. If the protocol
        # orders these differently, missing slots degrade to zero/default cd.
        skill_features = []
        for slot in [slots[1], slots[2], slots[3], slots[5], slots[4]]:
            skill_features += self.process_skill(slot)

        level = int(get_any(hero, "level", default=1) or 1)
        x_level = [0.0] * Args.LEVEL_MAX
        x_level[clip(level, 1, Args.LEVEL_MAX) - 1] = 1.0

        x_money = self.process_money(self.money_total(hero), is_enemy)
        x_grass = [float(bool(get_any(hero, "is_in_grass", "isInGrass", default=False)))]
        x_tower = [0.0, 0.0]
        if opposed_tower:
            x_tower[0] = float(math.dist(pos2(hero, self.mirror), pos2(opposed_tower, self.mirror)) <= float(get_any(opposed_tower, "attack_range", default=0) or 0))
            x_tower[1] = float(runtime_id(hero) == get_any(opposed_tower, "attack_target", default=-1))

        x_buff = [0.0] * Args.DIM_BUFF
        buff_state = hero.get("buff_state", {}) or {}
        for buff in buff_state.get("buff_skills", []) or []:
            buff_id = int(get_any(buff, "configId", "config_id", default=-1) or -1)
            if buff_id in Args.COMMON_BUFFS:
                x_buff[Args.COMMON_BUFFS.index(buff_id)] = 1.0
            elif buff_id > 0 and buff_id // 1000 in Args.HERO_BUFF_PREFIXES:
                bucket = len(Args.COMMON_BUFFS) + min((buff_id // 10) % 12, 11)
                x_buff[bucket] = 1.0
            else:
                x_buff[-1] = 1.0

        return x_hero_id + x_behave + x_ep + skill_features + x_level + x_money + x_grass + x_tower + x_buff + x_unit

    def process_soldiers(self, soldiers, opposed_tower):
        soldiers = sorted(soldiers, key=lambda s: math.dist(pos2(s, self.mirror), self.pos))[: Args.SOLDIER_MAX_NUM]
        x_soldiers = []
        for soldier in soldiers:
            x = [0.0] * (Args.DIM_SOLDIER - Args.DIM_UNIT)
            behave = get_any(soldier, "behav_mode", default="")
            idx = Args.SOLDIER_BEHAVE.index(behave) if behave in Args.SOLDIER_BEHAVE else len(Args.SOLDIER_BEHAVE)
            x[idx] = 1.0
            base = len(Args.SOLDIER_BEHAVE) + 1
            config_id = int(get_any(soldier, "config_id", default=-1) or -1)
            for j, ids in enumerate(Args.SOLDIER_CONFIG_ID):
                if config_id in ids:
                    x[base + j] = 1.0
            base += len(Args.SOLDIER_CONFIG_ID)
            if opposed_tower:
                x[base] = float(math.dist(pos2(soldier, self.mirror), pos2(opposed_tower, self.mirror)) <= float(get_any(opposed_tower, "attack_range", default=0) or 0))
                x[base + 1] = float(runtime_id(soldier) == get_any(opposed_tower, "attack_target", default=-1))
            x_soldiers += x + self.process_unit(soldier)
        x_soldiers += [0.0] * (Args.DIM_SOLDIERS - len(x_soldiers))
        return x_soldiers

    def process_monster(self, monsters):
        if not monsters:
            return [0.0] * Args.DIM_MONSTER
        monster = sorted(monsters, key=lambda s: math.dist(pos2(s, self.mirror), self.pos))[0]
        x_behave = [0.0] * (len(Args.MONSTER_BEHAVE) + 1)
        behave = get_any(monster, "behav_mode", default="")
        idx = Args.MONSTER_BEHAVE.index(behave) if behave in Args.MONSTER_BEHAVE else len(Args.MONSTER_BEHAVE)
        x_behave[idx] = 1.0
        return x_behave + self.process_unit(monster)

    def process_tower(self, tower, cakes, is_enemy):
        if not tower:
            return [0.0] * Args.DIM_ORGAN
        x_target = [0.0] * 5
        target = get_any(tower, "attack_target", default=0)
        if not target:
            x_target[0] = 1.0
        elif self.id2type.get(target) == "hero":
            x_target[1] = 1.0
        elif self.id2type.get(target) == "soldier":
            x_target[2] = 1.0

        cake = self.nearest_cake(tower, cakes)
        idx = int(is_enemy)
        x_target[3] = float(cake is not None)
        if cake is not None:
            self.next_cake_frame[idx] = self.n_frame + 76 * 30
            x_target[4] = 0.0
        else:
            x_target[4] = min((self.next_cake_frame[idx] - self.n_frame) / (75 * 30), 1.0)
        return x_target + self.process_unit(tower)

    def nearest_cake(self, tower, cakes):
        if not cakes:
            return None
        tower_pos = pos2(tower, False)
        best = None
        best_dist = 10**18
        for cake in cakes:
            collider = cake.get("collider", {}) if isinstance(cake, dict) else {}
            cake_pos = pos2(collider.get("location", {}), False)
            dist = math.dist(cake_pos, tower_pos)
            if dist < best_dist:
                best, best_dist = cake, dist
        return best if best_dist < 12000 else None

    def process_bullet(self, bullet):
        x_slot = [0.0] * (Args.DIM_BULLET - Args.DIM_DISTANCE)
        slot = get_any(bullet, "slot_type", default="other")
        if isinstance(slot, str) and slot.startswith("SLOT_SKILL_"):
            suffix = slot.rsplit("_", 1)[-1]
            slot = int(suffix) if suffix.isdigit() else "other"
        idx = Args.BULLET_SLOT.index(slot) if slot in Args.BULLET_SLOT else len(Args.BULLET_SLOT) - 1
        x_slot[idx] = 1.0
        return x_slot + self.process_position(pos2(bullet, self.mirror))

    def process_bullets(self, bullets):
        enemy_camp = 1 if self.main_camp == 2 else 2
        enemy_bullets = [b for b in bullets if camp_id(b.get("camp")) == enemy_camp]
        hero_bullets, tower_bullets = [], []
        for bullet in enemy_bullets:
            source_type = self.id2type.get(get_any(bullet, "source_actor", default=-1), "")
            if source_type == "organ":
                tower_bullets.append(bullet)
            else:
                hero_bullets.append(bullet)

        hero_bullets = sorted(hero_bullets, key=lambda b: math.dist(pos2(b, self.mirror), self.pos))[:9]
        x_bullets = []
        for bullet in hero_bullets:
            x_bullets += self.process_bullet(bullet)
        x_bullets += [0.0] * (9 * Args.DIM_BULLET - len(x_bullets))

        tower_bullets = sorted(tower_bullets, key=lambda b: math.dist(pos2(b, self.mirror), self.pos))[:1]
        for bullet in tower_bullets:
            x_bullets += self.process_bullet(bullet)
        x_bullets += [0.0] * (Args.DIM_BULLETS - len(x_bullets))
        return x_bullets
