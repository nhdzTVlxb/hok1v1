#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from enum import Enum
from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math
from collections import OrderedDict


class HeroProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_tower_dict = {}
        self.enemy_camp_tower_dict = {}
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 65
        self.unit_buff_num = 1

    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.hero_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_hero(self, frame_state):

        self.generate_tower_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)

        # Generate hero features for our camp
        # 生成我方阵营的英雄特征
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")

        # Generate hero features for enemy camp
        enemy_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.enemy_camp_hero_dict, "enemy_camp")

        # Return the combined hero features
        return main_camp_hero_vector_feature + enemy_camp_hero_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["actor_state"]["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["actor_state"]["config_id"]] = hero
                self.enemy_hero_info = hero

    def generate_tower_info_dict(self, frame_state):
        self.main_camp_tower_dict.clear()
        self.enemy_camp_tower_dict.clear()

        # Find our towers and number them in order
        # 找到我方塔并按照顺序编号
        for tower in frame_state["npc_states"]:
            if tower["sub_type"] != "ACTOR_SUB_TOWER" or tower["hp"] <= 0:
                continue
            if tower["camp"] == self.main_camp:
                self.main_camp_tower_dict[0] = tower
            else:
                self.enemy_camp_tower_dict[0] = tower

        # Find enemy heroes and number them in order
        # 找到敌方英雄并按照顺序编号
        for hero in frame_state["npc_states"]:
            if hero["sub_type"] != "ACTOR_SUB_hero" or hero["hp"] <= 0:
                continue
            if hero["camp"] != self.main_camp:
                self.enemy_camp_hero_dict[hero["runtime_id"]] = hero
        self.enemy_camp_hero_dict = OrderedDict(sorted(self.enemy_camp_hero_dict.items()))

    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break

            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    value_vec = []
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            self.no_hero_feature(vector_feature, num_heros_considered)
        return vector_feature

    def no_hero_feature(self, vector_feature, num_heros_considered):
        for _ in range((self.unit_buff_num - num_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["actor_state"]["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    # Hero stats methods
    def get_level(self, hero, value, feature_name):
        value.append(hero.get("level", 0))

    def get_exp(self, hero, value, feature_name):
        value.append(hero.get("exp", 0))

    def get_money(self, hero, value, feature_name):
        value.append(hero.get("money", 0))

    def get_revive_time(self, hero, value, feature_name):
        value.append(hero.get("revive_time", 0))

    def get_kill_count(self, hero, value, feature_name):
        value.append(hero.get("killCnt", 0))

    def get_dead_count(self, hero, value, feature_name):
        value.append(hero.get("deadCnt", 0))

    def get_assist_count(self, hero, value, feature_name):
        value.append(hero.get("assistCnt", 0))

    def get_money_count(self, hero, value, feature_name):
        value.append(hero.get("moneyCnt", 0))

    def get_total_hurt(self, hero, value, feature_name):
        value.append(hero.get("totalHurt", 0))

    def get_total_hurt_to_hero(self, hero, value, feature_name):
        value.append(hero.get("totalHurtToHero", 0))

    def get_total_be_hurt_by_hero(self, hero, value, feature_name):
        value.append(hero.get("totalBeHurtByHero", 0))

    # Status flags methods
    def get_is_in_grass(self, hero, value, feature_name):
        value.append(1 if hero.get("isInGrass", False) else 0)

    def get_can_buy_equip(self, hero, value, feature_name):
        value.append(1 if hero.get("canBuyEquip", False) else 0)

    # Actor state attributes methods
    def is_main_camp(self, hero, value, feature_name):
        output = 0
        actor_state = hero.get("actor_state", {})
        if actor_state.get("camp", "PLAYERCAMP_1") == self.main_camp:
            output = 1
        value.append(output)

    def get_hp(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        value.append(actor_state.get("hp", 0))

    def get_max_hp(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        value.append(actor_state.get("max_hp", 0))

    def get_attack_range(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        value.append(actor_state.get("attack_range", 0))

    def get_attack_target(self, hero, value, feature_name):
        # Get tower and enemy hero runtime IDs
        main_tower_runtime_id = self.main_camp_tower_dict.get(0, {}).get("runtime_id", 0)
        enemy_tower_runtime_id = self.enemy_camp_tower_dict.get(0, {}).get("runtime_id", 0)
        enemy_hero_runtime_id = self.enemy_hero_info.get("actor_state", {}).get("runtime_id", 0)
        tower_ids = [main_tower_runtime_id, enemy_tower_runtime_id]
        actor_state = hero.get("actor_state", {})
        attack_target_id = actor_state.get("attack_target", 0)

        output = 0
        if attack_target_id in tower_ids:
            output = 3
        elif attack_target_id == enemy_hero_runtime_id:
            output = 2
        elif attack_target_id != 0:
            output = 1
        else:
            output = 0
        value.append(output)

    def get_kill_income(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        value.append(actor_state.get("kill_income", 0))

    def get_sight_area(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        value.append(actor_state.get("sight_area", 0))

    def is_visible(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        camp_visible = actor_state.get("camp_visible", [True, True])

        # Determine which camp index to use
        if self.main_camp == "PLAYERCAMP_1":  # Blue camp
            output = 1 if camp_visible[0] else 0
        else:  # PLAYERCAMP_2 (Red camp)
            output = 1 if camp_visible[1] else 0
        value.append(output)

    # Actor values methods
    def get_phy_atk(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("phy_atk", 0))

    def get_phy_def(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("phy_def", 0))

    def get_mov_spd(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("mov_spd", 0))

    def get_atk_spd(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("atk_spd", 0))

    def get_ep(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("ep", 0))

    def get_max_ep(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("max_ep", 0))

    def get_hp_recover(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("hp_recover", 0))

    def get_ep_recover(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("ep_recover", 0))

    def get_phy_armor_hurt(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("phy_armor_hurt", 0))

    def get_crit_rate(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("crit_rate", 0))

    def get_crit_effe(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("crit_effe", 0))

    def get_phy_vamp(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("phy_vamp", 0))

    def get_cd_reduce(self, hero, value, feature_name):
        actor_state = hero.get("actor_state", {})
        values = actor_state.get("values", {})
        value.append(values.get("cd_reduce", 0))

    # Helper method for skill extraction
    def _get_skill_by_slot(self, hero, slot_index):
        """Helper method to get skill by slot index"""
        skill_state = hero.get("skill_state", {})
        skill_slots = skill_state.get("slot_states", [])
        if slot_index < len(skill_slots):
            return skill_slots[slot_index]
        return None

    # Skill slot 0 methods
    def get_skill_0_level(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 0)
        value.append(skill.get("level", 0) if skill else 0)

    def get_skill_0_usable(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 0)
        value.append(1 if skill and skill.get("usable", False) else 0)

    def get_skill_0_cooldown(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 0)
        value.append(skill.get("cooldown", 0) if skill else 0)

    def get_skill_0_cooldown_max(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 0)
        value.append(skill.get("cooldown_max", 0) if skill else 0)

    # Skill slot 1 methods
    def get_skill_1_level(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 1)
        value.append(skill.get("level", 0) if skill else 0)

    def get_skill_1_usable(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 1)
        value.append(1 if skill and skill.get("usable", False) else 0)

    def get_skill_1_cooldown(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 1)
        value.append(skill.get("cooldown", 0) if skill else 0)

    def get_skill_1_cooldown_max(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 1)
        value.append(skill.get("cooldown_max", 0) if skill else 0)

    # Skill slot 2 methods
    def get_skill_2_level(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 2)
        value.append(skill.get("level", 0) if skill else 0)

    def get_skill_2_usable(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 2)
        value.append(1 if skill and skill.get("usable", False) else 0)

    def get_skill_2_cooldown(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 2)
        value.append(skill.get("cooldown", 0) if skill else 0)

    def get_skill_2_cooldown_max(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 2)
        value.append(skill.get("cooldown_max", 0) if skill else 0)

    # Skill slot 3 methods
    def get_skill_3_level(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 3)
        value.append(skill.get("level", 0) if skill else 0)

    def get_skill_3_usable(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 3)
        value.append(1 if skill and skill.get("usable", False) else 0)

    def get_skill_3_cooldown(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 3)
        value.append(skill.get("cooldown", 0) if skill else 0)

    def get_skill_3_cooldown_max(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 3)
        value.append(skill.get("cooldown_max", 0) if skill else 0)

    # Skill slot 4 methods
    def get_skill_4_level(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 4)
        value.append(skill.get("level", 0) if skill else 0)

    def get_skill_4_usable(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 4)
        value.append(1 if skill and skill.get("usable", False) else 0)

    def get_skill_4_cooldown(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 4)
        value.append(skill.get("cooldown", 0) if skill else 0)

    def get_skill_4_cooldown_max(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 4)
        value.append(skill.get("cooldown_max", 0) if skill else 0)

    # Skill slot 5 methods
    def get_skill_5_level(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 5)
        value.append(skill.get("level", 0) if skill else 0)

    def get_skill_5_usable(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 5)
        value.append(1 if skill and skill.get("usable", False) else 0)

    def get_skill_5_cooldown(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 5)
        value.append(skill.get("cooldown", 0) if skill else 0)

    def get_skill_5_cooldown_max(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 5)
        value.append(skill.get("cooldown_max", 0) if skill else 0)

    # Skill slot 6 methods
    def get_skill_6_level(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 6)
        value.append(skill.get("level", 0) if skill else 0)

    def get_skill_6_usable(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 6)
        value.append(1 if skill and skill.get("usable", False) else 0)

    def get_skill_6_cooldown(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 6)
        value.append(skill.get("cooldown", 0) if skill else 0)

    def get_skill_6_cooldown_max(self, hero, value, feature_name):
        skill = self._get_skill_by_slot(hero, 6)
        value.append(skill.get("cooldown_max", 0) if skill else 0)
