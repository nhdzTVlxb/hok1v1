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


class OrganProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp

        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        # Change from dict to dict of lists
        self.main_camp_organ_dict = {"tower": [], "soldier": []}
        self.enemy_camp_organ_dict = {"tower": [], "soldier": []}


        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_organ_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.organ_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 16
        self.unit_buff_num = 20

    def get_organ_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "organ_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.organ_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.organ_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_organ(self, frame_state):
        self.generate_organ_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)

        local_vector_feature = []

        # Generate features for enemy team's tower and soldiers
        enemy_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.enemy_camp_organ_dict, "enemy_camp")
        local_vector_feature.extend(enemy_camp_organ_vector_feature)

        # Generate features for main team's tower and soldiers
        main_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.main_camp_organ_dict, "main_camp")
        local_vector_feature.extend(main_camp_organ_vector_feature)

        vector_feature = local_vector_feature
        return vector_feature

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

    def generate_organ_info_dict(self, frame_state):
        self.main_camp_organ_dict = {"tower": [], "soldier": []}
        self.enemy_camp_organ_dict = {"tower": [], "soldier": []}

        for organ in frame_state["npc_states"]:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == self.main_camp:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    self.main_camp_organ_dict["tower"].append(organ)
                if organ_subtype == "ACTOR_SUB_SOLDIER":
                    self.main_camp_organ_dict["soldier"].append(organ)

            else:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    self.enemy_camp_organ_dict["tower"].append(organ)
                if organ_subtype == "ACTOR_SUB_SOLDIER":
                    self.enemy_camp_organ_dict["soldier"].append(organ)
                if organ_subtype == "ACTOR_SUB_NONE":
                    self.enemy_camp_organ_dict["soldier"].append(organ)

    def generate_one_type_organ_feature(self, one_type_organ_info, camp):
        vector_feature = []
        num_organs_considered = 0

        def process_organ(organ):
            nonlocal num_organs_considered
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](organ, value)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_organs_considered += 1

        if "tower" in one_type_organ_info:
            organ = one_type_organ_info["tower"][0]  
            process_organ(organ)

        if "soldier" in one_type_organ_info:
            list_of_soldiers = one_type_organ_info["soldier"]
            for soldier in list_of_soldiers:
                process_organ(soldier)

        if num_organs_considered < self.unit_buff_num:
            self.no_organ_feature(vector_feature, num_organs_considered)
        return vector_feature

    def no_organ_feature(self, vector_feature, num_organs_considered):
        for _ in range((self.unit_buff_num - num_organs_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def get_hp_rate(self, organ, vector_feature):
        value = 0
        if organ["max_hp"] > 0:
            value = organ["hp"] / organ["max_hp"]
        vector_feature.append(value)

    def judge_in_view(self, main_hero_location, obj_location):
        if (
            (main_hero_location["x"] - obj_location["x"] >= 0 - self.view_dist)
            and (main_hero_location["x"] - obj_location["x"] <= self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] >= 0 - self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] <= self.view_dist)
        ):
            return True
        return False

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt((pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2 + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2)
        return dist

    def is_alive(self, organ, vector_feature):
        value = 0.0
        if organ["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def belong_to_main_camp(self, organ, vector_feature):
        value = 0.0
        if organ["camp"] == self.main_hero_info["actor_state"]["camp"]:
            value = 1.0
        vector_feature.append(value)

    def get_normal_organ_location_x(self, organ, vector_feature):
        value = organ["location"]["x"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def get_normal_organ_location_z(self, organ, vector_feature):
        value = organ["location"]["z"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def relative_location_x(self, organ, vector_feature):
        organ_location_x = organ["location"]["x"]
        location_x = self.main_hero_info["actor_state"]["location"]["x"]
        x_diff = organ_location_x - location_x
        if self.transform_camp2_to_camp1 and organ_location_x != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def relative_location_z(self, organ, vector_feature):
        organ_location_z = organ["location"]["z"]
        location_z = self.main_hero_info["actor_state"]["location"]["z"]
        z_diff = organ_location_z - location_z
        if self.transform_camp2_to_camp1 and organ_location_z != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    # Additional ActorState features
    def is_main_camp(self, organ, vector_feature):
        value = 0
        if organ.get("camp", "PLAYERCAMP_1") == self.main_camp:
            value = 1
        vector_feature.append(value)

    def get_attack_range(self, organ, vector_feature):
        value = organ.get("attack_range", 0)
        vector_feature.append(value)

    def get_attack_target(self, organ, vector_feature):
        value = 0
        target = organ.get("attack_target", 0)
        main_hero_id = self.main_hero_info["actor_state"]["runtime_id"]
        enemy_hero_id = self.enemy_hero_info["actor_state"]["runtime_id"]
        hero_ids = [main_hero_id, enemy_hero_id]
        if target in hero_ids:
            value = 1
        vector_feature.append(value)

    def get_kill_income(self, organ, vector_feature):
        value = organ.get("kill_income", 0)
        vector_feature.append(value)

    def get_sight_area(self, organ, vector_feature):
        value = organ.get("sight_area", 0)
        vector_feature.append(value)

    def get_visibility(self, organ, vector_feature):
        camp_visible = organ.get("camp_visible", [True, True])
        
        # Determine which camp index to use
        if self.main_camp == "PLAYERCAMP_1":  # Blue camp
            value = 1 if camp_visible[0] else 0
        else:  # PLAYERCAMP_2 (Red camp)
            value = 1 if camp_visible[1] else 0
        
        vector_feature.append(value)

    # Actor values features
    def get_phy_atk(self, organ, vector_feature):
        values = organ.get("values", {})
        value = values.get("phy_atk", 0)
        vector_feature.append(value)

    def get_phy_def(self, organ, vector_feature):
        values = organ.get("values", {})
        value = values.get("phy_def", 0)
        vector_feature.append(value)

    def get_mov_spd(self, organ, vector_feature):
        values = organ.get("values", {})
        value = values.get("mov_spd", 0)
        vector_feature.append(value)
