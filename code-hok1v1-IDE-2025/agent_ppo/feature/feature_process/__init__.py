#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_cake(self, frame_state):
        cakes = frame_state.get("cakes", [])
        
        # Initialize 6-element vector: [cake1_exists, cake1_x, cake1_z, cake2_exists, cake2_x, cake2_z]
        cake_features = [0, 0, 0, 0, 0, 0]
        
        # Process up to 2 cakes
        for i, cake in enumerate(cakes[:2]):  # take first 2 cakes (if they exist)
            base_idx = i * 3  # Each cake uses 3 features
            
            # Cake exists
            cake_features[base_idx] = 1
            
            # Get cake location
            collider = cake.get("collider", {})
            center = collider.get("center", {})
            cake_x = center.get("x", 0)
            cake_z = center.get("z", 0)
            
            # Transform coordinates for camp 2 (similar to hero coordinate transformation)
            if self.camp == "PLAYERCAMP_2":
                cake_x = -cake_x
                cake_z = -cake_z

            # Normalize coordinates
            normalized_x = (cake_x + 60000.0) / 120000.0
            normalized_z = (cake_z + 60000.0) / 120000.0
            
            cake_features[base_idx + 1] = normalized_x
            cake_features[base_idx + 2] = normalized_z
        
        return cake_features
    
    def process_bullet(self, frame_state):
        bullets = frame_state.get("bullets", [])
        output_feature = []
        
        # Early return with proper padding if no bullets
        if not bullets:
            return [0] * (6 * 6)  # Return padded feature vector

        heros = frame_state["hero_states"]
        main_hero_id = -1
        enemy_hero_id = -1
        for hero in heros:
            actor_state = hero.get("actor_state", {})
            hero_id = actor_state.get("runtime_id", -1)
            camp = actor_state["camp"]
            if camp == self.camp:
                main_hero_id = hero_id
            else:
                enemy_hero_id = hero_id
        
        for bullet in bullets: 
            source_id = bullet.get("source_actor", -1)
            slot_type = bullet.get("slot_type", 0)  # Move slot_type here for both branches

            if source_id == main_hero_id:
                first_dim = [1]
                second_dim = [0, 0, 0]
                if slot_type == "SLOT_SKILL_0":
                    second_dim = [1, 0, 0]
                elif slot_type == "SLOT_SKILL_2":
                    second_dim = [0, 1, 0]
                elif slot_type == "SLOT_SKILL_3":
                    second_dim = [0, 0, 1]
                # Default case: second_dim remains [0, 0, 0] for unknown slot types
                
                # Handle position
                bullet_x = bullet.get("location", {}).get("x", 0)
                bullet_z = bullet.get("location", {}).get("z", 0)

                # Transform coordinates for camp 2
                if self.camp == "PLAYERCAMP_2":
                    bullet_x = -bullet_x
                    bullet_z = -bullet_z

                # Normalize coordinates
                normalized_x = (bullet_x + 60000.0) / 120000.0
                normalized_z = (bullet_z + 60000.0) / 120000.0
                third_dim = [normalized_x, normalized_z]

                bullet_feature = first_dim + second_dim + third_dim
                output_feature.extend(bullet_feature)

            elif source_id == enemy_hero_id:
                first_dim = [0]
                second_dim = [0, 0, 0]
                if slot_type == "SLOT_SKILL_0":
                    second_dim = [1, 0, 0]
                elif slot_type == "SLOT_SKILL_2":
                    second_dim = [0, 1, 0]
                elif slot_type == "SLOT_SKILL_3":
                    second_dim = [0, 0, 1]
                # Default case: second_dim remains [0, 0, 0] for unknown slot types

                # Handle position
                bullet_x = bullet.get("location", {}).get("x", 0)
                bullet_z = bullet.get("location", {}).get("z", 0)

                # Transform coordinates for camp 2
                if self.camp == "PLAYERCAMP_2":
                    bullet_x = -bullet_x
                    bullet_z = -bullet_z

                # Normalize coordinates
                normalized_x = (bullet_x + 60000.0) / 120000.0
                normalized_z = (bullet_z + 60000.0) / 120000.0
                third_dim = [normalized_x, normalized_z]

                bullet_feature = first_dim + second_dim + third_dim
                output_feature.extend(bullet_feature)

            # Minion bullets - skip
            else: 
                continue
        
        # Ensure dimension is exactly 6 * 6 = 36
        target_length = 6 * 6
        if len(output_feature) < target_length:
            output_feature.extend([0] * (target_length - len(output_feature)))
        elif len(output_feature) > target_length:
            output_feature = output_feature[:target_length]  # Truncate if too long
        
        return output_feature

    def process_relative_location_to_spring(self, frame_state):
        spring = None
        main_hero = None
        for organ in frame_state["npc_states"]:
            if organ["sub_type"] == "ACTOR_SUB_TOWER_SPRING" and organ["camp"] == self.camp:
                spring = organ
                break
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == self.camp:
                main_hero = hero
                break
        if not spring or not main_hero: 
            return [0,0]

        # copied from relative location in organ_process
        organ_location_x = spring["location"]["x"]
        location_x = main_hero["actor_state"]["location"]["x"]
        x_diff = location_x - organ_location_x
        if self.camp == "PLAYERCAMP_2" and organ_location_x != 100000:
            x_diff = -x_diff
        x_value = (x_diff) / 90000.0
        
        organ_location_z = spring["location"]["z"]
        location_z = main_hero["actor_state"]["location"]["z"]
        z_diff = location_z - organ_location_z
        if self.camp == "PLAYERCAMP_2" and organ_location_z != 100000:
            z_diff = -z_diff
        z_value = (z_diff) / 90000.0

        # fixed norms
        x_value = max(0.0, min(1.0, x_value))
        z_value = max(0.0, min(1.0, z_value))

        return [x_value, z_value]

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        both_camp_hero_vector_feature = self.process_hero_feature(frame_state) # shape of 65 * 2 for main and enemy hero
        organ_feature = self.process_organ_feature(frame_state) # shape of 16 features * 20 organs * 2 camps
        cake_feature = self.process_cake(frame_state) # shape of 6 features for cakes
        bullet_feature = self.process_bullet(frame_state) # shape of 6 * 6 for bullets
        relative_spring_location = self.process_relative_location_to_spring(frame_state) # shape of 2 for relative location to spring

        feature = both_camp_hero_vector_feature + organ_feature + cake_feature + bullet_feature + relative_spring_location

        return feature
