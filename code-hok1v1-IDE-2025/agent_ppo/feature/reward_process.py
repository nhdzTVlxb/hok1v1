#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_ppo.conf.conf import GameConfig


# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        # 监控主英雄的阵营 0-1（PLAYCAMP0-PLAYCAMP1）
        self.main_hero_camp = -1
        # 代表主英雄血量，代码没有使用
        self.main_hero_hp = -1
        # 代表主英雄防御塔血量，代码没有使用
        self.main_hero_organ_hp = -1
        # 全局的奖励字典，经过get_result函数进行初始化，每次初始化都是当前帧
        self.m_reward_value = {}
        # 代表上一帧号，代码没有使用
        self.m_last_frame_no = -1
        # 这个代表的是权重状态map，和下面两个的区别就是没有带前后帧值
        self.m_cur_calc_frame_map = init_calc_frame_map()
        # 主英雄的状态帧map
        self.m_main_calc_frame_map = init_calc_frame_map()
        # 敌方英雄的状态帧map
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        # 时间折扣因子
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        # 代表当前英雄的id号，代码没有使用
        self.m_main_hero_config_id = -1
        # 初始化英雄各个等级的最大经验值字典
        self.m_each_level_max_exp = {}
        # cakes
        self.cakes = []

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        # 获取当前帧号
        frame_no = frame_data["frameNo"]
        # 初始化英雄的各个等级经验
        self.init_max_exp_of_each_hero()
        # 对英雄的特征数据进行处理
        self.frame_data_process(frame_data)
        # 得到对应的奖励信息
        self.get_reward(frame_data, self.m_reward_value)
        # 如果设置了时间折扣因子，则进行开始奖励退火
        if self.time_scale_arg > 0:
            # 遍历所有的奖励，并且对对应的值进行衰减
            for key in self.m_reward_value:
                # 下面是实现退火逻辑，默认的逻辑函数是0.6^(frame_no/scale_arg)
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体

        # 主视角英雄和敌人英雄
        main_hero, enemy_hero = None, None
        # 英雄列表
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        # 主英雄当前血量
        main_hero_hp = main_hero["actor_state"]["hp"]
        # 主英雄最大血量
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        # 主英雄法力值
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        # 主英雄最大法力值
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

        # Get both defense towers and soldiers
        # 获取双方防御塔
        main_tower, main_soldiers, enemy_tower, enemy_soldiers = None, [], None, []
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            # 防御塔阵营
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            # 下面的逻辑是确定此时的防御塔阵营
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":  
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_SOLDIER":  
                    main_soldiers.append(organ)
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":  
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_SOLDIER":  
                    enemy_soldiers.append(organ)

        for reward_name, reward_struct in cul_calc_frame_map.items():
            # 将前一个帧的信息给last_frame_value
            reward_struct.last_frame_value = reward_struct.cur_frame_value

            # Money
            # 金钱
            if reward_name == "money":
                # 计算当前的经济
                reward_struct.cur_frame_value = main_hero["moneyCnt"]
            # Health points
            # 生命值
            elif reward_name == "hp_point":
                # 计算对应的血量
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            # Game win status
            elif reward_name == "game_win":
                if enemy_tower["hp"] <= 0 : # win
                    reward_struct.cur_frame_value = 1.0
                else:
                    reward_struct.cur_frame_value = 0.0
            # Energy points
            # 法力值
            elif reward_name == "ep_rate":
                # 统计法力百分比
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            # Kills
            # 击杀
            elif reward_name == "kill":
                # 统计击杀次数
                reward_struct.cur_frame_value = main_hero["killCnt"]
            # Deaths
            # 死亡
            elif reward_name == "death":
                # 统计死亡次数
                reward_struct.cur_frame_value = main_hero["deadCnt"]
            # Tower health points
            # 塔血量
            elif reward_name == "tower_hp_point":
                # 统计塔血量百分比
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            # Last hit
            # 补刀
            elif reward_name == "last_hit":
                # 先默认初始化当前帧信息为0
                reward_struct.cur_frame_value = 0.0
                # 取出当前的死亡事件信息
                frame_action = frame_data["frame_action"]
                # 如果发生了死亡事件
                if "dead_action" in frame_action:
                    # 取出当前的死亡事件信息
                    dead_actions = frame_action["dead_action"]

                    for dead_action in dead_actions:
                        # 这个部分鼓励我方英雄去杀小兵/野怪
                        if (
                            # 杀人者-我方英雄
                            dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                            # 死亡者-敌方小兵
                            and (dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER" or dead_action["death"]["sub_type"] == "ACTOR_SUB_NONE")
                        ):
                            # 此时当前帧的value + 1
                            reward_struct.cur_frame_value += 1.0
                        # 这个部分阻碍敌方英雄去杀小兵/野怪
                        elif (
                            # 杀人者-敌方英雄
                            dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                            # 死亡者-我方小兵
                            and (dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER" or dead_action["death"]["sub_type"] == "ACTOR_SUB_NONE")
                        ):
                            # 此时当前帧value-1
                            reward_struct.cur_frame_value -= 1.0
            # Experience points
            # 经验值
            elif reward_name == "exp":
                # 根据self.calculate_exp_sum获取经验value
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            elif reward_name == "forward":
                # 根据self.calculate_forward获取forward的value
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)

            # Skill 2 reward
            elif reward_name == "sk2":
                usedTimes = main_hero["skill_state"]["slot_states"][2]["usedTimes"]
                hitHeroTimes = main_hero["skill_state"]["slot_states"][2]["hitHeroTimes"]
                reward_struct.cur_frame_value = (usedTimes, hitHeroTimes)

            # Skill 3 reward
            elif reward_name == "sk3":
                usedTimes = main_hero["skill_state"]["slot_states"][3]["usedTimes"]
                hitHeroTimes = main_hero["skill_state"]["slot_states"][3]["hitHeroTimes"]
                reward_struct.cur_frame_value = (usedTimes, hitHeroTimes)
            
            # Skill 5 reward 
            elif reward_name == "sk5":
                usedTime = main_hero["skill_state"]["slot_states"][5]["usedTimes"]
                hitHeroTimes = main_hero["skill_state"]["slot_states"][5]["hitHeroTimes"]
                reward_struct.cur_frame_value = (usedTime, hitHeroTimes)

            # minion hp
            elif reward_name == "minion_hp":
                # 统计我方小兵的血量百分比
                total_hp = sum(soldier["hp"] for soldier in main_soldiers) if main_soldiers else 0.0
                total_max_hp = sum(soldier["max_hp"] for soldier in main_soldiers) if main_soldiers else 1.0
                reward_struct.cur_frame_value = (total_hp, total_max_hp)
            
            # recover rwd
            elif reward_name == "recvr_rwd":
                # extract hp/ep percentage for main hero 
                usedTimes = main_hero["skill_state"]["slot_states"][4]["usedTimes"]
                main_cur_hp = main_hero["actor_state"]["hp"]
                main_max_hp = main_hero["actor_state"]["max_hp"]
                main_cur_ep = main_hero["actor_state"]["values"]["ep"]  
                main_max_ep = main_hero["actor_state"]["values"]["max_ep"]  
                main_hp_percent = main_cur_hp / main_max_hp if main_max_hp > 0 else 0
                main_ep_percent = main_cur_ep / main_max_ep if main_max_ep > 0 else 0

                # extract hp/ep percentage for enemy hero
                enemy_hp = enemy_hero["actor_state"]["hp"]
                enemy_max_hp = enemy_hero["actor_state"]["max_hp"]
                enemy_ep = enemy_hero["actor_state"]["values"]["ep"]
                enemy_max_ep = enemy_hero["actor_state"]["values"]["max_ep"]
                enemy_hp_percent = enemy_hp / enemy_max_hp if enemy_max_hp > 0 else 0
                enemy_ep_percent = enemy_ep / enemy_max_ep if enemy_max_ep > 0 else 0

                # extract main hero location info
                main_location_x = main_hero["actor_state"]["location"]["x"]
                main_location_z = main_hero["actor_state"]["location"]["z"]

                # get cake info
                cakes = self.cakes

                # Save stats
                reward_struct.cur_frame_value = (usedTimes, main_hp_percent, main_ep_percent, 
                                                 enemy_hp_percent, enemy_ep_percent, 
                                                 main_location_x, main_location_z,
                                                 cakes)

            elif reward_name == "go_to_spring":
                main_cur_hp = main_hero["actor_state"]["hp"]
                main_max_hp = main_hero["actor_state"]["max_hp"]
                main_hp_percent = main_cur_hp / main_max_hp if main_max_hp > 0 else 0

                # extract main hero location info
                main_location_x = main_hero["actor_state"]["location"]["x"]
                main_location_z = main_hero["actor_state"]["location"]["z"]

                # Save stats
                reward_struct.cur_frame_value = (main_hp_percent,  
                                                 main_location_x, 
                                                 main_location_z)

            # lv 2 and 4 rwd 
            elif reward_name == "lv_rwd":
                reward_struct.cur_frame_value = main_hero["level"]

            elif reward_name == "took_hit_from_tower":
                reward_struct.cur_frame_value = 1.0 if (enemy_tower.get("hit_target_info") is not None and 
                                                        enemy_tower["hit_target_info"][0]["hit_target"] == self.main_hero_player_id) else 0.0


    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        # 通过我方当前的等级去遍历
        for i in range(1, this_hero_info["level"]):
            # 统计累计经验
            exp_sum += self.m_each_level_max_exp[i]
        # 因为前面sum是加到1-当前角色等级-1，所以此处要补充当前的角色等级的经验,这样做的目的是避免当前角色没有升级满，但是加了满经验
        exp_sum += this_hero_info["exp"]
        return exp_sum

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        # if hero is less than 0.75 percent, forward reward should be 0
        if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] < 0.75:
            return 0
        # 获取主防御塔在地图里面的x和z
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        # 获取敌人防御塔在地图里面的x和z
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        # 获取英雄当前在地图里面的x和z
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        # 初始化forward_value
        forward_value = 0
        # 用math里面的距离函数去计算当前主英雄和敌人防御塔的距离
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        # 计算我方防御塔和敌方防御塔直接的距离
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        # and左边代表英雄血量健康， and右边是此时的主英雄在两个防御塔之间的战争之外
        if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
            # 计算逻辑为（两塔之间的距离 - 主英雄与敌方防御塔的距离）/ 主英雄与敌方防御塔的距离
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy

        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        # update cakes info
        self.cakes = []
        for cake in frame_data.get("cakes", []):
            self.cakes.append(cake)

        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]
        # 获取主英雄的各项信息
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        # 获取敌方英雄的各项信息
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        # 清空当前的奖励字典
        reward_dict.clear()
        # 初始化总奖励和总权重
        reward_sum, weight_sum = 0.0, 0.0
        # 从权重map取出对应的权重名和权重值来着GameConfig里面
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():

            # 英雄血量
            if reward_name == "hp_point":
                if (
                    # 前一帧主英雄和敌方英雄血量都为空的时候
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
                    # 给reward_struct的前一帧和当前帧初始化为0
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                # 前一帧我方主英雄血量为空
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    # 给reward_struct的前一帧和当前帧初始化为（0 - 敌方血量）
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                # 前一帧敌方英雄血量为空
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    # 给reward_struct的前一帧和当前帧初始化为（我方血量 - 0）
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                # 都不为空
                else:
                    # 给reward_struct的前一帧初始化为（我方血量 - 敌方血量）
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    # 给reward_struct的当前帧初始化为（我方血量 - 敌方血量）
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                # 最后给reward_struct的value赋值为（当前帧的value - 前一帧的value）
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            # 法力值
            elif reward_name == "ep_rate":
                # 前一帧和当前帧初始化为主英雄对应的数值
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                # 如果前一帧对应的数值是正数即为大于0
                if reward_struct.last_frame_value > 0:
                    # 给reward_struct的value赋值为（当前帧的value - 前一帧的value）
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                # 如果没了则为0
                else:
                    reward_struct.value = 0
            # 经验值
            elif reward_name == "exp":
                # 默认初始化主英雄为空
                main_hero = None
                # 遍历所有的英雄
                for hero in frame_data["hero_states"]:
                    # 如果此时满足为主英雄则将此时的主英雄指定
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                # 主英雄已经制定了，而且主英雄的等级满了，将reward_struct的value赋值为0
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                # 如果没有满
                else:
                    # 对应的reward_struct的当前帧为（我方经验 - 敌方经验）
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    # 对应的reward_struct的前一帧为（我方经验 - 敌方经验）
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    # 然后将reward_struct的value赋值为（当前帧的value - 前一帧的value）
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            # 英雄与塔之间的距离，判断英雄是否进入战场，此为惩罚
            elif reward_name == "forward":
                # 具体参考奖励函数
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            # 最后一次击杀（对击杀事件的分析）
            elif reward_name == "last_hit":
                # 直接将reward_struct的value赋值给主英雄当前帧对应的值，对应的实现在初始化里面，详细参考set_cur_calc_frame_vec函数，对应179行之后
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            
            elif reward_name == "sk2":
                reward = 0.0
                # main hero reward calculation
                main_cur_usedTimes, main_cur_hitHeroTimes = self.m_main_calc_frame_map[reward_name].cur_frame_value
                try:
                    main_prev_usedTimes, main_prev_hitHeroTimes = self.m_main_calc_frame_map[reward_name].last_frame_value
                except TypeError:
                    reward_struct.value = 0.0
                    continue
                is_main_sk2_used = main_cur_usedTimes > main_prev_usedTimes
                if is_main_sk2_used:
                    reward -= 0.5
                
                is_main_sk2_hit = main_cur_hitHeroTimes > main_prev_hitHeroTimes
                if is_main_sk2_hit:
                    reward += 1.0
                

                # enemy hero reward calculation
                '''enemy_cur_usedTimes, enemy_cur_hitHeroTimes = self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                enemy_prev_usedTimes, enemy_prev_hitHeroTimes = self.m_enemy_calc_frame_map[reward_name].last_frame_value
                is_enemy_sk2_used = enemy_cur_usedTimes > enemy_prev_usedTimes
                is_enemy_sk2_hit = enemy_cur_hitHeroTimes > enemy_prev_hitHeroTimes
                if is_enemy_sk2_used:
                    if is_enemy_sk2_hit:
                        reward -= 1.0
                    else:
                        reward += 0.5'''

                reward_struct.value = reward

            # Skill 3 reward calculation
            elif reward_name == "sk3":
                reward = 0.0
                # main hero reward calculation
                main_cur_usedTimes, main_cur_hitHeroTimes = self.m_main_calc_frame_map[reward_name].cur_frame_value
                try:
                    main_prev_usedTimes, main_prev_hitHeroTimes = self.m_main_calc_frame_map[reward_name].last_frame_value
                except TypeError:
                    reward_struct.value = 0.0
                    continue
                is_main_sk3_used = main_cur_usedTimes > main_prev_usedTimes
                if is_main_sk3_used:
                    reward -= 0.5
                is_main_sk3_hit = main_cur_hitHeroTimes > main_prev_hitHeroTimes
                if is_main_sk3_hit:
                    reward += 1.0

                # # enemy hero reward calculation
                # enemy_cur_usedTimes, enemy_cur_hitHeroTimes = self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                # enemy_prev_usedTimes, enemy_prev_hitHeroTimes = self.m_enemy_calc_frame_map[reward_name].last_frame_value
                # is_enemy_sk3_used = enemy_cur_usedTimes > enemy_prev_usedTimes
                # is_enemy_sk3_hit = enemy_cur_hitHeroTimes > enemy_prev_hitHeroTimes
                # if is_enemy_sk3_used:
                #     if is_enemy_sk3_hit:
                #         reward -= 1.0
                #     else:
                #         reward += 0.5
                
                reward_struct.value = reward
            
            elif reward_name == "sk5":
                reward = 0.0
                # main hero reward calculation
                main_cur_usedTimes, main_cur_hitHeroTimes = self.m_main_calc_frame_map[reward_name].cur_frame_value
                try:
                    main_prev_usedTimes, main_prev_hitHeroTimes = self.m_main_calc_frame_map[reward_name].last_frame_value
                except TypeError:
                    reward_struct.value = 0.0
                    continue
                is_main_sk5_used = main_cur_usedTimes > main_prev_usedTimes
                if is_main_sk5_used:
                    reward -= 1.0
                # is_main_sk5_hit = main_cur_hitHeroTimes > main_prev_hitHeroTimes
                # if is_main_sk5_used:
                #     if is_main_sk5_hit:
                #         reward += 1.0
                #     else:
                #         reward -= 1.0

                # # enemy hero reward calculation
                # enemy_cur_usedTimes, enemy_cur_hitHeroTimes = self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                # enemy_prev_usedTimes, enemy_prev_hitHeroTimes = self.m_enemy_calc_frame_map[reward_name].last_frame_value
                # is_enemy_sk5_used = enemy_cur_usedTimes > enemy_prev_usedTimes
                # is_enemy_sk5_hit = enemy_cur_hitHeroTimes > enemy_prev_hitHeroTimes
                # if is_enemy_sk5_used:
                #     if is_enemy_sk5_hit:
                #         reward -= 1.0
                #     else:
                #         reward += 0.5

                reward_struct.value = reward

            elif reward_name == "minion_hp":
                reward = 0.0
                # Handle potential empty tuples or missing data
                try:
                    prev_enemy_total_hp = self.m_enemy_calc_frame_map[reward_name].last_frame_value[0]
                    prev_enemy_max_hp = self.m_enemy_calc_frame_map[reward_name].last_frame_value[1]
                    cur_enemy_total_hp = self.m_enemy_calc_frame_map[reward_name].cur_frame_value[0]
                    cur_enemy_max_hp = self.m_enemy_calc_frame_map[reward_name].cur_frame_value[1]
                except (IndexError, TypeError):
                    reward = 0.0
                    reward_struct.value = reward
                    continue

                # Case of no soldiers died
                if prev_enemy_max_hp > 0:
                    if prev_enemy_max_hp == cur_enemy_max_hp:
                        reward += 1.0 * (prev_enemy_total_hp - cur_enemy_total_hp) / prev_enemy_max_hp
                    elif prev_enemy_max_hp < cur_enemy_max_hp:  # one enemy solider is killed
                        reward += 1.0 * (prev_enemy_total_hp - cur_enemy_total_hp) / cur_enemy_max_hp
                    else:  # new soldiers spawning
                        reward += 1.0 * (prev_enemy_total_hp - cur_enemy_total_hp) / prev_enemy_max_hp
                
                # Penalty for decrease main minion HP
                prev_main_total_hp = self.m_main_calc_frame_map[reward_name].last_frame_value[0]
                prev_main_max_hp = self.m_main_calc_frame_map[reward_name].last_frame_value[1]
                cur_main_total_hp = self.m_main_calc_frame_map[reward_name].cur_frame_value[0]
                cur_main_max_hp = self.m_main_calc_frame_map[reward_name].cur_frame_value[1]

                # Case of no soldiers died
                if prev_main_max_hp > 0:
                    if prev_main_max_hp == cur_main_max_hp:
                        reward -= 1.0 * (prev_main_total_hp - cur_main_total_hp) / prev_main_max_hp
                    elif prev_main_max_hp < cur_main_max_hp:  # one main solider is killed
                        reward -= 1.0 * (prev_main_total_hp - cur_main_total_hp) / cur_main_max_hp
                    else:  # new soldiers spawning
                        reward -= 1.0 * (prev_main_total_hp - cur_main_total_hp) / prev_main_max_hp
                
                reward_struct.value = reward
            
            elif reward_name == "recvr_rwd":
                rwd = 0.0
                # calculate reward for recvr skill 
                try:
                    prev_used_times = self.m_main_calc_frame_map[reward_name].last_frame_value[0]
                    prev_cakes = self.m_main_calc_frame_map[reward_name].last_frame_value[7]
                    prev_hp_pct = self.m_main_calc_frame_map[reward_name].last_frame_value[1]
                    prev_ep_pct = self.m_main_calc_frame_map[reward_name].last_frame_value[2]
                    prev_enemy_hp_pct = self.m_main_calc_frame_map[reward_name].last_frame_value[3]
                    prev_loc_x = self.m_main_calc_frame_map[reward_name].last_frame_value[5]
                    prev_loc_z = self.m_main_calc_frame_map[reward_name].last_frame_value[6]
                except (TypeError, IndexError):
                    reward_struct.value = 0.0
                    continue
                
                cur_used_times = self.m_main_calc_frame_map[reward_name].cur_frame_value[0]
                cur_loc_x = self.m_main_calc_frame_map[reward_name].cur_frame_value[5]
                cur_loc_z = self.m_main_calc_frame_map[reward_name].cur_frame_value[6]
                
                is_recvr_used = cur_used_times > prev_used_times
                if is_recvr_used:
                    if self.is_recvr_useful(prev_hp_pct, prev_ep_pct):
                        rwd += 0.1
                    else:
                        rwd -= 1.0
                
                reward_struct.value = rwd

            elif reward_name == "go_to_spring":
                rwd = 0.0
                try:
                    prev_main_hp_pct = self.m_main_calc_frame_map[reward_name].last_frame_value[0]
                    prev_loc_x = self.m_main_calc_frame_map[reward_name].last_frame_value[1]
                    prev_loc_z = self.m_main_calc_frame_map[reward_name].last_frame_value[2]
                except (TypeError, IndexError):
                    reward_struct.value = 0.0
                    continue

                cur_loc_x = self.m_main_calc_frame_map[reward_name].cur_frame_value[1]
                cur_loc_z = self.m_main_calc_frame_map[reward_name].cur_frame_value[2]

                # difference in HP percentage (if enemy has more hp)
                delta_hp_rate = 1 - prev_main_hp_pct
                if delta_hp_rate > 0:
                    if self.main_hero_camp == "PLAYERCAMP_2":
                        spring_x = 50000
                        spring_z = 50000
                    else: 
                        spring_x = -50000
                        spring_z = -50000
                    prev_distance_2_spring = ((prev_loc_x - spring_x) ** 2 + (prev_loc_z - spring_z) ** 2) ** 0.5
                    cur_distance_2_spring = ((cur_loc_x - spring_x) ** 2 + (cur_loc_z - spring_z) ** 2) ** 0.5
                    del_distance = prev_distance_2_spring - cur_distance_2_spring
                    rwd += (del_distance / 50000) * delta_hp_rate
                reward_struct.value = rwd

            elif reward_name == "lv_rwd":
                rwd = 0.0
                prev_main_lv = self.m_main_calc_frame_map[reward_name].last_frame_value
                cur_main_lv = self.m_main_calc_frame_map[reward_name].cur_frame_value
                prev_enemy_lv = self.m_enemy_calc_frame_map[reward_name].last_frame_value
                cur_enemy_lv = self.m_enemy_calc_frame_map[reward_name].cur_frame_value

                is_main_lv2_first = (cur_main_lv == 2 and prev_main_lv == 1 and cur_enemy_lv == 1)
                is_enemy_lv2_first = (cur_enemy_lv == 2 and prev_enemy_lv == 1 and cur_main_lv == 1)

                is_main_lv4_first = (cur_main_lv == 4 and prev_main_lv == 3 and cur_enemy_lv == 3)
                is_enemy_lv4_first = (cur_enemy_lv == 4 and prev_enemy_lv == 3 and cur_main_lv == 3)

                if is_main_lv2_first:
                    rwd += 1.0
                elif is_enemy_lv2_first:
                    rwd -= 1.0
                elif is_main_lv4_first:
                    rwd += 0.5
                elif is_enemy_lv4_first:
                    rwd -= 0.5
                reward_struct.value = rwd




            # 其他的奖励包括tower_hp_point、money、death、kill
            else:
                # 对应的reward_struct的当前帧为（我方 - 敌方）
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                # 对应的reward_struct的前一帧为（我方 - 敌方）
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                # 然后将reward_struct的value赋值为（当前帧的value - 前一帧的value）
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            # 权重相加
            weight_sum += reward_struct.weight
            # 奖励*权重 = 总奖励 ， 总奖励累加
            reward_sum += reward_struct.value * reward_struct.weight
            # 将对应的奖励名称的奖励值赋值给奖励字典
            reward_dict[reward_name] = reward_struct.value
        # 总奖励的值赋值给reward_dict
        reward_dict["reward_sum"] = reward_sum
    
    # Check if recovery is useful
    def is_recvr_useful(self, hp, ep):
        # 判断当前的血量和能量值是否都大于0
        if hp < 0.75:
            return True
        return False

    # Check if moving to spring is useful
    def should_mv_to_spring(self, prev_hp_pct, prev_enemy_hp_pct, prev_loc_x, prev_loc_z):
        # Check if hero is at the spring and still needs to recover
        if self.main_hero_camp == "PLAYERCAMP_2":
            spring_x = 40000
            spring_z = 40000
            if prev_loc_x > spring_x and prev_loc_z > spring_z and prev_hp_pct < 0.75:
                return True
        else:
            spring_x = -40000
            spring_z = -40000
            if prev_loc_x < spring_x and prev_loc_z < spring_z and prev_hp_pct < 0.75:
                return True

        # Trigger mv to spring if hp less than 1/8 
        if prev_hp_pct < 0.125:
            return True
        return False

    # Caculate reward for moving to spring 
    def calculate_spring_reward(self, prev_loc_x, prev_loc_z, cur_loc_x, cur_loc_z):
        # Get spring location
        if self.main_hero_camp == "PLAYERCAMP_2":
            spring_x = 50000
            spring_z = 50000
        else: 
            spring_x = -50000
            spring_z = -50000
        
        # calculate prev and cur distance to spring 
        prev_distance_2_spring = ((prev_loc_x - spring_x) ** 2 + (prev_loc_z - spring_z) ** 2) ** 0.5
        cur_distance_2_spring = ((cur_loc_x - spring_x) ** 2 + (cur_loc_z - spring_z) ** 2) ** 0.5
        del_distance = prev_distance_2_spring - cur_distance_2_spring
        norm_del_distance = del_distance / prev_distance_2_spring if prev_distance_2_spring > 0 else 0
        return norm_del_distance

    # Check if moving to cakes is useful
    def should_mv_to_cakes(self, prev_hp_pct, prev_enemy_hp_pct, prev_cakes):
        if self.main_hero_camp == "PLAYERCAMP_2":
            cake_x = 15340
            cake_z = 15100
        else: 
            cake_x = -15220
            cake_z = -15120
        is_cake_exist = False
        for cake in prev_cakes:
            if cake["collider"]["location"]["x"] == cake_x and cake["collider"]["location"]["z"] == cake_z:
                is_cake_exist = True
                break

        if (prev_hp_pct < 0.5 * prev_enemy_hp_pct or prev_hp_pct < 0.25) and is_cake_exist:
            return True
        return False

    # Caculate reward for moving to cake
    def calculate_cake_reward(self, prev_loc_x, prev_loc_z, cur_loc_x, cur_loc_z):
        # Get cake location
        if self.main_hero_camp == "PLAYERCAMP_2":
            cake_x = 15340
            cake_z = 15100
        else: 
            cake_x = -15220
            cake_z = -15120

        # calculate prev and cur distance to cake
        prev_distance_2_cake = ((prev_loc_x - cake_x) ** 2 + (prev_loc_z - cake_z) ** 2) ** 0.5
        cur_distance_2_cake = ((cur_loc_x - cake_x) ** 2 + (cur_loc_z - cake_z) ** 2) ** 0.5
        del_distance = prev_distance_2_cake - cur_distance_2_cake
        norm_del_distance = del_distance / prev_distance_2_cake if prev_distance_2_cake > 0 else 0
        return norm_del_distance