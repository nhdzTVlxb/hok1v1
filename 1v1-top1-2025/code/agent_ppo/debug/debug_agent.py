"""
Action一个长度为6的列表: (6,), 下面一共有6项, 分别表示action的对应维度上的含义
1. button: (1,) 取值范围 0~11 (后面0/1向量为对应的mask, sub_action_mask)
(这里sub_action_mask是指做了第一维动作后, 整个action的mask就会变成对应的sub_action_mask)
(这里的sub_action_mask对于每个英雄都是固定不变的, 技能释放和该技能是否为指向性技能有关)
    0, 1: 无动作    [1. 0. 0. 0. 0. 0.]
    2: 移动英雄     [1. 1. 1. 0. 0. 0.]
    3: 普攻         [1. 0. 0. 0. 0. 1.]
                         李元芳         
    4: 释放1技能     [1. 0. 0. 0. 0. 0.]
    5: 释放2技能     [1. 0. 0. 1. 1. 1.]
    6: 释放3技能     [1. 0. 0. 1. 1. 1.]
    7: 回复             [1. 0. 0. 0. 0. 0.]
    8: 闪现 (召唤师技能)    [1. 0. 0. 1. 1. 1.]
    9: 回城                [1. 0. 0. 0. 0. 0.]
    10: 释放4技能 (无用)    [1. 0. 0. 0. 0. 0.]
    11: 装备技能            [1. 0. 0. 0. 0. 1.]

以下x,z方向取值的legal_action中第0维都是False, 其他都是True

2. move: (2,) 相对网格化移动方向, 当前英雄位于中心点(8, 8)
    x: 取值范围 0~15    
    z: 取值范围 0~15
3. skill: (2,) 相对网格花技能释放点, 以target为中心点(8, 8), 英雄与其的连线
    x: 取值范围 0~15
    z: 取值范围 0~15
4. target: (1,) 取值范围 0~8
|编号 | target |
| - | - |
| 0 | None |
| 1 | 敌方英雄 |
| 2 | 我方英雄 |
| 3~6 | 分别4个最近的小兵 (TODO: 最近小兵是如何定义的) |
| 7 | 敌方防御塔 |
| 8 | |

1. 在没有放任何技能的情况下, 狄仁杰进行普攻, 子弹出现最大次数7, 单帧最大金钱300
2. 漏兵不会增加金币
3. 英雄从复活到(0,0)处线上大概用时414帧, 13s, 而回城只需8s
4. 6min公孙离打没有防御的狄仁杰4下平a就死, 最高伤害[1690, 1332, 1247](公孙离, 多次测试结果) [](伽罗), 而初始时最低伤害只有129
5. 小兵伤害从50增加到70, 炮车可能更高
6. frame_action中出现非常多的遗漏问题, 减去6个防御塔, 跑12230帧总共出现了86-6=80个单位(其中2个英雄, 其余78个小兵,
     这里还没计算出现的1~3只河蟹), 但是总共出现的frame_action死亡记录才[11,18](多次测试结果)次, 因此不能依赖此项, 判断单位死亡, 也无需补刀奖励了
7. 释放动作的下一帧才会处于UseSkill_[1,2,3]状态中
8. 虽然无法看到对方英雄的技能level, cd, 但是可以通过usable看到对方技能是否可用, 因此优先通过usable判断cd是否为0s
9. 回复技能好像不是智能体控制, 而是掉血之后自动使用了
"""
import numpy as np
from agent_ppo.utils import get_dist
from agent_ppo.feature.unpack_state_dict import Info, ActorInfo, info2dict
from agent_ppo.utils.dfs_iterable_struct import dfs_iter_apply_fn
from agent_ppo.conf.conf import GameConfig
    
NO_ACTION = [0] * 6    # 不执行动作
class DebugAgent:

    def __init__(self):
        self.last_position = None
        self.avg_move_dist = 0
        self.count = 0
        self.behaves = set()
        self.soldier_behaves = set()
        self.max_bullets = 0
        self.last_level = 0
        self.find_target = False
        self.near_target = False
        self.first_in_grass = False
        self.use_skill_count = 0
        self.first_use_skill1 = False
        self.first_use_skill2 = False
        self.first_use_skill3 = False
        self.frame_debug = 0
        self.has_been_near = False
        self.max_money_delta = 0
        self.last_money = 0
        self.last_hp = 0
        self.max_hp_delta_minus = 0
        self.buff_dict = {'skills': set(), 'marks': set()}
        self.last_at_target = False
        self.last_use_skill = False
        
    def act(self, info: Info):
        """基于当前帧信息, 给出想要debug目标所需的动作"""
        self.info = info
        self.camp = info.player_camp
        self.n_frame = info.n_frame

        ### 统计共有多少种behave ###
        # def fn(x, key, passby: list):
        #     if key == 'behave': passby.append(x)
        # passby = []
        # dfs_iter_apply_fn(info2dict(info, skip_keys=False)[0], fn, only_dict=False, input_key=True, passby=passby)
        # self.behaves = self.behaves.union(passby)
        ### 统计英雄一共有多少种behave ###
        self.behaves.add(info.hero_our.info.behave)
        self.behaves.add(info.hero_enemy.info.behave)
        for soldier in [info.soldiers_our, info.soldiers_enemy]:
            for s in soldier.merge:
                self.soldier_behaves.add(s.behave)

        ### 查看hit_target_info出现的帧数 ###
        def fn(x, key, passby: list):
            if key == 'hit_target_infos' and len(x):
                passby.append([(i['hit_target'], i['slot_type']) for i in x])
        tmp = []
        dfs_iter_apply_fn(info2dict(info)[0], fn, only_dict=True, input_key=True, passby=tmp, only_leaf=False)
        if len(tmp):
            self.print(f"hit_target_info={tmp}")

        ### 查看防御塔的状态 ###
        assert info.organ_enemy.sub_tower.behave == 'Attack_Move'
        assert info.organ_our.sub_tower.behave == 'Attack_Move'

        ### 统计有多少种buff ###
        def fn(x, key, passby: dict):
            if key == 'buff_skill_id': passby['skills'].add(x)
            if key == 'buff_mark_id': passby['marks'].add(x)
        dfs_iter_apply_fn(info2dict(info, skip_keys=False)[0], fn, only_dict=False, input_key=True, passby=self.buff_dict)

        ### 检查是否可以看到己方在敌方加的印记 (2025比赛可以看到了) ###
        # if self.camp == 1:
        #     if len(info.hero_enemy.info.buff.marks):
        #         self.print(f"红方看到蓝方印记信息, exit!")
        #         exit()

        self.action = NO_ACTION
        """ 注意修改action的顺序, 后面的action可能覆盖上面的动作 """
        if self.camp == 0:    # 蓝方动作
            dist = self.move_target(0, 0)
            if dist < 1000:
                if not self.last_at_target:
                    self.print('at (0, 0)')
                self.last_at_target = True
            else:
                self.last_at_target = False
            
            """ 测试技能, 可能需要两帧连续释放才能放出, 因此判断cd是最准确的 """
            use_skill = False
            if self.n_frame > 1000 and not self.first_use_skill1:
                use_skill = self.use_skill(0)
                if use_skill and not self.first_use_skill1:
                    self.print("Use skill 1")
            if info.hero_our.skill.first.cd != 0 and not self.first_use_skill1:
                self.print("Use skill 1 Good")
                self.first_use_skill1 = True
            if self.n_frame > 1000 and not use_skill and not self.first_use_skill2:
                use_skill = self.use_skill(1)
                if use_skill and not self.first_use_skill2:
                    self.print("Use skill 2")
            if info.hero_our.skill.second.cd != 0 and not self.first_use_skill2:
                self.print("Use skill 2 Good")
                self.first_use_skill2 = True
            if self.n_frame > 1000 and not use_skill and not self.first_use_skill3:
                use_skill = self.use_skill(2)
                if use_skill and not self.first_use_skill3:
                    self.print("Use skill 3")
            if info.hero_our.skill.thrid.cd != 0 and not self.first_use_skill3:
                self.print("Use skill 3 Good")
                self.first_use_skill3 = True
            # if self.n_frame > 1000:
            #     self.print(info.hero_enemy.info.buff.skill_ids)
            # dist = self.move_target(4000, -10000)    # 蓝方的下草, 红方的上草
            # # dist = self.move_target(0, 0)
            # if dist < 1000:
            #     self.action = NO_ACTION
            #     flag = self.use_skill(3)
            #     if flag:
            #         self.print("Use recover!")
            # if info.hero_our.flag_in_grass and not self.first_in_grass:
            #     self.first_in_grass = True
            #     self.print(f"In Grass")
            """ 上帧使用技能下帧普攻可能导致技能被吞, 需要延迟 """
            if (
                dist < 1000 and self.n_frame > 1000 and
                not use_skill and not self.last_use_skill
            ):
                self.normal_attack()
            # if dist < 1000:
            #     self.action = NO_ACTION
                # self.print(f"in grass={info.hero_our.flag_in_grass}, position={info.hero_our.info.position}")
            #     flag = self.use_skill(0)
            #     if not flag:
            #         self.normal_attack()
            # self.normal_attack()
            self.last_use_skill = use_skill

        if self.camp == 1:    # 红方动作
            dist = self.move_target(0, 0)
            if dist < 1000:
                if not self.last_at_target:
                    self.print('at (0, 0)')
                self.last_at_target = True
            else:
                self.last_at_target = False
            ### 调试每帧的扣血量 ###
            now = info.hero_our.info.hp
            delta = now - self.last_hp
            self.last_hp = now
            self.max_hp_delta_minus = min(self.max_hp_delta_minus, delta)
            if delta != 0:    # 扣血
                self.print(f"hero hp delta={delta}, now hp={now}")
            # if self.n_frame > 1000:
            #     self.print(info.hero_our.info.buff.skill_ids)
            # if now < 500:
            #     self.print(f"hero behave: {info.hero_our.info.behave}, now hp={now}")

            # if self.n_frame > 1000 and not self.first_use_skill2_:
            #     flag = self.use_skill(1)
            #     if flag:
            #         self.print("Use skill 2")
            #         self.first_use_skill2_ = True
            # if dist < 1000:
            #     self.action = NO_ACTION
            #     self.print(f"in grass: {info.hero_our.flag_in_grass}, position={info.hero_our.info.position}")
            #     self.print(f"see camp1 position: {info.hero_enemy.info.position}")
            # if info.river_crab is not None:    # 跟踪河蟹
            #     # self.print(f"carb hp: {info.river_crab.hp}, behave: {info.river_crab.behave}")
            #     dist = self.follow_target(info.river_crab)
            # else:
            # if info.hero_enemy.info.behave == 'State_Dead':
            #     self.print(f"敌方英雄死亡, 结束!")
            #     exit()
            # """先升级, 然后进草找到蓝方英雄, 使用技能对其进行测试, 看是否有新的状态产生"""
            # if info.hero_our.level != self.last_level and self.n_frame > 300:
            #     self.last_level = info.hero_our.level
            #     self.print(f"LEVEL={self.last_level}")
            #     self.print(f"available skill={info.legal_actions[0]}")
            # if info.hero_our.level < 4:
            #     dist = self.move_target(2000, 2000)
            #     if dist < 2000:
            #         self.normal_attack()
            #     # if dist < 1000 and not self.first_use_skil:
            #     #     self.first_use_skil = True
            #     #     self.print("Use Skill 1")
            #     #     self.use_skill(0)
            #         # self.normal_attack(target='soldier1')
            #         # for i in range(3):    # 轮流释放技能
            #         #     flag = self.use_skill(i)
            #         #     if flag:
            #         #         self.print(f"Use skill{i+1}")
            #         #         break
            #         # if not flag:
            #         # self.normal_attack()
            # else:

            #     ### 公孙离释放1,2技能分别等1s后回伞 ###
            #     # flag = False
            #     # skill_id = 0 if self.use_skill_count in [0, 1] else 1
            #     # if self.use_skill_count in [0,2] or (self.use_skill_count in [1, 3] and self.n_frame - self.frame_debug > 30):
            #     #     flag = self.use_skill(skill_id)
            #     # if flag:
            #     #     self.print(f"Use skill {skill_id+1}")
            #     #     self.use_skill_count += 1
            #     #     self.frame_debug = self.n_frame
            #     # else:
            #     #     self.print(f"Wait skill {skill_id+1}, delta_frame={self.n_frame - self.frame_debug}")
            #     
            #     dist = self.move_target(-10000, 4000)    # 蓝方的下草, 红方的上草
            #     if info.hero_enemy.info.position[0] != info.UNSEEN_PADDING:
            #         if not self.find_target:
            #             self.print(f"FIND camp1")
            #             self.find_target = True
            #         dist = self.follow_target(info.hero_enemy.info)
            #         if dist < 500 or self.has_been_near:
            #             self.has_been_near = True
            #             # self.print(f"{dist=}, position our={info.hero_our.info.position}, enemy={info.hero_enemy.info.position}")
            #             if not self.near_target:
            #                 self.print(f"NEAR camp1")
            #                 self.near_target = True
            #             self.action = NO_ACTION
            #             ### 轮流使用1,2,3技能 ###
            #             if not self.first_use_skill1:
            #                 flag = self.use_skill(0)
            #                 if flag:
            #                     self.print("Use skill 1")
            #                     self.first_use_skill1 = True
            #             elif not self.first_use_skill2:
            #                 flag = self.use_skill(1)
            #                 if flag:
            #                     self.print("Use skill 2")
            #                     self.first_use_skill2 = True
            #             elif not self.first_use_skill3:
            #                 flag = self.use_skill(2)
            #                 if flag:
            #                     self.print("Use skill 3")
            #                     self.first_use_skill3 = True
            #             else:
            #                 self.normal_attack()
            #             ### 调狄仁杰1,2,3技能, 伽罗2技能 ###
            #             # flag = self.use_skill(1, 'hero')
            #             # if not self.first_use_skill2:
            #             #     flag = self.use_skill(1)
            #             #     if flag:
            #             #         self.first_use_skill2 = True
            #             #         self.print(f"Use skill 2")
            #             # else:
            #             #     self.normal_attack()
            #             ### 调伽罗1,3技能 公孙离3技能 ###
            #             # if self.use_skill_count < 2:
            #             #     flag = False
            #             #     if self.use_skill_count == 0 or (self.use_skill_count == 1 and self.n_frame - self.frame_debug > 30):
            #             #         # flag = self.use_skill(0)    # 伽罗1技能
            #             #         flag = self.use_skill(2, 'hero')    # 公孙离3技能
            #             #     if flag:
            #             #         self.use_skill_count += 1
            #             #         self.print(f"Use skill 3 count={self.use_skill_count}")
            #             #         self.frame_debug = self.n_frame
            #             #     else:
            #             #         self.print(f"Wait skill 3, delta_frame={self.n_frame - self.frame_debug}")
            #             # elif not self.first_use_skill3:
            #             #     self.first_use_skill3 = True
            #             #     flag = self.use_skill(2)
            #             #     if flag:
            #             #         self.print(f"Use skill 3")
            #             # else:
            #             #     self.print(f"Move to (-1000, -1000)")
            #             #     self.move_target(-1000, -1000)
                            

            ### 检查一步移动距离 ###
            # if self.last_position is not None:
            #     self.count += 1
            #     dist = get_dist(self.last_position, info.hero_our.info.position)
            #     self.avg_move_dist += (dist - self.avg_move_dist) / self.count
            #     self.print(f"Move dist={dist:.2f} (Avg={self.avg_move_dist:.2f})")
            # self.last_position = info.hero_our.info.position.copy()

        # self.print_position()

        ### 检查阵亡信息 ###
        # if self.camp == 1:
        if len(info.deads.soldier) > 0:
            for dead in info.deads.soldier:
                self.print(f"soldier{dead.death.camp+1} was killed by: {dead.killer.type}")
        if len(info.deads.hero) > 0:
            for dead in info.deads.hero:
                self.print(f"hero was killed by: {dead.killer.type}")
        if len(info.deads.river_crab) > 0:
            for dead in info.deads.river_crab:
                self.print(f"river crab was killed by: {dead.killer.type}")

        ### 检查子弹信息 ###
        n = len(info.bullets_our.merge) + len(info.bullets_enemy.merge)
        self.max_bullets = max(n, self.max_bullets)
        # if n:
        #     # self.print(f"Total bullets: {n}")
        #     for bullet in info.bullets_our.hero:
        #         self.print(f"Hero Bullet: {bullet.type}, slot={bullet.slot_type}, cmap={bullet.camp+1}")
        #         # self.print(f"Bullet: {bullet.type}, slot={bullet.slot_id}, skill={bullet.skill_id}, cmap={bullet.camp+1}")
        
        ### 金币增长量 ###
        # if self.camp == 1:
        #     now = info.hero_our.money_total
        #     delta = now - self.last_money
        #     self.max_money_delta = max(self.max_money_delta, delta)
        #     if delta != 0:
        #         self.print(f"Money delta={delta}")
        #     self.last_money = now
        
        if info.n_frame % 1000 < 6 or info.n_frame > GameConfig.debug_total_frames - 10:
            self.print(f"behaves={self.behaves}")
            self.print(f"soldier behaves={self.soldier_behaves}")
            self.print(f"buff_dict[skills]={sorted(self.buff_dict['skills'])}, buff_dict[marks]={sorted(self.buff_dict['marks'])}")
            # self.print(f"max_hp_delta_minus={self.max_hp_delta_minus}")
            # self.print(f"all id2type={info.id2type}, len={len(info.id2type)}")
            self.print(f"max_bullets={self.max_bullets}")
            # self.print(f"Max money delta={self.max_money_delta}")
        
        ### 检查不同hero的sub_action_mask ###
        # self.print(f"{Args.ID2HERO_NAME[info.hero_our.info.config_id]}, {info.sub_action_mask}")
        
        return self.action
    
    def normal_attack(self, target=None):
        """target选择为
        'hero': 敌方英雄
        'soldier[0,1,2,3]: 距离依次最近的四个小兵    (TODO: 检查下)
        'organ': 最近敌方防御塔
        """
        a = self.action = [3, 0, 0, 0, 0, 0]
        if target is None:
            return a
        if target == 'hero':
            a[-1] = 1
        elif 'soldier' in target:
            print(f"GG: Use target action {target}")
            a[-1] = int(target[-1]) + 3
        elif 'organ' in target:
            a[-1] = 7
        else:
            raise Exception(f"Don't know {target=}")
        return a
    
    def print_position(self):
        self.print(f"position: {self.info.hero_our.info.position}")

    def print(self, message):
        message = f"[DEBUG] Frame{self.n_frame} Camp{self.camp+1}: {message}"
        # self.logger.info(message)
        print(message)
    
    def move_target(self, x, z):
        """ 到达指定位置所需的动作, camp默认None, 即两方都到达该位置
        Returns:
            distance: 二者相差的L2距离
        """
        info = self.info
        target = np.array([x, z], np.float32)
        now = info.hero_our.info.position
        delta = self.delta_action_16x16(now, target)
        self.action = [2, delta[0], delta[1], 0, 0, 0]
        return get_dist(now, target)
    
    def follow_target(self, actor_info: ActorInfo):
        x, z = actor_info.position
        return self.move_target(x, z)

    def use_skill(self, skill_id: int, target: str=None, skill_x: int=1, skill_z: int=1):
        """
        Args:
            skill_id: int, 0,1,2,3,4,5分别表示一二三技能,回复,闪现,回城
            target: str | None, 选择小兵从近到远'soldier0, soldier1, soldier2, soldier3'
        Returns:
            success: [bool] 是否成功执行动作
        """
        assert skill_id in [0,1,2,3,4,5]
        id = 4 + skill_id
        info = self.info
        if len(info.legal_actions) == 0:
            self.print(f"[WARNING] No legal_actions skip use skill!")
            return False
        if not info.legal_actions[0][id]: return False
        legal_target = info.legal_actions[-1][id]
        if target is None:
            target = np.argwhere(legal_target).reshape(-1)[0]
        else:
            if 'soldier' in target: target = 3 + int(target[-1])
            elif target == 'hero':    # 可以选择 1, 2, 8 (只执行了一次, 没有任何作用)
                target = 1
            elif target == 'moster': target = 7
            else: ValueError
            if not legal_target[target]: return False
        mask = np.array(info.sub_action_mask[id], np.int32)
        self.action = np.array([id, 0, 0, skill_x, skill_z, target], np.int32) * mask
        self.action = self.action.tolist()
        return True
    
    @staticmethod
    def delta_action_16x16(center, target):
        delta = target - center
        return np.ceil(delta / np.max(np.abs(delta)) * 7).astype(np.int32) + [8, 8]
