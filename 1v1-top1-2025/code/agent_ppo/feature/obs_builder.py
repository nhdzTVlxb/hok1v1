import math
import numpy as np
from agent_ppo.feature.unpack_state_dict import (
    Info, ActorInfo, HeroInfo, SlotInfo, OrganInfo, SoldierInfo,
    BulletInfo, CakeInfo
)
from agent_ppo.conf.conf import Args
from typing import List

def clip(x, mn, mx):
    return max(min(x, mx), mn)

def fix(x):
    return math.floor(x) if x > 0 else math.ceil(x)

class ObsBuilder:
    def __init__(self, logger=None):
        self.logger = logger
        self.reset()
    
    def reset(self):
        self.last_money = [None, None]
        self.next_cake_frame = [1778, 1778]    # 首次血包生成帧
    
    def process_position(self, position):
        """将坐标信息转化为相对位置以及全局位置的one_hot"""
        # 相对位置特征 x_rpos
        p = position
        if p[0] == Info.UNSEEN_PADDING:
            return [0] * Args.DIM_DISTANCE
        unit_size, max_size = Args.RELATIVE_DISTANCE_UNIT_SIZE, Args.RELATIVE_DISTANCE_MAX_SIZE
        max_idx = max_size / unit_size + 1    # 向下取整41x41左右各加一个维度作为溢出, 0~42
        max_idx = max_idx / 2    # 单边最大索引 21
        rpos = [    # 相对位置
            int(clip(fix((p[i] - self.pos[i]) / unit_size), -max_idx, max_idx) + max_idx) for i in range(2)
        ]
        x_dis = clip(math.dist(p, self.pos) / (max_size / 2), 0, 1)    # 相对距离
        rpos_dim = int(2 * max_idx + 1)    # 维度大小 43
        x_rpos = [0] * (rpos_dim * 2 + 1)
        x_rpos[rpos[0]] = x_rpos[rpos[1] + rpos_dim] = 1.0
        x_rpos[-1] = x_dis

        # 全局坐标特征 x_wpos
        unit_size = Args.WHOLE_DISTANCE_UNIT_SIZE
        max_size = Args.WHOLE_DISTANCE_MAX_SIZE
        wpos_dim = int(max_size / unit_size)
        wpos = [    # 相对位置
            int(clip(math.floor((p[i] + max_size / 2) / unit_size), 0, wpos_dim-1)) for i in range(2)
        ]
        x_wratio = [    # 相对小区域的比例位置
            clip((p[i] + max_size / 2) / unit_size - wpos[i], -1, 1) for i in range(2)
        ]
        x_wpos = [0] * (wpos_dim * 2 + 2)
        x_wpos[wpos[0]] = x_wpos[wpos[1] + wpos_dim] = 1.0
        x_wpos[-2] = x_wratio[0]
        x_wpos[-1] = x_wratio[1]

        return x_rpos + x_wpos

    def process_unit(self, info: ActorInfo):
        """处理所有单位的共用信息 (相对位置, 全局位置, 血量, 公孙离印记buff)"""
        x_pos = self.process_position(info.position)
        # HP
        unit_size, max_size = Args.HP_UNIT_SIZE, Args.HP_MAX_SIZE
        hp_dim = int(max_size / unit_size) + 2    # 离散hp维度26 (向上取证)
        x_hp = [0] * (1 + hp_dim)     # (now/max, 离散hp) = (27,)
        x_hp[0] = info.hp / info.hp_max
        x_hp[1+min(int(math.ceil(info.hp/unit_size)), hp_dim-1)] = 1.0
        # 印记
        x_mark = [0] * Args.DIM_MARK
        now_feature_idx = 0
        use_marks = 0
        for id, max_layer in Args.MARK_ID_LAYERS.items():
            if id in info.buff.marks_ids:
                index = info.buff.marks_ids.index(id)
                layer = max(min(info.buff.marks_layers[index], max_layer), 0)
                x_mark[now_feature_idx+layer] = 1.0
                use_marks += 1
            now_feature_idx += max_layer + 1
        if use_marks < len(info.buff.marks_ids):  # 其他未知印记
            x_mark[-1] = 1.0
        return x_hp + x_mark + x_pos
    
    def process_skill(self, info: SlotInfo):
        unit_size, max_size = Args.CD_UNIT_SIZE, Args.CD_MAX_SIZE
        cd_dim = int(max_size / unit_size) + 2    # 离散cd维度11 (向上取整)
        x_cd = [0] * (2 + cd_dim)    # (now/max, 离散cd, 没学技能或看不到对方cd) = (13,)
        x_cd[0] = info.cd / info.cd_max
        x_cd[1+min(int(math.ceil(info.cd/unit_size)), cd_dim-1)] = 1.0
        if not info.usable and info.level == 0:    # 还有一种情况是看不见对方cd且level=0, 但是对方技能usable=True, 这时对方也可以用该技能
            x_cd[-1] = 1.0
        return x_cd
    
    def process_money(self, money, is_enemy: bool):
        idx = int(is_enemy)
        delta = money - self.last_money[idx]
        self.last_money[idx] = money
        unit_size, max_size = Args.MONEY_UNIT_SIZE, Args.MONEY_MAX_SIZE
        money_dim = int(max_size / unit_size) + 1  # 离散money维度16 (向下取整+1)
        x_money = [0] * (2 + money_dim)    # (离散money, 当增长量<10时(每秒自动+4~6块), 总金钱比例)
        if 0 < delta < 20:
            x_money[-2] = 1.0
        else:
            x_money[max(min(int(math.floor(delta/unit_size)), money_dim-1), 0)] = 1.0
        x_money[-1] = min(money / 10000, 1.0)
        return x_money
    
    def process_organ(self, hero: HeroInfo, enemy_sub_tower: ActorInfo):
        tower = enemy_sub_tower
        x_organ = [0] * 2    # (是否在塔的攻击范围内, 是否为塔的攻击目标)
        x_organ[0] = float(math.dist(hero.info.position, tower.position) <= tower.attack_range)
        x_organ[1] = float(hero.info.id == tower.attack_target)
        return x_organ
    
    def process_hero(self, hero: HeroInfo, is_enemy: bool, enemy_sub_tower: ActorInfo):
        # 通用特征 x_unit
        x_unit = self.process_unit(hero.info)
        # 类型 x_hero_id
        config_ids = Args.HERO_CONFIG_ID
        # x_hero_id = [0] * len(config_ids)  # 这是v8后面的写法, 但是和多输出头方案冲突
        # x_hero_id[config_ids.index(hero.info.config_id)] = 1.0
        x_hero_id = [config_ids.index(hero.info.config_id) - 1]  # (-1, 0, 1)
        # 专属行为 x_behave
        behaves = Args.HERO_BEHAVE
        x_behave = [0] * (len(behaves)+1)
        idx = -1
        if hero.info.behave in behaves:
            idx = behaves.index(hero.info.behave)
        x_behave[idx] = 1.0
        # 法力 x_ep
        unit_size, max_size = Args.EP_UNIT_SIZE, Args.EP_MAX_SIZE
        ep_dim = int(max_size / unit_size) + 1    # 离散ep维度9 (向下取证)
        x_ep = [0] * (1 + ep_dim)     # (now/max, 离散hp) = (10,)
        x_ep[0] = hero.info.ep / hero.info.ep_max
        x_ep[1+min(int(math.floor(hero.info.ep/unit_size)), ep_dim-1)] = 1.0
        # 冷却 x_skill1, x_skill2, x_skill3, x_skill_flash, x_skill_recover
        s = hero.skill
        x_skill1, x_skill2, x_skill3, x_skill_flash, x_skill_recover = (
            self.process_skill(x) for x in [s.first, s.second, s.thrid, s.flash, s.recover]
        )
        # 等级 x_level
        x_level = [0] * Args.LEVEL_MAX
        x_level[hero.level-1] = 1.0
        # 金币获取 x_money
        x_money = self.process_money(hero.money_total, is_enemy)
        # 是否在草丛 x_grass
        x_grass = [float(hero.flag_in_grass)]
        # (是否在塔的攻击范围内, 是否为塔的攻击目标) x_organ
        x_organ = self.process_organ(hero, enemy_sub_tower)
        # buff
        x_buff = [0] * Args.DIM_BUFF
        for id in hero.info.buff.skill_ids:
            if id in Args.BUFFS:
                x_buff[Args.BUFFS.index(id)] = 1.0
            else:
                x_buff[-1] = 1.0  # 未知buff
        
        return (
            x_hero_id + x_behave + x_ep +
            x_skill1 + x_skill2 + x_skill3 + x_skill_flash + x_skill_recover +
            x_level + x_money + x_grass + x_organ +
            x_buff +
            x_unit
        )
    
    def process_soldier(self, soldiers: List[SoldierInfo], opposed_sub_tower: ActorInfo):
        """按照距离从近到远排序, 专属行为, 类型, 防御塔相关参数"""
        soldiers = sorted(soldiers, key=lambda s: math.dist(s.position, self.pos))    # 按照距离从从小到大排序
        max_num = Args.SOLDIER_MAX_NUM
        soldiers = soldiers[:max_num]    # 最大小兵数目
        behavers = Args.SOLDIER_BEHAVE
        config_ids = Args.SOLDIER_CONFIG_ID
        soldier_dim = Args.DIM_SOLDIER    # len(SOLDIER_BEHAVE) + 1 + len(SOLDIER_CONFIG_ID) + 2 + DIM_UNIT
        x_soldiers = []
        mask_soldiers = [0] * max_num
        for i, soldier in enumerate(soldiers):
            x_soldier = [0] * (soldier_dim - Args.DIM_UNIT)
            # 专属行为
            base_idx = 0
            idx = 2
            if soldier.behave in behavers:
                idx = behavers.index(soldier.behave)
            x_soldier[base_idx + idx] = 1.0
            base_idx += len(behavers) + 1    # 3
            # 类型
            for j, ids in enumerate(config_ids):
                if soldier.config_id in ids:
                    x_soldier[base_idx + j] = 1.0
            base_idx += len(config_ids)
            # (是否在防御塔攻击范围内, 是否为防御塔攻击目标)
            if math.dist(soldier.position, opposed_sub_tower.position) <= opposed_sub_tower.attack_range:
                x_soldier[base_idx] = 1.0
            if soldier.id == opposed_sub_tower.attack_target:
                x_soldier[base_idx+1] = 1.0
            # 通用信息
            x_soldiers += x_soldier + self.process_unit(soldier)
            mask_soldiers[i] = 1.0
        x_soldiers += [0] * (Args.DIM_SOLDIERS - len(x_soldiers))
        return x_soldiers, mask_soldiers

    def process_river_crab(self):
        behave = Args.RIVER_CRAB_BEHAVE
        x_behave = [0] * (len(behave) + 1)
        crab = self.info.river_crab
        mask = [0]
        if crab is None:
            return [0] * Args.DIM_RIVER_CRAB, mask
        mask[0] = 1.0
        idx = -1
        if crab.behave in behave:
            idx = behave.index(self.info.river_crab.behave)
        x_behave[idx] = 1.0
        return x_behave + self.process_unit(crab), mask
    
    def process_sub_tower(self, sub_tower: ActorInfo, cake: CakeInfo, is_enemy: bool):
        x_target = [0] * 5  # 无, 英雄, 小兵, 塔后是否有血包, 血包生成剩余时间
        if sub_tower.attack_target == 0:
            x_target[0] = 1.0
        else:
            type = self.info.id2type.get(sub_tower.attack_target, None)
            if type is None:
                message = f"sub_tower target={sub_tower.attack_target} not in id2type{self.info.id2type}"
                if self.logger is not None: self.logger.warning(message)
                else: print(message)
            elif type == 'hero':
                x_target[1] = 1.0
            elif type == 'soldier':
                x_target[2] = 1.0
            else:
                message = f"sub_tower target type={type} not in [hero, soldier]"
                if self.logger is not None: self.logger.warning(message)
                else: print(message)
        x_target[3] = float(cake is not None)
        is_enemy = int(is_enemy)
        if cake is not None:
            self.next_cake_frame[is_enemy] = self.n_frame + 76 * 30  # 计算下次生成血包的帧数
            x_target[4] = 0
        else:
            x_target[4] = min((self.next_cake_frame[is_enemy] - self.n_frame) / (75 * 30), 1)  # 血包生成时间为75s
        return x_target + self.process_unit(sub_tower)

    def process_bullet(self, bullet: BulletInfo):
        x_slot = [0] * (Args.DIM_BULLET - Args.DIM_DISTANCE)
        # 来源技能
        slots = Args.BULLET_SLOT
        base_idx = 0
        x_slot[base_idx+slots.index(bullet.slot_type)] = 1.0
        base_idx += len(slots)
        return x_slot + self.process_position(bullet.position)
        
    def process_bullets(self):
        # 只考虑最近9个敌方英雄子弹
        hero_bullets = self.info.bullets_enemy.hero
        hero_bullets = sorted(hero_bullets, key=lambda b: math.dist(b.position, self.pos))
        hero_bullets = hero_bullets[:Args.BULLET_MAX_NUM-1]
        x_bullets = []
        dim_bullet = Args.DIM_BULLET
        masks = [0] * Args.BULLET_MAX_NUM
        for i, b in enumerate(hero_bullets):
            x_bullets += self.process_bullet(b)
            masks[i] = 1.0
        x_bullets += [0] * ((Args.BULLET_MAX_NUM-1) * Args.DIM_BULLET - len(x_bullets))
        # 考虑1个敌方防御塔子弹
        organ_bullets = self.info.bullets_enemy.organ
        if len(organ_bullets) > 1:
            organ_bullets = sorted(organ_bullets, key=lambda b: math.dist(b.position, self.pos))
        if len(organ_bullets):
            x_bullets += self.process_bullet(organ_bullets[0])
            masks[-1] = 1.0
        x_bullets += [0] * (Args.DIM_BULLETS - len(x_bullets))
        
        return x_bullets, masks

    def process_gsl_bullet(self, bullet: BulletInfo):
        x_gsl_bullet = [0] * (Args.DIM_GSL_BULLET - Args.DIM_DISTANCE)
        # 来源技能
        slots = Args.BULLET_GSL_SLOT
        x_gsl_bullet[slots.index(bullet.slot_type)] = 1.0
        x_gsl_bullet += self.process_position(bullet.position)
        return x_gsl_bullet
                
    def build_observation(self, info: Info, need_mask=False):
        """ 将Info解包的信息构造为Obs
        Return:
            obs: [np.ndarray] 长度为Args.DIM_ALL的向量
            masks (need_mask=True): [Tuple[np.ndarray]] 包含3个mask, 分别为
                mask_solider: [np.ndarray] 长度为Args.DIM_SOLDIER * 2, 表示敌我双方小兵的mask
                mask_bullet: [np.ndarray] 长度为Args.DIM_BULLETS, 表示敌方子弹的mask
                mask_gsl_bullet: []
        """
        self.info = info
        self.n_frame = info.n_frame
        self.pos = info.hero_our.info.position
        if self.last_money[0] is None:  # 如果是第一次计算last_money直接设置为当下的money, 否则在计算完money后保存
            self.last_money = [info.hero_our.money_total, info.hero_enemy.money_total]

        """将info中信息转化为obs向量"""
        # 英雄
        x_hero_our = self.process_hero(info.hero_our, False, info.organ_enemy.sub_tower)
        x_hero_enemy = self.process_hero(info.hero_enemy, True, info.organ_our.sub_tower)
        # 小兵
        x_soldier_our, mask_soldier_our = self.process_soldier(info.soldiers_our.merge, info.organ_enemy.sub_tower)
        x_soldier_enemy, mask_soldier_enemy = self.process_soldier(info.soldiers_enemy.merge, info.organ_our.sub_tower)
        mask_soldier = mask_soldier_our + mask_soldier_enemy
        # 河蟹
        x_river_crab, mask_river_crab = self.process_river_crab()
        # 防御塔
        x_sub_tower_our = self.process_sub_tower(info.organ_our.sub_tower, info.cake_our, False)
        x_sub_tower_enemy = self.process_sub_tower(info.organ_enemy.sub_tower, info.cake_enemy, True)

        """ DEBUG 固定维度数值, 查看model是否正确划分 """
        # for x0, value in zip(
        #     [x_hero_our, x_hero_enemy, x_soldier_our, x_soldier_enemy, x_river_crab, x_sub_tower_our, x_sub_tower_enemy],
        #     [0, 1, 2, 3, 4, 5, 6]
        # ):
        #     for i in range(len(x0)):
        #         x0[i] = value

        x_all_units = (
            x_hero_our + x_hero_enemy + x_soldier_our + x_soldier_enemy +
            x_river_crab + x_sub_tower_our + x_sub_tower_enemy
        )
        assert len(x_all_units) == Args.DIM_ALL_UNITS, f"{len(x_all_units)=}, {Args.DIM_ALL_UNITS=}"

        # 子弹
        x_bullet, mask_bullet = self.process_bullets()

        """ DEBUG 固定维度数值, 查看model是否正确划分 """
        # x_bullet = [7] * Args.DIM_BULLET * 9 + [8] * Args.DIM_BULLET

        x = np.array(x_all_units + x_bullet, np.float32)
        if need_mask:
            return x, (mask_soldier, mask_bullet)
        return x
        
def debug_position():
    # p = np.array([600*20+900, -600*20-500])
    p = np.array([12300, 0])
    # p = np.array([-45000+4000+600, 45000-700-600])
    # p = np.array([0, 0])
    print(p)
    pos = np.array([0, 0])
    unit_size, max_size = Args.RELATIVE_DISTANCE_UNIT_SIZE, Args.RELATIVE_DISTANCE_MAX_SIZE
    max_idx = max_size / unit_size + 1    # 向下取整41x41左右各加一个维度作为溢出, 0~42
    max_idx = max_idx / 2    # 单边最大索引 21
    rpos = (np.clip(np.fix((p - pos) / unit_size), -max_idx, max_idx) + max_idx).astype(np.int32)
    print(rpos)
    rpos_dim = int(2 * max_idx + 1)    # 维度大小 43
    x_dis = np.clip(math.dist(p, pos) / (max_size / 2), 0, 1)
    x_rpos = np.zeros(rpos_dim * 2 + 1, np.float32)
    x_rpos[rpos[0]] = x_rpos[rpos[1] + rpos_dim] = 1.0
    x_rpos[-1] = x_dis
    print(x_dis, x_rpos, np.argwhere(x_rpos).reshape(-1))

    unit_size = Args.WHOLE_DISTANCE_UNIT_SIZE
    max_size = Args.WHOLE_DISTANCE_MAX_SIZE
    wpos_dim = int(max_size / unit_size)
    wpos = (np.clip(np.floor((p + max_size / 2) / unit_size), 0, wpos_dim-1)).astype(np.int32)
    x_wratio = np.clip((p+max_size/2) / unit_size - wpos, -1, 1)
    x_wpos = np.zeros(wpos_dim * 2 + 2, np.float32)
    x_wpos[wpos[0]] = x_wpos[wpos[1] + wpos_dim] = 1.0
    x_wpos[-2:] = x_wratio
    print(x_wratio, wpos, x_wpos, np.argwhere(x_wpos).reshape(-1))

def debug_hp():
    # HP
    hp = 2700
    hp_max = 5000
    unit_size, max_size = Args.HP_UNIT_SIZE, Args.HP_MAX_SIZE
    hp_dim = int(max_size / unit_size) + 2    # 离散hp维度26 (向上取证)
    x_hp = np.zeros(1 + hp_dim, np.float32)     # (now/max, 离散hp) = (27,)
    x_hp[0] = hp / hp_max
    x_hp[1+min(int(math.ceil(hp/unit_size)), hp_dim-1)] = 1.0
    print(x_hp, x_hp.shape)

def debug_ep():
    ep = 230
    ep_max = 300
    unit_size, max_size = Args.EP_UNIT_SIZE, Args.EP_MAX_SIZE
    ep_dim = int(max_size / unit_size) + 1    # 离散ep维度9 (向下取证)
    x_ep = np.zeros(1 + ep_dim, np.float32)     # (now/max, 离散hp) = (10,)
    x_ep[0] = ep / ep_max
    x_ep[1+min(int(math.floor(ep/unit_size)), ep_dim-1)] = 1.0
    print(x_ep)

def debug_cd():
    cd, cd_max = 4, 11
    usable, level = False, 1
    unit_size, max_size = Args.CD_UNIT_SIZE, Args.CD_MAX_SIZE
    cd_dim = int(max_size / unit_size) + 2    # 离散cd维度11 (向上取整)
    x_cd = np.zeros(2 + cd_dim, np.float32)    # (now/max, 离散cd, 没学技能或看不到对方cd) = (13,)
    x_cd[0] = cd / cd_max
    x_cd[1+min(int(math.ceil(cd/unit_size)), cd_dim-1)] = 1.0
    if not usable and level == 0:    # 还有一种情况是看不见对方cd且level=0, 但是对方技能usable=True, 这时对方也可以用该技能
        x_cd[-1] = 1.0
    print(x_cd, x_cd.shape)

def debug_money():
    delta = 270
    unit_size, max_size = Args.MONEY_UNIT_SIZE, Args.MONEY_MAX_SIZE
    money_dim = int(max_size / unit_size) + 1    # 离散money维度16 (向下取整)
    x_money = np.zeros(1 + money_dim, np.float32)    # (离散money, 当增长量<10时(每秒自动+4~6块))
    if 0 < delta < 20:
        x_money[-1] = 1.0
    else:
        x_money[min(int(math.floor(delta/unit_size)), money_dim-1)] = 1.0
    print(x_money, x_money.shape)

def debug_builder():
    import json, time
    # with open("/data/projects/hok1v1/agent_ppo/debug/important_frames/后羿3层印记1148.json", 'r') as file:
    # with open("/data/projects/hok1v1/agent_ppo/debug/important_frames/start/56.json", 'r') as file:
    with open("/data/projects/hok1v1/agent_ppo/debug/important_frames/出现cake_1778.json", 'r') as file:
    # with open("/data/projects/hok1v1/agent_ppo/debug/important_frames/李元芳4层印记1094.json", 'r') as file:
        state_dicts = json.load(file)
    info = Info(state_dicts['0'])
    print(info.cake_our, info.cake_enemy)
    obs_builder = ObsBuilder()
    start_time = time.time()
    obs, masks = obs_builder.build_observation(info, need_mask=True)
    print("list version time used:", time.time() - start_time)
    print(obs, obs.shape, Args.DIM_ALL)
    print(masks)

if __name__ == '__main__':
    # debug_position()
    # debug_hp()
    # debug_ep()
    # debug_cd()
    # debug_money()
    debug_builder()
