import numpy as np
from agent_ppo.conf.conf import Args

class ListRearrange:
    """ 将List按照玩家当前控制hero编号分组排序 (包含正向与反向变换) """
    def __init__(self, total_hero_class: int = len(Args.HERO_CONFIG_ID)):
        self.C = total_hero_class
    
    def update(self, hero_idx: list):
        """ 基于英雄编号构建正反向映射列表 """
        self.n = len(hero_idx)
        hero_idx = np.array(hero_idx, np.int32)
        assert hero_idx.min() >= 0 and hero_idx.max() < self.C
        self.forward_idx = np.argsort(hero_idx)
        self.inverse_idx = np.argsort(self.forward_idx)
        self.split_nums = []
        for i in range(self.C):
            self.split_nums.append(sum(hero_idx==i))
    
    def forward(self, xs: list):
        return [xs[self.forward_idx[i]] for i in range(self.n)]
    
    def inverse(self, ys: list):
        return [ys[self.inverse_idx[i]] for i in range(self.n)]

if __name__ == '__main__':
    list_rearrange = ListRearrange(4)
    a = [3, 3, 1, 1, 2, 0, 2, 3]
    list_rearrange.update(a)
    b = list_rearrange.forward(a)
    print(b, list_rearrange.split_nums)
    c = list_rearrange.inverse(b)
    print(c)
