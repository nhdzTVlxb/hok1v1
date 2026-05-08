import itertools
import random

# Loop through camps, shuffling camps before each major loop
# 循环返回camps, 每次大循环前对camps进行shuffle
def _lineup_iterator_shuffle_cycle(camps):
    while True:
        random.shuffle(camps)
        for camp in camps:
            yield camp


# Specify single-side multi-agent lineups, looping through all pairwise combinations
# 指定单边多智能体阵容，两两组合循环
def lineup_iterator_roundrobin_camp_heroes(camp_heroes=None):
    if not camp_heroes:
        raise Exception(f"camp_heroes is empty")

    try:
        # 英雄ID，类型整数，取值范围: 169:后羿，173:李元芳，174:虞姬
        valid_ids = [169, 173, 174]
        for camp in camp_heroes:
            hero_id = camp[0]
            if hero_id not in valid_ids:
                raise Exception(f"hero_id {hero_id} not valid, valid is {valid_ids}")
    except Exception as e:
        raise Exception(f"check hero valid, exception is {str(e)}")

    camps = []
    for lineups in itertools.product(camp_heroes, camp_heroes):
        camp = []
        for lineup in lineups:
            camp.append(lineup[0])
        camps.append(camp)
    return _lineup_iterator_shuffle_cycle(camps)

if __name__ == '__main__':
    from agent_ppo.conf.conf import GameConfig
    random.seed(42)
    # Lineup iterator
    # 阵容生成器
    lineup_iter = lineup_iterator_roundrobin_camp_heroes(camp_heroes=GameConfig.CAMP_HEROES)
    for i in range(30):
        print(next(lineup_iter))
        if ((i+1)%9 == 0):
            print()
"""
[173, 169]
[174, 169]
[174, 173]
[173, 173]
[174, 174]
[169, 174]
[173, 174]
[169, 169]
[169, 173]

[169, 169]
[174, 173]
[169, 174]
[173, 174]
[173, 169]
[173, 173]
[174, 174]
[174, 169]
[169, 173]
...
"""