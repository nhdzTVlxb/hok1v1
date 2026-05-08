from typing import Dict, Set

""" 用户名相关 """
user_email = ""
user_password = ""

""" 自动评估 """
# 评估界面, 注意需要给出stage后面完整的编号
base_url = "https://aiarena.tencent.com/p/competition/stage/178/5516/1546"  # 2025王者高级赛道1v1复赛
# 评估界面的标题, 用于判断是否进入评估界面
eval_page_title = "智能体决策挑战-1v1"
# 一个包含已处理模型名称的集合，用于避免重复评估
add_model_set = set()
# add_model_set: Set[str] = {'v1_2_2_191475'}
# 之前预训练的总时长 (格式例如79h20min50s)
# previous_train_time = "54h22min"  # 1.2.1 90k
# previous_train_time = "127h6min"  # 1.2.2 211k
# previous_train_time = "176h39min"  # 1.2.6 293k
# previous_train_time = "221h59min"  # 1.2.8 368k
# previous_train_time = "272h36min"  # 1.2.9 452k
# previous_train_time = "349h52min24s"  # 1.2.10 581k
# previous_train_time = "378h17min26s"  # 1.2.10 628k
# previous_train_time = "454h5min5s"  # 1.2.11 753k
# previous_train_time = "406h58min"  # 1.2.15 675k
previous_train_time = "505h32min"  # 1.2.12 839k
# 存储待评估的对手模型及其在任务名称中的缩写
eval_model_dict: Dict[str, str] = {
    # '90245': '90k',
    # '150877': '151k',
    # '203214': '203k',
    # '318607': '319k',
    # '421060': '421k',
    # '435387': '435k',
    # '580517': '581k',
    '659920': '660k',
    '753337': '753k',
    '799513': '800k',
    'baseline7': 'b7',
}
# 对战的回合数 (最大为5)
battle_rounds = 3
