""" 从tools.env_conf_manager拷贝过来
1. 加入对eval_opponent_type按照eval_opponent_types进行随机轮换
"""
import random
from tools.train_env_conf_validate import read_usr_conf


class EnvConfManager:
    def __init__(self, config_path, logger):
        self.config_path = config_path
        self.logger = logger
        self.usr_conf = None
        self.episode_cnt = 0
        self.eval_interval = 0
        self.random_eval_start = 0
        self.default_opponent_agent = None
        self.auto_switch_monitor_side = False
        self.monitor_side = 0
        self.initialize()

    def initialize(self):
        # Load configuration file
        # 读取并验证配置文件
        self.usr_conf = read_usr_conf(self.config_path, self.logger)
        if self.usr_conf is None:
            raise ValueError("usr_conf is None, please check the configuration file")

        # Get evaluation interval and default opponent type
        # 获取评估间隔和默认对手类型
        self.eval_interval = self.usr_conf["episode"]["eval_interval"] + 1
        self.default_opponent_agent = self.usr_conf["episode"]["opponent_agent"]

        # Determine whether to automatically switch the training agent's side
        # 确定是否自动切换训练智能体的阵营
        self.auto_switch_monitor_side = self.usr_conf["monitor"]["auto_switch_monitor_side"]
        self.monitor_side = self.usr_conf["monitor"]["monitor_side"]

        # Randomly initialize the random evaluation start point
        # 初始化随机评估起始点
        if self.eval_interval != 0:
            self.random_eval_start = random.randint(0, self.eval_interval)
        else:
            self.random_eval_start = 0

    def get_current_config(self):
        # Get the current episode configuration
        # 获取当前对局的配置
        return self.usr_conf

    def get_monitor_side(self):
        # Get the current monitor side
        # 获取当前对局的上报阵营
        return self.monitor_side

    def get_opponent_agent(self):
        # Get the current opponent agent
        # 获取当前对局的对手类型
        return self.usr_conf["episode"]["opponent_agent"]

    def update_config(self, lineup=None):
        # Update the configuration
        # 更新对局配置，包括阵容和训练智能体的ID
        if lineup:
            # 确保阵容是有效的英雄ID列表
            if len(lineup) == 2 and all(type(hero_id) == int for hero_id in lineup[:2]):
                self.usr_conf["lineups"]["blue_camp"][0]["hero_id"] = lineup[0]
                self.usr_conf["lineups"]["red_camp"][0]["hero_id"] = lineup[1]
            else:
                raise ValueError("Invalid lineup format, expected list of 2 integers")

        # Determine whether to switch the monitor side
        # 确定是否自动切换上报监控阵营
        if self.auto_switch_monitor_side:
            self.monitor_side = 1 - self.monitor_side
        self.usr_conf["monitor"]["monitor_side"] = self.monitor_side

        # Determine whether to evaluate
        # 确定是否进行评估
        is_eval = (
            (self.episode_cnt + self.random_eval_start) % self.eval_interval == 0
        )
        if is_eval:
            eval_opponent_idx = random.randint(0, len(self.usr_conf['episode']['eval_opponent_types'])-1)
            self.usr_conf['episode']['eval_opponent_type'] = self.usr_conf['episode']['eval_opponent_types'][eval_opponent_idx]
        opponent_agent = self.default_opponent_agent if not is_eval else self.usr_conf["episode"]["eval_opponent_type"]
        self.usr_conf["episode"]["opponent_agent"] = opponent_agent

        self.episode_cnt += 1

        return self.get_current_config(), is_eval, self.get_monitor_side()
