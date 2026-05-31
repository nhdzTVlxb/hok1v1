#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.utils.train_test_utils import run_train_test

# To run the train_test, you must modify the algorithm name here. It must be one of ppo, diy.
# Simply modify the value of the algorithm_name variable.
# 运行train_test前必须修改这里的算法名字, 必须是ppo、diy里的一个, 修改algorithm_name的值即可
algorithm_name_list = ["ppo", "diy"]
algorithm_name = "ppo"


if __name__ == "__main__":
    run_train_test(
        algorithm_name=algorithm_name,
        algorithm_name_list=algorithm_name_list,
        env_vars={
            "replay_buffer_capacity": "10",
            "preload_ratio": "1.0",
            "train_batch_size": "2",
            "dump_model_freq": "1",
            "DUMP_OBS": "1",
            "DUMP_OBS_PRINT_SCHEMA": "1",
            "DUMP_OBS_MAX_EPISODES": "2",
            "DUMP_OBS_FRAMES": "56,500,1000,1778,2500,4000,6000,9000",
            "DUMP_OBS_EVERY_N": "500",
            "DUMP_OBS_DIR": "debug_obs",
            "FEATURE_AUDIT": "1",
            "FEATURE_AUDIT_FRAMES": "56,500,1000,1778,2500,4000,6000,9000",
            "FEATURE_AUDIT_EVERY_N": "500",
            "FEATURE_AUDIT_DIR": "feature_audit",
            "ACTION_AUDIT": "1",
            "ACTION_AUDIT_FRAMES": "56,100,200,300,500,1000,1500,2000,4000,6000",
            "ACTION_AUDIT_EVERY_N": "500",
            "ACTION_AUDIT_DIR": "action_audit",
            "TRAIN_TEST_MAX_FRAME": "6000",
        },
    )
