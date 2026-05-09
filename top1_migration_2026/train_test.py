#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import json
import os
from pathlib import Path

from kaiwudrl.common.utils.train_test_utils import run_train_test

# To run the train_test, you must modify the algorithm name here. It must be one of ppo, diy.
# Simply modify the value of the algorithm_name variable.
# 运行train_test前必须修改这里的算法名字, 必须是ppo、diy里的一个, 修改algorithm_name的值即可
algorithm_name_list = ["ppo", "diy"]
algorithm_name = "ppo"


DEFAULT_DUMP_OBS_FRAMES = "56,500,1000,1094,1148,1500,1778,2500,4000,6000,9000,12000"


def build_env_vars():
    env_vars = {
        "replay_buffer_capacity": "10",
        "preload_ratio": "1.0",
        "train_batch_size": "2",
        "dump_model_freq": "1",
    }
    if os.environ.get("DUMP_OBS"):
        dump_config = {
            "DUMP_OBS": os.environ.get("DUMP_OBS", "1"),
            "DUMP_OBS_DIR": os.environ.get("DUMP_OBS_DIR", "/data/projects/hok1v1/debug_obs"),
            "DUMP_OBS_FRAMES": os.environ.get("DUMP_OBS_FRAMES", DEFAULT_DUMP_OBS_FRAMES),
            "DUMP_OBS_MAX_EPISODES": os.environ.get("DUMP_OBS_MAX_EPISODES", "4"),
            "DUMP_OBS_PRINT_SCHEMA": os.environ.get("DUMP_OBS_PRINT_SCHEMA", "1"),
            "DUMP_OBS_TRAIN_TEST_FRAMES": os.environ.get("DUMP_OBS_TRAIN_TEST_FRAMES", "12500"),
        }
        env_vars.update(dump_config)
        written_paths = []
        for config_path in (Path("/data/projects/hok1v1/.dump_obs_config.json"), Path("/workspace/code/.dump_obs_config.json")):
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with config_path.open("w", encoding="utf-8") as file_obj:
                    json.dump(dump_config, file_obj, ensure_ascii=False, indent=2)
                written_paths.append(str(config_path))
            except Exception:
                continue
        print(
            f"DUMP_OBS config written to {written_paths}, frames {dump_config['DUMP_OBS_FRAMES']}, "
            f"output dir {dump_config['DUMP_OBS_DIR']}",
            flush=True,
        )
    return env_vars


if __name__ == "__main__":
    run_train_test(
        algorithm_name=algorithm_name,
        algorithm_name_list=algorithm_name_list,
        env_vars=build_env_vars(),
    )
