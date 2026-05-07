# READNE2

## 这次平台现象

- `train_test.py` 可以启动训练，但局内智能体长期停在泉水。
- 监控里 `reward/value_loss/policy_loss/total_loss` 基本贴近 0，`entropy_loss` 还在动。
- common_ai 评估胜率为 0，敌方塔血几乎不掉，己方持续死亡。

## 主要问题

- `agent_ppo/feature/definition.py`
  - `build_frame()` 用原始 `hero["camp"] == agent.hero_camp` 判断是否训练。
  - 平台里 camp 可能混用 `0/1`、`1/2`、`PLAYERCAMP_1/2`，原始值不相等时 `is_train=False`。
  - 结果是样本进了 learner，但训练 loss 基本不生效。

- `agent_ppo/feature/reward_process.py`
  - reward 只用 `player_id` 匹配 `hero.runtime_id/player_id`。
  - 如果平台上的 `observation["player_id"]` 和英雄 `runtime_id` 不一致，主英雄找不到，reward 子项不更新。

- `agent_ppo/feature/top1_feature_builder.py`
  - camp 字符串 `"0"`、`"1"` 的归一化顺序不严谨，可能把 `"0"` 保留成 0，而不是平台侧常用的 1 号阵营。

## 已修改

- `agent_ppo/agent.py`
  - 新增 `camp_id()`，reset 时统一归一化 `hero_camp`。
  - `GameRewardManager` 初始化时额外传入 `hero_camp`。
  - `_main_hero_state()` 增加 camp 归一化 fallback。

- `agent_ppo/feature/definition.py`
  - 新增 `camp_id()`。
  - `build_frame()` 用归一化后的 camp 判断 `is_train`，避免有效样本被标成不训练。

- `agent_ppo/feature/reward_process.py`
  - 新增/修正 `camp_id()`。
  - `GameRewardManager` 支持 `main_hero_camp`。
  - 当 `player_id/runtime_id` 匹配失败时，用主阵营 camp fallback 找主英雄和敌方阵营。

- `agent_ppo/feature/top1_feature_builder.py`
  - 修正 camp 字符串 `"0"`、`"1"`、`"2"` 的归一化顺序。

## 验证

- 本地通过 `py_compile` 语法检查。
- 用合成帧验证：即使 `player_id` 对不上 `runtime_id`，只要 camp 对得上，第二帧 reward 能从 0 变成非零。

## 下一步看平台

- 先看 `reward/value_loss/policy_loss/total_loss` 是否不再长期为 0。
- 如果 reward 恢复但仍不出门，再单独修动作兜底或开局引导。
