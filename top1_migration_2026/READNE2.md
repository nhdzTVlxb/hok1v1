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
- 看局内英雄是否离开泉水。
- 如果 reward 恢复但仍不出门，再单独修动作兜底或开局引导。

## 第二轮现象

- 平台环境指标开始有波动，但 common_ai 评估仍然很差。
- 局内表现为：要么泉水不动，要么一顿一顿走到中间，几乎不普攻/技能，最后被打死。
- `log-dead1` 中能看到模型正常加载、评估正常结束，但没有逐帧 action 日志，无法直接统计 action 分布。

## 第二轮修改

- `agent_ppo/conf/conf.py`
  - `LOW_HP_HEAL_THRESHOLD` 从 `0.80` 降到 `0.30`。
  - 原 80% 血量就强制治疗，太容易在残血后反复覆盖移动/普攻。
  - 新增 `PASSIVE_BUTTONS`，标记无动作、回复、召唤师、回城等被动按钮。

- `agent_ppo/agent.py`
  - 新增轻量动作兜底 `_fallback_active_action()`。
  - 只在模型输出明显被动按钮时接管，不覆盖模型主动的移动、普攻、技能。
  - 兜底优先尝试普通攻击合法目标：敌方英雄、小兵、敌塔。
  - 如果普攻不可用，则根据己方位置向最近的敌方英雄/小兵/塔移动。
  - 修正兜底拆 `legal_action` 的维度判断：平台 legal_action 是 `184` 维，最后 target mask 是 `12*9`。

## 第二轮验证

- 本地 `py_compile` 通过。
- 这轮重点看：英雄是否还会长时间泉水不动；到中路后是否开始普攻；common_ai 的 `hurt_to_hero_common_ai` 和敌方塔血是否明显变动。

## Obs Dump 工具

- 新增 `agent_ppo/utils_obs_dump.py`。
- `agent_ppo/workflow/train_workflow.py` 已接入 dump 调用。
- `train_test.py` 会把 `DUMP_OBS*` 环境变量显式转发给平台子进程。
- `train_test.py` 还会写 `.dump_obs_config.json`，用于平台子进程读不到环境变量时兜底。
- 默认关闭，不影响正式训练。

平台运行方式：

```bash
DUMP_OBS=1 /bin/python3 /data/projects/hok1v1/train_test.py
```

默认输出目录：

```text
/data/projects/hok1v1/debug_obs/raw/
/data/projects/hok1v1/debug_obs/schema/
```

默认只保存前 4 局的这些帧：

```text
0,1,2,10,100,500,1000
```

可选参数：

```bash
DUMP_OBS=1 \
DUMP_OBS_DIR=/data/projects/hok1v1/debug_obs \
DUMP_OBS_FRAMES=0,1,2,10,100,500,1000 \
DUMP_OBS_MAX_EPISODES=4 \
/bin/python3 /data/projects/hok1v1/train_test.py
```

如果想把 schema 摘要也打印到日志：

```bash
DUMP_OBS=1 DUMP_OBS_PRINT_SCHEMA=1 /bin/python3 /data/projects/hok1v1/train_test.py
```

如果平台终端里没有看到 `DUMP_OBS enabled...`，先确认是否出现：

```text
DUMP_OBS config written to /data/projects/hok1v1/.dump_obs_config.json
```

说明：

- `raw` 是完整原始 observation，适合后续对字段和数值。
- `schema` 是结构摘要，适合先快速看 key、类型、list 长度。
- frame 0 会在 reward 初始化后保存，因此 observation 里会带上当前 reward 字段。
