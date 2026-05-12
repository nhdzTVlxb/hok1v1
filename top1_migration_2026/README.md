# 2026 Top1 Migration PPO

这套代码从 `baseline1v1-2026` 复制而来，只改 `agent_ppo` 训练提交链路，目标是在 2026 新环境下迁移 2025 top1 中确定有效、且接口风险可控的部分。

## 环境差异核对

2025 环境和方案重点：

- 英雄：后羿、李元芳、虞姬。
- top1 手工解包 `state_dict`，构造约 3900 维结构化 obs。
- 模型按实体切分：英雄、小兵、河蟹、防御塔、子弹分别编码，再 LSTM + MLP 融合。
- reward 使用换血、塔血、经济、经验、蓝量、击杀/死亡、补刀、前进奖励。
- 最终经验：不要使用分英雄多输出头；不要迁移 DreamerV3 tricks。

2026 环境变化：

- 英雄改为鲁班七号 `112` 和狄仁杰 `133`。
- 观测协议变成更扁平的 `frame_state` 字段，例如 `hero_states/npc_states/bullets/cakes`。
- `env.reset` 返回 dict，训练 workflow 使用 `observation["0"] / observation["1"]`。
- 动作空间仍是 `[12, 16, 16, 16, 16, 9]`，但新增 reset 前 `init_config` 召唤师技能选择。
- 官方 baseline 只保留很少的示例特征和 reward，需要重建 obs/reward。

## 改动文件

- `agent_ppo/conf/conf.py`
  - 新增 `Args`/`DimConfig`，定义 2026 结构化 obs 维度。
  - 英雄 ID 改为 `[112, 133]`。
  - 保留 2025 风格奖励权重、LSTM=512、PPO 动作维度。
  - 召唤师技能默认固定为治疗术 `80102`，避免随机选择带来的训练噪声。

- `agent_ppo/feature/top1_feature_builder.py`
  - 新增 2026 flat schema 版手工 obs builder。
  - 迁移实体特征：双方英雄、双方小兵、野怪/河蟹占位、双方塔、敌方子弹、血包。
  - 保留位置 one-hot、全局位置、hp 离散、技能 cd、经济增量、buff/mark 桶、塔攻击目标等思路。

- `agent_ppo/model/model.py`
  - 迁移 2025 top1 的实体分块模型。
  - 保留共享 position/unit 编码、敌方目标 embedding、LSTM + MLP 双路融合。
  - 不迁移分英雄多输出头，因为 2025 README 总结其训练慢且收益差。

- `agent_ppo/feature/reward_process.py`
  - 从 baseline 的 `tower_hp_point/forward` 扩展为 2025 风格完整 reward。
  - 按 2026 flat 字段读取 hp、ep、money、exp、kill/death、dead_action。

- `agent_ppo/agent.py`
  - 接入 `Top1FeatureBuilder`。
  - `init_config` 从随机召唤师技能改为确定性选择。
  - 当前选择治疗术 `80102`，并在自身血量低于 80% 且 chosen summoner button 合法时，硬规则覆盖动作为使用治疗术。
  - 推理输出转 numpy 时兼容 GPU tensor。

- `agent_ppo/algorithm/algorithm.py`
  - 样本进入 learner 时用 `np.stack -> torch.as_tensor`，兼容当前 `SampleData.sample` 为 numpy array 的情况。

- `agent_ppo/workflow/train_workflow.py`
  - 阵容轮转改用 `GameConfig.CAMP_HEROES = [[112], [133]]`。
  - 增加 reward 子项监控：`diy_1~diy_5`。

- `conf/configure_app.toml`
  - 加入 `learner_train_sleep_seconds = 2.00`，迁移 2025 最终版对样本生产/消费节奏的经验。
  - `train_batch_size` 从 `1024` 调整为 `512`，与 2025 top1 配置更接近。

- `agent_ppo/conf/train_env_conf.toml`
  - `eval_interval` 从 `10` 调整为 `32`，减少训练中评估打断。

## 没有硬迁移的内容

- 2025 后羿/李元芳/虞姬的 buff/mark 精确编号没有直接迁移。
  - 原因：2026 英雄是鲁班七号/狄仁杰，编号不同，硬迁会污染特征。
  - 当前实现只保留通用 buff、英雄前缀桶和 unknown 桶，建议在平台 debug 后补精确 ID。

- 2025 debug_agent 和重要帧 JSON 没有迁移。
  - 原因：今年 protocol 已变，先保证训练提交主链路干净。

- 多输出头、多模型、DreamerV3 tricks 没有迁移。
  - 原因：2025 README 明确总结为慢或负收益。

## 预期收益

- 比 baseline 更充分利用环境信息，尤其是双方英雄/小兵/塔/子弹和经济节奏。
- reward 从只会推塔/前进，扩展到对线、经济、补刀、击杀死亡和蓝量管理。
- 模型结构更贴近 1v1 MOBA 的目标选择逻辑，target head 能看见敌方英雄、小兵、塔等实体 embedding。
- 召唤师技能不再随机，训练分布更稳定；治疗术使用由规则即时触发，同时 PPO 会从这些样本中学习低血量治疗行为。

## 主要风险

- 2026 buff/mark ID 未经平台 debug，当前只是风险较低的泛化桶。
- `sub_type/actor_type/behav_mode` 在平台若全部为整数枚举，部分 one-hot 行为桶会落到 unknown，但不会崩。
- 血包归属按“离塔最近”处理，若 2026 地图血包逻辑变化，需要用平台帧数据校准。
- 低血量治疗覆盖默认使用 button index `8`（Chosen Skill）和 target index `2`（Self）。如果平台动作定义调整，需要根据首轮日志修正。
- 本地没有 torch，无法做模型前向运行检查；已完成 Python 静态编译检查。

## 提交建议

优先把 `top1_migration_2026` 作为一套独立代码包提交训练。第一轮平台日志重点看：

- obs 构造是否有维度 assert。
- `reward_sum`、`diy_1~diy_5` 是否正常上报。
- `win_rate:common_ai` 是否随训练抬升。
- 如果训练正常，下一步用 debug 帧补齐鲁班/狄仁杰 buff 和 mark 精确编号。
