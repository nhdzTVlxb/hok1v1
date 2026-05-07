# README3 - Feature/Reward 第一版改动说明

## 目标

这次不再继续 dump obs 数据结构，而是进入正式训练改动：

- 用 raw observation 构造固定长度 feature。
- 去掉临时 obs dump 调试接入。
- 简化模型输入，避免旧 2025 大实体特征维度难 debug。
- 加一版简单 reward，重点解决“不出泉水、走到中间挨打、不主动做事”。

## 去掉的调试代码

- 删除 `agent_ppo/utils_obs_dump.py`。
- 从 `agent_ppo/workflow/train_workflow.py` 移除 `ObsDumper` import、初始化和 dump 调用。
- 从 `train_test.py` 移除 `DUMP_OBS*` 环境变量写配置逻辑。

说明：之前 dump 出来的 `raw/` 和 `schema/` 已经完成数据结构确认。正式训练不再需要每次输出 obs 文件。

## Feature 改动

文件：

- `agent_ppo/feature/top1_feature_builder.py`
- `agent_ppo/conf/conf.py`
- `agent_ppo/model/model.py`

核心变化：

- feature 维度固定为 128。
- 坐标统一按 `100000` 归一化，并裁剪到 `[-1, 1]`。
- hp / ep / cooldown 都使用比例。
- 敌方不可见时，不使用 `100000,100000,100000` 坐标，改为位置 0，并额外给 `enemy_visible`。
- 从以下路径提取 feature：
  - `frame_state.hero_states`
  - `frame_state.npc_states`
  - `hero.skill_state.slot_states`
  - `legal_action`

第一版 feature 包含：

- 我方英雄：血量、蓝量、等级、经济、位置、朝向、攻击目标、是否被塔威胁。
- 敌方英雄：是否可见、血量、相对位置、距离、是否在攻击范围内。
- 技能：7 个 slot 的等级、usable、cooldown ratio。
- 防御塔：双方最近塔血量、相对位置、距离、是否在塔范围内。
- 兵线：双方可见兵数量、附近兵数量、最近兵距离、低血量兵、总血量。
- 全局：frame_no、离中线距离、低血状态、敌方可见性。
- 动作：主按钮 legal mask、移动维度合法比例、总合法比例。

## 模型改动

旧模型依赖 2025 风格的大实体向量，维度很大，不适合第一阶段 debug。

现在模型改为：

- 输入：128 维 feature。
- 结构：MLP -> LSTM -> policy heads/value head。
- 输出动作头仍保持 `[12, 16, 16, 16, 16, 9]`，不改 PPO 动作协议。
- `legal_action` 训练时仍使用压缩后的 85 维，推理时仍使用平台给的 184 维。

## Reward 改动

文件：

- `agent_ppo/conf/conf.py`
- `agent_ppo/feature/reward_process.py`

主要变化：

- 修正 `kill` 权重为正，死亡权重为负。
- 修正小兵/塔识别：
  - 小兵：`actor_type=1` 或 `sub_type=11`
  - 塔/建筑：`actor_type=2` 且 `sub_type in (21, 23, 24)`
- 保留并简化以下 reward：
  - 血量优势
  - 塔血优势
  - 经济/经验
  - 击杀/死亡
  - 补刀
  - 向敌塔推进
  - 低血进敌塔危险区惩罚
  - 高血长时间几乎不移动惩罚

这版 reward 的目标不是最终强策略，而是先让 agent 有明确基础方向：

- 不要一直待在泉水。
- 不要低血硬进敌塔。
- 有经济、补刀、消耗、推塔的正反馈。
- 死亡有明显负反馈。

## 平台验证方式

先跑：

```bash
/bin/python3 /data/projects/hok1v1/train_test.py
```

预期：

- 不再出现 `DUMP_OBS config written...`
- 不再生成新的 `/data/projects/hok1v1/debug_obs`
- `train_test.py` 能通过
- learner 能正常 save model

然后上平台训练，看这些指标：

- `money_per_frame`：不应长期贴近 0。
- `hurt_to_hero`：不应长期为 0。
- `hurt_by_hero`：如果很高且 death 高，说明太激进。
- `death`：应比之前“一路送死”下降。
- `enemy_tower_hp`：长期训练后应开始下降。
- `reward`：不应长期为 0。
- 局内观察：至少应离开泉水，并围绕兵线/塔做动作。

## 下一步

如果这版能跑通但行为仍弱，优先调：

1. reward 权重。
2. fallback action 的主动攻击/移动策略。
3. feature 中兵线和塔危险区的特征。
4. 等 feature 稳定后，再进一步增强 LSTM 或做更复杂 reward。

