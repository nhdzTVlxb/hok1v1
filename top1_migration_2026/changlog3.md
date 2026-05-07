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

## 2026-05-08 追加：安全推塔与红方移动修复

### 背景

局内观察发现：

- agent 已经能离开泉水，会补兵、刷野。
- 但在安全窗口不会主动攻击防御塔。
- 自训练对战时，蓝方能赢，红方会往反方向移动，卡在出生点附近。
- reward 图仍可能显示为 0，loss 变化较小。

这说明基础 feature/reward 已经让智能体“活了”，但推塔目标和红蓝方对称性还不够明确。

### 安全推塔定义

这次把“安全推塔”明确成以下条件：

- 敌方防御塔正在攻击我方小兵。
- 我方英雄已经在可攻击敌塔的距离内。
- 敌方英雄阵亡、不可见，或者不在我方英雄的危险范围内。
- 敌塔没有正在攻击我方英雄。

对应改动：

- `agent_ppo/feature/top1_feature_builder.py`
  - 增加 `safe_push` 特征。
  - 增加 `tower_tanking_minion` 特征。
  - 增加 `enemy_threat` 特征。
- `agent_ppo/feature/reward_process.py`
  - 增加 `tower_attack` reward。
  - 增加 `safe_push` reward。
- `agent_ppo/conf/conf.py`
  - 增加 `tower_attack: 0.25`。
  - 增加 `safe_push: 0.04`。
- `agent_ppo/agent.py`
  - fallback action 在安全推塔窗口优先选择普通攻击敌方防御塔。
  - 目标 7 沿用 2025 top1 target 约定，表示敌方 organ/tower。

### 红方反向移动问题

观察到红方卡在出生点、往反方向移动，主要怀疑两类问题：

1. 红蓝方视角没有统一。
2. 不可见敌方英雄的坐标 `100000,100000,100000` 被当成真实移动目标。

本次改动：

- `agent_ppo/feature/top1_feature_builder.py`
  - 当 `camp == 2` 时，对 feature 中的位置做镜像，让红方也以“我方向敌方推进”的统一视角看地图。
- `agent_ppo/agent.py`
  - fallback move 不再把不可见英雄的 `100000` 坐标作为移动目标。
  - 如果没有合适敌人或小兵，会优先向最近敌方防御塔移动，而不是默认走向错误方向。

说明：

- 这次先修最可能导致红方反向走的实际 bug：不可见目标坐标污染移动目标。
- 暂未做完整 action 坐标反向映射，因为动作 mask / PPO label 也要同步镜像，否则容易引入新的训练标签错位。若后续红方仍反向，再单独做 action mirror。

### loss 与 reward 观察

如果 `kill_common_ai` 高于 `death_common_ai`，说明行为本身不是崩的。

如果 reward 图仍是 0、loss 几乎不动，优先判断：

- reward 可能被监控 round 成 0。
- reward 可能主要来自稀疏事件，平均后很小。
- policy/value loss 变化小，说明训练信号仍偏弱。

下一轮观察重点：

- 安全窗口是否开始攻击敌塔。
- `enemy_tower_hp_common_ai` 是否下降更明显。
- 红方自训练是否还会往出生点反方向走。
- `hurt_to_hero` 是否保持，`death` 是否不要明显上升。
- `reward` / `total_loss` / `policy_loss` / `value_loss` 是否开始有非零变化。

如果推塔仍弱，下一步优先继续加密 tower reward，而不是先加大模型。

## 2026-05-08 追加：塔伤诊断、攻击奖励与镜像修复

### 这次主要修什么

- 修正 `camp_id()`：
  - 之前把整数 `1` 误映射成 `2`，可能导致蓝/红阵营、敌我塔、红方镜像判断错位。
  - 现在只有 `0` 会兼容映射到 `1`，`1/2` 保持原值。
- 新增 `agent_ppo/diagnostics.py`：
  - 从每帧 `frame_state` 和实际 action 中统计塔伤、英雄造成塔伤、安全推塔窗口、普攻频率、攻击力、攻速。
- 修改 `agent_ppo/workflow/train_workflow.py`：
  - 每局累计诊断数据，并上报到 monitor。
- 修改 `agent_ppo/conf/monitor_builder.py`：
  - 新增塔伤、安全推塔帧、攻击频率、攻击属性、打塔动作面板。
- 修改 `agent_ppo/feature/reward_process.py`：
  - 新增 `attack_hit`、`attack_power`、`attack_speed`、`safe_tower_damage` reward。
  - 安全推塔时，英雄真实命中防御塔并造成塔血下降，给高奖励。
  - 被塔攻击或低血进塔区的惩罚加重。
- 修改 `agent_ppo/agent.py`：
  - 安全推塔窗口内强制优先普通攻击防御塔目标 7。
  - 如果敌方防御塔正在攻击自己，优先离开塔区。

### 新增监控含义

- `diy_6`：敌方塔总掉血，可能包含小兵伤害。
- `diy_7`：估算的英雄对敌方塔伤害，要求同帧英雄在打塔/命中塔。
- `diy_8`：安全推塔窗口帧数。
- `diy_9`：安全推塔窗口内选择普攻塔的帧数。
- `diy_10`：英雄处于打塔/命中塔的帧数。
- `diy_11`：平均物理攻击力 `phy_atk`。
- `diy_12`：平均攻速 `atk_spd`。
- `diy_13`：每 1000 帧命中目标次数。
- `diy_14`：每 1000 帧普通攻击动作次数。
- `diy_15`：平均命中间隔帧数，越低表示攻击越频繁。
- `diy_16`：本局物理攻击力增长。
- `diy_17`：本局攻速增长。
- `diy_18`：普攻塔目标 7 合法的帧数。
- `diy_19`：实际选择普攻塔目标 7 的帧数。
- `diy_20`：自己被敌塔锁定的帧数。

### 下一轮重点看

- `diy_8 > 0` 且 `diy_18 > 0` 时，`diy_19` 应该同步上升。
- `diy_19 > 0` 后，`diy_7` 应该开始大于 0。
- 如果 `diy_8/diy_18` 有，但 `diy_19=0`，说明动作目标 7 仍未正确生效。
- 如果 `diy_19` 有但 `diy_7=0`，说明 target 7 可能不是塔，或 hit/tower hp 对齐方式还要继续查。
- 红方仍不出门时，优先看是否仍有阵营识别错误，而不是先调 reward。
