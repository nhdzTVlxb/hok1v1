# README7 - 1536 特征、兵线目标与 Target Attention 版本

## 这次主要解决什么

当前 agent 的 KDA 已经偏高，但胜负目标仍然不够清楚：

- 回泉水次数太多，后期容易因为补血丢兵线、丢塔，甚至被一波。
- 击杀后没有稳定转化成清线、推塔、结束比赛。
- 塔血量偏高、每帧伤害偏低，说明进攻目标和塔目标还不够明确。
- 兵线决策弱，agent 更会保命，但不够理解“赢比赛靠兵线和塔，不是靠英雄自己站得靠前”。
- 新拿到的 `ObjAbility_*` 信息说明之前控制判断可能误把正常状态当成受控，需要修正。

本版改动方向：

- `forward` 只保留开局上线导航，700 帧后关闭。
- 把中后期推进的学习信号转移到兵线、守塔清线、安全推塔、target entity。
- feature 从 512 提到 1536。
- 网络 public dim 从 384 提到 512。
- 最后 288 维固定为 9 个 target entity 槽，并用 target attention 输出 target head。
- 降低后期回城优先级，让守塔清线和终局推进可以盖过补给。


## 配置变化

文件：`agent_ppo/conf/conf.py`

关键参数：

```python
REMOVE_FORWARD_AFTER = 700
DIM_OF_FEATURE = [1536]
DIM_PUBLIC = 512
TARGET_ENTITY_COUNT = 9
TARGET_ENTITY_DIM = 32
TARGET_FEATURE_DIM = 288
```

回城相关阈值下调：

```python
CRITICAL_HOME_HP_THRESHOLD = 0.16
SUPPLY_RETURN_HOME_HP_THRESHOLD = 0.16
SAFE_RECALL_HP_THRESHOLD = 0.24
```

新增守塔清线 reward：

```python
"defend_tower_clear": 0.25
```


## Feature 布局

总 feature 维度：1536。

其中：

- 前 1248 维：普通状态、宏观、兵线、塔、控制、子弹、经济阶段等。
- 后 288 维：固定 target entity 特征，即 `9 * 32`。

### 原有特征保留

原先约 508 维继续保留：

- self hero
- enemy hero
- skills
- towers
- soldiers summary
- monsters
- cakes
- unit slots
- projectiles
- combat risk
- objective
- global
- status macro
- strategic macro
- legal summary

### 新增特征

新增特征会被放进前 1248 维：

1. `ability_control_features`，64 维

利用 `abilities` 的真实含义：

- `0`: NoControl
- `1`: NoMove
- `2`: NoSkill
- `7`: Blindness
- `10`: Freeze
- `14`: AbortMove
- `21`: Repressed

派生出：

- 是否硬控
- 能否移动
- 能否放技能
- 能否普攻造成伤害
- 是否免控/冲刺/不可选中等

2. `lane_entity_features`，256 维

参考 2025 的做法，保留：

- 我方最近 4 个小兵
- 敌方最近 4 个小兵
- 每个小兵 32 维

一波兵通常是 3 个，但两波汇合、残兵、炮车存在时会超过 3 个，所以保留 4 槽。

3. `tower_wave_features`，96 维

强化兵线和塔关系：

- 我方兵在敌塔附近数量/总血量
- 敌方兵在我塔附近数量/总血量
- 我塔/敌塔血量
- 塔是否正在打英雄或小兵
- 是否安全推塔
- 是否需要守塔清线

4. `projectile_detail_features`，160 维，当前暂时占位为 0

这块原计划最多 10 个敌方弹道，每个 16 维：

- 相对位置/距离
- source_actor
- skill_id
- slot_type
- 是否塔弹
- 是否疑似关键技能弹道

但目前本地能核到的是 2025 important_frames JSON，不一定兼容当前 2026 obs。为了避免把 2025 schema 当成 2026 事实，本版先保留 160 维占位，不使用这些字段。等 2026 debug JSON 确认 `bullets` 结构后再启用。

注意：鲁班强化普攻 `112045` 已在 `debug_buff_summary2.json` 的 `buff_skills` 中确认，所以它作为 buff 特征使用，不放在 projectile detail 里。

5. `position_bucket_features`，64 维

给关键实体增加粗粒度位置 bucket：

- 敌方英雄
- 我塔
- 敌塔
- 最近敌兵
- 最近我兵

这个借鉴 2025 的离散位置思想，但比 2025 的 125 维位置编码更轻。

6. `buff_mark_detail_features`，64 维

强化 buff/mark：

- buff skill 数量
- buff mark 数量
- 鲁班强化普攻 buff `112045`
- 鲁班当前 debug 已观测相关 buff 集合：`112044/112046/112047/112048`
- 狄仁杰当前 debug 已观测相关 buff 集合：`133950/133951/133260`
- 若干 config id hash 归一化
- mark 层数

这些 ID 只来自当前工作区的 `eyes_debug_buff.json` 和 `debug_buff_summary2.json`。没有在这两个文件里确认的 2025 buff/mark 映射，本版不直接使用；当前两个 summary 里的 `buff_marks` 也是空的，所以 mark 只保留数量/层数通道，不绑定旧语义。

7. `economy_phase_features`，36 维

强化发育阶段：

- 金钱、等级、经济差、等级差
- 4000/6000/8000/10000 经济阶段
- fight score
- 野怪价值递减
- 双方塔血
- 双方兵线数量


## Target Entity 布局

最后 288 维固定给 target attention：

```text
target 0: 空/无目标
target 1: 敌方英雄
target 2: 自己
target 3: 最近敌兵 1
target 4: 最近敌兵 2
target 5: 最近敌兵 3
target 6: 最近敌兵 4
target 7: 敌方塔
target 8: 最近野怪/中立目标
```

每个 target 32 维：

- 是否存在
- target 类型
- 可见性
- 血量
- 是否低血
- 相对位置
- 距离
- 是否在攻击范围
- 是否被塔攻击
- 是否正在攻击自己
- 击杀收益
- 攻击范围
- config id 摘要
- 阵营信息

这样 target head 不再只是普通分类，而是能根据目标实体本身的信息决定“该打谁”。


## 网络结构

文件：`agent_ppo/model/model.py`

当前结构：

```text
feature: 1536
   ├─ feature_mlp: 1536 -> 512 -> 512
   │      └─ LSTM: input 512, hidden 512, time_steps 16
   │             └─ lstm_output: 512
   │
   └─ side_mlp: 1536 -> 512

concat:
   LSTM 512 + side 512 = 1024

public_mlp:
   1024 -> 512
```

前 5 个动作头：

```text
512 -> 128 -> [12, 16, 16, 16, 16]
```

target head 使用 target attention：

```text
target_features: 9 * 32
target_entity_mlp: 32 -> 64 -> 32
target_query_mlp: public 512 -> 128 -> 32
matmul(target_embed, query) -> 9 target logits
```

value head：

```text
512 -> 128 -> 1
```

这个结构比原先的普通 `384 -> 9` target head 更适合 MOBA 目标选择。


## Reward 变化

### Forward

`forward` 改为 700 帧后关闭。

作用定位：

- 只负责前期开局上线，避免 agent 原地发呆。
- 不负责中后期推进。
- 中后期推进应该靠兵线和塔相关 reward，而不是靠英雄位置 forward。

### 新增 defend_tower_clear

新增 `defend_tower_clear`：

- 敌方小兵靠近我塔时，如果英雄攻击/命中这些小兵，给正奖励。
- 如果敌兵压塔而英雄不处理，给小惩罚。
- 我塔血量低时，惩罚更明显。

目标：

- 防止 agent 后期为了补血/刷野放兵线进塔。
- 让“保护自己的塔”成为明确学习信号。


## 行为规则变化

### 回城/补给降优先级

后期经济达到 `ENDGAME_MONEY` 后：

- 只要血量高于 `CRITICAL_HOME_HP_THRESHOLD`，不轻易触发安全回城。
- 自家塔前有敌兵时，不触发普通补给，先让清线逻辑接管。
- 终局窗口中优先推塔/带线，而不是回去补血。

### abilities 控制判断修正

旧逻辑可能把 `abilities[1] == False` 误判为不能动。

新逻辑按你的枚举理解：

```text
True 才表示对应禁用/控制状态生效
```

硬控/行动受限判断使用：

```python
blocked_indices = (0, 1, 2, 7, 10, 14, 21)
```

即：

- NoControl
- NoMove
- NoSkill
- Blindness
- Freeze
- AbortMove
- Repressed


## 和 2025 做法的关系

这版不是直接复刻 2025 的 3910 维，但吸收了几个关键思想：

- 固定实体槽位。
- 同类目标结构化表达。
- 小兵最多取最近 4 个，而不是只做数量摘要。
- 子弹保留多个槽位。
- target head 利用实体 embedding，而不是普通分类。
- `forward` 只做前期引导，后面让目标/兵线/塔 reward 起作用。

区别：

- 2025 每个单位用大规模离散化位置和血量编码，维度很高。
- 当前仍保留连续归一化 + 轻量 bucket，先控制在 1536。
- 当前英雄是鲁班/狄仁杰，并且有闪现/狂暴，所以额外加入了控制、闪现、弹道危险、后期反打等信息。


## 后续还能怎么继续加

如果 1536 后仍然不够，可以继续往这些方向走：

1. 更接近 2025 的位置编码
   - 给英雄、小兵、塔、子弹加更细的相对位置 one-hot。

2. 更完整的实体分支网络
   - hero branch
   - soldier branch
   - tower/objective branch
   - projectile branch
   - macro branch

3. target attention 加强
   - 现在 target attention 已经加上。
   - 后续可把 action button 与 target 也做条件耦合，比如 skill1/skill2/skill3 各自有不同 target query。

4. buff/mark 字典化
   - 如果 debug 里能稳定提取鲁班/狄仁杰关键 buff id，可以从 hash 摘要升级为 one-hot 字典。

5. 兵线时间序列
   - 记录最近几帧兵线重心变化，用于判断兵线是推入还是回推。


## 训练注意

这版 feature 维度和网络结构都变了，不兼容旧权重。

建议：

- 从零训练。
- 重点看监控：
  - 回泉水次数是否下降。
  - 自家塔血是否更健康。
  - 敌方塔血下降是否更快。
  - 每帧伤害是否上升。
  - 击杀后是否清线/推塔而不是回家。
  - 后期野怪行为是否下降，兵线/推塔行为是否上升。
