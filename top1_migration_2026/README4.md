# README4: 技能、野怪、推塔奖励修正

## 本次目标

针对 2h 训练后暴露的问题，继续把行为目标做明确：

- 解决安全时仍不攻击防御塔的问题。
- 修正技能释放逻辑：1 技能用于清线，2/3 技能主要用于英雄对战。
- 给普通攻击、技能命中、野怪争抢、推塔建立可 debug 的奖励。
- 按 2026 raw obs 字段重新确认英雄/小兵/防御塔/野怪映射。

## 2026 字段映射

主要使用 `frame_state.hero_states` 和 `frame_state.npc_states`：

- 英雄：`actor_type == 0`，来自 `hero_states`。
- 小兵：`actor_type == 1` 或 `sub_type in (1, 11, ACTOR_SUB_SOLDIER)`，来自 `npc_states`。
- 防御塔/水晶类建筑：`actor_type == 2` 且 `sub_type in (21, 23, 24, ACTOR_SUB_TOWER)`。
- 野怪/中立资源：`actor_type == 3` 或 `actor_type == ACTOR_TYPE_MONSTER`，或 `sub_type in (ACTOR_SUB_MONSTER, ACTOR_SUB_NEUTRAL_MONSTER)`，来自 `npc_states`。

## 修改文件

- `agent_ppo/conf/conf.py`
  - `DIM_OF_FEATURE` 从 128 调整到 144，给野怪特征留输入空间。
  - 增加技能命中、野怪击杀、线权打英雄、争抢野怪、塔下追击风险等奖励权重。
  - 增加 `LOW_HP_RECOVER_THRESHOLD = 0.85`，血量低于 85% 时尝试使用恢复技能；召唤师治疗仍是 75%。

- `agent_ppo/feature/top1_feature_builder.py`
  - 从 `npc_states` 里提取野怪特征。
  - 增加最近野怪的血量、相对位置、距离、可击杀状态等固定长度输入。

- `agent_ppo/feature/reward_process.py`
  - 普通攻击命中奖励：防御塔 > 英雄 > 野怪 > 小兵。
  - 技能命中奖励：英雄 > 小兵 > 野怪；大招命中英雄额外放大。
  - 安全推塔伤害继续给高奖励，必须是英雄对塔造成的有效伤害。
  - 击杀野怪给较大奖励，被对方抢野给惩罚。
  - 敌方兵线清完、血量差在 15% 内、且不进塔时，鼓励攻击敌方英雄。
  - 如果不适合打英雄，则鼓励争抢最近野怪。
  - 进入敌方塔危险范围追人给惩罚，除非对方残血且我方血量安全。

- `agent_ppo/agent.py`
  - 低血量恢复：85% 以下尝试恢复技能，75% 以下尝试召唤师治疗。
  - 2/3 技能只在英雄交战窗口内主动兜底释放。
  - 1 技能在约 3 秒内没有英雄交互时优先用于清线。
  - 敌方兵线清完时，如果血量差距 15% 以内且不需要冲塔，优先打英雄；否则转向争抢野怪。
  - 安全推塔仍强制使用普通攻击打塔；如果防御塔正在攻击自己，立即走出塔范围。

- `agent_ppo/diagnostics.py`
  - 增加技能与野怪行为统计。

- `agent_ppo/workflow/train_workflow.py`
  - 接入新增监控数据 `diy_21` 到 `diy_26`。

- `agent_ppo/conf/monitor_builder.py`
  - 增加“技能与野怪”面板。

## 新增监控指标

- `diy_21`: 1 技能释放次数。
- `diy_22`: 2 技能释放次数。
- `diy_23`: 3 技能释放次数。
- `diy_24`: 技能命中帧数。
- `diy_25`: 技能命中英雄帧数。
- `diy_26`: 命中野怪帧数。

## 训练时重点观察

- `hero_tower_damage` 是否开始大于 0，这是是否真正推塔的核心指标。
- `safe_push_attack_frames` 是否随安全推塔机会增加。
- `skill3_actions` 和 `skill_hero_hit_frames` 是否开始非 0，判断大招是否进入英雄战斗。
- `monster_hit_frames` 是否非 0，判断野怪字段和目标选择是否对上。
- 如果 `tower_target_legal_frames` 很高但 `tower_target_action_frames` 仍为 0，说明目标索引或 legal_action 映射还要继续查。

## 注意

`skill_misuse` 目前只保留奖励项入口，没有强行扣分。原因是 reward 侧主要看到的是帧结果，不稳定知道“本帧选择了哪个技能但没命中”。这一版先用 agent 规则限制 2/3 技能只在英雄战斗中释放，用命中奖励引导技能质量；如果后续仍乱放，再把 action 记录传入 reward 或在 collector 侧做动作级惩罚。

## 关键帧 Dump

重新加回轻量 obs dump，用于对齐 2025 的关键帧调试方式。

默认抓取帧：

`56,500,1000,1094,1148,1500,1778,2500,4000,6000,9000,12000`

含义：

- `56`：开局基础结构，对齐 2025 的 `start/56.json`。
- `1094`、`1148`：对齐 2025 用来观察英雄特殊状态/印记/技能阶段的关键帧。
- `1778`：对齐 2025 的蛋糕/血包出现帧。
- `2500` 以后：补充中后期，用于观察野怪、敌塔、推塔机会和技能命中。

运行方式：

```bash
DUMP_OBS=1 DUMP_OBS_PRINT_SCHEMA=1 /bin/python3 train_test.py
```

输出目录默认是：

```text
/data/projects/hok1v1/debug_obs/raw
/data/projects/hok1v1/debug_obs/schema
```

注意：dump 现在不是要求环境帧号“精确等于”关键帧，而是只要本次 step 跨过关键帧，就会用当前最近一帧写出对应文件。例如跨过 1094 时，会生成 `frame_001094.json`，文件内部会记录真实 `frame_no` 和 `state_frame_no`。

`train_test.py` 默认会在 1000 帧提前结束。开启 `DUMP_OBS=1` 时，会临时把测试截止帧提高到 `12500`，这样才能抓到 1778、4000、9000、12000 这些中后期关键帧。

如果要临时覆盖帧列表：

```bash
DUMP_OBS=1 DUMP_OBS_FRAMES=56,1778,4000,9000 /bin/python3 train_test.py
```

如果只想快速跑短测：

```bash
DUMP_OBS=1 DUMP_OBS_TRAIN_TEST_FRAMES=1000 /bin/python3 train_test.py
```

## 2026 Raw2 校正

根据 `raw2` 中 `56/500/1000` 三个关键帧，补充两个字段映射修正：

- 中立怪/野怪：`camp == 0 && actor_type == 1 && sub_type == 0`。这类单位不能算作双方小兵，否则会干扰清线和野怪争抢逻辑。
- 技能命中：`hit_target_info.slot_type == 0` 是普攻，`1/2/3` 分别对应 1/2/3 技能。因此技能命中奖励、技能监控都改为统计 `slot_type in (1,2,3)`，其中 `slot_type == 3` 按大招处理。

已同步修改：

- `agent_ppo/feature/top1_feature_builder.py`
- `agent_ppo/feature/reward_process.py`
- `agent_ppo/diagnostics.py`
- `agent_ppo/agent.py`

## Raw2 中后期帧分析

继续读取 `1778/2500/4000/6000` 后确认：

- `cakes` 在 1778 已出现，结构为 `frame_state.cakes[*].collider.location`，位置大约在双方一塔附近。
- 中立怪在 1778/6000 出现，字段为 `camp=0, actor_type=1, sub_type=0`。之前用 `camp_id()` 判断会把 `0` 转成 `1`，导致中立怪仍可能被误判；已改为直接判断原始 camp。
- 4000 和 6000 帧都出现了安全推塔窗口：
  - 敌方一塔 `sub_type=21` 正在攻击我方小兵。
  - 普攻 button 合法，target mask 中 `target=7` 合法。
  - 英雄位置在攻击范围内。
  - 因此如果后续 `tower_target_action_frames` 仍为 0，问题不在 obs 字段，而更可能在 action 被其他逻辑覆盖或模型版本未更新。
- 2/3 技能曾在敌方英雄不可见时被使用，说明仅靠 reward 不足以防止模型乱放。现在增加动作保护：非英雄战斗窗口内采样到 2/3 技能，会改成清线、打野或移动。

本次补充修复：

- 新增 `is_neutral_camp()`，避免 `camp_id(0) -> 1` 造成中立怪识别失败。
- 2/3 技能只允许在英雄战斗窗口内保留；非战斗乱放会被替换。

## Small-3 监控后调整

根据 small-3 训练表现：

- common_ai 胜率提高，是正向信号。
- 但环境指标 `hurt_to_hero` 和 `kill` 为 0，说明策略偏清线/推塔/避战，英雄对抗不足。
- 自博弈仍出现一方不出水晶/泉水。

本次补充：

- 强制出门：血量大于 80%，且英雄仍在我方基地水晶附近时，强制移动到中路 `(0, 0)`。
- 恢复技能：除了血量低于 85%，蓝量/能量低于 50% 时也尝试使用恢复技能。
- 放宽安全打英雄：敌方英雄可见、在攻击距离附近、且我方不处于敌塔危险时，允许主动普攻和 2/3 技能打英雄。
- 奖励轻微强化英雄对抗：`kill`、`attack_hit`、`skill_hit` 权重略微上调，保留推塔奖励不回滚。

观察重点：

- `hurt_to_hero` 是否从 0 变为非 0。
- `kill` 是否仍长期为 0。
- 自博弈是否还出现一方长时间停在水晶附近。
- `diy_23` 和 `diy_25` 是否同时上升，判断大招是否真的命中英雄。

## 镜像与技能定向修复

本次针对红方不出水晶、技能乱放继续修复：

- 参考 2025 的镜像思路：红方动作方向也用镜像后的坐标计算，但字段仍适配 2026 raw obs 的 `camp/location/frame_state`，没有照搬旧解包结构。
- 强制出泉水、移动 fallback、撤离防御塔等规则动作，现在都会先把红方坐标转到蓝方视角再算 16x16 移动方向，避免红方朝反方向走。
- 1/2/3 技能不再“按钮合法就放”。技能必须有可见目标、目标 mask 合法，并按英雄到目标的方向计算 skill x/z。
- 英雄战斗时优先级调整为：3 技能、2 技能、1 技能、普攻；因此 1 技能冷却好且敌方英雄可打时，会优先对英雄释放，而不是先清兵。
- 非英雄战斗时，2/3 技能会被替换成移动或普攻；1 技能只允许对小兵/野怪等明确目标释放，避免空放和乱放。

下一次训练优先看：

- 红方自博弈是否还能长时间停在水晶附近。
- `hurt_to_hero` 是否恢复为非 0。
- `diy_21/diy_23/diy_25` 是否随英雄交战同步上升。
- 1 技能命中是否主要发生在英雄/兵线/野怪附近，而不是无目标空放。
