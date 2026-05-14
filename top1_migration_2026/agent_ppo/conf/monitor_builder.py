#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("智能决策1v1")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="round(avg(reward{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="round(avg(total_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="round(avg(value_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="round(avg(policy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="round(avg(entropy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="目标推进",
            name_en="objective",
            type="line",
        )
        .add_metric(metrics_name="objective_tower_damage", expr="round(avg(objective_tower_damage{}), 0.01)")
        .add_metric(metrics_name="objective_hero_tower_damage", expr="round(avg(objective_hero_tower_damage{}), 0.01)")
        .add_metric(metrics_name="objective_safe_push_frames", expr="round(avg(objective_safe_push_frames{}), 0.01)")
        .add_metric(metrics_name="objective_finish_window_frames", expr="round(avg(objective_finish_window_frames{}), 0.01)")
        .end_panel()
        .add_panel(
            name="兵线防守",
            name_en="lane_defense",
            type="line",
        )
        .add_metric(metrics_name="lane_own_tower_pressure_frames", expr="round(avg(lane_own_tower_pressure_frames{}), 0.01)")
        .add_metric(metrics_name="lane_own_tower_clear_hit_frames", expr="round(avg(lane_own_tower_clear_hit_frames{}), 0.01)")
        .add_metric(metrics_name="lane_enemy_tower_wave_frames", expr="round(avg(lane_enemy_tower_wave_frames{}), 0.01)")
        .end_panel()
        .add_panel(
            name="战斗结果",
            name_en="fight_result",
            type="line",
        )
        .add_metric(metrics_name="fight_kills", expr="round(avg(fight_kills{}), 0.01)")
        .add_metric(metrics_name="fight_deaths", expr="round(avg(fight_deaths{}), 0.01)")
        .add_metric(metrics_name="fight_enemy_deaths", expr="round(avg(fight_enemy_deaths{}), 0.01)")
        .add_metric(metrics_name="fight_combat_frames", expr="round(avg(fight_combat_frames{}), 0.01)")
        .add_metric(metrics_name="fight_bad_frames", expr="round(avg(fight_bad_frames{}), 0.01)")
        .end_panel()
        .add_panel(
            name="技能释放",
            name_en="skill_usage",
            type="line",
        )
        .add_metric(metrics_name="fight_skill1_actions", expr="round(avg(fight_skill1_actions{}), 0.01)")
        .add_metric(metrics_name="fight_skill2_actions", expr="round(avg(fight_skill2_actions{}), 0.01)")
        .add_metric(metrics_name="fight_skill3_actions", expr="round(avg(fight_skill3_actions{}), 0.01)")
        .add_metric(metrics_name="fight_skill_hit_frames", expr="round(avg(fight_skill_hit_frames{}), 0.01)")
        .add_metric(metrics_name="fight_skill_hero_hit_frames", expr="round(avg(fight_skill_hero_hit_frames{}), 0.01)")
        .end_panel()
        .add_panel(
            name="草丛先手",
            name_en="grass_ambush",
            type="line",
        )
        .add_metric(metrics_name="grass_frames", expr="round(avg(grass_frames{}), 0.01)")
        .add_metric(metrics_name="grass_combat_frames", expr="round(avg(grass_combat_frames{}), 0.01)")
        .end_panel()
        .add_panel(
            name="发育资源",
            name_en="resource",
            type="line",
        )
        .add_metric(metrics_name="resource_rune_pickups", expr="round(avg(resource_rune_pickups{}), 0.01)")
        .add_metric(metrics_name="resource_monster_kills", expr="round(avg(resource_monster_kills{}), 0.01)")
        .add_metric(metrics_name="resource_monster_hit_frames", expr="round(avg(resource_monster_hit_frames{}), 0.01)")
        .add_metric(metrics_name="resource_monster_near_frames", expr="round(avg(resource_monster_near_frames{}), 0.01)")
        .end_panel()
        .add_panel(
            name="续航回泉",
            name_en="sustain",
            type="line",
        )
        .add_metric(metrics_name="sustain_recall_actions", expr="round(avg(sustain_recall_actions{}), 0.01)")
        .add_metric(metrics_name="sustain_base_returns", expr="round(avg(sustain_base_returns{}), 0.01)")
        .add_metric(metrics_name="sustain_endgame_base_returns", expr="round(avg(sustain_endgame_base_returns{}), 0.01)")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
