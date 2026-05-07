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
            name="英雄塔伤",
            name_en="hero_tower_damage",
            type="line",
        )
        .add_metric(metrics_name="diy_7", expr="round(avg(diy_7{}), 0.01)")
        .end_panel()
        .add_panel(
            name="总塔伤",
            name_en="tower_damage",
            type="line",
        )
        .add_metric(metrics_name="diy_6", expr="round(avg(diy_6{}), 0.01)")
        .end_panel()
        .add_panel(
            name="安全推塔帧",
            name_en="safe_push_frames",
            type="line",
        )
        .add_metric(metrics_name="diy_8", expr="round(avg(diy_8{}), 0.01)")
        .add_metric(metrics_name="diy_9", expr="round(avg(diy_9{}), 0.01)")
        .end_panel()
        .add_panel(
            name="攻击频率",
            name_en="attack_frequency",
            type="line",
        )
        .add_metric(metrics_name="diy_13", expr="round(avg(diy_13{}), 0.01)")
        .add_metric(metrics_name="diy_14", expr="round(avg(diy_14{}), 0.01)")
        .add_metric(metrics_name="diy_15", expr="round(avg(diy_15{}), 0.01)")
        .end_panel()
        .add_panel(
            name="攻击属性",
            name_en="attack_stats",
            type="line",
        )
        .add_metric(metrics_name="diy_11", expr="round(avg(diy_11{}), 0.01)")
        .add_metric(metrics_name="diy_12", expr="round(avg(diy_12{}), 0.01)")
        .add_metric(metrics_name="diy_16", expr="round(avg(diy_16{}), 0.01)")
        .add_metric(metrics_name="diy_17", expr="round(avg(diy_17{}), 0.01)")
        .end_panel()
        .add_panel(
            name="打塔动作",
            name_en="tower_action",
            type="line",
        )
        .add_metric(metrics_name="diy_10", expr="round(avg(diy_10{}), 0.01)")
        .add_metric(metrics_name="diy_18", expr="round(avg(diy_18{}), 0.01)")
        .add_metric(metrics_name="diy_19", expr="round(avg(diy_19{}), 0.01)")
        .add_metric(metrics_name="diy_20", expr="round(avg(diy_20{}), 0.01)")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
