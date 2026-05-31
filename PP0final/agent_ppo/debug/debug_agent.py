#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Deterministic action script used only for DEBUG_AGENT train_test runs.

The goal is to actively trigger skills, recovery, recall, grass movement and
combat so raw observations can be compared frame by frame for buff ids.
"""

import math
import os


NO_ACTION = [0, 15, 15, 15, 15, 0]
MOVE_BUTTON = 2
ATTACK_BUTTON = 3
SKILL1_BUTTON = 4
SKILL2_BUTTON = 5
SKILL3_BUTTON = 6
RECOVER_BUTTON = 7
SUMMONER_BUTTON = 8
RECALL_BUTTON = 9
ENEMY_HERO_TARGET = 1
ENEMY_TOWER_TARGET = 7


class DebugAgent:
    def __init__(self, agent_idx=0, logger=None):
        self.agent_idx = agent_idx
        self.logger = logger
        self.last_print_frame = {}
        self.mode = os.environ.get("DEBUG_SCENARIO", "buff_probe").lower()
        self.eye_probe_grass_pos = None
        self.eye_probe_last_phase = None

    def act(self, observation):
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = int(frame_state.get("frame_no", frame_state.get("frameNo", 0)) or 0)
        hero = self._self_hero(observation, frame_state)
        if not hero:
            return NO_ACTION.copy()

        hero_id = int(hero.get("config_id", hero.get("configId", 0)) or 0)
        enemy = self._enemy_hero(observation, frame_state, hero)
        enemy_id = int((enemy or {}).get("config_id", (enemy or {}).get("configId", 0)) or 0)
        camp = int(hero.get("camp", observation.get("camp", 0)) or 0)

        # Keep both sides close enough for skill hit tests. DiRenjie ult / skill2
        # states are short, so the first half of the script is deliberately
        # deterministic instead of policy-like.
        if self.mode in ("buff_probe", "control_probe", "stun_probe", "133_probe"):
            control_action = self._control_probe_action(frame_no, hero, hero_id, enemy_id)
            if control_action is not None:
                return control_action

        if self.mode == "eye_probe":
            eye_action = self._eye_probe_action(frame_no, hero)
            if eye_action is not None:
                return eye_action

        if self.mode in ("133_probe", "vision_probe"):
            vision_action = self._vision_probe_action(frame_no, hero)
            if vision_action is not None:
                return vision_action

        if self.mode in ("buff_probe", "grass_probe"):
            grass_action = self._grass_probe_action(frame_no, hero)
            if grass_action is not None:
                return grass_action

        # 0-700: leave spring and meet at middle lane.
        if frame_no < 700:
            return self._move_to(hero, self._mid_target(camp))

        # 700-980: stand around middle and keep vision/combat stable.
        if frame_no < 980:
            if self.agent_idx == 0:
                return self._move_to(hero, (1200.0, 1200.0))
            return self._move_to(hero, (-1200.0, -1200.0))

        # 980-1700: trigger hero skills in a deterministic cycle.
        if frame_no < 1700:
            return self._skill_cycle_action(frame_no, hero_id)

        # 1700-2300: normal attacks and directed combat for hit/bullet/buff data.
        if frame_no < 2300:
            if frame_no % 90 < 18:
                return self._skill_action(SKILL1_BUTTON)
            return [ATTACK_BUTTON, 15, 15, 15, 15, ENEMY_HERO_TARGET]

        # 2300-2800: test recover buff while the other side keeps attacking.
        if frame_no < 2800:
            if self.agent_idx == 0 and frame_no % 120 < 18:
                return [RECOVER_BUTTON, 15, 15, 15, 15, 0]
            return [ATTACK_BUTTON, 15, 15, 15, 15, ENEMY_HERO_TARGET]

        # 2800-3400: test recall and recall interruption.
        if frame_no < 3400:
            if self.agent_idx == 0:
                return [RECALL_BUTTON, 15, 15, 15, 15, 0]
            return [ATTACK_BUTTON, 15, 15, 15, 15, ENEMY_HERO_TARGET]

        # 3400-6200 is handled by _grass_probe_action in buff_probe mode.

        # 6200+: keep light combat until DEBUG_TOTAL_FRAMES.
        if frame_no % 180 < 24:
            return self._skill_action(SKILL3_BUTTON)
        if frame_no % 90 < 18:
            return [SUMMONER_BUTTON, 15, 15, 15, 15, ENEMY_HERO_TARGET]
        return [ATTACK_BUTTON, 15, 15, 15, 15, ENEMY_HERO_TARGET]

    def _control_probe_action(self, frame_no, hero, hero_id, enemy_id):
        """Create repeatable control / cleanse windows.

        In 133 vs 133 episodes, agent0 fires DiRenjie ult first and agent1 waits
        briefly before using skill2. Later the roles swap. This lets the dump
        reveal both the victim's control state and the caster's cleanse/immune
        state.
        """
        if frame_no < 760:
            return self._move_to(hero, (0.0, 0.0))

        if frame_no < 1120:
            target = (-650.0, -650.0) if self.agent_idx == 0 else (650.0, 650.0)
            return self._move_to(hero, target)

        if hero_id == 133 and enemy_id == 133:
            # Agent0 ult -> agent1 skill2 cleanse.
            if 1120 <= frame_no < 1360:
                if self.agent_idx == 0 and frame_no % 120 < 18:
                    return self._skill_action(SKILL3_BUTTON)
                if self.agent_idx == 1 and 42 <= frame_no % 120 < 72:
                    return self._skill_action(SKILL2_BUTTON)
                return [ATTACK_BUTTON, 15, 15, 15, 15, ENEMY_HERO_TARGET]

            # Agent1 ult -> agent0 skill2 cleanse.
            if 1360 <= frame_no < 1600:
                if self.agent_idx == 1 and frame_no % 120 < 18:
                    return self._skill_action(SKILL3_BUTTON)
                if self.agent_idx == 0 and 42 <= frame_no % 120 < 72:
                    return self._skill_action(SKILL2_BUTTON)
                return [ATTACK_BUTTON, 15, 15, 15, 15, ENEMY_HERO_TARGET]

            # A few direct skill2 casts without incoming ult, to separate
            # cleanse/immune self buff from victim-control buff.
            if 1600 <= frame_no < 1850:
                if frame_no % 150 < 24:
                    return self._skill_action(SKILL2_BUTTON)
                return [ATTACK_BUTTON, 15, 15, 15, 15, ENEMY_HERO_TARGET]

        # For non-133 matchups, keep the original skill cycle useful.
        if frame_no < 1850:
            return self._skill_cycle_action(frame_no, hero_id)

        return None

    def _grass_probe_action(self, frame_no, hero):
        """Scan likely grass areas and fire skills into/around them.

        Previous runs never reached is_in_grass=True. This uses a wider sweep
        across the diagonal 1v1 lane and pauses if the env reports grass.
        """
        if frame_no < 3400 or frame_no >= 6200:
            return None

        if bool(hero.get("is_in_grass", False)):
            if frame_no % 180 < 30:
                return self._skill_action(SKILL1_BUTTON)
            return NO_ACTION.copy()

        grass_path = self._grass_path(self.agent_idx)
        slot = min((frame_no - 3400) // 260, len(grass_path) - 1)
        target = grass_path[slot]

        # Agent not currently walking scans the area with skills so visibility
        # changes / eye buffs can be correlated with nearby grass frames.
        if frame_no % 210 < 24:
            return self._skill_action(SKILL1_BUTTON)
        if frame_no % 330 < 24:
            return self._skill_action(SKILL3_BUTTON)
        return self._move_to(hero, target)

    def _vision_probe_action(self, frame_no, hero):
        """Dedicated grass-vision probe inside 133 vs 133 episodes.

        Agent0 tries to enter and hold grass first while agent1 probes with
        skills from outside. Then the roles swap. This makes it easier to
        identify whether the eye/vision exposure is a buff id or only a
        camp_visible transition.
        """
        if frame_no < 3400 or frame_no >= 7000:
            return None

        first_half = frame_no < 5200
        hiding_agent = 0 if first_half else 1
        probing_agent = 1 - hiding_agent

        if self.agent_idx == hiding_agent:
            if bool(hero.get("is_in_grass", False)):
                return NO_ACTION.copy()
            grass_path = self._grass_path(self.agent_idx)
            slot = min((frame_no - (3400 if first_half else 5200)) // 220, len(grass_path) - 1)
            return self._move_to(hero, grass_path[slot])

        if self.agent_idx == probing_agent:
            # Stay near middle but outside the grass sweep, then fire probing
            # skills into the hidden enemy's likely area.
            if frame_no % 180 < 36:
                return self._skill_action(SKILL1_BUTTON)
            if frame_no % 300 < 36:
                return self._skill_action(SKILL3_BUTTON)
            outside_target = (-1800.0, -1800.0) if self.agent_idx == 0 else (1800.0, 1800.0)
            return self._move_to(hero, outside_target)

        return None

    def _eye_probe_action(self, frame_no, hero):
        """Force repeated grass hide / skill-probe windows for eye buff search.

        The existing vision probe can pass over grass too quickly. This variant
        remembers the first grass position seen by the hider and keeps walking
        back to it while the other agent repeatedly casts skills from outside.
        Buff ids can then be compared against is_in_grass and camp_visible
        transitions in the dumped JSON.
        """
        if frame_no < 700:
            return self._move_to(hero, self._mid_target(int(hero.get("camp", 0) or 0)))

        if frame_no < 1100:
            return self._move_to(hero, (-1200.0, -1200.0) if self.agent_idx == 0 else (1200.0, 1200.0))

        if frame_no >= 7600:
            if frame_no % 180 < 24:
                return self._skill_action(SKILL1_BUTTON)
            return [ATTACK_BUTTON, 15, 15, 15, 15, ENEMY_HERO_TARGET]

        phase = 0 if frame_no < 4300 else 1
        if self.eye_probe_last_phase != phase:
            self.eye_probe_last_phase = phase
            self.eye_probe_grass_pos = None

        hiding_agent = 0 if phase == 0 else 1
        probing_agent = 1 - hiding_agent
        phase_start = 1100 if phase == 0 else 4300
        local_frame = frame_no - phase_start

        if self.agent_idx == hiding_agent:
            if bool(hero.get("is_in_grass", False)):
                self.eye_probe_grass_pos = self._position(hero)
                # Do not cast while hidden; we want the exposure to come from
                # the other hero's probe skill, not our own action.
                return NO_ACTION.copy()

            if self.eye_probe_grass_pos is not None:
                return self._move_to(hero, self.eye_probe_grass_pos)

            grass_path = self._tight_grass_path(self.agent_idx)
            slot = min(local_frame // 220, len(grass_path) - 1)
            return self._move_to(hero, grass_path[slot])

        if self.agent_idx == probing_agent:
            # Keep the prober near but usually outside grass, then alternate
            # skill1/skill3 probe windows. Use larger periods so buff segments
            # around each cast remain separable.
            if local_frame % 240 < 36:
                return self._skill_action(SKILL1_BUTTON)
            if 96 <= local_frame % 360 < 132:
                return self._skill_action(SKILL3_BUTTON)
            outside_target = self._probe_outside_target(self.agent_idx, phase)
            return self._move_to(hero, outside_target)

        return None

    def _tight_grass_path(self, agent_idx):
        """Likely grass points sampled around the diagonal lane."""
        if agent_idx == 0:
            return [
                (-5200.0, -9000.0),
                (-2600.0, -9800.0),
                (2600.0, -9200.0),
                (7200.0, -5200.0),
                (9800.0, -1200.0),
                (9000.0, 3200.0),
                (4500.0, 8200.0),
                (-1200.0, 9800.0),
                (-7200.0, 5200.0),
                (-9800.0, 1200.0),
            ]
        return [
            (5200.0, 9000.0),
            (2600.0, 9800.0),
            (-2600.0, 9200.0),
            (-7200.0, 5200.0),
            (-9800.0, 1200.0),
            (-9000.0, -3200.0),
            (-4500.0, -8200.0),
            (1200.0, -9800.0),
            (7200.0, -5200.0),
            (9800.0, -1200.0),
        ]

    def _probe_outside_target(self, agent_idx, phase):
        if phase == 0:
            return (1800.0, 1800.0) if agent_idx == 1 else (-1800.0, -1800.0)
        return (-1800.0, -1800.0) if agent_idx == 0 else (1800.0, 1800.0)

    def _mid_target(self, camp):
        if camp == 1:
            return (-1200.0, -1200.0)
        if camp == 2:
            return (1200.0, 1200.0)
        return (0.0, 0.0)

    def _grass_path(self, agent_idx):
        # Broad diagonal sweep. Values are intentionally spread out because the
        # 2026 map coordinates differ between local dumps and live episodes.
        if agent_idx == 0:
            return [
                (-12000.0, -4000.0),
                (-9000.0, 2500.0),
                (-4500.0, 9000.0),
                (2500.0, 9000.0),
                (9000.0, 4500.0),
                (12000.0, -3500.0),
                (4500.0, -9500.0),
                (-3500.0, -9500.0),
                (0.0, 0.0),
            ]
        return [
            (12000.0, 4000.0),
            (9000.0, -2500.0),
            (4500.0, -9000.0),
            (-2500.0, -9000.0),
            (-9000.0, -4500.0),
            (-12000.0, 3500.0),
            (-4500.0, 9500.0),
            (3500.0, 9500.0),
            (0.0, 0.0),
        ]

    def _skill_cycle_action(self, frame_no, hero_id):
        cycle = (frame_no - 980) // 120
        phase = (frame_no - 980) % 120
        if phase >= 24:
            return [ATTACK_BUTTON, 15, 15, 15, 15, ENEMY_HERO_TARGET]
        buttons = [SKILL1_BUTTON, SKILL2_BUTTON, SKILL3_BUTTON]
        # Luban and DiRenjie both use 1/2/3 skill buttons in this action space.
        return self._skill_action(buttons[cycle % len(buttons)])

    def _skill_action(self, button):
        # Aim at enemy hero. If the skill is non-directional, the env will ignore
        # the direction sub-action by its legal mask.
        return [button, 15, 15, 15, 15, ENEMY_HERO_TARGET]

    def _move_to(self, hero, target):
        x, z = self._position(hero)
        dx = target[0] - x
        dz = target[1] - z
        if math.hypot(dx, dz) < 900.0:
            return NO_ACTION.copy()
        move_x, move_z = self._delta_action_16x16((x, z), target)
        return [MOVE_BUTTON, move_x, move_z, 0, 0, 0]

    def _delta_action_16x16(self, center, target):
        dx = float(target[0]) - float(center[0])
        dz = float(target[1]) - float(center[1])
        max_abs = max(abs(dx), abs(dz))
        if max_abs <= 1e-6:
            return 8, 8
        move_x = int(math.ceil(dx / max_abs * 7.0) + 8)
        move_z = int(math.ceil(dz / max_abs * 7.0) + 8)
        return max(0, min(15, move_x)), max(0, min(15, move_z))

    def _self_hero(self, observation, frame_state):
        hero_states = frame_state.get("hero_states", []) or []
        player_id = observation.get("player_id", frame_state.get("player_id"))
        for hero in hero_states:
            if player_id is not None and hero.get("runtime_id") == player_id:
                return hero
            if hero.get("player_id") == player_id:
                return hero
        if self.agent_idx < len(hero_states):
            return hero_states[self.agent_idx]
        return hero_states[0] if hero_states else None

    def _enemy_hero(self, observation, frame_state, self_hero):
        hero_states = frame_state.get("hero_states", []) or []
        self_runtime = self_hero.get("runtime_id", self_hero.get("runtimeId", self_hero.get("player_id")))
        self_camp = self_hero.get("camp")
        for hero in hero_states:
            runtime = hero.get("runtime_id", hero.get("runtimeId", hero.get("player_id")))
            if self_runtime is not None and runtime == self_runtime:
                continue
            if self_camp is not None and hero.get("camp") == self_camp:
                continue
            return hero
        for hero in hero_states:
            if hero is not self_hero:
                return hero
        return None

    def _position(self, actor):
        location = actor.get("location", {}) or actor.get("position", {}) or {}
        x = location.get("x", location.get("X", 0.0))
        z = location.get("z", location.get("Z", location.get("y", 0.0)))
        return float(x or 0.0), float(z or 0.0)
