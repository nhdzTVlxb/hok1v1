#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Optional action/position audit dumps for platform train_test runs."""

import json
import math
import os
from pathlib import Path

LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
LEGAL_ACTION_SIZE_LIST = [12, 16, 16, 16, 16, 108]
UNSEEN_PADDING = 100000


def _truthy(value):
    return str(value).lower() in ("1", "true", "yes", "on")


def _int_value(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_pos(actor):
    loc = (actor or {}).get("location", {}) or {}
    if isinstance(loc, dict):
        return {
            "x": float(loc.get("x", 0) or 0),
            "z": float(loc.get("z", 0) or 0),
        }
    if isinstance(loc, (list, tuple)) and len(loc) >= 3:
        return {"x": float(loc[0] or 0), "z": float(loc[2] or 0)}
    return {"x": 0.0, "z": 0.0}


def _hp_ratio(actor):
    max_hp = float((actor or {}).get("max_hp", 0) or 0)
    if max_hp <= 0:
        return 0.0
    return float((actor or {}).get("hp", 0) or 0) / max_hp


def _actor_id(actor):
    return (actor or {}).get("runtime_id", (actor or {}).get("player_id", 0))


def _distance(a, b):
    apos = _get_pos(a)
    bpos = _get_pos(b)
    return math.dist([apos["x"], apos["z"]], [bpos["x"], bpos["z"]])


def _is_unseen(actor):
    pos = _get_pos(actor)
    return abs(pos["x"]) >= UNSEEN_PADDING or abs(pos["z"]) >= UNSEEN_PADDING


def _json_safe(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _camp_id(camp):
    if isinstance(camp, str):
        if camp in ("0", "1", "2"):
            value = int(camp)
            return 1 if value == 0 else value
        if camp[-1:].isdigit():
            return int(camp[-1])
    if isinstance(camp, int):
        return 1 if camp == 0 else camp
    return camp


def _is_soldier(actor):
    cfg = int((actor or {}).get("config_id", (actor or {}).get("configId", 0)) or 0)
    return (actor or {}).get("actor_type") == 1 and ((actor or {}).get("sub_type") == 11 or cfg in (6800, 6801, 6802, 6803, 6804, 6805))


def _skill_summary(hero):
    slots = (((hero or {}).get("skill_state", {}) or {}).get("slot_states", []) or [])
    result = []
    for slot in slots:
        result.append(
            {
                "slot_type": slot.get("slot_type"),
                "usable": slot.get("usable"),
                "cooldown": slot.get("cooldown"),
                "level": slot.get("level"),
            }
        )
    return result


def _split_legal_action(legal_action):
    if len(legal_action) != sum(LEGAL_ACTION_SIZE_LIST):
        return None
    chunks = []
    cursor = 0
    for size in LEGAL_ACTION_SIZE_LIST:
        chunks.append(legal_action[cursor : cursor + size])
        cursor += size
    return chunks


def _target_legal_info(legal_action, action):
    legal_chunks = _split_legal_action(list(legal_action or []))
    if legal_chunks is None or not action:
        return {}
    button = _int_value(action[0], -1)
    target = _int_value(action[-1], -1)
    if button < 0 or button >= LABEL_SIZE_LIST[0]:
        return {"button": button, "target": target, "button_valid": False}
    target_rows = legal_chunks[-1]
    start = button * LABEL_SIZE_LIST[-1]
    row = target_rows[start : start + LABEL_SIZE_LIST[-1]]
    return {
        "button": button,
        "target": target,
        "button_valid": bool(legal_chunks[0][button]) if button < len(legal_chunks[0]) else False,
        "target_valid": bool(0 <= target < len(row) and row[target] > 0),
        "target_row": _json_safe(row),
        "target_legal_indices": [idx for idx, value in enumerate(row) if value > 0],
    }


def _hero_summary(hero):
    return {
        "runtime_id": _actor_id(hero),
        "hero_id": (hero or {}).get("config_id", (hero or {}).get("configId")),
        "hp": (hero or {}).get("hp"),
        "max_hp": (hero or {}).get("max_hp"),
        "hp_ratio": _hp_ratio(hero),
        "dead_cnt": (hero or {}).get("dead_cnt", (hero or {}).get("deadCnt")),
        "revive_time": (hero or {}).get("revive_time"),
        "pos": _get_pos(hero),
        "unseen": _is_unseen(hero),
        "attack_target": (hero or {}).get("attack_target"),
        "money": (hero or {}).get("money", (hero or {}).get("money_cnt")),
        "level": (hero or {}).get("level"),
        "is_in_grass": (hero or {}).get("is_in_grass"),
        "skill_state": _skill_summary(hero),
    }


def _load_config():
    config = dict(os.environ)
    for path in (Path("/data/projects/hok1v1/.dump_obs_config.json"), Path("/workspace/code/.dump_obs_config.json")):
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as file_obj:
                file_config = json.load(file_obj)
            config.update({str(k): str(v) for k, v in file_config.items()})
        except Exception:
            continue
    return config


class ActionAuditDumper:
    def __init__(self, logger=None):
        self.logger = logger
        config = _load_config()
        self.enabled = _truthy(config.get("ACTION_AUDIT", ""))
        frame_text = config.get("ACTION_AUDIT_FRAMES", "56,100,200,300,500,1000,1500,2000,4000,6000")
        self.frames = {_int_value(item.strip()) for item in frame_text.split(",") if item.strip()}
        self.every_n = _int_value(config.get("ACTION_AUDIT_EVERY_N", 0), 0)
        self.output_dir = Path(config.get("ACTION_AUDIT_DIR", "/data/projects/hok1v1/action_audit"))
        self.dumped = set()
        self.last_seen_frame = {}
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"ACTION_AUDIT enabled, frames={sorted(self.frames)}, every_n={self.every_n}, dir={self.output_dir}")

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message, flush=True)

    def target_frames_to_dump(self, episode, agent_idx, frame_no, state_frame_no):
        if not self.enabled:
            return []

        current_frame = max(int(frame_no), int(state_frame_no))
        last_key = (episode, agent_idx)
        last_frame = self.last_seen_frame.get(last_key, -1)
        self.last_seen_frame[last_key] = max(last_frame, current_frame)

        target_frames = [target for target in self.frames if last_frame < target <= current_frame]
        if self.every_n > 0 and current_frame > last_frame:
            first = max(last_frame + 1, 0)
            first = first + ((self.every_n - first % self.every_n) % self.every_n)
            target_frames.extend(range(first, current_frame + 1, self.every_n))

        result = []
        for target_frame in sorted(set(target_frames)):
            key = (episode, agent_idx, target_frame)
            if key in self.dumped:
                continue
            self.dumped.add(key)
            result.append(target_frame)
        return result

    def dump(self, episode, agent_idx, frame_no, observation, action, action_source, usr_conf=None, is_eval=False, do_predict=True):
        observation = observation or {}
        frame_state = observation.get("frame_state", {}) or {}
        state_frame_no = _int_value(frame_state.get("frame_no", frame_state.get("frameNo", frame_no)), int(frame_no))
        target_frames = self.target_frames_to_dump(episode, agent_idx, int(frame_no), state_frame_no)
        if not target_frames:
            return

        main_camp = _camp_id(observation.get("camp", observation.get("player_camp", 1)))
        player_id = observation.get("player_id")
        hero, enemy = None, None
        for item in frame_state.get("hero_states", []) or []:
            if item.get("runtime_id", item.get("player_id")) == player_id or _camp_id(item.get("camp")) == main_camp:
                hero = item
            else:
                enemy = item

        organs = []
        soldiers = []
        for npc in frame_state.get("npc_states", []) or []:
            if npc.get("actor_type") == 2:
                organs.append(
                    {
                        "camp": _camp_id(npc.get("camp")),
                        "sub_type": npc.get("sub_type"),
                        "config_id": npc.get("config_id", npc.get("configId")),
                        "hp": npc.get("hp"),
                        "max_hp": npc.get("max_hp"),
                        "pos": _get_pos(npc),
                    }
                )
            elif _is_soldier(npc):
                soldiers.append(npc)

        near_soldiers = []
        if hero:
            for soldier in sorted(soldiers, key=lambda item: _distance(hero, item))[:8]:
                near_soldiers.append(
                    {
                        "runtime_id": _actor_id(soldier),
                        "camp": _camp_id(soldier.get("camp")),
                        "config_id": soldier.get("config_id", soldier.get("configId")),
                        "hp": soldier.get("hp"),
                        "max_hp": soldier.get("max_hp"),
                        "hp_ratio": _hp_ratio(soldier),
                        "pos": _get_pos(soldier),
                        "distance": _distance(hero, soldier),
                        "attack_target": soldier.get("attack_target"),
                    }
                )

        legal_action = observation.get("legal_action", []) or []
        hero_payload = _hero_summary(hero)
        enemy_payload = _hero_summary(enemy)
        payload = {
            "episode": episode,
            "agent_idx": agent_idx,
            "frame_no": int(frame_no),
            "state_frame_no": int(state_frame_no),
            "dump_target_frames": _json_safe(target_frames),
            "is_eval": bool(is_eval),
            "do_predict": bool(do_predict),
            "camp": main_camp,
            "player_id": player_id,
            "action": _json_safe(list(action or [])),
            "action_source": action_source,
            "legal_buttons": _json_safe(list(legal_action[:12])),
            "target_legal": _target_legal_info(legal_action, action),
            "hero": hero_payload,
            "enemy": enemy_payload,
            "enemy_hero": enemy_payload,
            "organs": organs,
            "near_soldiers": near_soldiers,
            "entity_counts": {
                "heroes": len(frame_state.get("hero_states", []) or []),
                "npcs": len(frame_state.get("npc_states", []) or []),
                "soldiers": len(soldiers),
                "organs": len(organs),
                "bullets": len(frame_state.get("bullets", []) or []),
                "cakes": len(frame_state.get("cakes", []) or []),
            },
            "episode_conf": _json_safe((usr_conf or {}).get("episode", {})),
            "lineups": _json_safe((usr_conf or {}).get("lineups", {})),
        }
        safe_payload = _json_safe(payload)
        for target_frame in target_frames:
            path = self.output_dir / f"action_audit_episode_{episode:04d}_agent_{agent_idx}_frame_{int(target_frame):06d}.json"
            with path.open("w", encoding="utf-8") as file_obj:
                json.dump(safe_payload, file_obj, ensure_ascii=False, indent=2)
