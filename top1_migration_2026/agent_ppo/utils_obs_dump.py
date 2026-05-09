#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Optional observation dumper for 2026 raw frame_state debugging."""

import json
import os
from pathlib import Path


DEFAULT_DUMP_FRAMES = "56,500,1000,1094,1148,1500,1778,2500,4000,6000,9000,12000"
CONFIG_PATHS = (
    Path("/data/projects/hok1v1/.dump_obs_config.json"),
    Path("/workspace/code/.dump_obs_config.json"),
)


def _to_jsonable(value, depth=0):
    if depth > 8:
        return "<max_depth>"
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v, depth + 1) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return repr(value)


def _schema(value, depth=0):
    if depth > 8:
        return {"type": "max_depth"}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys()),
            "fields": {str(k): _schema(v, depth + 1) for k, v in value.items()},
        }
    if isinstance(value, (list, tuple)):
        item_schema = _schema(value[0], depth + 1) if value else {"type": "empty"}
        return {"type": "list", "len": len(value), "item": item_schema}
    return {"type": type(value).__name__}


def _load_config():
    config = dict(os.environ)
    for path in CONFIG_PATHS:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as file_obj:
                file_config = json.load(file_obj)
            config.update({str(k): str(v) for k, v in file_config.items()})
        except Exception:
            continue
    return config


class ObsDumper:
    def __init__(self, logger=None):
        self.logger = logger
        config = _load_config()
        self.enabled = str(config.get("DUMP_OBS", "")).lower() in ("1", "true", "yes", "on")
        self.print_schema = str(config.get("DUMP_OBS_PRINT_SCHEMA", "")).lower() in ("1", "true", "yes", "on")
        self.max_episodes = int(config.get("DUMP_OBS_MAX_EPISODES", 4) or 4)
        frame_text = config.get("DUMP_OBS_FRAMES", DEFAULT_DUMP_FRAMES)
        self.frames = {int(item.strip()) for item in str(frame_text).split(",") if item.strip()}
        self.output_dir = Path(config.get("DUMP_OBS_DIR", "/data/projects/hok1v1/debug_obs"))
        self.dumped = set()
        self.last_seen_frame = {}
        if self.enabled:
            (self.output_dir / "raw").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "schema").mkdir(parents=True, exist_ok=True)
            self._log(f"DUMP_OBS enabled, frames={sorted(self.frames)}, dir={self.output_dir}")

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message, flush=True)

    def dump(self, episode, agent_idx, frame_no, observation, usr_conf=None, extra_info=None, env_meta=None):
        if not self.enabled or episode > self.max_episodes:
            return
        state_frame_no = int(((observation or {}).get("frame_state", {}) or {}).get("frame_no", frame_no) or frame_no)
        current_frame = max(int(frame_no), state_frame_no)
        last_key = (episode, agent_idx)
        last_frame = self.last_seen_frame.get(last_key, -1)
        self.last_seen_frame[last_key] = max(last_frame, current_frame)

        target_frames = sorted(target for target in self.frames if last_frame < target <= current_frame)
        if not target_frames:
            return

        payload = {
            "episode": episode,
            "agent_idx": agent_idx,
            "frame_no": frame_no,
            "state_frame_no": state_frame_no,
            "dump_target_frames": target_frames,
            "usr_conf": usr_conf,
            "extra_info": extra_info,
            "env_meta": env_meta,
            "observation": observation,
        }
        safe_payload = _to_jsonable(payload)
        for target_frame in target_frames:
            key = (episode, agent_idx, target_frame)
            if key in self.dumped:
                continue
            self.dumped.add(key)
            filename = f"episode_{episode:04d}_agent_{agent_idx}_frame_{target_frame:06d}.json"
            raw_path = self.output_dir / "raw" / filename
            schema_path = self.output_dir / "schema" / filename
            with raw_path.open("w", encoding="utf-8") as file_obj:
                json.dump(safe_payload, file_obj, ensure_ascii=False, indent=2)
            with schema_path.open("w", encoding="utf-8") as file_obj:
                json.dump(_schema(safe_payload), file_obj, ensure_ascii=False, indent=2)
            if self.print_schema:
                self._log(
                    f"DUMP_OBS wrote target={target_frame}, env_frame={frame_no}, "
                    f"state_frame={state_frame_no}, raw={raw_path}, schema={schema_path}"
                )
