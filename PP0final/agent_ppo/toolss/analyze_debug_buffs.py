#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Summarise debug observation dumps for buff/config id discovery.

Default mode only scans hero_states because short hero buffs are the target and
dense per-frame dumps can be large enough for npc/organ analysis to be killed.

Usage:
    python3 toolss/analyze_debug_buffs.py /data/projects/hok1v1/debug_obs_probe/raw
    python3 toolss/analyze_debug_buffs.py /data/projects/hok1v1/debug_obs_probe/raw --all-actors
"""

import json
import re
import sys
from pathlib import Path


DEFAULT_GROUPS = ("hero_states",)
ALL_GROUPS = ("hero_states", "npc_states", "organ_states")
MAX_FRAME_SAMPLES = 20
MAX_SEGMENTS = 30
JSON_DECODER = json.JSONDecoder()
FRAME_RE = re.compile(r'"state_frame_no"\s*:\s*(\d+)')
FRAME_FALLBACK_RE = re.compile(r'"frame_no"\s*:\s*(\d+)')
AGENT_RE = re.compile(r'"agent_idx"\s*:\s*(\d+)')
EPISODE_RE = re.compile(r'"episode"\s*:\s*(\d+)')


class Stat:
    def __init__(self):
        self.first = None
        self.last = None
        self.count = 0
        self.samples = []
        self.segments = []
        self.extra_samples = []
        self._seg_start = None
        self._seg_end = None
        self._seg_count = 0
        self._base_gap = None

    def add(self, frame, extra=None):
        frame = int(frame)
        if self.last is not None and frame == self.last:
            # The same frame can appear in repeated dump files. Keep duration
            # statistics frame-based instead of duplicate-file-based.
            return
        if self.first is None:
            self.first = frame
            self.last = frame
            self._seg_start = frame
            self._seg_end = frame
            self._seg_count = 1
        else:
            gap = frame - self.last
            if gap < 0:
                self._close_segment()
                self._seg_start = frame
                self._seg_end = frame
                self._seg_count = 1
                self.last = frame
                self.count += 1
                if len(self.samples) < MAX_FRAME_SAMPLES:
                    self.samples.append(frame)
                return
            if gap > 0 and self._base_gap is None:
                self._base_gap = gap
            split_gap = max((self._base_gap or 1) * 2, (self._base_gap or 1) + 1)
            if gap > split_gap:
                self._close_segment()
                self._seg_start = frame
                self._seg_end = frame
                self._seg_count = 1
            elif gap >= 0:
                self._seg_end = frame
                self._seg_count += 1
            self.last = frame
        self.count += 1
        if len(self.samples) < MAX_FRAME_SAMPLES:
            self.samples.append(frame)
        if extra is not None and len(self.extra_samples) < MAX_FRAME_SAMPLES:
            self.extra_samples.append({"frame": frame, **extra})

    def _close_segment(self):
        if self._seg_start is None:
            return
        if len(self.segments) < MAX_SEGMENTS:
            self.segments.append(
                {
                    "start": self._seg_start,
                    "end": self._seg_end,
                    "duration_frames": self._seg_end - self._seg_start,
                    "sample_count": self._seg_count,
                }
            )

    def to_row(self, key):
        self._close_segment()
        return {
            "key": key,
            "first_frame": self.first,
            "last_frame": self.last,
            "count": self.count,
            "frames_sample": self.samples,
            "segments": self.segments,
            "segment_count": len(self.segments),
            "max_duration_frames": max((seg["duration_frames"] for seg in self.segments), default=0),
            "extra_samples": self.extra_samples,
        }


def extract_int(pattern, text, default=0):
    match = pattern.search(text)
    return int(match.group(1)) if match else default


def extract_actor_group(text, group_name):
    key = f'"{group_name}"'
    key_pos = text.find(key)
    if key_pos < 0:
        return []
    colon_pos = text.find(":", key_pos + len(key))
    if colon_pos < 0:
        return []
    pos = colon_pos + 1
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != "[":
        return []
    try:
        actors, _ = JSON_DECODER.raw_decode(text, pos)
    except json.JSONDecodeError:
        return []
    return actors if isinstance(actors, list) else []


def iter_actors_from_text(text, groups):
    for group_name in groups:
        for actor in extract_actor_group(text, group_name):
            yield group_name, actor


def actor_label(group_name, actor):
    config_id = actor.get("config_id", actor.get("configId", 0))
    runtime_id = actor.get("runtime_id", actor.get("runtimeId", actor.get("player_id", "")))
    camp = actor.get("camp", "")
    return f"{group_name}:config={config_id}:runtime={runtime_id}:camp={camp}"


def add_stat(mapping, key, frame, extra=None):
    if key not in mapping:
        mapping[key] = Stat()
    mapping[key].add(frame, extra=extra)


def actor_position(actor):
    location = actor.get("location", {}) or actor.get("position", {}) or {}
    x = location.get("x", location.get("X", 0.0))
    z = location.get("z", location.get("Z", location.get("y", 0.0)))
    try:
        return {"x": round(float(x or 0.0), 2), "z": round(float(z or 0.0), 2)}
    except (TypeError, ValueError):
        return {"x": x, "z": z}


def summarise(raw_dir, groups):
    buff_skills = {}
    buff_marks = {}
    behav_modes = {}
    grass = {}
    visible = {}
    loaded = 0

    paths = sorted(Path(raw_dir).glob("*.json"), key=path_sort_key)
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        episode = extract_int(EPISODE_RE, text, 0)
        frame = extract_int(FRAME_RE, text, extract_int(FRAME_FALLBACK_RE, text, 0))
        agent_idx = extract_int(AGENT_RE, text, 0)
        loaded += 1
        if loaded % 2000 == 0:
            print(f"processed={loaded}/{len(paths)}", flush=True)

        for group_name, actor in iter_actors_from_text(text, groups):
            label = actor_label(group_name, actor)
            prefix = f"episode={episode}:agent={agent_idx}:{label}"
            pos_sample = actor_position(actor)
            behav = actor.get("behav_mode", actor.get("behave"))
            if behav is not None:
                add_stat(behav_modes, (prefix, behav), frame, pos_sample)
            if "is_in_grass" in actor:
                add_stat(grass, (prefix, actor.get("is_in_grass")), frame, pos_sample)
            if "camp_visible" in actor:
                add_stat(visible, (prefix, tuple(actor.get("camp_visible") or [])), frame, pos_sample)

            buff_state = actor.get("buff_state", {}) or {}
            for buff in buff_state.get("buff_skills", []) or []:
                config_id = buff.get("configId", buff.get("config_id"))
                if config_id is not None:
                    add_stat(buff_skills, (prefix, int(config_id)), frame, pos_sample)
            for buff in buff_state.get("buff_marks", []) or []:
                config_id = buff.get("configId", buff.get("config_id"))
                layer = buff.get("layer", buff.get("count", buff.get("times", "")))
                if config_id is not None:
                    add_stat(buff_marks, (prefix, int(config_id), layer), frame, pos_sample)

    summary = {
        "meta": {
            "loaded_frames": loaded,
            "actor_groups": list(groups),
            "hero_states_only": tuple(groups) == DEFAULT_GROUPS,
        },
        "buff_skills": compact(buff_skills),
        "buff_marks": compact(buff_marks),
        "behav_modes": compact(behav_modes),
        "is_in_grass": compact(grass),
        "camp_visible": compact(visible),
    }
    return loaded, summary


def path_sort_key(path):
    name = path.name
    episode = 0
    agent = 0
    frame = 0
    for part in name.split("_"):
        if part.startswith("episode"):
            continue
    match = re.search(r"episode_(\d+)_agent_(\d+)_frame_(\d+)", name)
    if match:
        episode = int(match.group(1))
        agent = int(match.group(2))
        frame = int(match.group(3))
    return episode, agent, frame, name


def compact(mapping):
    rows = []
    for key, stat in sorted(mapping.items(), key=lambda item: str(item[0])):
        rows.append(stat.to_row(key))
    return rows


def main():
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/projects/hok1v1/debug_obs/raw")
    groups = ALL_GROUPS if "--all-actors" in sys.argv[2:] else DEFAULT_GROUPS
    loaded, summary = summarise(raw_dir, groups)
    out_path = raw_dir.parent / "debug_buff_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    print(f"loaded_frames={loaded}")
    print(f"actor_groups={','.join(groups)}")
    print(f"summary={out_path}")
    print("buff_skill_config_ids=")
    ids = sorted({row["key"][1] for row in summary["buff_skills"]})
    print(ids)


if __name__ == "__main__":
    main()
