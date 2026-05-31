#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Optional Phase 1 feature audit dumps for platform train_test runs."""

import json
import os
from pathlib import Path

import numpy as np


FEATURE_SECTIONS = [
    ("hero_frd", 0, 113),
    ("hero_emy", 113, 230),
    ("soldier", 230, 518),
    ("organ", 518, 620),
    ("river_crab", 620, 634),
    ("bullet", 634, 1114),
    ("environment", 1114, 1128),
    ("game_meta", 1128, 1129),
]


def _truthy(value):
    return str(value).lower() in ("1", "true", "yes", "on")


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


class FeatureAuditDumper:
    def __init__(self, logger=None):
        self.logger = logger
        config = _load_config()
        self.enabled = _truthy(config.get("FEATURE_AUDIT", ""))
        frame_text = config.get("FEATURE_AUDIT_FRAMES", "56,500,1000,1778,2500,4000,6000")
        self.frames = {int(item.strip()) for item in str(frame_text).split(",") if item.strip()}
        self.every_n = int(config.get("FEATURE_AUDIT_EVERY_N", 0) or 0)
        self.output_dir = Path(config.get("FEATURE_AUDIT_DIR", "/data/projects/hok1v1/feature_audit"))
        self.dumped = set()
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"FEATURE_AUDIT enabled, frames={sorted(self.frames)}, every_n={self.every_n}, dir={self.output_dir}")

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message, flush=True)

    def should_dump(self, episode, agent_idx, frame_no, force=False):
        if not self.enabled:
            return False
        key = (episode, agent_idx, frame_no)
        if key in self.dumped:
            return False
        if force or frame_no in self.frames or (self.every_n > 0 and frame_no % self.every_n == 0):
            self.dumped.add(key)
            return True
        return False

    def dump(self, episode, agent_idx, observation, feature, force=False):
        frame_state = (observation or {}).get("frame_state", {}) or {}
        frame_no = int(frame_state.get("frame_no", frame_state.get("frameNo", 0)) or 0)
        if not self.should_dump(episode, agent_idx, frame_no, force=force):
            return

        feature = np.asarray(feature, dtype=np.float32)
        sections = {}
        for name, start, end in FEATURE_SECTIONS:
            values = feature[start:end]
            sections[name] = {
                "start": start,
                "end": end,
                "dim": end - start,
                "nonzero": int(np.count_nonzero(values)),
                "min": float(np.min(values)) if values.size else 0.0,
                "max": float(np.max(values)) if values.size else 0.0,
                "mean": float(np.mean(values)) if values.size else 0.0,
                "std": float(np.std(values)) if values.size else 0.0,
            }

        heroes = frame_state.get("hero_states", []) or []
        npcs = frame_state.get("npc_states", []) or []
        bullets = frame_state.get("bullets", []) or []
        cakes = frame_state.get("cakes", []) or []
        hero_keys = sorted({key for hero in heroes for key in hero.keys()})
        npc_keys = sorted({key for npc in npcs for key in npc.keys()})
        bullet_keys = sorted({key for bullet in bullets for key in bullet.keys()})
        cake_keys = sorted({key for cake in cakes for key in cake.keys()})

        required_probe = {
            "hero_has_total_hurt_to_organ": "total_hurt_to_organ" in hero_keys,
            "hero_has_total_be_hurt_by_organ": "total_be_hurt_by_organ" in hero_keys,
            "npc_has_total_hurt_to_hero": "total_hurt_to_hero" in npc_keys,
            "npc_has_total_hurt_to_organ": "total_hurt_to_organ" in npc_keys,
            "organ_has_attack_count": any(key in npc_keys for key in ("attack_count", "attack_seq", "conti_hit_count")),
            "skill_slot_has_ep_cost": any(
                any(key in slot for key in ("cost", "ep_cost"))
                for hero in heroes
                for slot in (((hero.get("skill_state", {}) or {}).get("slot_states", []) or []))
            ),
            "cake_has_spawn_timer": any(key in cake_keys for key in ("spawn_timer", "next_refresh_time", "refresh_time")),
        }

        payload = {
            "episode": episode,
            "agent_idx": agent_idx,
            "frame_no": frame_no,
            "feature_dim": int(feature.size),
            "feature_finite": bool(np.all(np.isfinite(feature))),
            "feature_min": float(np.min(feature)) if feature.size else 0.0,
            "feature_max": float(np.max(feature)) if feature.size else 0.0,
            "sections": sections,
            "entity_counts": {
                "heroes": len(heroes),
                "npcs": len(npcs),
                "bullets": len(bullets),
                "cakes": len(cakes),
            },
            "schema_keys": {
                "hero": hero_keys,
                "npc": npc_keys,
                "bullet": bullet_keys,
                "cake": cake_keys,
            },
            "required_probe": required_probe,
        }
        path = self.output_dir / f"feature_audit_episode_{episode:04d}_agent_{agent_idx}_frame_{frame_no:06d}.json"
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
