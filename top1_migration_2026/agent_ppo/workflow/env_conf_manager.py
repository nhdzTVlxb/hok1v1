#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Local train_env_conf manager with eval opponent pool support.

The stock validator treats eval_opponent_type as a single string.  This
manager keeps that field intact and adds optional eval_opponent_types for
random eval rotation.
"""

import copy
import random

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


class EnvConfManager:
    def __init__(self, config_path, logger):
        self.config_path = config_path
        self.logger = logger
        self.usr_conf = None
        self.episode_cnt = 0
        self.eval_interval = 0
        self.random_eval_start = 0
        self.default_opponent_agent = None
        self.auto_switch_monitor_side = False
        self.monitor_side = 0
        self.initialize()

    def initialize(self):
        self.usr_conf = self._read_usr_conf()
        episode = self.usr_conf["episode"]
        self.eval_interval = int(episode.get("eval_interval", 10)) + 1
        self.default_opponent_agent = str(episode.get("opponent_agent", "selfplay"))
        self.auto_switch_monitor_side = bool(self.usr_conf["monitor"].get("auto_switch_monitor_side", False))
        self.monitor_side = int(self.usr_conf["monitor"].get("monitor_side", 0))
        self.random_eval_start = random.randint(0, self.eval_interval) if self.eval_interval != 0 else 0

    def _read_usr_conf(self):
        if tomllib is not None:
            with open(self.config_path, "rb") as f:
                conf = tomllib.load(f)
        else:
            with open(self.config_path, "r", encoding="utf-8") as f:
                conf = self._parse_minimal_toml(f.read())
        self._validate_usr_conf(conf)
        return conf

    def _parse_minimal_toml(self, text):
        conf = {"monitor": {}, "episode": {}, "lineups": {"blue_camp": [], "red_camp": []}}
        section = None
        current_list_item = None
        lines = text.splitlines()
        idx = 0
        while idx < len(lines):
            raw = lines[idx].split("#", 1)[0].strip()
            idx += 1
            if not raw:
                continue
            if raw == "[monitor]":
                section = conf["monitor"]
                current_list_item = None
                continue
            if raw == "[episode]":
                section = conf["episode"]
                current_list_item = None
                continue
            if raw == "[[lineups.blue_camp]]":
                current_list_item = {}
                conf["lineups"]["blue_camp"].append(current_list_item)
                section = current_list_item
                continue
            if raw == "[[lineups.red_camp]]":
                current_list_item = {}
                conf["lineups"]["red_camp"].append(current_list_item)
                section = current_list_item
                continue
            if "=" not in raw or section is None:
                continue
            key, value = [part.strip() for part in raw.split("=", 1)]
            if value == "[":
                values = []
                while idx < len(lines):
                    item = lines[idx].split("#", 1)[0].strip()
                    idx += 1
                    if not item:
                        continue
                    if item == "]":
                        break
                    values.append(self._parse_toml_scalar(item.rstrip(",")))
                section[key] = values
            else:
                section[key] = self._parse_toml_scalar(value)
        return conf

    def _parse_toml_scalar(self, value):
        value = value.strip().rstrip(",")
        if len(value) >= 2 and value[0] == value[-1] == '"':
            return value[1:-1]
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            return value

    def _validate_usr_conf(self, conf):
        if not isinstance(conf.get("monitor"), dict):
            raise ValueError("train_env_conf.toml missing [monitor]")
        if not isinstance(conf.get("episode"), dict):
            raise ValueError("train_env_conf.toml missing [episode]")
        if not isinstance(conf.get("lineups"), dict):
            raise ValueError("train_env_conf.toml missing [lineups]")

        episode = conf["episode"]
        if not isinstance(episode.get("opponent_agent"), str):
            raise ValueError("episode.opponent_agent must be string")
        if not isinstance(episode.get("eval_opponent_type"), str):
            raise ValueError("episode.eval_opponent_type must be string")
        if not isinstance(episode.get("eval_interval"), int) or episode["eval_interval"] < 2:
            raise ValueError("episode.eval_interval must be integer >= 2")
        eval_pool = episode.get("eval_opponent_types")
        if eval_pool is not None:
            if not isinstance(eval_pool, list) or not eval_pool:
                raise ValueError("episode.eval_opponent_types must be a non-empty string list")
            if not all(isinstance(item, str) and item for item in eval_pool):
                raise ValueError("episode.eval_opponent_types must contain only non-empty strings")

        for camp_key in ("blue_camp", "red_camp"):
            camp = conf["lineups"].get(camp_key)
            if not isinstance(camp, list) or not camp:
                raise ValueError(f"lineups.{camp_key} must be a non-empty list")
            for item in camp:
                if not isinstance(item, dict) or not isinstance(item.get("hero_id"), int):
                    raise ValueError(f"lineups.{camp_key} item must contain integer hero_id")

    def get_current_config(self):
        return self.usr_conf

    def get_monitor_side(self):
        return self.monitor_side

    def get_opponent_agent(self):
        return self.usr_conf["episode"]["opponent_agent"]

    def update_config(self, lineup=None):
        if lineup:
            if len(lineup) == 2 and all(type(hero_id) == int for hero_id in lineup[:2]):
                self.usr_conf["lineups"]["blue_camp"][0]["hero_id"] = lineup[0]
                self.usr_conf["lineups"]["red_camp"][0]["hero_id"] = lineup[1]
            else:
                raise ValueError("Invalid lineup format, expected list of 2 integers")

        if self.auto_switch_monitor_side:
            self.monitor_side = 1 - self.monitor_side
        self.usr_conf["monitor"]["monitor_side"] = self.monitor_side

        is_eval = (self.episode_cnt + self.random_eval_start) % self.eval_interval == 0
        episode = self.usr_conf["episode"]
        if is_eval:
            eval_pool = episode.get("eval_opponent_types") or [episode.get("eval_opponent_type", "common_ai")]
            episode["eval_opponent_type"] = random.choice(eval_pool)
        episode["opponent_agent"] = self.default_opponent_agent if not is_eval else episode["eval_opponent_type"]

        self.episode_cnt += 1
        return self.get_current_config(), is_eval, self.get_monitor_side()

    @staticmethod
    def extract_hero_ids_from_usr_conf(usr_conf):
        lineups = usr_conf.get("lineups", {})
        blue = [int(item["hero_id"]) for item in lineups.get("blue_camp", [])]
        red = [int(item["hero_id"]) for item in lineups.get("red_camp", [])]
        return blue, red

    @staticmethod
    def inject_select_skills(usr_conf, camp_key, select_skills):
        for hero_conf in usr_conf.get("lineups", {}).get(camp_key, []):
            hero_id = int(hero_conf.get("hero_id", 0))
            if hero_id in select_skills:
                hero_conf["select_skill"] = int(select_skills[hero_id])

    def snapshot(self):
        return copy.deepcopy(self.usr_conf)
