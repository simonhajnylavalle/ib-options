"""
paths.py
─────────────────────────────────────────────────────────────────────────────
Application path helpers.

Centralises user-writable config/state locations so installed wheels do not
try to read or write runtime files inside site-packages.

Overrides:
  OPTION_ALGORITHM_HOME        → base dir for both config + state files
  OPTION_ALGORITHM_CONFIG      → full path to config.toml
  OPTION_ALGORITHM_STATE_FILE  → full path to plays.json
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = "option_algorithm"


def _expand(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _xdg_dir(env_name: str, fallback: Path) -> Path:
    raw = os.getenv(env_name)
    return _expand(raw) if raw else fallback


def app_home() -> Path | None:
    raw = os.getenv("OPTION_ALGORITHM_HOME")
    return _expand(raw) if raw else None


def default_config_path(filename: str = "config.toml") -> Path:
    home = app_home()
    if home is not None:
        return home / filename
    base = _xdg_dir("XDG_CONFIG_HOME", Path.home() / ".config")
    return base / APP_DIR / filename


def default_state_path(filename: str = "plays.json") -> Path:
    home = app_home()
    if home is not None:
        return home / filename
    base = _xdg_dir("XDG_STATE_HOME", Path.home() / ".local" / "state")
    return base / APP_DIR / filename


def configured_config_path() -> Path | None:
    raw = os.getenv("OPTION_ALGORITHM_CONFIG")
    return _expand(raw) if raw else None


def configured_state_path() -> Path | None:
    raw = os.getenv("OPTION_ALGORITHM_STATE_FILE")
    return _expand(raw) if raw else None


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
