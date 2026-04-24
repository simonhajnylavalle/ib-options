"""
config.py
─────────────────────────────────────────────────────────────────────────────
Load config.toml and produce typed objects for the strategy.

Runtime config is resolved in this order:
  1. Explicit path passed to load(...)
  2. OPTION_ALGORITHM_CONFIG
  3. ./config.toml
  4. user config dir (~/.config/option_algorithm/config.toml by default)
  5. legacy module-adjacent config.toml

If no config file exists, a default one is bootstrapped into the user config
location so installed wheels still have an editable runtime config.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # pip install tomli for Python 3.10

from execution import PriceMode, RetryProfile
from paths import configured_config_path, default_config_path, ensure_parent
from strategy import ContractSpec, ExitProfile

_LEGACY_CFG_PATH = Path(__file__).resolve().parent / "config.toml"

_DEFAULT_RAW: dict[str, Any] = {
    "general": {
        "loop_interval": 300,
        "risk_ceiling": 0.40,
        "approach_max_nav_pct": 0.03,
        "sentinel_max_nav_pct": 0.03,
        "sniper_max_nav_pct": 0.40,
        "base_currency": "CHF",
        "account_id": "",
        "ib_host": "127.0.0.1",
        "ib_port": 4001,
        "ib_client_id": 1,
    },
    "execution": {
        "patient": {
            "fill_timeout_secs": 60,
            "max_retries": 29,
            "mode": "IB_MODEL",
            "fallback_mode": "MID",
            "fallback_after": 10,
            "last_resort_mode": "",
        },
        "urgent": {
            "fill_timeout_secs": 60,
            "max_retries": 7,
            "mode": "IB_MODEL",
            "fallback_mode": "MID",
            "fallback_after": 3,
            "last_resort_mode": "NATURAL",
        },
    },
    "thesis": {
        "exit": {
            "stop_loss_pct": -0.45,
            "full_exit_pct": 1.50,
            "trailing_stop_pct": 0.30,
            "dte_floor": 5,
            "tranches": [[0.50, 0.25], [1.00, 0.35]],
            "spike_pct": 1.00,
            "spike_window_hours": 6.0,
            "spike_sell_ratio": 0.50,
        },
        "contract": {
            "delta_min": 0.40,
            "delta_max": 0.55,
            "dte_min": 25,
            "dte_max": 50,
            "strike_width": 0.25,
        },
    },
    "approach": {
        "exit": {
            "stop_loss_pct": -0.35,
            "full_exit_pct": 0.60,
            "trailing_stop_pct": 0.20,
            "dte_floor": 8,
            "tranches": [[0.15, 0.30], [0.30, 0.40]],
            "spike_pct": 0.80,
            "spike_window_hours": 4.0,
            "spike_sell_ratio": 0.60,
        },
        "contract": {
            "delta_min": 0.30,
            "delta_max": 0.45,
            "dte_min": 25,
            "dte_max": 45,
            "strike_width": 0.20,
        },
    },
    "sentinel": {
        "exit": {
            "stop_loss_pct": -0.50,
            "full_exit_pct": 1.00,
            "trailing_stop_pct": 0.25,
            "dte_floor": 15,
            "tranches": [[0.30, 0.30], [0.60, 0.40]],
            "spike_pct": 1.00,
            "spike_window_hours": 6.0,
            "spike_sell_ratio": 0.50,
        },
        "contract": {
            "delta_min": 0.15,
            "delta_max": 0.30,
            "dte_min": 60,
            "dte_max": 120,
            "strike_width": 0.30,
            "max_spread_pct": 25.0,
            "min_open_interest": 10,
            "min_volume": 1,
        },
    },
    "sniper": {
        "exit": {
            "stop_loss_pct": -0.50,
            "full_exit_pct": 1.00,
            "max_hold_days": 7,
            "dte_floor": 1,
        },
        "contract": {
            "delta_min": 0.35,
            "delta_max": 0.55,
            "dte_min": 2,
            "dte_max": 14,
            "strike_width": 0.20,
        },
        "scanner": {
            "watchlist": ["BNTX", "NVAX", "MRNA", "REGN"],
            "drop_pct": 0.15,
        },
    },
}

_DEFAULT_CONFIG_TEXT = """# ─────────────────────────────────────────────────────────────────────────────
# config.toml — Single source of truth for every tunable runtime parameter.
#
# Runtime config search order:
#   1. OPTION_ALGORITHM_CONFIG
#   2. ./config.toml
#   3. ~/.config/option_algorithm/config.toml
#
# If a value or section is omitted, config.py fills it from built-in defaults.
# ─────────────────────────────────────────────────────────────────────────────

[general]
loop_interval        = 300      # seconds between strategy ticks
risk_ceiling         = 0.40     # max options notional / NAV
approach_max_nav_pct = 0.03     # hard cap for APPROACH sizing
sentinel_max_nav_pct = 0.03     # hard cap for SENTINEL sizing
sniper_max_nav_pct   = 0.40     # hard cap for SNIPER sizing (% of NAV)
base_currency        = "CHF"    # account base currency for IB value fetching
account_id           = ""       # optional IB account code; blank = auto-select
ib_host              = "127.0.0.1"
ib_port              = 4001
ib_client_id         = 1


# ── EXECUTION ─────────────────────────────────────────────────────────────────
# Retry and pricing behaviour for order placement.
#
# Two profiles: "patient" (entries, profit-taking, DTE floor, max hold)
#               "urgent"  (stop loss, trailing stop)
#
# Mode ladder: start at `mode`, switch to `fallback_mode` after
# `fallback_after` attempts, optionally use `last_resort_mode` on the
# final attempt only (set to "" to disable). Set fallback_mode="" to disable
# fallback-mode switching entirely.

[execution.patient]
fill_timeout_secs = 60        # seconds per attempt
max_retries       = 29        # total attempts = 30  (~30 min)
mode              = "IB_MODEL"
fallback_mode     = "MID"
fallback_after    = 10        # first 10 attempts IB_MODEL, rest MID
last_resort_mode  = ""        # no NATURAL ever

[execution.urgent]
fill_timeout_secs = 60        # seconds per attempt
max_retries       = 7         # total attempts = 8   (~8 min)
mode              = "IB_MODEL"
fallback_mode     = "MID"
fallback_after    = 3         # first 3 attempts IB_MODEL, rest MID
last_resort_mode  = "NATURAL" # NATURAL on final attempt only (stop loss)


# ── THESIS ──────────────────────────────────────────────────────────────────

[thesis.exit]
stop_loss_pct      = -0.45
full_exit_pct      =  1.50
trailing_stop_pct  =  0.30
dte_floor          =  5
tranches           = [[0.50, 0.25], [1.00, 0.35]]
# velocity spike
spike_pct          =  1.00
spike_window_hours =  6.0
spike_sell_ratio   =  0.50

[thesis.contract]
delta_min    = 0.40
delta_max    = 0.55
dte_min      = 25
dte_max      = 50
strike_width = 0.25


# ── APPROACH ────────────────────────────────────────────────────────────────

[approach.exit]
stop_loss_pct      = -0.35
full_exit_pct      =  0.60
trailing_stop_pct  =  0.20
dte_floor          =  8
tranches           = [[0.15, 0.30], [0.30, 0.40]]
# velocity spike
spike_pct          =  0.80
spike_window_hours =  4.0
spike_sell_ratio   =  0.60

[approach.contract]
delta_min    = 0.30
delta_max    = 0.45
dte_min      = 25
dte_max      = 45
strike_width = 0.20


# ── SENTINEL ────────────────────────────────────────────────────────────────

[sentinel.exit]
stop_loss_pct      = -0.50
full_exit_pct      =  1.00
trailing_stop_pct  =  0.25
dte_floor          =  15
tranches           = [[0.30, 0.30], [0.60, 0.40]]
# velocity spike
spike_pct          =  1.00
spike_window_hours =  6.0
spike_sell_ratio   =  0.50

[sentinel.contract]
delta_min         = 0.15
delta_max         = 0.30
dte_min           = 60
dte_max           = 120
strike_width      = 0.30
max_spread_pct    = 25.0      # relaxed for illiquid biotech
min_open_interest = 10
min_volume        = 1


# ── SNIPER ──────────────────────────────────────────────────────────────────

[sniper.exit]
stop_loss_pct = -0.50
full_exit_pct =  1.00
max_hold_days =  7
dte_floor     =  1

[sniper.contract]
delta_min    = 0.35
delta_max    = 0.55
dte_min      = 2
dte_max      = 14
strike_width = 0.20

[sniper.scanner]
watchlist = ["BNTX", "NVAX", "MRNA", "REGN"]
drop_pct  = 0.15
"""


@dataclass
class Config:
    """Typed, read-only view of config.toml."""

    # general
    loop_interval:        int
    risk_ceiling:         float
    approach_max_nav_pct: float
    sentinel_max_nav_pct: float
    sniper_max_nav_pct:   float
    base_currency:        str
    account_id:           str
    ib_host:              str
    ib_port:              int
    ib_client_id:         int

    # per-play-type (keyed by "THESIS", "APPROACH", etc.)
    exit_profiles:  dict[str, ExitProfile]
    contract_specs: dict[str, ContractSpec]

    # execution
    patient: RetryProfile
    urgent:  RetryProfile

    # sniper scanner
    sniper_watchlist: list[str]
    sniper_drop_pct:  float

    # provenance
    path: Path


def _deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_price_mode(raw: Any) -> PriceMode | None:
    if raw in (None, ""):
        return None
    return PriceMode(str(raw))


def _bootstrap_default_config(path: Path) -> Path:
    ensure_parent(path)
    if not path.exists():
        path.write_text(_DEFAULT_CONFIG_TEXT)
        print(f"[CONFIG] Bootstrapped default config at {path}")
    return path


def resolve_path(path: Path | None = None) -> Path:
    """Resolve the runtime config path, bootstrapping one if needed."""
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")
        return resolved

    configured = configured_config_path()
    if configured is not None:
        if not configured.exists():
            raise FileNotFoundError(f"OPTION_ALGORITHM_CONFIG does not exist: {configured}")
        return configured

    cwd_path = Path.cwd() / "config.toml"
    if cwd_path.exists():
        return cwd_path.resolve()

    user_path = default_config_path()
    if user_path.exists():
        return user_path

    if _LEGACY_CFG_PATH.exists():
        return _LEGACY_CFG_PATH

    return _bootstrap_default_config(user_path)


def load(path: Path | None = None) -> Config:
    """Parse config.toml and return a fully resolved Config."""
    resolved = resolve_path(path)
    with open(resolved, "rb") as f:
        raw = tomllib.load(f)

    raw = _deep_merge(_DEFAULT_RAW, raw)
    gen = raw.get("general", {})

    exit_profiles: dict[str, ExitProfile] = {}
    contract_specs: dict[str, ContractSpec] = {}

    for play_type in ("thesis", "approach", "sentinel", "sniper"):
        section = raw.get(play_type, {})
        key = play_type.upper()

        exit_kw = dict(section.get("exit", {}))
        if "tranches" in exit_kw:
            exit_kw["tranches"] = [tuple(t) for t in exit_kw["tranches"]]
        exit_profiles[key] = ExitProfile(**exit_kw)

        contract_specs[key] = ContractSpec(**section.get("contract", {}))

    scanner = raw.get("sniper", {}).get("scanner", {})

    def _retry_profile(section: dict[str, Any]) -> RetryProfile:
        return RetryProfile(
            fill_timeout_secs=int(section["fill_timeout_secs"]),
            max_retries=int(section["max_retries"]),
            mode=PriceMode(section["mode"]),
            fallback_mode=_parse_price_mode(section.get("fallback_mode")),
            fallback_after=(
                int(section["fallback_after"])
                if section.get("fallback_after") not in (None, "")
                else None
            ),
            last_resort_mode=_parse_price_mode(section.get("last_resort_mode")),
        )

    exec_raw = raw.get("execution", {})

    return Config(
        loop_interval=int(gen.get("loop_interval", 300)),
        risk_ceiling=float(gen.get("risk_ceiling", 0.40)),
        approach_max_nav_pct=float(gen.get("approach_max_nav_pct", 0.03)),
        sentinel_max_nav_pct=float(gen.get("sentinel_max_nav_pct", 0.03)),
        sniper_max_nav_pct=float(gen.get("sniper_max_nav_pct", 0.05)),
        base_currency=str(gen.get("base_currency", "CHF")),
        account_id=str(gen.get("account_id", "")),
        ib_host=str(gen.get("ib_host", "127.0.0.1")),
        ib_port=int(gen.get("ib_port", 4001)),
        ib_client_id=int(gen.get("ib_client_id", 1)),
        exit_profiles=exit_profiles,
        contract_specs=contract_specs,
        patient=_retry_profile(exec_raw.get("patient", {})),
        urgent=_retry_profile(exec_raw.get("urgent", {})),
        sniper_watchlist=list(scanner.get("watchlist", ["BNTX", "NVAX", "MRNA", "REGN"])),
        sniper_drop_pct=float(scanner.get("drop_pct", 0.15)),
        path=resolved,
    )
