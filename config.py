"""
config.py
─────────────────────────────────────────────────────────────────────────────
Load config.toml and produce typed objects for the strategy.

config.toml is the single source of truth for all parameter values.
ExitProfile and ContractSpec in strategy.py define the structure;
this module reads concrete values from TOML and constructs the objects.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # pip install tomli for Python 3.10

from execution import PriceMode, RetryProfile
from strategy import ContractSpec, ExitProfile

_CFG_PATH = Path(__file__).resolve().parent / "config.toml"


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

    # per-play-type (keyed by "THESIS", "APPROACH", etc.)
    exit_profiles:  dict[str, ExitProfile]
    contract_specs: dict[str, ContractSpec]

    # execution
    patient: RetryProfile
    urgent:  RetryProfile

    # sniper scanner
    sniper_watchlist: list[str]
    sniper_drop_pct:  float


def load(path: Path = _CFG_PATH) -> Config:
    """Parse config.toml and return a fully resolved Config."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    gen = raw.get("general", {})

    exit_profiles:  dict[str, ExitProfile]  = {}
    contract_specs: dict[str, ContractSpec]  = {}

    for play_type in ("thesis", "approach", "sentinel", "sniper"):
        section = raw.get(play_type, {})
        key = play_type.upper()

        exit_kw = dict(section.get("exit", {}))
        if "tranches" in exit_kw:
            exit_kw["tranches"] = [tuple(t) for t in exit_kw["tranches"]]
        exit_profiles[key] = ExitProfile(**exit_kw)

        contract_specs[key] = ContractSpec(**section.get("contract", {}))

    scanner = raw.get("sniper", {}).get("scanner", {})

    def _retry_profile(section: dict, defaults: dict) -> RetryProfile:
        merged = {**defaults, **section}
        lr = merged.get("last_resort_mode", "")
        return RetryProfile(
            fill_timeout_secs = merged["fill_timeout_secs"],
            max_retries       = merged["max_retries"],
            mode              = PriceMode(merged["mode"]),
            fallback_mode     = PriceMode(merged["fallback_mode"]),
            fallback_after    = merged["fallback_after"],
            last_resort_mode  = PriceMode(lr) if lr else None,
        )

    exec_raw = raw.get("execution", {})
    patient_defaults = dict(
        fill_timeout_secs=60, max_retries=29, mode="IB_MODEL",
        fallback_mode="MID", fallback_after=10, last_resort_mode="",
    )
    urgent_defaults = dict(
        fill_timeout_secs=60, max_retries=7, mode="IB_MODEL",
        fallback_mode="MID", fallback_after=3, last_resort_mode="NATURAL",
    )

    return Config(
        loop_interval        = gen.get("loop_interval", 300),
        risk_ceiling         = gen.get("risk_ceiling", 0.40),
        approach_max_nav_pct = gen.get("approach_max_nav_pct", 0.03),
        sentinel_max_nav_pct = gen.get("sentinel_max_nav_pct", 0.03),
        sniper_max_nav_pct   = gen.get("sniper_max_nav_pct", 0.05),
        base_currency        = gen.get("base_currency", "CHF"),
        account_id           = gen.get("account_id", ""),
        exit_profiles        = exit_profiles,
        contract_specs       = contract_specs,
        patient              = _retry_profile(exec_raw.get("patient", {}), patient_defaults),
        urgent               = _retry_profile(exec_raw.get("urgent", {}), urgent_defaults),
        sniper_watchlist     = scanner.get("watchlist", ["BNTX", "NVAX", "MRNA", "REGN"]),
        sniper_drop_pct      = scanner.get("drop_pct", 0.15),
    )
