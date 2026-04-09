"""
state.py
─────────────────────────────────────────────────────────────────────────────
Minimal persistence for strategy.plays.

One JSON file (plays.json) written after every mutation.
On startup: load plays, reconcile against live IB positions, warn about orphans.

Old plays.json files with legacy fields (partial_trigger_pct, fast_move_*)
are loaded safely — unknown keys are simply ignored.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    import pandas as pd
    from strategy import Play

STATE_FILE = Path(__file__).resolve().parent / "plays.json"

_PLAY_TYPE_MIGRATION = {
    "RECOVERY": "THESIS",
    "CATALYST": "THESIS",
    "BASELINE": "APPROACH",
}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC
# ─────────────────────────────────────────────────────────────────────────────

def save(plays: list[Play], account_id: str | None = None) -> None:
    merged = list(plays)
    if account_id:
        existing = _read_all_plays()
        merged.extend(
            p for p in existing
            if p.account_id and p.account_id != account_id
        )
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps([_to_dict(p) for p in merged], indent=2))
    os.replace(tmp, STATE_FILE)


def read_raw(account_id: str | None = None) -> list[dict]:
    """Return the raw JSON dicts from plays.json (no deserialization)."""
    if not STATE_FILE.exists():
        return []
    raw = json.loads(STATE_FILE.read_text())
    if not account_id:
        return raw
    return [
        d for d in raw
        if not d.get("account_id") or d.get("account_id") == account_id
    ]


def load(
    ib_positions: "pd.DataFrame | None" = None,
    account_id: str | None = None,
) -> "list[Play]":
    plays = _read_all_plays()
    if account_id:
        migrated = False
        for play in plays:
            if not play.account_id:
                play.account_id = account_id
                migrated = True
        plays = [p for p in plays if p.account_id == account_id]
        if migrated and plays:
            save(plays, account_id=account_id)
    if ib_positions is not None:
        _reconcile(plays, ib_positions)
        _adopt_orphans(plays, ib_positions)
        if plays:
            save(plays, account_id=account_id)
    open_count    = sum(1 for p in plays if p.status.value in ("OPEN", "SCALING"))
    pending_count = sum(1 for p in plays if p.status.value == "PENDING")
    parts = [f"{open_count} active"]
    if pending_count:
        parts.append(f"{pending_count} pending")
    scope = f" account={account_id}" if account_id else ""
    print(f"[STATE]{scope} {len(plays)} plays ({', '.join(parts)})")
    return plays


# ─────────────────────────────────────────────────────────────────────────────
# SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _to_dict(play: Play) -> dict:
    ep = play.exit_profile
    return {
        "play_id":      play.play_id,
        "account_id":   play.account_id,
        "play_type":    play.play_type.value,
        "symbol":       play.symbol,
        "con_id":       play.con_id,
        "qty_initial":  play.qty_initial,
        "qty_open":     play.qty_open,
        "entry_time":   play.entry_time.isoformat(),
        "entry_price":  play.entry_price,
        "entry_nav":    play.entry_nav,
        "entry_time_known": play.entry_time_known,
        "status":       play.status.value,
        "peak_pnl_pct": play.peak_pnl_pct,
        "tranche_idx":  play.tranche_idx,
        "spike_fired":  play.spike_fired,
        "pnl_history":  [[t.isoformat(), p] for t, p in play.pnl_history],
        "exit_profile": {
            "stop_loss_pct":        ep.stop_loss_pct,
            "full_exit_pct":        ep.full_exit_pct,
            "dte_floor":            ep.dte_floor,
            "trailing_stop_pct":    ep.trailing_stop_pct,
            "tranches":             ep.tranches,
            "max_hold_days":        ep.max_hold_days,
            "spike_pct":            ep.spike_pct,
            "spike_window_hours":   ep.spike_window_hours,
            "spike_sell_ratio":     ep.spike_sell_ratio,
        },
    }


def _from_dict(d: dict) -> Play:
    from strategy import ExitProfile, Play, PlayStatus, PlayType

    raw_type = d["play_type"]
    if raw_type in _PLAY_TYPE_MIGRATION:
        migrated = _PLAY_TYPE_MIGRATION[raw_type]
        print(f"[STATE] Migrating {raw_type} → {migrated} ({d.get('symbol', '?')})")
        raw_type = migrated

    ep_d = d["exit_profile"]
    exit_profile = ExitProfile(
        stop_loss_pct      = ep_d["stop_loss_pct"],
        full_exit_pct      = ep_d["full_exit_pct"],
        dte_floor          = ep_d["dte_floor"],
        trailing_stop_pct  = ep_d.get("trailing_stop_pct"),
        tranches           = ep_d.get("tranches", []),
        max_hold_days      = ep_d.get("max_hold_days", 0),
        spike_pct          = ep_d.get("spike_pct", 0.0),
        spike_window_hours = ep_d.get("spike_window_hours", 0.0),
        spike_sell_ratio   = ep_d.get("spike_sell_ratio", 0.0),
    )

    return Play(
        play_id      = d.get("play_id", uuid4().hex[:12]),
        account_id   = d.get("account_id", ""),
        play_type    = PlayType(raw_type),
        symbol       = d["symbol"],
        con_id       = d["con_id"],
        qty_initial  = d["qty_initial"],
        qty_open     = d["qty_open"],
        entry_time   = datetime.fromisoformat(d["entry_time"]),
        entry_price  = d["entry_price"],
        entry_nav    = d["entry_nav"],
        entry_time_known = d.get("entry_time_known", True),
        exit_profile = exit_profile,
        status       = PlayStatus(d["status"]),
        peak_pnl_pct = d.get("peak_pnl_pct", 0.0),
        tranche_idx  = d.get("tranche_idx", 0),
        spike_fired  = d.get("spike_fired", False),
        pnl_history  = [
            (datetime.fromisoformat(t), p)
            for t, p in d.get("pnl_history", [])
        ],
    )


def _read_all_plays() -> list[Play]:
    if not STATE_FILE.exists():
        return []
    raw = json.loads(STATE_FILE.read_text())
    return [_from_dict(d) for d in raw]


# ─────────────────────────────────────────────────────────────────────────────
# RECONCILIATION
# ─────────────────────────────────────────────────────────────────────────────

def _reconcile(plays: list, ib_positions: "pd.DataFrame") -> None:
    """Startup-only reconciliation of OPEN/SCALING plays against IB positions.
    PENDING promotion is handled by Strategy._monitor_plays() on first tick."""
    from strategy import PlayStatus

    ib_qty: dict[int, int] = {}
    if not ib_positions.empty and "con_id" in ib_positions.columns:
        for _, row in ib_positions.iterrows():
            cid = int(row["con_id"])
            ib_qty[cid] = int(abs(float(row.get("position", 0))))

    for play in plays:
        if play.status not in (PlayStatus.OPEN, PlayStatus.SCALING):
            continue

        live = ib_qty.get(play.con_id, 0)
        if live == 0:
            play.status = PlayStatus.CLOSED
            print(f"[STATE] {play.symbol} con_id={play.con_id} gone → CLOSED")
        elif live != play.qty_open:
            print(
                f"[STATE] {play.symbol} qty {play.qty_open}→{live} (reconcile)"
            )
            play.qty_open = live
            play.status   = (
                PlayStatus.OPEN if live >= play.qty_initial else PlayStatus.SCALING
            )


def _adopt_orphans(plays: list, ib_positions: "pd.DataFrame") -> None:
    """Log untracked IB positions but do NOT auto-create plays.
    Use 'track CON_ID TYPE [SYM]' to track them explicitly."""
    from strategy import PlayStatus

    if ib_positions is None or ib_positions.empty:
        return
    tracked = {
        p.con_id for p in plays
        if p.status in (PlayStatus.OPEN, PlayStatus.SCALING, PlayStatus.PENDING)
    }
    opt_rows = (
        ib_positions[ib_positions.get("sec_type", "") == "OPT"]
        if "sec_type" in ib_positions.columns
        else ib_positions
    )
    for _, row in opt_rows.iterrows():
        con_id = int(row.get("con_id", 0))
        if not con_id or con_id in tracked:
            continue
        qty    = int(abs(float(row.get("position", 0))))
        symbol = str(row.get("symbol", "UNKNOWN")).upper()
        if qty < 1:
            continue
        print(
            f"[STATE] Untracked position: {symbol} con_id={con_id} qty={qty}. "
            f"Use 'track {con_id} thesis {symbol}' to track."
        )
