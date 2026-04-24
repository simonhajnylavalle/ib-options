"""
state.py
─────────────────────────────────────────────────────────────────────────────
Minimal persistence for strategy.plays.

Default storage is user-writable (~/.local/state/option_algorithm/plays.json on
Linux/XDG). Legacy module-adjacent plays.json files are still read for
migration, but new saves go to the user state path unless explicitly
overridden.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from paths import configured_state_path, default_state_path, ensure_parent

if TYPE_CHECKING:
    import pandas as pd
    from strategy import Play

_LEGACY_STATE_FILE = Path(__file__).resolve().parent / "plays.json"

_PLAY_TYPE_MIGRATION = {
    "RECOVERY": "THESIS",
    "CATALYST": "THESIS",
    "BASELINE": "APPROACH",
}


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

def state_file() -> Path:
    configured = configured_state_path()
    if configured is not None:
        return configured
    return default_state_path()


def _read_path() -> Path:
    current = state_file()
    if current.exists():
        return current
    if current == default_state_path() and _LEGACY_STATE_FILE.exists():
        return _LEGACY_STATE_FILE
    return current


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC
# ─────────────────────────────────────────────────────────────────────────────

def save(plays: list[Play], account_id: str | None = None) -> None:
    target = state_file()
    merged = list(plays)
    if account_id:
        existing = _read_all_plays()
        merged.extend(
            p for p in existing
            if p.account_id and p.account_id != account_id
        )
    ensure_parent(target)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps([_to_dict(p) for p in merged], indent=2))
    os.replace(tmp, target)


def read_raw(account_id: str | None = None) -> list[dict]:
    """Return the raw JSON dicts from plays.json (no deserialization)."""
    path = _read_path()
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
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
    open_count = sum(1 for p in plays if p.status.value in ("OPEN", "SCALING"))
    pending_count = sum(1 for p in plays if p.status.value == "PENDING")
    parts = [f"{open_count} active"]
    if pending_count:
        parts.append(f"{pending_count} pending")
    scope = f" account={account_id}" if account_id else ""
    print(f"[STATE]{scope} {len(plays)} plays ({', '.join(parts)})  path={state_file()}")
    return plays


# ─────────────────────────────────────────────────────────────────────────────
# SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _tracker_to_dict(tracker) -> dict | None:
    if tracker is None:
        return None
    return {
        "remaining_qty": tracker.remaining_qty,
        "attempts_used": tracker.attempts_used,
        "submitted_at": tracker.submitted_at.isoformat(),
        "retry_kind": tracker.retry_kind,
        "reason": tracker.reason,
        "status": getattr(tracker, "status", "WORKING"),
        "accounted_fills": tracker.accounted_fills,
        "cancel_requested": tracker.cancel_requested,
        "order_id": getattr(tracker, "order_id", None),
        "perm_id": getattr(tracker, "perm_id", None),
        "native_order_id": getattr(tracker, "native_order_id", None),
        "client_id": getattr(tracker, "client_id", None),
        "account_id": getattr(tracker, "account_id", None),
        "side": getattr(tracker, "side", None),
        "limit_px": getattr(tracker, "limit_px", None),
        "submitted_qty": getattr(tracker, "submitted_qty", None),
    }


def _working_order_to_dict(play: Play) -> dict | None:
    wo = play.working_order
    data = _tracker_to_dict(wo)
    if data is None:
        return None
    data.update({
        "reserved_tranche_idx": wo.reserved_tranche_idx,
        "reserve_spike_fired": wo.reserve_spike_fired,
    })
    return data


def _working_entry_to_dict(play: Play) -> dict | None:
    we = getattr(play, "working_entry", None)
    data = _tracker_to_dict(we)
    if data is None:
        return None
    data["requested_qty"] = we.requested_qty
    return data


def _to_dict(play: Play) -> dict:
    ep = play.exit_profile
    return {
        "play_id": play.play_id,
        "account_id": play.account_id,
        "play_type": play.play_type.value,
        "symbol": play.symbol,
        "con_id": play.con_id,
        "qty_initial": play.qty_initial,
        "qty_open": play.qty_open,
        "entry_time": play.entry_time.isoformat(),
        "entry_price": play.entry_price,
        "entry_nav": play.entry_nav,
        "entry_time_known": play.entry_time_known,
        "status": play.status.value,
        "peak_pnl_pct": play.peak_pnl_pct,
        "tranche_idx": play.tranche_idx,
        "spike_fired": play.spike_fired,
        "pnl_history": [[t.isoformat(), p] for t, p in play.pnl_history],
        "working_order": _working_order_to_dict(play),
        "working_entry": _working_entry_to_dict(play),
        "exit_profile": {
            "stop_loss_pct": ep.stop_loss_pct,
            "full_exit_pct": ep.full_exit_pct,
            "dte_floor": ep.dte_floor,
            "trailing_stop_pct": ep.trailing_stop_pct,
            "tranches": ep.tranches,
            "max_hold_days": ep.max_hold_days,
            "spike_pct": ep.spike_pct,
            "spike_window_hours": ep.spike_window_hours,
            "spike_sell_ratio": ep.spike_sell_ratio,
        },
    }


def _from_dict(d: dict) -> Play:
    from strategy import ExitProfile, Play, PlayStatus, PlayType, WorkingEntry, WorkingOrder

    raw_type = d["play_type"]
    if raw_type in _PLAY_TYPE_MIGRATION:
        migrated = _PLAY_TYPE_MIGRATION[raw_type]
        print(f"[STATE] Migrating {raw_type} → {migrated} ({d.get('symbol', '?')})")
        raw_type = migrated

    ep_d = d["exit_profile"]
    exit_profile = ExitProfile(
        stop_loss_pct=ep_d["stop_loss_pct"],
        full_exit_pct=ep_d["full_exit_pct"],
        dte_floor=ep_d["dte_floor"],
        trailing_stop_pct=ep_d.get("trailing_stop_pct"),
        tranches=ep_d.get("tranches", []),
        max_hold_days=ep_d.get("max_hold_days", 0),
        spike_pct=ep_d.get("spike_pct", 0.0),
        spike_window_hours=ep_d.get("spike_window_hours", 0.0),
        spike_sell_ratio=ep_d.get("spike_sell_ratio", 0.0),
    )

    def _common_tracker_kwargs(raw: dict) -> dict:
        return {
            "trade_result": None,
            "remaining_qty": int(raw["remaining_qty"]),
            "attempts_used": int(raw["attempts_used"]),
            "submitted_at": datetime.fromisoformat(raw["submitted_at"]),
            "retry_kind": str(raw.get("retry_kind", "patient")),
            "reason": str(raw.get("reason", "")),
            "status": str(raw.get("status", "UNBOUND")),
            "accounted_fills": int(raw.get("accounted_fills", 0)),
            "cancel_requested": bool(raw.get("cancel_requested", False)),
            "order_id": raw.get("order_id"),
            "perm_id": raw.get("perm_id"),
            "native_order_id": raw.get("native_order_id"),
            "client_id": raw.get("client_id"),
            "account_id": raw.get("account_id"),
            "side": str(raw.get("side") or ""),
            "limit_px": raw.get("limit_px"),
            "submitted_qty": raw.get("submitted_qty"),
        }

    wo_d = d.get("working_order")
    working_order = None
    if wo_d:
        kwargs = _common_tracker_kwargs(wo_d)
        kwargs["side"] = kwargs["side"] or "SELL"
        working_order = WorkingOrder(
            **kwargs,
            reserved_tranche_idx=wo_d.get("reserved_tranche_idx"),
            reserve_spike_fired=bool(wo_d.get("reserve_spike_fired", False)),
        )

    we_d = d.get("working_entry")
    working_entry = None
    if we_d:
        kwargs = _common_tracker_kwargs(we_d)
        kwargs["side"] = kwargs["side"] or "BUY"
        working_entry = WorkingEntry(
            **kwargs,
            requested_qty=int(we_d.get("requested_qty", we_d.get("remaining_qty", 0))),
        )

    return Play(
        play_id=d.get("play_id", uuid4().hex[:12]),
        account_id=d.get("account_id", ""),
        play_type=PlayType(raw_type),
        symbol=d["symbol"],
        con_id=d["con_id"],
        qty_initial=d["qty_initial"],
        qty_open=d["qty_open"],
        entry_time=datetime.fromisoformat(d["entry_time"]),
        entry_price=d["entry_price"],
        entry_nav=d["entry_nav"],
        entry_time_known=d.get("entry_time_known", True),
        exit_profile=exit_profile,
        status=PlayStatus(d["status"]),
        peak_pnl_pct=d.get("peak_pnl_pct", 0.0),
        tranche_idx=d.get("tranche_idx", 0),
        spike_fired=d.get("spike_fired", False),
        pnl_history=[
            (datetime.fromisoformat(t), p)
            for t, p in d.get("pnl_history", [])
        ],
        working_order=working_order,
        working_entry=working_entry,
    )


def _read_all_plays() -> list[Play]:
    path = _read_path()
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [_from_dict(d) for d in raw]


# ─────────────────────────────────────────────────────────────────────────────
# RECONCILIATION
# ─────────────────────────────────────────────────────────────────────────────

def _reconcile(plays: list, ib_positions: "pd.DataFrame") -> None:
    """Startup-only reconciliation of active plays against IB positions.

    We intentionally do not close active plays from a single empty IB positions
    snapshot.  IB can briefly return empty/stale portfolio data during startup;
    closing persisted state on that one view is more dangerous than deferring.
    """
    from strategy import PlayStatus

    active = [
        p for p in plays
        if p.status in (PlayStatus.OPEN, PlayStatus.SCALING)
    ]
    if not active:
        return
    if ib_positions is None or ib_positions.empty or "con_id" not in ib_positions.columns:
        print(
            "[STATE] ⚠ IB positions snapshot is empty/incomplete; "
            "skipping startup reconciliation to avoid closing active plays on stale data"
        )
        return

    ib_qty: dict[int, int] = {}
    for _, row in ib_positions.iterrows():
        cid = int(row.get("con_id", 0) or 0)
        if not cid:
            continue
        raw_qty = float(row.get("position", 0) or 0)
        # This app is long-call-only. Negative positions must not be converted
        # into a positive long quantity by abs().
        ib_qty[cid] = max(0, int(raw_qty))

    for play in active:
        live = ib_qty.get(play.con_id, 0)
        if live == 0:
            play.qty_open = 0
            play.status = PlayStatus.CLOSED
            play.working_order = None
            play.working_entry = None
            print(f"[STATE] {play.symbol} con_id={play.con_id} gone/unsupported → CLOSED")
        elif live != play.qty_open:
            print(f"[STATE] {play.symbol} qty {play.qty_open}→{live} (reconcile)")
            play.qty_open = live
            play.status = (
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
        con_id = int(row.get("con_id", 0) or 0)
        if not con_id or con_id in tracked:
            continue
        raw_qty = float(row.get("position", 0) or 0)
        symbol = str(row.get("symbol", "UNKNOWN")).upper()
        if raw_qty == 0:
            continue
        if raw_qty < 0:
            print(
                f"[STATE] Unsupported short option position: {symbol} con_id={con_id} "
                f"qty={raw_qty:g} — not auto-tracked; manage manually"
            )
            continue
        if str(row.get("right", "")).upper() != "C":
            print(
                f"[STATE] Unsupported non-CALL option position: {symbol} con_id={con_id} "
                "— not auto-tracked"
            )
            continue
        qty = int(raw_qty)
        if qty < 1:
            continue
        print(
            f"[STATE] Orphan CALL option position: {symbol} con_id={con_id} qty={qty} "
            f"— use 'track {con_id} thesis|approach|sentinel|sniper {symbol}' if intentional"
        )
