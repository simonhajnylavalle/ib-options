"""
main.py — Entry point and interactive console.
─────────────────────────────────────────────────────────────────────────────

ALL tunable parameters live in config.toml.
Nothing in strategy.py or portfolio.py needs editing during normal use.

Strategy loop ticks on interval in the main thread.
Console input is read from a daemon thread; commands dispatch on main.
"""

from __future__ import annotations

import queue
import shlex
import sys
import threading
import time
from dataclasses import replace
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
import state
from execution import OrderSide
from ib_core   import OptionChain, Right, connect
from portfolio import CashPolicy
from strategy  import (
    ConvictionLevel, PlayStatus, PlayType,
    SniperScanner, Strategy,
)


# ── active config instance (loaded from config.toml) ──────────────────────
CFG = config.load()


# ═════════════════════════════════════════════════════════════════════════════
# LOOKUPS
# ═════════════════════════════════════════════════════════════════════════════

_CONV = {
    "low": ConvictionLevel.LOW,    "l": ConvictionLevel.LOW,
    "med": ConvictionLevel.MEDIUM, "medium": ConvictionLevel.MEDIUM,
    "m": ConvictionLevel.MEDIUM,
    "high": ConvictionLevel.HIGH,  "h": ConvictionLevel.HIGH,
}

_PTYPE = {
    "thesis":   PlayType.THESIS,   "t": PlayType.THESIS,
    "approach": PlayType.APPROACH, "a": PlayType.APPROACH,
    "sentinel": PlayType.SENTINEL,
    "sniper":   PlayType.SNIPER,
}


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_con = Console()

def _d(v: float)  -> str: return f"${v:,.2f}"
def _ds(v: float) -> str: return f"${v:+,.2f}"

def _kvtable(*rows: tuple[str, str], title: str | None = None) -> Table:
    """Two-column key–value table (compact, no header row)."""
    t = Table(show_header=False, box=None, padding=(0, 1),
              title=title, title_style="bold")
    t.add_column(style="dim", no_wrap=True)
    t.add_column(no_wrap=True)
    for k, v in rows:
        t.add_row(k, v)
    return t


def _err(message: str, detail: str | None = None) -> None:
    if detail:
        _con.print(f"  [bold red]Error:[/] {message}\n  [dim]{detail}[/]")
    else:
        _con.print(f"  [bold red]Error:[/] {message}")


def _pop_flag(args: list[str], *names: str) -> tuple[list[str], bool]:
    flags = {n.lower() for n in names}
    out: list[str] = []
    seen = False
    for a in args:
        if a.lower() in flags:
            seen = True
        else:
            out.append(a)
    return out, seen


def _yes(args: list[str]) -> tuple[list[str], bool]:
    return _pop_flag(args, "--yes", "-y")


def _retry_desc(profile) -> str:
    total = profile.max_retries + 1
    fallback = profile.fallback_mode.value if profile.fallback_mode else "─"
    last = profile.last_resort_mode.value if profile.last_resort_mode else "─"
    after = profile.fallback_after if profile.fallback_after is not None else "─"
    return (
        f"{profile.mode.value}→{fallback} after {after}; "
        f"last={last}; {total}×{profile.fill_timeout_secs}s"
    )


def _ticket(title: str, *rows: tuple[str, str], subtitle: str | None = None) -> None:
    table = _kvtable(*rows)
    _con.print()
    _con.print(Panel(table, title=title, subtitle=subtitle or "",
                     border_style="yellow", padding=(1, 2)))


def _confirm_order_or_preview(title: str, rows: list[tuple[str, str]], yes: bool) -> bool:
    if not CFG.terminal.confirm_orders or yes:
        return True
    _ticket(
        title,
        *rows,
        ("Confirmation", "re-run the command with --yes to submit"),
        ("Fidelity", "execution refreshes quote/order state at submission time"),
        subtitle="[yellow]No order submitted[/]",
    )
    return False


def _confirm_cancel_all_or_preview(rows: list[tuple[str, str]], yes: bool) -> bool:
    if not CFG.terminal.confirm_cancel_all or yes:
        return True
    _ticket(
        "Cancel-All Preview",
        *rows,
        ("Confirmation", "re-run cancel-all with --yes to send reqGlobalCancel"),
        subtitle="[yellow]No cancel request sent[/]",
    )
    return False


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value != value:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _row_value(row, key: str, default: str = "─") -> str:
    try:
        value = row.get(key, None)
    except Exception:
        value = None
    if value is None or (isinstance(value, float) and value != value):
        return default
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _money_or_dash(value) -> str:
    if value is None:
        return "─"
    try:
        if value != value:
            return "─"
        return _d(float(value))
    except Exception:
        return "─"


def _entry_cap_pct(strat: Strategy, play_type: PlayType, conviction: ConvictionLevel | None) -> float:
    if play_type == PlayType.THESIS:
        return strat.thesis_max_nav_pct * (conviction or ConvictionLevel.MEDIUM).value
    if play_type == PlayType.APPROACH:
        return strat.approach_max_nav_pct
    if play_type == PlayType.SENTINEL:
        return strat.sentinel_max_nav_pct
    return strat.sniper_max_nav_pct


def _entry_preview(
    strat: Strategy,
    play_type: PlayType,
    symbol: str,
    conviction: ConvictionLevel | None = None,
    spot_price: float | None = None,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("Action", "BUY TO OPEN"),
        ("Play", play_type.value),
        ("Symbol", symbol.upper()),
        ("Account", strat.account.account_id or "AUTO"),
        ("Order type", "LMT option entry"),
        ("Retry", _retry_desc(CFG.entry)),
    ]
    try:
        ctx = strat.context()
        spec = strat._contract_spec(play_type)
        chain = OptionChain(strat.ib, symbol.upper())
        kwargs = spec.to_kwargs()
        if play_type == PlayType.SNIPER:
            picks = chain.select(spot_price=spot_price, **kwargs)
        else:
            picks = chain.select(**kwargs)
        if picks.empty:
            rows.append(("Contract", "no qualifying CALL found"))
            return rows

        row = picks.iloc[0]
        ask = _safe_float(row.get("ask"))
        cap_pct = _entry_cap_pct(strat, play_type, conviction)
        desired = ctx.snapshot.nav * cap_pct
        budget = strat._entry_budget(ctx, desired)
        qty = int(budget / (ask * 100)) if ask > 0 else 0
        capital = qty * ask * 100
        nav_pct = (capital / ctx.snapshot.nav) if ctx.snapshot.nav else 0.0
        contract_bits = [
            f"con_id={_row_value(row, 'con_id')}",
            f"{_row_value(row, 'expiry')} {_row_value(row, 'right')}{_row_value(row, 'strike')}",
            f"DTE={_row_value(row, 'dte')}",
            f"Δ={_row_value(row, 'delta')}",
        ]
        rows.extend([
            ("Contract", "  ".join(contract_bits)),
            ("Bid/Ask", f"{_row_value(row, 'bid')} / {_row_value(row, 'ask')}"),
            ("Estimated qty", str(qty)),
            ("Estimated capital", f"{_d(capital)}  ({nav_pct:.1%} NAV)"),
            ("Sizing cap", f"{cap_pct:.1%} NAV"),
            ("Available headroom", _d(strat._available_headroom(ctx))),
        ])
        warnings = strat._soft_contract_warnings(row, spec)
        if warnings:
            rows.append(("Contract warnings", "; ".join(warnings)))
    except Exception as exc:
        rows.append(("Preview error", str(exc)))
    return rows


def _manual_entry_preview(
    strat: Strategy,
    con_id: int,
    qty_requested: int,
    play_type: PlayType,
    conviction: ConvictionLevel | None,
    symbol: str,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("Action", "BUY TO OPEN"),
        ("Play", play_type.value),
        ("Con ID", str(con_id)),
        ("Requested qty", str(qty_requested)),
        ("Account", strat.account.account_id or "AUTO"),
        ("Order type", "LMT option entry"),
        ("Retry", _retry_desc(CFG.entry)),
    ]
    try:
        ctx = strat.context()
        contract = strat._resolve_contract(con_id, ctx)
        sym = symbol.upper() or getattr(contract, "symbol", "")
        bid, ask, last = strat._quote_option_contract(contract)
        ref_px = ((bid + ask) / 2) if (bid is not None and ask is not None) else (ask or last or 0.0)
        cap_pct = _entry_cap_pct(strat, play_type, conviction)
        budget = strat._entry_budget(ctx, ctx.snapshot.nav * cap_pct)
        algo_qty = int(budget / (ref_px * 100)) if ref_px > 0 else 0
        qty = min(qty_requested, algo_qty)
        capital = qty * ref_px * 100
        rows.extend([
            ("Symbol", sym or "UNKNOWN"),
            ("Quote", f"bid={bid if bid is not None else '─'} ask={ask if ask is not None else '─'} last={last if last is not None else '─'}"),
            ("Reference price", f"{ref_px:.2f}" if ref_px > 0 else "unavailable"),
            ("Estimated qty", str(qty)),
            ("Estimated capital", _d(capital)),
            ("Sizing cap", f"{cap_pct:.1%} NAV"),
        ])
    except Exception as exc:
        rows.append(("Preview error", str(exc)))
    return rows


def _tracker_badge(kind: str, tracker) -> str:
    status = str(getattr(tracker, "status", "WORKING") or "WORKING")
    color = "green" if status == "WORKING" else "yellow" if status == "UNBOUND" else "red"
    return f"{kind.lower()}:[{color}]{status}[/]"


def _play_status_cell(play) -> str:
    parts = [play.status.value]
    if getattr(play, "spike_fired", False):
        parts.append("⚡")
    if getattr(play, "working_entry", None) is not None:
        parts.append(_tracker_badge("ENTRY", play.working_entry))
    if getattr(play, "working_order", None) is not None:
        parts.append(_tracker_badge("EXIT", play.working_order))
    return " ".join(parts)


def _order_int(order, name: str) -> int | None:
    value = getattr(order, name, None)
    try:
        ivalue = int(value or 0)
    except (TypeError, ValueError):
        return None
    return ivalue or None


def _live_trades_for_orders(
    strat: Strategy,
    *,
    con_id: int | None = None,
    side: OrderSide | None = None,
    broad: bool = True,
):
    """Use the broad open-order view when the executor supports it."""
    try:
        return strat.executor.live_trades(con_id=con_id, side=side, broad=broad)
    except TypeError:
        return strat.executor.live_trades(con_id=con_id, side=side)


def _tracker_candidates(strat: Strategy):
    for idx, play in enumerate(strat.plays):
        if getattr(play, "working_entry", None) is not None:
            yield idx, play, "ENTRY", play.working_entry, OrderSide.BUY
        if getattr(play, "working_order", None) is not None:
            yield idx, play, "EXIT", play.working_order, OrderSide.SELL


def _tracker_matches_trade(strat: Strategy, trade, tracker, side: OrderSide, play) -> bool:
    contract = trade.contract
    order = trade.order
    if int(getattr(contract, "conId", 0) or 0) != int(play.con_id):
        return False
    if str(getattr(order, "action", "")).upper() != side.value:
        return False
    order_account = getattr(order, "account", None)
    tracker_account = getattr(tracker, "account_id", None) or strat.account.account_id or None
    if tracker_account and order_account and order_account != tracker_account:
        return False
    perm_id = _order_int(order, "permId")
    native_id = _order_int(order, "orderId")
    if getattr(tracker, "perm_id", None) and perm_id:
        return int(tracker.perm_id) == int(perm_id)
    if getattr(tracker, "native_order_id", None) and native_id:
        return int(tracker.native_order_id) == int(native_id)
    return True


def _find_tracker_for_trade(strat: Strategy, trade, used: set[tuple[int, str]]):
    strong: list[tuple[int, Any, str, Any]] = []
    fallback: list[tuple[int, Any, str, Any]] = []
    for idx, play, kind, tracker, side in _tracker_candidates(strat):
        key = (idx, kind)
        if key in used:
            continue
        if not _tracker_matches_trade(strat, trade, tracker, side, play):
            continue
        order = trade.order
        perm_id = _order_int(order, "permId")
        native_id = _order_int(order, "orderId")
        if (
            (getattr(tracker, "perm_id", None) and perm_id and int(tracker.perm_id) == int(perm_id))
            or (getattr(tracker, "native_order_id", None) and native_id and int(tracker.native_order_id) == int(native_id))
        ):
            strong.append((idx, play, kind, tracker))
        else:
            fallback.append((idx, play, kind, tracker))
    return (strong or fallback or [None])[0]


def _order_rows(strat: Strategy) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: set[tuple[int, str]] = set()

    for trade in _live_trades_for_orders(strat, broad=True):
        c = trade.contract
        o = trade.order
        st = trade.orderStatus
        match = _find_tracker_for_trade(strat, trade, used)
        play_idx = play = tracker_kind = tracker = None
        if match:
            play_idx, play, tracker_kind, tracker = match
            used.add((play_idx, tracker_kind))

        lmt = getattr(o, "lmtPrice", None) if getattr(o, "orderType", "") == "LMT" else None
        rows.append({
            "_trade": trade,
            "_play": play,
            "_tracker": tracker,
            "live": True,
            "play_row": str(play_idx) if play_idx is not None else "─",
            "tracker": tracker_kind or "UNTRACKED",
            "tstate": getattr(tracker, "status", "LIVE") if tracker else "LIVE",
            "account": getattr(o, "account", None) or strat.account.account_id or "─",
            "perm": str(_order_int(o, "permId") or "─"),
            "native": str(_order_int(o, "orderId") or "─"),
            "side": str(getattr(o, "action", "─")),
            "symbol": str(getattr(c, "symbol", "─")),
            "con_id": str(getattr(c, "conId", "─")),
            "qty": str(getattr(o, "totalQuantity", "─")),
            "filled": str(getattr(st, "filled", "─")),
            "remaining": str(strat.executor.remaining_qty_from_trade(trade)),
            "limit": f"{float(lmt):.2f}" if lmt else "MKT",
            "status": str(getattr(st, "status", "─")),
            "reason": str(getattr(tracker, "reason", "") if tracker else ""),
        })

    for idx, play, kind, tracker, side in _tracker_candidates(strat):
        if (idx, kind) in used:
            continue
        tr = getattr(tracker, "trade_result", None)
        status = str(getattr(tracker, "status", "UNBOUND") or "UNBOUND")
        rows.append({
            "_trade": getattr(tr, "trade", None) if tr is not None and getattr(tr, "trade", None) is not None else None,
            "_play": play,
            "_tracker": tracker,
            "live": False,
            "play_row": str(idx),
            "tracker": kind,
            "tstate": status,
            "account": getattr(tracker, "account_id", None) or strat.account.account_id or "─",
            "perm": str(getattr(tracker, "perm_id", None) or getattr(tracker, "order_id", None) or "─"),
            "native": str(getattr(tracker, "native_order_id", None) or "─"),
            "side": getattr(tracker, "side", side.value),
            "symbol": play.symbol,
            "con_id": str(play.con_id),
            "qty": str(getattr(tracker, "submitted_qty", None) or "─"),
            "filled": str(getattr(tracker, "accounted_fills", 0)),
            "remaining": str(getattr(tracker, "remaining_qty", "─")),
            "limit": f"{float(tracker.limit_px):.2f}" if getattr(tracker, "limit_px", None) else "─",
            "status": status,
            "reason": str(getattr(tracker, "reason", "")),
        })

    for i, row in enumerate(rows):
        row["row"] = i
    return rows


def _render_orders(strat: Strategy, rows: list[dict[str, Any]] | None = None) -> None:
    rows = rows if rows is not None else _order_rows(strat)
    if not rows:
        print("  No live orders or working trackers.\n")
        return
    t = Table(box=None, padding=(0, 1))
    for name, justify in (
        ("#", "right"), ("LIVE", "left"), ("PLAY", "right"), ("TRACKER", "left"),
        ("TSTATE", "left"), ("ACCT", "left"), ("PERM", "right"), ("NATIVE", "right"),
        ("SIDE", "left"), ("SYM", "left"), ("CON_ID", "right"), ("QTY", "right"),
        ("FILLED", "right"), ("REM", "right"), ("LMT", "right"), ("STATUS", "left"),
        ("REASON", "left"),
    ):
        t.add_column(name, justify=justify, no_wrap=True, style="dim" if name in ("#", "PLAY") else "")
    for r in rows:
        tstate = str(r["tstate"])
        if tstate == "WORKING":
            tstate = "[green]WORKING[/]"
        elif tstate == "UNBOUND":
            tstate = "[yellow]UNBOUND[/]"
        elif tstate == "EXHAUSTED":
            tstate = "[red]EXHAUSTED[/]"
        t.add_row(
            str(r["row"]),
            "yes" if r["live"] else "no",
            r["play_row"],
            r["tracker"],
            tstate,
            r["account"],
            r["perm"],
            r["native"],
            r["side"],
            r["symbol"],
            r["con_id"],
            r["qty"],
            r["filled"],
            r["remaining"],
            r["limit"],
            r["status"],
            r["reason"] or "─",
        )
    _con.print()
    _con.print(Panel(t, title="Orders / Working Trackers",
                     border_style="dim", padding=(1, 2)))


def _position_rows(strat: Strategy) -> list[dict[str, str]]:
    ctx = strat.context()
    positions = ctx.snapshot.positions
    if positions is None or positions.empty:
        return []

    tracked = {
        int(p.con_id)
        for p in strat.plays
        if p.status is not PlayStatus.CLOSED
    }
    rows: list[dict[str, str]] = []
    for _, row in positions.iterrows():
        con_id = _safe_int(row.get("con_id"))
        sec_type = str(row.get("sec_type", "") or "").upper()
        right = str(row.get("right", "") or "").upper()
        qty = _safe_float(row.get("position"))
        symbol = str(row.get("symbol", "") or "").upper()
        avg_cost = _safe_float(row.get("avg_cost"))
        avg_px = avg_cost / 100 if sec_type == "OPT" and avg_cost > 0 else avg_cost

        track_cmd = "─"
        if (
            sec_type == "OPT"
            and right == Right.CALL.value
            and qty > 0
            and con_id
            and con_id not in tracked
        ):
            track_cmd = f"track {con_id} thesis {symbol}"

        rows.append({
            "symbol": symbol or "─",
            "con_id": str(con_id or "─"),
            "type": sec_type or "─",
            "contract": (
                f"{row.get('expiry') or '─'} {right or '─'}{row.get('strike') or '─'}"
                if sec_type == "OPT" else "─"
            ),
            "qty": f"{qty:g}",
            "avg": f"{avg_px:.2f}" if avg_px > 0 else "─",
            "value": _money_or_dash(row.get("market_value")),
            "pnl": _money_or_dash(row.get("unrealized_pnl")),
            "tracked": "yes" if con_id in tracked else "no",
            "track_cmd": track_cmd,
        })
    return rows


def _render_positions(strat: Strategy) -> None:
    rows = _position_rows(strat)
    if not rows:
        print("  No live IB positions.\n")
        return

    t = Table(box=None, padding=(0, 1))
    for name, justify in (
        ("SYM", "left"), ("CON_ID", "right"), ("TYPE", "left"),
        ("CONTRACT", "left"), ("QTY", "right"), ("AVG", "right"),
        ("VALUE", "right"), ("PnL", "right"), ("TRACKED", "left"),
        ("TRACK CMD", "left"),
    ):
        t.add_column(name, justify=justify, no_wrap=True)
    for r in rows:
        t.add_row(
            r["symbol"], r["con_id"], r["type"], r["contract"], r["qty"],
            r["avg"], r["value"], r["pnl"], r["tracked"], r["track_cmd"],
        )
    _con.print()
    _con.print(Panel(t, title="IB Positions",
                     border_style="dim", padding=(1, 2)))


def _clean_id(value: Any) -> str | None:
    text = str(value).strip()
    return None if not text or text == "─" else text


def _resolve_order_selector(
    strat: Strategy,
    selector: str,
    *,
    allow_row: bool,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Resolve an order selector.

    Confirmed cancels pass ``allow_row=False`` so regenerated table rows cannot
    silently target a different live order. Bare numbers prefer broker IDs over
    display rows.
    """
    rows = _order_rows(strat)
    raw = selector.strip()
    if not raw:
        return None, None, "empty selector"

    key = "bare"
    value = raw
    if ":" in raw:
        key_part, value_part = raw.split(":", 1)
        key = key_part.strip().lower()
        value = value_part.strip()
    if not value:
        return None, None, f"empty value in selector '{selector}'"

    def by_perm(v: str):
        return next((r for r in rows if r["perm"] == v), None)

    def by_native(v: str):
        return next((r for r in rows if r["native"] == v), None)

    def by_broker(v: str):
        return by_perm(v) or by_native(v)

    if key in ("perm", "perm_id"):
        row = by_perm(value)
        return row, "perm", None if row else f"no live order has perm:{value}"
    if key in ("native", "native_id", "order", "order_id"):
        row = by_native(value)
        return row, "native", None if row else f"no live order has native:{value}"
    if key in ("id", "broker"):
        row = by_broker(value)
        return row, "broker", None if row else f"no live order has perm/native id {value}"
    if key in ("row", "#"):
        if not allow_row:
            return None, "row", "row selectors are preview-only; confirm with perm:<id> or native:<id>"
        try:
            n = int(value)
        except ValueError:
            return None, "row", f"invalid row selector '{selector}'"
        row = next((r for r in rows if r["row"] == n), None)
        return row, "row", None if row else f"no current order row #{n}"
    if key != "bare":
        return None, None, f"unknown selector prefix '{key}:'"

    row = by_broker(value)
    if row is not None:
        return row, "broker", None
    if allow_row:
        try:
            n = int(value)
        except ValueError:
            return None, None, f"unknown selector '{selector}'"
        row = next((r for r in rows if r["row"] == n), None)
        return row, "row", None if row else f"no live order id or current row matches {value}"
    return None, None, "confirmed cancels require perm:<id> or native:<id>; row numbers can change"


def _resolve_order_row(strat: Strategy, selector: str) -> dict[str, Any] | None:
    row, _kind, _err = _resolve_order_selector(strat, selector, allow_row=True)
    return row


def _parse_order_guards(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    guards: dict[str, str] = {}
    unknown: list[str] = []
    for token in tokens:
        if ":" not in token:
            unknown.append(token)
            continue
        key, value = token.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in ("conid", "con_id"):
            guards["con_id"] = value
        elif key in ("perm", "perm_id"):
            guards["perm"] = value
        elif key in ("native", "native_id", "order", "order_id"):
            guards["native"] = value
        else:
            unknown.append(token)
    return guards, unknown


def _guard_failure(row: dict[str, Any], guards: dict[str, str]) -> str | None:
    checks = (
        ("con_id", "con_id", "conid"),
        ("perm", "perm", "perm"),
        ("native", "native", "native"),
    )
    for guard_key, row_key, label in checks:
        expected = guards.get(guard_key)
        if expected is None:
            continue
        actual = _clean_id(row.get(row_key))
        if actual != expected:
            return f"selector guard mismatch: expected {label}:{expected}, current row has {actual or '─'}"
    return None


def _stable_cancel_selector(row: dict[str, Any]) -> str | None:
    perm = _clean_id(row.get("perm"))
    if perm:
        return f"perm:{perm}"
    native = _clean_id(row.get("native"))
    if native:
        return f"native:{native}"
    return None


def _confirm_cancel_command(
    row: dict[str, Any],
    *,
    command_name: str,
    block_resubmit: bool,
) -> str | None:
    selector = _stable_cancel_selector(row)
    if selector is None:
        return None
    tokens = [command_name, selector]
    con_id = _clean_id(row.get("con_id"))
    if con_id:
        tokens.append(f"conid:{con_id}")
    if command_name == "cancel" and not block_resubmit:
        tokens.append("--retry")
    tokens.append("--yes")
    return " ".join(tokens)


_CHAIN_PAGE_SIZE = 30
_CHAIN_INT_COLS = frozenset({"dte", "open_interest", "con_id"})


def _chain_panel(df, avail_cols, title, page=1):
    """Render a paginated chain DataFrame as a Rich Panel."""
    total = len(df)
    pages = max(1, (total + _CHAIN_PAGE_SIZE - 1) // _CHAIN_PAGE_SIZE)
    page = max(1, min(page, pages))
    start = (page - 1) * _CHAIN_PAGE_SIZE
    end = min(start + _CHAIN_PAGE_SIZE, total)
    chunk = df.iloc[start:end]

    t = Table(box=None, padding=(0, 1))
    for col in avail_cols:
        t.add_column(col, justify="left" if col == "expiry" else "right",
                     no_wrap=True)

    for _, row in chunk.iterrows():
        vals = []
        for col in avail_cols:
            v = row[col]
            if v is None or (isinstance(v, float) and v != v):
                vals.append("─")
            elif col in _CHAIN_INT_COLS:
                vals.append(str(int(v)))
            elif isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        t.add_row(*vals)

    sub = f"{start + 1}–{end} of {total}"
    if pages > 1:
        sub += f"  (page {page}/{pages})"
    _con.print(Panel(t, title=title, subtitle=f"[dim]{sub}[/]",
                     border_style="dim", padding=(1, 2)))


# ═════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ═════════════════════════════════════════════════════════════════════════════

# ── status ──────────────────────────────────────────────────────────────────

def do_status(strat: Strategy, ib, args: list[str]):
    ctx  = strat.context()
    risk = ctx.risk
    nav  = risk.nav
    pct  = lambda v: f"{v / nav:.1%}" if nav else ""

    # ── top: balances (left) + risk (right) in one grid ──
    top = Table.grid(padding=(0, 1))
    top.add_column(min_width=34)
    top.add_column(min_width=34)

    bal = _kvtable(
        ("NAV",            _d(nav)),
        ("Cash",           f"{_d(risk.cash)}  {pct(risk.cash)}"),
        ("Spot",           f"{_d(risk.spot_value)}  {pct(risk.spot_value)}"),
        ("Options (risk)", f"{_d(risk.risk_capital)}  ({risk.risk_pct:.1%})"),
        title="Balances",
    )
    tag = "[green]OK[/]" if risk.risk_status == "OK" else "[bold red]BREACH[/]"
    active  = sum(1 for p in strat.plays
                  if p.status in (PlayStatus.OPEN, PlayStatus.SCALING))
    pending = sum(1 for p in strat.plays if p.status == PlayStatus.PENDING)
    parts = [f"{active} active"]
    if pending:
        parts.append(f"{pending} pending")
    pending_reserved = strat._pending_entry_capital()
    available_headroom = max(0.0, risk.headroom() - pending_reserved)
    rsk_rows = [
        ("Risk ceiling", f"{strat.policy.risk_ceiling:.0%}  {tag}"),
        ("Headroom",     _ds(risk.headroom())),
    ]
    if pending_reserved > 0:
        rsk_rows.append(("Reserved entries", _ds(pending_reserved)))
        rsk_rows.append(("Avail. headroom", _ds(available_headroom)))
    rsk_rows.append(("Plays", f"{', '.join(parts)}, {len(strat.plays)} total"))
    rsk = _kvtable(*rsk_rows, title="Risk")
    top.add_row(bal, rsk)

    # ── exposures table ──
    exp_t = None
    if risk.exposures:
        exp_t = Table(box=None, padding=(0, 1), title="Exposures",
                      title_style="bold")
        exp_t.add_column("SYM", justify="left", no_wrap=True, style="dim")
        for col in ("OPT", "SPOT", "NAV%", "PnL"):
            exp_t.add_column(col, justify="right", no_wrap=True)
        for sym, e in sorted(risk.exposures.items()):
            exp_t.add_row(sym, _d(e.option_notional), _d(e.spot_value),
                          f"{e.nav_pct:.1%}", _ds(e.unrealized_pnl))

    # ── render ──
    inner = Table.grid()
    inner.add_row(top)
    if exp_t is not None:
        inner.add_row("")
        inner.add_row(exp_t)
    _con.print()
    _con.print(Panel(inner, title="Portfolio", border_style="dim", padding=(1, 2)))


# ── plays / plays N ────────────────────────────────────────────────────────

def do_plays(strat: Strategy, ib, args: list[str]):
    if args:
        return _play_detail(strat, args)
    if not strat.plays:
        print("  No plays.\n")
        return

    ctx = strat.context()

    active  = sum(1 for p in strat.plays if p.status in (PlayStatus.OPEN, PlayStatus.SCALING))
    pending = sum(1 for p in strat.plays if p.status == PlayStatus.PENDING)
    closed  = sum(1 for p in strat.plays if p.status == PlayStatus.CLOSED)

    t = Table(box=None, padding=(0, 1))
    t.add_column("#", justify="right", style="dim", no_wrap=True)
    t.add_column("TYPE", no_wrap=True)
    t.add_column("SYM", no_wrap=True)
    t.add_column("QTY", justify="right", no_wrap=True)
    t.add_column("ENTRY", justify="right", no_wrap=True)
    t.add_column("NOW", justify="right", no_wrap=True)
    t.add_column("PnL", justify="right", no_wrap=True)
    t.add_column("STATUS", no_wrap=True)

    for i, p in enumerate(strat.plays):
        now_s, pnl_s = "─", "─"
        pos = ctx.position(p.con_id)
        if pos is not None:
            mv  = pos.get("market_value")
            qty = pos.get("position")
            if mv is not None and qty and abs(float(qty)) > 0:
                cpx   = abs(float(mv)) / (abs(float(qty)) * 100)
                now_s = f"{cpx:.2f}"
                pnl_s = f"{p.current_pnl_pct(cpx):+.0%}"

        t.add_row(
            str(i), p.play_type.value, p.symbol,
            f"{p.qty_open}/{p.qty_initial}",
            f"{p.entry_price:.2f}", now_s, pnl_s,
            _play_status_cell(p),
        )

    sub = f"{active} active, {pending} pending, {closed} closed"
    _con.print()
    _con.print(Panel(t, title=f"Plays  ({sub})", border_style="dim", padding=(1, 2)))


def _play_detail(strat: Strategy, args: list[str]):
    try:
        idx = int(args[0])
    except ValueError:
        print("  Usage: plays <row>    (row # from plays listing)")
        return
    if idx < 0 or idx >= len(strat.plays):
        print(f"  No play #{idx}.")
        return

    p  = strat.plays[idx]
    ep = p.exit_profile

    # ── left: position info ──
    pos_rows: list[tuple[str, str]] = [
        ("Play ID",     p.play_id),
        ("Account",     p.account_id or "AUTO"),
        ("Con ID",      str(p.con_id)),
        ("Status",      p.status.value),
        ("Quantity",    f"{p.qty_open} / {p.qty_initial}"),
        ("Entry price", f"{p.entry_price:.2f}"),
        ("Entry time" if p.entry_time_known else "Tracked at",
         p.entry_time.strftime("%Y-%m-%d %H:%M %Z")),
        ("Entry NAV",   _d(p.entry_nav)),
    ]

    ctx = strat.context()
    cpx = strat.price_for_play(p, ctx)
    if cpx is not None:
        pos_rows.append(("Current price", f"{cpx:.2f}"))
        pos_rows.append(("PnL",           f"{p.current_pnl_pct(cpx):+.1%}"))
        # Derive market value from price × position × 100 multiplier
        pos_rows.append(("Market value",   _d(cpx * p.qty_open * 100)))

    pos_rows.append((
        "Hours held" if p.entry_time_known else "Tracked for",
        f"{p.hours_since_entry():.1f}h",
    ))
    pos_rows.append(("Peak PnL",  f"{p.peak_pnl_pct:+.1%}"))
    if not p.entry_time_known:
        pos_rows.append(("Time exits", "disabled until true entry time is known"))

    vel = p.velocity_pct_per_hour()
    if vel is not None and p.entry_time_known:
        pos_rows.append(("Velocity (4h)", f"{vel:+.2%}/h"))

    gain = (p.pnl_gain_in_window(ep.spike_window_hours)
            if ep.spike_window_hours > 0 else None)
    if gain is not None and p.entry_time_known:
        pos_rows.append((f"Gain ({ep.spike_window_hours:.0f}h)", f"{gain:+.1%}"))
    pos_rows.append(("Spike fired", "yes" if p.spike_fired else "no"))
    if p.working_entry:
        pos_rows.append((
            "Entry tracker",
            f"{p.working_entry.remaining_qty} {p.working_entry.status}  "
            f"(attempt {p.working_entry.attempts_used}, id={p.working_entry.order_id or '─'})",
        ))
    if p.working_order:
        pos_rows.append((
            "Exit tracker",
            f"{p.working_order.remaining_qty} {p.working_order.status}  "
            f"(attempt {p.working_order.attempts_used}, id={p.working_order.order_id or '─'})",
        ))
    if ep.tranches:
        pos_rows.append(("Tranche", f"{p.tranche_idx}/{len(ep.tranches)}"))

    left = _kvtable(*pos_rows, title="Position")

    # ── right: exit rules ──
    exit_rows: list[tuple[str, str]] = [
        ("Stop loss", f"{ep.stop_loss_pct:+.0%}"),
        ("Full exit", f"{ep.full_exit_pct:+.0%}"),
    ]
    trail_activate = ep.trail_activate()
    trail_drawdown = ep.trail_drawdown()
    if trail_activate is not None and trail_drawdown is not None:
        exit_rows.append(("Trail stop", f"activate {trail_activate:.0%}, drawdown {trail_drawdown:.0%}"))
    if ep.spike_pct > 0:
        exit_rows.append(("Spike", f"+{ep.spike_pct:.0%} in <{ep.spike_window_hours:.0f}h "
                                   f"→ sell {ep.spike_sell_ratio:.0%}"))
    if ep.tranches:
        for i, (trig, frac) in enumerate(ep.tranches):
            mark = "✓" if i < p.tranche_idx else " "
            exit_rows.append((f"Tranche {i+1} {mark}", f"+{trig:.0%} → sell {frac:.0%}"))
    if ep.max_hold_days:
        exit_rows.append(("Max hold", f"{ep.max_hold_days}d"))
    exit_rows.append(("DTE floor", str(ep.dte_floor)))

    right = _kvtable(*exit_rows, title="Exit Rules")

    # ── orders (full-width below) ──
    ord_t = None
    if p.orders:
        ord_t = Table(box=None, padding=(0, 1), title="Orders",
                      title_style="bold")
        ord_t.add_column("SIDE", no_wrap=True)
        ord_t.add_column("QTY", justify="right", no_wrap=True)
        ord_t.add_column("PRICE", no_wrap=True)
        ord_t.add_column("FILL", no_wrap=True)
        for o in p.orders:
            fill = f"avg={o.avg_fill():.2f}" if o.avg_fill() else o.status()
            px   = f"lmt={o.limit_px:.2f}" if o.limit_px else "MKT"
            ord_t.add_row(o.side.value, f"{o.qty}x", px, fill)

    # ── assemble ──
    grid = Table.grid(padding=(0, 2))
    grid.add_column()
    grid.add_column()
    grid.add_row(left, right)

    inner = Table.grid()
    inner.add_row(grid)
    if ord_t is not None:
        inner.add_row("")
        inner.add_row(ord_t)

    title = f"Play [{idx}]  {p.play_type.value}  {p.symbol}"
    _con.print()
    _con.print(Panel(inner, title=title, border_style="dim", padding=(1, 2)))


# ── cfg ─────────────────────────────────────────────────────────────────────

def do_cfg(strat: Strategy, ib, args: list[str]):
    """Show current config (loaded from config.toml)."""
    ep = CFG.exit_profiles
    cs = CFG.contract_specs
    t, a, s, n = ep["THESIS"], ep["APPROACH"], ep["SENTINEL"], ep["SNIPER"]
    ct, ca, ss_, cn = cs["THESIS"], cs["APPROACH"], cs["SENTINEL"], cs["SNIPER"]

    # ── general ──
    gen = _kvtable(
        ("Config path",        str(CFG.path)),
        ("Risk loop",          f"{CFG.loop_interval}s"),
        ("Scanner interval",   f"{CFG.scanner_interval}s"),
        ("Risk ceiling",       f"{CFG.risk_ceiling:.0%}"),
        ("Thesis max NAV",     f"{CFG.thesis_max_nav_pct:.1%}"),
        ("Approach max NAV",   f"{CFG.approach_max_nav_pct:.1%}"),
        ("Sentinel max NAV",   f"{CFG.sentinel_max_nav_pct:.1%}"),
        ("Sniper max NAV",     f"{CFG.sniper_max_nav_pct:.1%}"),
        ("Base currency",      CFG.base_currency),
        ("IB",                 f"{CFG.ib_host}:{CFG.ib_port}  clientId={CFG.ib_client_id}"),
        ("Confirm orders",     "yes" if CFG.terminal.confirm_orders else "no"),
        ("Confirm cancel-all", "yes" if CFG.terminal.confirm_cancel_all else "no"),
        ("Tracebacks",         "yes" if CFG.terminal.show_tracebacks else "no"),
        ("Scanner auto-open",  "yes" if CFG.sniper_scanner_auto_open else "no"),
        title="General",
    )

    # ── exit rules: one table, Thesis/Approach/Sentinel/Sniper as columns ──
    def _tranche_str(tranches):
        if not tranches:
            return "─"
        return ", ".join(f"+{tr:.0%}→{fr:.0%}" for tr, fr in tranches)

    def _spike_str(ep_):
        if ep_.spike_pct <= 0:
            return "─"
        return f"+{ep_.spike_pct:.0%} <{ep_.spike_window_hours:.0f}h → {ep_.spike_sell_ratio:.0%}"

    def _trail_str(ep_):
        activate = ep_.trail_activate()
        drawdown = ep_.trail_drawdown()
        if activate is None or drawdown is None:
            return "─"
        return f"act {activate:.0%}, dd {drawdown:.0%}"

    exits = Table(box=None, padding=(0, 1), title="Exit Rules",
                  title_style="bold")
    exits.add_column(style="dim", no_wrap=True, min_width=10)
    exits.add_column("Thesis", min_width=18)
    exits.add_column("Approach", min_width=18)
    exits.add_column("Sentinel", min_width=18)
    exits.add_column("Sniper", min_width=18)

    exits.add_row("Stop/Full",
                  f"{t.stop_loss_pct:+.0%} / {t.full_exit_pct:+.0%}",
                  f"{a.stop_loss_pct:+.0%} / {a.full_exit_pct:+.0%}",
                  f"{s.stop_loss_pct:+.0%} / {s.full_exit_pct:+.0%}",
                  f"{n.stop_loss_pct:+.0%} / {n.full_exit_pct:+.0%}")
    exits.add_row("Trail",
                  _trail_str(t), _trail_str(a), _trail_str(s), _trail_str(n))
    exits.add_row("Tranches",
                  _tranche_str(t.tranches), _tranche_str(a.tranches),
                  _tranche_str(s.tranches), _tranche_str(n.tranches))
    exits.add_row("Spike",
                  _spike_str(t), _spike_str(a), _spike_str(s), _spike_str(n))
    exits.add_row("DTE floor",
                  str(t.dte_floor), str(a.dte_floor),
                  str(s.dte_floor), str(n.dte_floor))
    exits.add_row("Max hold",
                  f"{t.max_hold_days}d" if t.max_hold_days else "─",
                  f"{a.max_hold_days}d" if a.max_hold_days else "─",
                  f"{s.max_hold_days}d" if s.max_hold_days else "─",
                  f"{n.max_hold_days}d" if n.max_hold_days else "─")
    exits.add_row("Drop", "─", "─", "─", f"{CFG.sniper_drop_pct:.0%}")
    exits.add_row("Watchlist", "─", "─", "─", ", ".join(CFG.sniper_watchlist))

    # ── contract selection: same layout ──
    specs = Table(box=None, padding=(0, 1), title="Contract Selection",
                  title_style="bold")
    specs.add_column(style="dim", no_wrap=True, min_width=10)
    specs.add_column("Thesis", no_wrap=True, min_width=18)
    specs.add_column("Approach", no_wrap=True, min_width=18)
    specs.add_column("Sentinel", no_wrap=True, min_width=18)
    specs.add_column("Sniper", no_wrap=True, min_width=18)

    specs.add_row("Δ",
                  f"{ct.delta_min:.2f}–{ct.delta_max:.2f}",
                  f"{ca.delta_min:.2f}–{ca.delta_max:.2f}",
                  f"{ss_.delta_min:.2f}–{ss_.delta_max:.2f}",
                  f"{cn.delta_min:.2f}–{cn.delta_max:.2f}")
    specs.add_row("DTE",
                  f"{ct.dte_min}–{ct.dte_max}",
                  f"{ca.dte_min}–{ca.dte_max}",
                  f"{ss_.dte_min}–{ss_.dte_max}",
                  f"{cn.dte_min}–{cn.dte_max}")
    specs.add_row("±K",
                  f"{ct.strike_width:.0%}",
                  f"{ca.strike_width:.0%}",
                  f"{ss_.strike_width:.0%}",
                  f"{cn.strike_width:.0%}")

    # ── execution retry profiles ──
    def _retry_row(rp):
        total = rp.max_retries + 1
        mins  = total * rp.fill_timeout_secs / 60
        lr    = rp.last_resort_mode.value if rp.last_resort_mode else "─"
        fallback = rp.fallback_mode.value if rp.fallback_mode else "─"
        after = rp.fallback_after if rp.fallback_after is not None else "─"
        return (f"{rp.mode.value}→{fallback} "
                f"(after {after}), last={lr}",
                f"{total} × {rp.fill_timeout_secs}s  (~{mins:.0f} min)")

    e_modes, e_timing = _retry_row(CFG.entry)
    p_modes, p_timing = _retry_row(CFG.patient)
    u_modes, u_timing = _retry_row(CFG.urgent)

    ex_t = _kvtable(
        ("Entry   (opening buys)",     f"{e_modes}  {e_timing}"),
        ("Patient (profit exits)",     f"{p_modes}  {p_timing}"),
        ("Urgent  (stop/trail/DTE)",   f"{u_modes}  {u_timing}"),
        title="Execution",
    )

    # ── assemble ──
    inner = Table.grid()
    inner.add_row(gen)
    inner.add_row("")
    inner.add_row(exits)
    inner.add_row("")
    inner.add_row(specs)
    inner.add_row("")
    inner.add_row(ex_t)

    _con.print()
    _con.print(Panel(inner, title=f"Configuration  ({CFG.path.name})", border_style="dim", padding=(1, 2)))


# ── chain ───────────────────────────────────────────────────────────────────

_SPEC_NAMES = {
    "thesis": "THESIS", "approach": "APPROACH",
    "sentinel": "SENTINEL", "sniper": "SNIPER",
}

_CHAIN_RESEARCH_RIGHT = Right.CALL
_CHAIN_RESEARCH_ALLOW_PUTS = False  # OptionChain still supports puts; console research is CALL-only for now.
_CHAIN_DEFAULT_DTE_MIN = 21
_CHAIN_DEFAULT_DTE_MAX = 180
_CHAIN_DEFAULT_EXPIRY_COUNT = 6
_CHAIN_DEFAULT_TARGET_DTES = (30, 45, 60, 90, 120, 180)


def _chain_usage() -> None:
    print(
        "  Usage: chain <SYM> [N] [dte=45-90|45-90] "
        "[exp=YYYYMMDD[,YYYYMMDD]] [thesis|approach|sentinel|sniper] [p<N>]"
    )
    print("         chain <SYM> expiries [N] [dte=45-180] [p<N>]")
    print("  Default: CALLS only, laddered expiries across ~21-180 DTE; not only front expiries.")


def _parse_chain_int(text: str) -> int | None:
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _parse_chain_dte(token: str) -> tuple[int | None, int | None] | None:
    """Parse friendly DTE tokens: dte=45-90, 45-90, dte>=60, dte<=120, 90+."""

    text = token.strip().lower().replace(" ", "")
    for prefix in ("dte=", "dte:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith("dte>="):
        text = ">=" + text[5:]
    elif text.startswith("dte<="):
        text = "<=" + text[5:]

    if text.startswith(">=") and text[2:].isdigit():
        return int(text[2:]), None
    if text.startswith("<=") and text[2:].isdigit():
        return None, int(text[2:])
    if text.endswith("+") and text[:-1].isdigit():
        return int(text[:-1]), None
    if "-" in text:
        parts = text.split("-", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            lo, hi = int(parts[0]), int(parts[1])
            return (lo, hi) if lo <= hi else (hi, lo)
    if token.lower().startswith("dte=") and text.isdigit():
        dte = int(text)
        return dte, dte
    return None


def _parse_chain_expiries(token: str) -> list[str] | None:
    """Parse exp=20260619,20260918 or a single YYYYMMDD/YYYY-MM-DD token."""

    text = token.strip()
    low = text.lower()
    payload = None
    for prefix in ("exp=", "expiry=", "e="):
        if low.startswith(prefix):
            payload = text[len(prefix):]
            break
    if payload is None:
        compact = text.replace("-", "").replace("/", "")
        if len(compact) == 8 and compact.isdigit():
            payload = text
    if payload is None:
        return None

    expiries = []
    for item in payload.split(","):
        item = item.strip()
        if item:
            expiries.append(OptionChain.normalize_expiry(item))
    return expiries or None


def _chain_filter_expiries(available, dte_min: int | None, dte_max: int | None):
    out = available.copy()
    if dte_min is not None:
        out = out[out["dte"] >= int(dte_min)]
    if dte_max is not None:
        out = out[out["dte"] <= int(dte_max)]
    return out.sort_values(["dte", "expiry"]).reset_index(drop=True)


def _chain_targets(count: int, dte_min: int | None, dte_max: int | None) -> list[float]:
    count = max(1, int(count))
    if dte_min is None and dte_max is None:
        base = list(_CHAIN_DEFAULT_TARGET_DTES)
        if count == len(base):
            return [float(x) for x in base]
        if count == 1:
            return [float(base[len(base) // 2])]
        if count < len(base):
            return [float(base[round(i * (len(base) - 1) / (count - 1))]) for i in range(count)]

    lo = _CHAIN_DEFAULT_DTE_MIN if dte_min is None else int(dte_min)
    hi = _CHAIN_DEFAULT_DTE_MAX if dte_max is None else int(dte_max)
    if hi < lo:
        lo, hi = hi, lo
    if count == 1:
        return [float((lo + hi) / 2)]
    return [float(lo + (hi - lo) * i / (count - 1)) for i in range(count)]


def _chain_pick_ladder_expiries(candidates, count: int, targets: list[float]) -> list[str]:
    if candidates.empty or count <= 0:
        return []
    if len(candidates) <= count:
        return list(candidates["expiry"])

    rows = [(str(r["expiry"]), int(r["dte"])) for _, r in candidates.iterrows()]
    selected: list[str] = []
    used: set[str] = set()

    for target in targets:
        best = None
        for expiry, dte in rows:
            if expiry in used:
                continue
            key = (abs(dte - target), dte, expiry)
            if best is None or key < best[0]:
                best = (key, expiry)
        if best is not None:
            expiry = best[1]
            selected.append(expiry)
            used.add(expiry)
        if len(selected) >= count:
            break

    for expiry, _ in rows:
        if len(selected) >= count:
            break
        if expiry not in used:
            selected.append(expiry)
            used.add(expiry)

    return selected


def _chain_expiry_summary(available, expiries: list[str]) -> str:
    lookup = {str(r["expiry"]): int(r["dte"]) for _, r in available.iterrows()}
    parts = []
    for expiry in expiries:
        dte = lookup.get(str(expiry))
        parts.append(f"{expiry}({dte}d)" if dte is not None else str(expiry))
    return ", ".join(parts) if parts else "─"


def _chain_dte_label(dte_min: int | None, dte_max: int | None, empty: str = "custom") -> str:
    if dte_min is None and dte_max is None:
        return empty
    lo = str(dte_min) if dte_min is not None else "0"
    hi = str(dte_max) if dte_max is not None else "∞"
    return f"{lo}–{hi}"


def _print_chain_expiry_note(symbol: str, available, candidates, selected: list[str]) -> None:
    selected_text = _chain_expiry_summary(available, selected)
    more = len(candidates) - len(selected)
    _con.print(f"  [dim]{symbol} expiries selected:[/] {selected_text}")
    if more > 0:
        _con.print(
            "  [dim]Tip: use `chain {sym} expiries`, `chain {sym} dte=45-90`, "
            "or `chain {sym} exp=YYYYMMDD` to steer the chain.[/]".format(sym=symbol)
        )


def _print_chain_expiries(symbol: str, available, dte_min: int | None, dte_max: int | None,
                          limit: int | None, page: int) -> None:
    df = _chain_filter_expiries(available, dte_min, dte_max)
    if limit is not None:
        df = df.head(limit)
    if df.empty:
        print("  No expiries match that DTE filter.")
        return
    title = f"EXPIRIES  {symbol}"
    if dte_min is not None or dte_max is not None:
        title += f"  DTE={_chain_dte_label(dte_min, dte_max)}"
    _chain_panel(df, ["expiry", "dte"], title, page)


def do_chain(strat: Strategy, ib, args: list[str]):
    if not args:
        _chain_usage()
        return
    symbol = args[0].upper()
    expiry_count = _CHAIN_DEFAULT_EXPIRY_COUNT
    count_was_set = False
    explicit_expiries: list[str] | None = None
    dte_min: int | None = None
    dte_max: int | None = None
    dte_was_set = False
    expiry_mode = "ladder"
    list_expiries = False
    list_limit: int | None = None
    rights = [_CHAIN_RESEARCH_RIGHT.value]
    right_filter = _CHAIN_RESEARCH_RIGHT.value
    spec_type = None
    page = 1

    for a in args[1:]:
        low = a.lower()
        if len(low) > 1 and low[0] == "p" and low[1:].isdigit():
            page = int(low[1:])
            continue
        if low.startswith("page=") and low[5:].isdigit():
            page = int(low[5:])
            continue

        if low in ("calls", "call"):
            rights = [Right.CALL.value]
            right_filter = Right.CALL.value
        elif low in ("puts", "put"):
            if not _CHAIN_RESEARCH_ALLOW_PUTS:
                print("  PUT chain research is disabled in the console for now; CALLS are the active research side.")
                return
            rights = [Right.PUT.value]
            right_filter = Right.PUT.value
        elif low in _SPEC_NAMES:
            spec_type = _SPEC_NAMES[low]
        elif low in ("expiries", "expiry", "exps", "dates"):
            list_expiries = True
        elif low in ("front", "near", "nearest", "recent"):
            expiry_mode = "front"
        elif low in ("ladder", "spread", "research"):
            expiry_mode = "ladder"
        else:
            parsed_expiries = _parse_chain_expiries(a)
            if parsed_expiries is not None:
                explicit_expiries = parsed_expiries
                continue

            parsed_dte = _parse_chain_dte(a)
            if parsed_dte is not None:
                dte_min, dte_max = parsed_dte
                dte_was_set = True
                continue

            count_value = None
            for prefix in ("n=", "count="):
                if low.startswith(prefix):
                    count_value = _parse_chain_int(low[len(prefix):])
                    break
            if count_value is None:
                count_value = _parse_chain_int(a)
            if count_value is not None:
                if list_expiries:
                    list_limit = count_value
                else:
                    expiry_count = count_value
                    count_was_set = True
                continue

            print(f"  Unknown arg '{a}'")
            _chain_usage()
            return

    if list_expiries and list_limit is None and count_was_set:
        list_limit = expiry_count

    if (
        not explicit_expiries
        and not dte_was_set
        and spec_type is None
        and not list_expiries
        and expiry_mode != "front"
    ):
        dte_min = _CHAIN_DEFAULT_DTE_MIN
        dte_max = _CHAIN_DEFAULT_DTE_MAX

    cols = ["expiry", "dte", "strike", "bid", "ask", "mid",
            "spread_pct", "delta", "iv", "open_interest", "con_id"]

    try:
        chain = OptionChain(ib, symbol)
        available = chain.available_expiries()
        if available.empty:
            print(f"  No option expiries found for {symbol}.")
            return

        if list_expiries:
            _print_chain_expiries(symbol, available, dte_min, dte_max, list_limit, page)
            return

        if spec_type is not None:
            # ── spec-based: show what the strategy would pick ────────────
            spec = strat._contract_spec(PlayType(spec_type))
            spec = replace(spec, right=Right(right_filter))
            if dte_was_set:
                spec = replace(spec, dte_min=dte_min, dte_max=dte_max)
            elif explicit_expiries is not None:
                # Exact expiries mean “show this expiry with the play's delta/liquidity shape”,
                # not “drop it if it falls outside the play's normal DTE window”.
                spec = replace(spec, dte_min=None, dte_max=None)

            kwargs = spec.to_kwargs()
            if explicit_expiries is not None:
                kwargs["expiries"] = explicit_expiries

            picks = chain.select(**kwargs)
            spot  = chain.spot

            if picks.empty:
                print(f"  No CALL contracts match {spec_type} spec for {symbol}.")
                return

            avail = [c for c in cols if c in picks.columns]
            dte_label = _chain_dte_label(spec.dte_min, spec.dte_max)
            title = (f"{spec_type} CALLS  {symbol}  spot=${spot:.2f}  "
                     f"Δ={spec.delta_min}–{spec.delta_max}  "
                     f"DTE={dte_label}")
            if explicit_expiries is not None:
                title += f"  exp={', '.join(explicit_expiries)}"
            _con.print()
            _chain_panel(picks, avail, title, page)

            top = picks.iloc[0]
            d = top["delta"]
            delta_s = f"{d:.2f}" if d is not None and d == d else "N/A"
            print(f"  Top: con_id={int(top['con_id'])}  "
                  f"strike={top['strike']}  DTE={top['dte']}  delta={delta_s}")
        else:
            # ── raw chain research ───────────────────────────────────────
            if explicit_expiries is not None:
                selected_expiries = explicit_expiries
                candidates = available[available["expiry"].isin(selected_expiries)]
            else:
                candidates = _chain_filter_expiries(available, dte_min, dte_max)
                if candidates.empty:
                    print("  No expiries match that DTE filter.")
                    _print_chain_expiries(symbol, available, None, None, 12, 1)
                    return
                if expiry_mode == "front":
                    selected_expiries = list(candidates.head(expiry_count)["expiry"])
                else:
                    targets = _chain_targets(
                        expiry_count,
                        None if not dte_was_set else dte_min,
                        None if not dte_was_set else dte_max,
                    )
                    selected_expiries = _chain_pick_ladder_expiries(candidates, expiry_count, targets)

            if not selected_expiries:
                print("  No expiries selected.")
                return

            _con.print()
            _print_chain_expiry_note(symbol, available, candidates, selected_expiries)

            df = chain.fetch(expiries=selected_expiries, strike_width=0.25,
                             rights=rights)
            if df.empty:
                print("  No contracts found.")
                return

            avail = [c for c in cols if c in df.columns]
            sides = [("CALLS", right_filter)]
            for label, rv in sides:
                sub = df[df["right"] == rv].sort_values(["expiry", "strike"])
                if sub.empty:
                    continue
                title = f"{label}  {symbol}  spot=${chain.spot:.2f}"
                _chain_panel(sub, avail, title, page)

    except Exception as e:
        print(f"  Error: {e}")


# ── thesis ──────────────────────────────────────────────────────────────────

def do_thesis(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    if len(args) < 2:
        print("  Usage: thesis <SYM> <low|med|high> [--yes]")
        return
    if len(args) >= 3 and args[2].lower() == "put":
        print("  PUT entries are disabled; this strategy is CALL-only for now.")
        return
    sym = args[0].upper()
    if strat._has_open_play(sym, PlayType.THESIS):
        print(f"  Already have an open THESIS on {sym}. Use 'close' first.")
        return
    conv = _CONV.get(args[1].lower())
    if conv is None:
        print(f"  Unknown conviction '{args[1]}' — use low, med, or high")
        return
    rows = _entry_preview(strat, PlayType.THESIS, sym, conviction=conv)
    if not _confirm_order_or_preview("Order Preview — THESIS", rows, confirmed):
        return
    strat.open_thesis(sym, conv, right=Right.CALL)


# ── approach ────────────────────────────────────────────────────────────────

def do_approach(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    if not args:
        print("  Usage: approach <SYM> [--yes]")
        return
    if len(args) >= 2 and args[1].lower() == "put":
        print("  PUT entries are disabled; this strategy is CALL-only for now.")
        return
    sym = args[0].upper()
    if strat._has_open_play(sym, PlayType.APPROACH):
        print(f"  Already have an open APPROACH on {sym}. Use 'close' first.")
        return
    rows = _entry_preview(strat, PlayType.APPROACH, sym)
    if not _confirm_order_or_preview("Order Preview — APPROACH", rows, confirmed):
        return
    strat.open_approach(sym, right=Right.CALL)


# ── sentinel ────────────────────────────────────────────────────────────────

def do_sentinel(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    if not args:
        print("  Usage: sentinel <SYM> [--yes]")
        return
    if len(args) >= 2 and args[1].lower() == "put":
        print("  PUT entries are disabled; this strategy is CALL-only for now.")
        return
    sym = args[0].upper()
    if strat._has_open_play(sym, PlayType.SENTINEL):
        print(f"  Already have an open SENTINEL on {sym}. Use 'close' first.")
        return
    rows = _entry_preview(strat, PlayType.SENTINEL, sym)
    if not _confirm_order_or_preview("Order Preview — SENTINEL", rows, confirmed):
        return
    strat.open_sentinel(sym, right=Right.CALL)


# ── sniper ──────────────────────────────────────────────────────────────────

def do_sniper(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    if len(args) < 2:
        print("  Usage: sniper <SYM> <PRICE> [--yes]")
        return
    try:
        spot = float(args[1])
    except ValueError:
        print("  spot_price must be a number.")
        return
    sym = args[0].upper()
    rows = _entry_preview(strat, PlayType.SNIPER, sym, spot_price=spot)
    if not _confirm_order_or_preview("Order Preview — SNIPER", rows, confirmed):
        return
    strat.open_sniper(sym, spot, ctx=strat.context())


# ── manual ──────────────────────────────────────────────────────────────────

def do_manual(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    if len(args) < 3:
        print("  Usage: manual <CON_ID> <QTY> <thesis|approach|sentinel|sniper> [CONV] [SYM] [--yes]")
        return
    try:
        con_id, qty = int(args[0]), int(args[1])
    except ValueError:
        print("  con_id and qty must be integers.")
        return
    pt = _PTYPE.get(args[2].lower())
    if pt is None:
        print(f"  Unknown type '{args[2]}' — use thesis, approach, sentinel, or sniper")
        return
    conviction, symbol = None, ""
    if len(args) >= 4:
        conviction = _CONV.get(args[3].lower())
        if conviction is None:
            symbol = args[3].upper()
        elif len(args) >= 5:
            symbol = args[4].upper()
    rows = _manual_entry_preview(strat, con_id, qty, pt, conviction, symbol)
    if not _confirm_order_or_preview("Order Preview — MANUAL", rows, confirmed):
        return
    strat.open_manual(con_id=con_id, qty=qty, play_type=pt,
                      conviction=conviction, symbol=symbol)



# ── track ──────────────────────────────────────────────────────────────────


def do_track(strat: Strategy, ib, args: list[str]):
    if len(args) < 2:
        print("  Usage: track <CON_ID> <thesis|approach|sentinel|sniper> [SYM]")
        return
    try:
        con_id = int(args[0])
    except ValueError:
        print("  con_id must be an integer.")
        return
    pt = _PTYPE.get(args[1].lower())
    if pt is None:
        print(f"  Unknown type '{args[1]}' — use thesis, approach, sentinel, or sniper")
        return
    symbol = args[2].upper() if len(args) >= 3 else ""
    strat.track_position(con_id=con_id, play_type=pt, symbol=symbol)



# ── close ───────────────────────────────────────────────────────────────────

def do_close(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    if not args:
        print("  Usage: close <row> [QTY] [--yes]    (row # from plays listing)")
        return
    try:
        idx = int(args[0])
    except ValueError:
        print("  Play index must be a number.")
        return
    if idx < 0 or idx >= len(strat.plays):
        print(f"  No play #{idx}.")
        return
    play = strat.plays[idx]
    if play.status not in (PlayStatus.OPEN, PlayStatus.SCALING):
        print(f"  Play [{idx}] is {play.status.value}.")
        return
    qty = play.qty_open
    if len(args) >= 2:
        try:
            qty = min(int(args[1]), play.qty_open)
        except ValueError:
            print("  qty must be an integer.")
            return
    if qty < 1:
        print("  qty must be at least 1.")
        return
    rows = [
        ("Action", "SELL TO CLOSE"),
        ("Play", f"[{idx}] {play.play_type.value} {play.symbol}"),
        ("Con ID", str(play.con_id)),
        ("Quantity", f"{qty} / {play.qty_open} open"),
        ("Account", strat.account.account_id or "AUTO"),
        ("Order type", "LMT option exit"),
        ("Retry", _retry_desc(CFG.patient)),
        ("Risk effect", "closing risk"),
    ]
    if play.working_order:
        rows.append(("Existing tracker", f"{play.working_order.status}; duplicate close will be blocked"))
    if not _confirm_order_or_preview("Order Preview — CLOSE", rows, confirmed):
        return
    ok, submitted = strat.manual_close(play, qty, ctx=strat.context())
    if ok or submitted:
        if submitted:
            print(f"  Submitted close for {qty}x {play.symbol}  [{play.status.value}]")
        else:
            print(f"  Closed {qty}x {play.symbol}  [{play.status.value}]")
    else:
        print(f"  No fill for {qty}x {play.symbol}; state unchanged.")


# ── spot ────────────────────────────────────────────────────────────────────

def do_spot(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    if len(args) < 3:
        print("  Usage: spot <buy|sell> <SYM> <QTY> [LMT] [--yes]")
        return
    d = args[0].lower()
    if d not in ("buy", "sell"):
        print(f"  Unknown direction '{args[0]}' — use buy or sell")
        return
    try:
        qty = int(args[2])
    except ValueError:
        print("  qty must be an integer.")
        return
    if qty < 1:
        print("  qty must be at least 1.")
        return
    limit = None
    if len(args) >= 4:
        try:
            limit = float(args[3])
        except ValueError:
            print("  limit must be a number.")
            return
    sym = args[1].upper()
    desc = f"LMT {limit:.2f}" if limit else "MKT"
    rows = [
        ("Action", f"{d.upper()} STOCK"),
        ("Symbol", sym),
        ("Quantity", str(qty)),
        ("Order type", desc),
        ("Account", strat.account.account_id or "AUTO"),
        ("Risk effect", "opens/closes stock exposure outside option-play lifecycle"),
    ]
    if not _confirm_order_or_preview("Order Preview — SPOT", rows, confirmed):
        return
    result = (strat.executor.buy_stock(sym, qty, limit) if d == "buy"
              else strat.executor.sell_stock(sym, qty, limit))
    print(f"  {d.upper()} {qty}x {sym}  {desc}  id={result.order_id}")


# ── scan ────────────────────────────────────────────────────────────────────

def do_scan(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    args, open_flag = _pop_flag(args, "--open")
    if args:
        print("  Usage: scan [--open] [--yes]")
        return
    if not strat.scanner:
        print("  No scanner configured.")
        return
    print(f"  Scanning: {', '.join(strat.scanner.watchlist)}")
    hit = strat.scanner.scan()
    if hit:
        sym, spot = hit
        print(f"  Hit: {sym} @ {spot:.2f}")
        if strat._has_open_play(sym, PlayType.SNIPER):
            print(f"  Already have an open SNIPER on {sym}.")
            return
        if not open_flag:
            print("  No order submitted. Use 'scan --open' to submit a SNIPER entry.")
            return
        rows = _entry_preview(strat, PlayType.SNIPER, sym, spot_price=spot)
        if not _confirm_order_or_preview("Order Preview — SCAN/SNIPER", rows, confirmed):
            return
        strat.open_sniper(sym, spot, ctx=strat.context())
    else:
        print("  No qualifying drops.")


# ── pending ─────────────────────────────────────────────────────────────────

def do_orders(strat: Strategy, ib, args: list[str]):
    _render_orders(strat)


def do_pending(strat: Strategy, ib, args: list[str]):
    do_orders(strat, ib, args)


def do_positions(strat: Strategy, ib, args: list[str]):
    if args:
        print("  Usage: positions")
        return
    _render_positions(strat)


# ── order management / recovery ──────────────────────────────────────────────

def _cancel_order_common(
    strat: Strategy,
    args: list[str],
    confirmed: bool,
    block_resubmit: bool,
    *,
    command_name: str = "cancel",
) -> None:
    if not args:
        print("  Usage: cancel <perm:id|native:id|row:n> [conid:n] [--retry] [--yes]")
        return
    selector = args[0]
    guards, unknown = _parse_order_guards(args[1:])
    if unknown:
        print(f"  Unknown selector guard(s): {', '.join(unknown)}")
        print("  Valid guards: conid:<con_id>, perm:<perm_id>, native:<order_id>")
        return

    row, kind, err = _resolve_order_selector(strat, selector, allow_row=not confirmed)
    if row is None:
        print(f"  {err or f'No live order matches {selector!r}.'} Use 'orders' first.")
        if confirmed and kind == "row":
            print("  Re-run the preview and confirm with the printed perm:/native: command instead.")
        return
    if not row.get("live") or row.get("_trade") is None:
        print("  Selected row is not a live IB order. Use clear-working after broker verification.")
        return

    guard_err = _guard_failure(row, guards)
    if guard_err:
        print(f"  Refusing cancel: {guard_err}")
        print("  Run 'orders' and preview the cancel again before confirming.")
        return

    safe_cmd = _confirm_cancel_command(
        row,
        command_name=command_name,
        block_resubmit=block_resubmit,
    )
    rows = [
        ("Action", "CANCEL LIVE ORDER"),
        ("Selector", f"{selector} ({kind or 'unknown'})"),
        ("Row", str(row["row"])),
        ("Order", f"perm={row['perm']} native={row['native']}"),
        ("Side", f"{row['side']} {row['qty']}x {row['symbol']}"),
        ("Con ID", row["con_id"]),
        ("Status", row["status"]),
        ("Tracker", f"play={row['play_row']} {row['tracker']} {row['tstate']}"),
        ("After cancel", "block strategy retry" if block_resubmit else "allow strategy retry ladder"),
    ]
    if safe_cmd:
        rows.append(("Confirm with", safe_cmd))
    else:
        rows.append(("Confirm with", "unavailable: live order has no perm/native ID; cancel from TWS or refresh orders"))

    if CFG.terminal.confirm_orders and not confirmed:
        _ticket("Cancel Preview", *rows, subtitle="[yellow]No order submitted[/]")
        return

    if confirmed and kind == "row":
        print("  Refusing confirmed cancel by row number; order rows can change between preview and --yes.")
        if safe_cmd:
            print(f"  Use: {safe_cmd}")
        return
    if confirmed and _stable_cancel_selector(row) is None:
        print("  Refusing confirmed cancel: selected live order has no stable perm/native ID.")
        return

    result = strat.executor.result_from_trade(row["_trade"])
    strat.executor.cancel(result)
    try:
        final_status = strat.executor.wait_until_not_live(result, timeout_secs=3.0, poll_secs=0.25)
    except Exception:
        final_status = result.status()

    play = row.get("_play")
    tracker_kind = row.get("tracker")
    if block_resubmit and play is not None and tracker_kind in ("ENTRY", "EXIT"):
        ok, msg = strat.block_working_tracker_after_cancel(
            play,
            "entry" if tracker_kind == "ENTRY" else "exit",
        )
        if ok:
            print(f"  {msg}")
        else:
            print(f"  Cancel sent, but tracker was not blocked: {msg}")
    elif not block_resubmit and row.get("_tracker") is not None:
        row["_tracker"].cancel_requested = True
        state.save(strat.plays, account_id=strat.account.account_id)
        print("  Cancel sent; retry ladder remains enabled for the matched tracker.")
    else:
        print("  Cancel sent for untracked live order; no strategy tracker changed.")
    print(f"  Broker status after cancel wait: {final_status}")


def do_cancel(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    args, retry_flag = _pop_flag(args, "--retry")
    if not args:
        print("  Usage: cancel <perm:id|native:id|row:n> [conid:n] [--retry] [--yes]")
        return
    _cancel_order_common(
        strat,
        args,
        confirmed,
        block_resubmit=not retry_flag,
        command_name="cancel",
    )


def do_cancel_retry(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    if not args:
        print("  Usage: cancel-retry <perm:id|native:id|row:n> [conid:n] [--yes]")
        return
    _cancel_order_common(
        strat,
        args,
        confirmed,
        block_resubmit=False,
        command_name="cancel-retry",
    )

def do_cancel_all(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    args, retry_flag = _pop_flag(args, "--retry")
    if args:
        print("  Usage: cancel-all [--retry] [--yes]")
        return

    rows = _order_rows(strat)
    live_rows = [r for r in rows if r.get("live")]
    preview = [
        ("Action", "GLOBAL CANCEL"),
        ("Live orders", str(len(live_rows))),
        ("Working trackers", str(sum(1 for r in rows if r.get("_tracker") is not None))),
        ("After cancel", "block matched/all working trackers" if not retry_flag else "allow strategy retry ladders"),
    ]
    if not _confirm_cancel_all_or_preview(preview, confirmed):
        return

    strat.executor.cancel_all()
    blocked = 0
    if not retry_flag:
        for idx, play, kind, tracker, side in _tracker_candidates(strat):
            ok, _ = strat.block_working_tracker_after_cancel(
                play,
                "entry" if kind == "ENTRY" else "exit",
            )
            blocked += 1 if ok else 0
    print(f"  Global cancel sent. Blocked trackers: {blocked}" if not retry_flag else "  Global cancel sent; retry ladders remain enabled.")


def do_rebind(strat: Strategy, ib, args: list[str]):
    if args:
        print("  Usage: rebind")
        return
    before = _order_rows(strat)
    ctx = strat.context()
    dirty = strat.restore_working_entries(ctx)
    dirty |= strat.restore_working_orders(ctx)
    if dirty:
        state.save(strat.plays, account_id=strat.account.account_id)
    after = _order_rows(strat)
    unbound = sum(1 for r in after if r["tstate"] == "UNBOUND")
    exhausted = sum(1 for r in after if r["tstate"] == "EXHAUSTED")
    live = sum(1 for r in after if r["live"])
    print(
        f"  Rebind complete. Live rows: {live}; "
        f"unbound: {unbound}; exhausted: {exhausted}; "
        f"state saved: {'yes' if dirty else 'no'}."
    )
    _render_orders(strat, after)


def do_clear_working(strat: Strategy, ib, args: list[str]):
    args, confirmed = _yes(args)
    if len(args) < 2:
        print("  Usage: clear-working <play-row> entry|exit [--yes]")
        return
    try:
        idx = int(args[0])
    except ValueError:
        print("  play-row must be a number.")
        return
    if idx < 0 or idx >= len(strat.plays):
        print(f"  No play #{idx}.")
        return
    kind = args[1].lower()
    if kind not in ("entry", "exit"):
        print("  kind must be entry or exit.")
        return
    play = strat.plays[idx]
    tracker = play.working_entry if kind == "entry" else play.working_order
    if tracker is None:
        print(f"  Play [{idx}] has no working {kind} tracker.")
        return

    rows = [
        ("Action", "CLEAR LOCAL WORKING TRACKER"),
        ("Play", f"[{idx}] {play.play_type.value} {play.symbol}"),
        ("Kind", kind.upper()),
        ("Tracker status", getattr(tracker, "status", "WORKING")),
        ("Remaining", str(getattr(tracker, "remaining_qty", "─"))),
        ("Order", f"perm={getattr(tracker, 'perm_id', None) or '─'} native={getattr(tracker, 'native_order_id', None) or '─'}"),
        ("Broker check", f"must find no live {'BUY' if kind == 'entry' else 'SELL'} for con_id={play.con_id}"),
    ]
    if CFG.terminal.confirm_orders and not confirmed:
        _ticket(
            "Clear-Working Preview",
            *rows,
            ("Confirmation", "re-run clear-working with --yes to verify broker state and clear"),
            subtitle="[yellow]State unchanged[/]",
        )
        return

    if kind == "entry":
        ok, msg = strat.clear_working_entry_verified(play, ctx=strat.context())
    else:
        ok, msg = strat.clear_working_exit_verified(play, ctx=strat.context())

    if ok:
        _con.print(f"  [green]{msg}[/]")
    else:
        _con.print(f"  [bold red]{msg}[/]")


# ── run ─────────────────────────────────────────────────────────────────────

def do_run(strat: Strategy, ib, args: list[str]):
    try:
        strat.step()
        print("  Tick complete.")
    except Exception as e:
        print(f"  Error: {e}")


# ── help ────────────────────────────────────────────────────────────────────

def do_help(strat: Strategy, ib, args: list[str]):
    topic = args[0].lower() if args else ""

    def _panel(title: str, rows: list[tuple[str, str]]) -> None:
        t = Table(box=None, padding=(0, 2))
        t.add_column("COMMAND", no_wrap=True, style="cyan")
        t.add_column("USE")
        for cmd, use in rows:
            t.add_row(cmd, use)
        _con.print()
        _con.print(Panel(t, title=title, border_style="dim", padding=(1, 2)))

    if topic in ("orders", "order", "recovery"):
        _panel("Order / Recovery Commands", [
            ("orders | pending", "live IB orders plus matched ENTRY/EXIT trackers"),
            ("cancel row:<n>", "preview one live order cancel; prints stable confirmation command"),
            ("cancel perm:<id> [conid:n] --yes", "cancel one live order and block automatic resubmission"),
            ("cancel perm:<id> --retry --yes", "cancel this attempt but allow the retry ladder"),
            ("cancel-retry native:<id> --yes", "same as cancel --retry"),
            ("cancel-all [--yes]", "IB global cancel and block working trackers"),
            ("cancel-all --retry [--yes]", "IB global cancel without blocking retry ladders"),
            ("rebind", "re-run working entry/exit rebinding from live IB orders"),
            ("clear-working <play-row> entry|exit [--yes]", "broker-verified local tracker clear"),
        ])
        return

    if topic in ("trading", "trade"):
        _panel("Trading Commands", [
            ("thesis <SYM> <low|med|high> [--yes]", "open THESIS CALL entry"),
            ("approach <SYM> [--yes]", "open APPROACH CALL entry"),
            ("sentinel <SYM> [--yes]", "open SENTINEL CALL entry"),
            ("sniper <SYM> <PRICE> [--yes]", "open SNIPER CALL entry"),
            ("manual <CON> <QTY> <TYPE> [CONV] [SYM] [--yes]", "manual option entry"),
            ("track <CON> <TYPE> [SYM]", "adopt an existing long CALL position as a play"),
            ("close <row> [QTY] [--yes]", "manual SELL-to-close"),
            ("spot <buy|sell> <SYM> <QTY> [LMT] [--yes]", "stock order outside play lifecycle"),
        ])
        return

    if topic in ("scan", "scanner"):
        _panel("Scanner Commands", [
            ("scan", "scan watchlist only; does not submit"),
            ("scan --open [--yes]", "submit SNIPER on hit after preview/confirmation"),
            ("cfg", "shows whether strategy-loop scanner auto-open is enabled"),
        ])
        return

    if topic in ("chain", "chains", "research"):
        _panel("Option Chain Research", [
            ("chain <SYM>", "CALLS only; laddered expiries around 21-180 DTE"),
            ("chain <SYM> expiries [p<N>]", "list available expiries and DTEs without quote requests"),
            ("chain <SYM> dte=45-90", "show calls from a DTE window"),
            ("chain <SYM> exp=YYYYMMDD", "show calls for exact expiry; comma-list supported"),
            ("chain <SYM> front 4", "old-style nearest/front expiries when you explicitly want them"),
            ("chain <SYM> thesis|approach|sentinel|sniper", "show contracts matching a play's selection shape"),
            ("chain <SYM> ... p2", "show page 2 of a large result"),
        ])
        return

    g = Table.grid(padding=(0, 2))
    g.add_column(min_width=42)
    g.add_column(min_width=54)

    # fmt: off
    g.add_row("[bold]PORTFOLIO[/]",                                  "[bold]TRADING[/]")
    g.add_row("  [cyan]status[/]          overview",                 "  [cyan]thesis[/]     <SYM> <CONV> [--yes]")
    g.add_row("  [cyan]plays[/]           list plays",               "  [cyan]approach[/]    <SYM> [--yes]")
    g.add_row("  [cyan]plays[/] <row>     detail",                   "  [cyan]sentinel[/]    <SYM> [--yes]")
    g.add_row("  [cyan]orders[/]          live orders/trackers",      "  [cyan]sniper[/]      <SYM> <PRICE> [--yes]")
    g.add_row("  [cyan]pending[/]         alias for orders",          r"  [cyan]manual[/]      <CON> <QTY> <TYPE> \[CONV] \[SYM] [--yes]")
    g.add_row("  [cyan]positions[/]       IB positions + track cmd",  r"  [cyan]track[/]       <CON> <TYPE> \[SYM]")
    g.add_row("[bold]RECOVERY[/]",                                   r"  [cyan]close[/]       <row> \[QTY] [--yes]")
    g.add_row("  [cyan]cancel[/] row:<n>       preview",             r"  [cyan]spot[/]        <buy|sell> <SYM> <QTY> \[LMT] [--yes]")
    g.add_row("  [cyan]cancel[/] perm:<id> --yes",                   "")
    g.add_row("  [cyan]cancel-retry[/] native:<id> --yes",           "")
    g.add_row("  [cyan]cancel-all[/] [--yes]",                       "[bold]RESEARCH[/]")
    g.add_row("  [cyan]rebind[/]",                                   "  [cyan]chain[/] <SYM> [N|dte=45-90|exp=YYYYMMDD] [type]")
    g.add_row("  [cyan]clear-working[/] <row> entry|exit",           "  [cyan]scan[/] [--open] [--yes]")
    g.add_row("[bold]SYSTEM[/]",                                     "")
    g.add_row("  [cyan]run[/]             strategy cycle",            "  [cyan]help orders[/]    order/recovery help")
    g.add_row("  [cyan]cfg[/]             show parameters",           "  [cyan]help trading[/]   trading help")
    g.add_row("  [cyan]state[/]           plays.json + sync check",   "  [cyan]help chain[/]     chain research help")
    g.add_row("  [cyan]quit[/]            disconnect",                "")
    g.add_row(r"[dim]<> required  \[] optional[/]",                  r"[dim]order rows preview only; confirm with perm:/native:[/]")
    # fmt: on

    _con.print()
    _con.print(Panel(g, border_style="dim", padding=(1, 2)))


# ── state ───────────────────────────────────────────────────────────────────

def do_state(strat: Strategy, ib, args: list[str]):
    """Show persisted plays.json and flag any drift from in-memory state."""
    disk = state.read_raw(strat.account.account_id)

    if not disk and not strat.plays:
        print("  No plays on disk or in memory.\n")
        return

    # ── disk table ──
    t = Table(box=None, padding=(0, 1))
    t.add_column("#", justify="right", style="dim", no_wrap=True)
    t.add_column("ID", no_wrap=True, style="dim")
    t.add_column("TYPE", no_wrap=True)
    t.add_column("SYM", no_wrap=True)
    t.add_column("CON_ID", justify="right", no_wrap=True)
    t.add_column("QTY", justify="right", no_wrap=True)
    t.add_column("ENTRY", justify="right", no_wrap=True)
    t.add_column("STATUS", no_wrap=True)
    t.add_column("PEAK", justify="right", no_wrap=True)
    t.add_column("TRANCHE", justify="right", no_wrap=True)
    t.add_column("SPIKE", no_wrap=True)

    for i, d in enumerate(disk):
        t.add_row(
            str(i),
            str(d.get("play_id", "?")),
            d.get("play_type", "?"),
            d.get("symbol", "?"),
            str(d.get("con_id", "?")),
            f"{d.get('qty_open', '?')}/{d.get('qty_initial', '?')}",
            f"{d.get('entry_price', 0):.2f}",
            d.get("status", "?"),
            f"{d.get('peak_pnl_pct', 0):+.1%}",
            f"{d.get('tranche_idx', 0)}/{len(d.get('exit_profile', {}).get('tranches', []))}",
            "yes" if d.get("spike_fired") else "no",
        )

    _con.print()
    title = f"plays.json  ({len(disk)} entries)"
    if strat.account.account_id:
        title += f"  account={strat.account.account_id}"
    _con.print(Panel(t, title=title,
                     border_style="dim", padding=(1, 2)))

    # ── sync check: compare disk vs memory ──
    mem_keys = [
        (p.play_id, p.con_id, p.qty_open, p.status.value, p.tranche_idx, p.spike_fired)
        for p in strat.plays
    ]
    disk_keys = [
        (d.get("play_id"), d.get("con_id"), d.get("qty_open"), d.get("status"),
         d.get("tranche_idx", 0), d.get("spike_fired", False))
        for d in disk
    ]

    if mem_keys == disk_keys:
        print("  [green]Sync OK[/] — disk matches memory.\n")
    else:
        diffs = []
        max_len = max(len(mem_keys), len(disk_keys))
        for i in range(max_len):
            mk = mem_keys[i] if i < len(mem_keys) else None
            dk = disk_keys[i] if i < len(disk_keys) else None
            if mk != dk:
                sym = (strat.plays[i].symbol if i < len(strat.plays)
                       else disk[i].get("symbol", "?") if i < len(disk) else "?")
                diffs.append(f"  [{i}] {sym}:  disk={dk}  mem={mk}")

        _con.print(f"  [bold red]DRIFT DETECTED[/] — {len(diffs)} mismatch(es):")
        for line in diffs:
            print(line)
        print()


# ═════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═════════════════════════════════════════════════════════════════════════════

_CMD = {
    "status":   do_status,   "s": do_status,
    "plays":    do_plays,    "p": do_plays,
    "orders":   do_orders,   "o": do_orders,
    "pending":  do_pending,
    "positions": do_positions, "pos": do_positions,
    "cancel":   do_cancel,
    "cancel-retry": do_cancel_retry,
    "cancel-all": do_cancel_all,
    "rebind":   do_rebind,
    "clear-working": do_clear_working,
    "clear":    do_clear_working,
    "chain":    do_chain,    "c": do_chain,
    "scan":     do_scan,
    "thesis":   do_thesis,   "t": do_thesis,
    "approach": do_approach,
    "sentinel": do_sentinel,
    "sniper":   do_sniper,
    "manual":   do_manual,
    "track":    do_track,
    "close":    do_close,    "x": do_close,
    "spot":     do_spot,
    "run":      do_run,      "r": do_run,
    "cfg":      do_cfg,
    "state":    do_state,    "st": do_state,
    "help":     do_help,     "h": do_help, "?": do_help,
}


def _dispatch(raw: str, ib, strat: Strategy, stop: threading.Event):
    try:
        tokens = shlex.split(raw)
    except ValueError as e:
        _err("Could not parse command.", str(e))
        return
    if not tokens:
        return

    name   = tokens[0].lower()
    args   = tokens[1:]

    if name in ("quit", "exit", "q"):
        stop.set()
        return

    handler = _CMD.get(name)
    if handler:
        try:
            handler(strat, ib, args)
        except ValueError as e:
            _err("Invalid command argument.", str(e))
        except Exception as e:
            _err("Command failed.", str(e))
            if CFG.terminal.show_tracebacks:
                _con.print_exception()
    else:
        print(f"  Unknown command '{name}'. Type 'help' for help.")


# ═════════════════════════════════════════════════════════════════════════════
# CONSOLE READER
# ═════════════════════════════════════════════════════════════════════════════

def _read_console(q: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            stop.set()
            break
        if line:
            q.put(line)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("Connecting to IB Gateway…")
    ib = connect(CFG.ib_host, CFG.ib_port, CFG.ib_client_id)

    policy  = CashPolicy(risk_ceiling=CFG.risk_ceiling)
    scanner = SniperScanner(
        ib, watchlist=CFG.sniper_watchlist,
        drop_threshold=CFG.sniper_drop_pct,
    )
    strat = Strategy(
        ib                   = ib,
        policy               = policy,
        exit_profiles        = CFG.exit_profiles,
        contract_specs       = CFG.contract_specs,
        sniper_scanner       = scanner,
        thesis_max_nav_pct   = CFG.thesis_max_nav_pct,
        approach_max_nav_pct = CFG.approach_max_nav_pct,
        sentinel_max_nav_pct = CFG.sentinel_max_nav_pct,
        sniper_max_nav_pct   = CFG.sniper_max_nav_pct,
        scanner_interval_secs = CFG.scanner_interval,
        scanner_auto_open    = CFG.sniper_scanner_auto_open,
        entry_retry          = CFG.entry,
        patient_retry        = CFG.patient,
        urgent_retry         = CFG.urgent,
        base_currency        = CFG.base_currency,
        account_id           = CFG.account_id or None,
    )

    strat.plays = state.load(
        strat.account.snapshot().positions,
        account_id=strat.account.account_id,
    )
    startup_ctx = strat.context()
    startup_dirty = strat.restore_working_entries(startup_ctx)
    startup_dirty |= strat.restore_working_orders(startup_ctx)
    if startup_dirty:
        state.save(strat.plays, account_id=strat.account.account_id)

    stop = threading.Event()
    q: queue.Queue[str] = queue.Queue()
    console_enabled = sys.stdin.isatty()
    if console_enabled:
        threading.Thread(
            target=_read_console, args=(q, stop), daemon=True, name="console",
        ).start()
    else:
        print("[console] stdin is not interactive; command console disabled.")

    hint = "Type 'help' for commands." if console_enabled else "Running headless."
    print(f"\nReady  (risk loop every {CFG.loop_interval}s; scanner every {CFG.scanner_interval}s).  {hint}\n")

    try:
        strat.step()
    except Exception as e:
        print(f"  [loop] {e}")

    last = time.monotonic()

    while not stop.is_set():
        while not q.empty():
            try:
                _dispatch(q.get_nowait(), ib, strat, stop)
            except queue.Empty:
                break

        now = time.monotonic()
        if now - last >= CFG.loop_interval:
            try:
                strat.step()
            except Exception as e:
                print(f"  [loop] {e}")
            last = now

        try:
            if not ib.isConnected():
                raise ConnectionError("IB disconnected")
            ib.sleep(1)
        except (ConnectionError, OSError) as e:
            print(f"  [loop] Connection lost: {e} — attempting reconnect…")
            for attempt in range(1, 6):
                try:
                    time.sleep(min(5 * attempt, 30))
                    ib.disconnect()
                    ib.connect(CFG.ib_host, CFG.ib_port, clientId=CFG.ib_client_id)
                    reconnect_ctx = strat.context()
                    reconnect_dirty = strat.restore_working_entries(reconnect_ctx)
                    reconnect_dirty |= strat.restore_working_orders(reconnect_ctx)
                    if reconnect_dirty:
                        state.save(strat.plays, account_id=strat.account.account_id)
                    print(f"  [loop] Reconnected on attempt {attempt}")
                    break
                except Exception as re:
                    print(f"  [loop] Reconnect attempt {attempt} failed: {re}")
            else:
                print("  [loop] ⚠  Could not reconnect after 5 attempts — exiting")
                stop.set()

    print("\nShutting down…")
    ib.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    main()
