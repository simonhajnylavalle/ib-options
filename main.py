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
import threading
import time
from dataclasses import replace

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
import state
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
    rsk = _kvtable(
        ("Risk ceiling", f"{strat.policy.risk_ceiling:.0%}  {tag}"),
        ("Headroom",     _ds(risk.headroom())),
        ("Plays",        f"{', '.join(parts)}, {len(strat.plays)} total"),
        title="Risk",
    )
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

        spike_mark = " ⚡" if p.spike_fired else ""
        working_mark = ""
        if p.working_order:
            working_mark = " [exit]"
        t.add_row(
            str(i), p.play_type.value, p.symbol,
            f"{p.qty_open}/{p.qty_initial}",
            f"{p.entry_price:.2f}", now_s, pnl_s,
            p.status.value + spike_mark + working_mark,
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
    if p.working_order:
        pos_rows.append((
            "Exit order",
            f"{p.working_order.remaining_qty} working  "
            f"(attempt {p.working_order.attempts_used})",
        ))
    if ep.tranches:
        pos_rows.append(("Tranche", f"{p.tranche_idx}/{len(ep.tranches)}"))

    left = _kvtable(*pos_rows, title="Position")

    # ── right: exit rules ──
    exit_rows: list[tuple[str, str]] = [
        ("Stop loss", f"{ep.stop_loss_pct:+.0%}"),
        ("Full exit", f"{ep.full_exit_pct:+.0%}"),
    ]
    if ep.trailing_stop_pct is not None:
        exit_rows.append(("Trail stop", f"{ep.trailing_stop_pct:.0%} from peak"))
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
        ("Loop interval",      f"{CFG.loop_interval}s"),
        ("Risk ceiling",       f"{CFG.risk_ceiling:.0%}"),
        ("Approach max NAV",   f"{CFG.approach_max_nav_pct:.1%}"),
        ("Sentinel max NAV",   f"{CFG.sentinel_max_nav_pct:.1%}"),
        ("IB",                 f"{CFG.ib_host}:{CFG.ib_port}  clientId={CFG.ib_client_id}"),
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
        if ep_.trailing_stop_pct is None:
            return "─"
        return f"{ep_.trailing_stop_pct:.0%} from peak"

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

    p_modes, p_timing = _retry_row(CFG.patient)
    u_modes, u_timing = _retry_row(CFG.urgent)

    ex_t = _kvtable(
        ("Patient (entries, profit)",  f"{p_modes}  {p_timing}"),
        ("Urgent  (stop, trail)",      f"{u_modes}  {u_timing}"),
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


def do_chain(strat: Strategy, ib, args: list[str]):
    if not args:
        print("  Usage: chain <SYM> [N] [calls|puts] [thesis|approach|sentinel|sniper]")
        return
    symbol = args[0].upper()
    expiry_count = 3
    rights = None
    right_filter = None
    spec_type = None
    page = 1

    for a in args[1:]:
        low = a.lower()
        if low in ("calls", "call"):
            rights = [Right.CALL.value]
            right_filter = "C"
        elif low in ("puts", "put"):
            rights = [Right.PUT.value]
            right_filter = "P"
        elif low in _SPEC_NAMES:
            spec_type = _SPEC_NAMES[low]
        elif len(low) > 1 and low[0] == "p" and low[1:].isdigit():
            page = int(low[1:])
        else:
            try:
                expiry_count = int(a)
            except ValueError:
                print(f"  Unknown arg '{a}'")
                return

    cols = ["expiry", "dte", "strike", "bid", "ask", "mid",
            "spread_pct", "delta", "iv", "open_interest", "con_id"]

    try:
        chain = OptionChain(ib, symbol)

        if spec_type is not None:
            # ── spec-based: show what the strategy would pick ────────────
            spec = strat._contract_spec(PlayType(spec_type))
            if right_filter:
                spec = replace(spec, right=Right(right_filter))
            picks = chain.select(**spec.to_kwargs())
            spot  = chain.spot

            if picks.empty:
                print(f"  No contracts match {spec_type} spec for {symbol}.")
                return

            avail = [c for c in cols if c in picks.columns]
            title = (f"{spec_type}  {symbol}  spot=${spot:.2f}  "
                     f"Δ={spec.delta_min}–{spec.delta_max}  "
                     f"DTE={spec.dte_min}–{spec.dte_max}")
            _con.print()
            _chain_panel(picks, avail, title, page)

            top = picks.iloc[0]
            d = top["delta"]
            delta_s = f"{d:.2f}" if d is not None and d == d else "N/A"
            print(f"  Top: con_id={int(top['con_id'])}  "
                  f"strike={top['strike']}  DTE={top['dte']}  delta={delta_s}")
        else:
            # ── raw chain (unfiltered) ───────────────────────────────────
            df = chain.fetch(expiry_count=expiry_count, strike_width=0.25,
                             rights=rights)
            if df.empty:
                print("  No contracts found.")
                return

            avail = [c for c in cols if c in df.columns]
            sides = [("CALLS", "C"), ("PUTS", "P")]
            if right_filter:
                sides = [(s, r) for s, r in sides if r == right_filter]

            _con.print()
            for label, rv in sides:
                sub = df[df["right"] == rv].sort_values(["expiry", "strike"])
                if sub.empty:
                    continue
                _chain_panel(sub, avail, label, page)

    except Exception as e:
        print(f"  Error: {e}")


# ── thesis ──────────────────────────────────────────────────────────────────

def do_thesis(strat: Strategy, ib, args: list[str]):
    if len(args) < 2:
        print("  Usage: thesis <SYM> <low|med|high>")
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
    strat.open_thesis(sym, conv, right=Right.CALL)


# ── approach ────────────────────────────────────────────────────────────────

def do_approach(strat: Strategy, ib, args: list[str]):
    if not args:
        print("  Usage: approach <SYM>")
        return
    if len(args) >= 2 and args[1].lower() == "put":
        print("  PUT entries are disabled; this strategy is CALL-only for now.")
        return
    sym = args[0].upper()
    if strat._has_open_play(sym, PlayType.APPROACH):
        print(f"  Already have an open APPROACH on {sym}. Use 'close' first.")
        return
    strat.open_approach(sym, right=Right.CALL)


# ── sentinel ────────────────────────────────────────────────────────────────

def do_sentinel(strat: Strategy, ib, args: list[str]):
    if not args:
        print("  Usage: sentinel <SYM>")
        return
    if len(args) >= 2 and args[1].lower() == "put":
        print("  PUT entries are disabled; this strategy is CALL-only for now.")
        return
    sym = args[0].upper()
    if strat._has_open_play(sym, PlayType.SENTINEL):
        print(f"  Already have an open SENTINEL on {sym}. Use 'close' first.")
        return
    strat.open_sentinel(sym, right=Right.CALL)


# ── sniper ──────────────────────────────────────────────────────────────────

def do_sniper(strat: Strategy, ib, args: list[str]):
    if len(args) < 2:
        print("  Usage: sniper <SYM> <PRICE>")
        return
    try:
        spot = float(args[1])
    except ValueError:
        print("  spot_price must be a number.")
        return
    strat.open_sniper(args[0].upper(), spot, ctx=strat.context())


# ── manual ──────────────────────────────────────────────────────────────────

def do_manual(strat: Strategy, ib, args: list[str]):
    if len(args) < 3:
        print("  Usage: manual <CON_ID> <QTY> <thesis|approach|sentinel|sniper> [CONV] [SYM]")
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
    strat.open_manual(con_id=con_id, qty=qty, play_type=pt,
                      conviction=conviction, symbol=symbol)


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
    if not args:
        print("  Usage: close <row> [QTY]    (row # from plays listing)")
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
    if len(args) < 3:
        print("  Usage: spot <buy|sell> <SYM> <QTY> [LMT]")
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
    result = (strat.executor.buy_stock(sym, qty, limit) if d == "buy"
              else strat.executor.sell_stock(sym, qty, limit))
    desc = f"LMT {limit:.2f}" if limit else "MKT"
    print(f"  {d.upper()} {qty}x {sym}  {desc}  id={result.order_id}")


# ── scan ────────────────────────────────────────────────────────────────────

def do_scan(strat: Strategy, ib, args: list[str]):
    if not strat.scanner:
        print("  No scanner configured.")
        return
    print(f"  Scanning: {', '.join(strat.scanner.watchlist)}")
    hit = strat.scanner.scan()
    if hit:
        sym, spot = hit
        print(f"  Hit: {sym} @ {spot:.2f}")
        if not strat._has_open_play(sym, PlayType.SNIPER):
            strat.open_sniper(sym, spot, ctx=strat.context())
    else:
        print("  No qualifying drops.")


# ── pending ─────────────────────────────────────────────────────────────────

def do_pending(strat: Strategy, ib, args: list[str]):
    df = strat.executor.pending_orders()
    if df.empty:
        print("  No pending orders.\n")
        return
    _con.print()
    _con.print(Panel(df.to_string(index=False), title="Pending Orders",
                     border_style="dim", padding=(1, 2)))


# ── run ─────────────────────────────────────────────────────────────────────

def do_run(strat: Strategy, ib, args: list[str]):
    try:
        strat.step()
        print("  Tick complete.")
    except Exception as e:
        print(f"  Error: {e}")


# ── help ────────────────────────────────────────────────────────────────────

def do_help(strat: Strategy, ib, args: list[str]):
    g = Table.grid(padding=(0, 2))
    g.add_column(min_width=40)
    g.add_column(min_width=48)

    # fmt: off
    g.add_row("[bold]PORTFOLIO[/]",                                 "[bold]TRADING[/]")
    g.add_row("  [cyan]status[/]          overview",                "  [cyan]thesis[/]     <SYM> <CONV>")
    g.add_row("  [cyan]plays[/]           list plays",              "  [cyan]approach[/]    <SYM>")
    g.add_row("  [cyan]plays[/] <row>     detail",                  "  [cyan]sentinel[/]    <SYM>")
    g.add_row("  [cyan]pending[/]         open orders",             "  [cyan]sniper[/]      <SYM> <PRICE>")
    g.add_row("",                                                   r"  [cyan]manual[/]      <CON> <QTY> <TYPE> \[CONV] \[SYM]")
    g.add_row("",                                                   r"  [cyan]track[/]       <CON> <TYPE> \[SYM]")
    g.add_row("[bold]RESEARCH[/]",                                  r"  [cyan]close[/]       <row> \[QTY]")
    g.add_row("  [cyan]chain[/] <SYM>     option chain",            r"  [cyan]spot[/]        <buy|sell> <SYM> <QTY> \[LMT]")
    g.add_row(r"    \[N] \[calls|puts]",                            "")
    g.add_row(r"    \[thesis|approach|sentinel|sniper]",            "[bold]SYSTEM[/]")
    g.add_row("  [cyan]scan[/]            sniper scan",             "  [cyan]run[/]         strategy cycle")
    g.add_row("",                                                   "  [cyan]cfg[/]         show parameters")
    g.add_row(r"[dim]<> required  \[] optional[/]",                 "  [cyan]state[/]       plays.json + sync check")
    g.add_row("[dim]CONV   low · med · high[/]",                    "  [cyan]help[/]        this help")
    g.add_row("[dim]TYPE   thesis · approach · sentinel · sniper[/]","  [cyan]quit[/]        disconnect")
    g.add_row(r"[dim]<row>   play # from[/] [cyan]plays[/] [dim]listing[/]", "")
    g.add_row(r"[dim]\[N]     how many expiry dates[/]",            "")
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
    "pending":  do_pending,
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
    tokens = raw.split()
    name   = tokens[0].lower()
    args   = tokens[1:]

    if name in ("quit", "exit", "q"):
        stop.set()
        return

    handler = _CMD.get(name)
    if handler:
        try:
            handler(strat, ib, args)
        except Exception as e:
            print(f"  Error: {e}")
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
        approach_max_nav_pct = CFG.approach_max_nav_pct,
        sentinel_max_nav_pct = CFG.sentinel_max_nav_pct,
        sniper_max_nav_pct   = CFG.sniper_max_nav_pct,
        patient_retry        = CFG.patient,
        urgent_retry         = CFG.urgent,
        base_currency        = CFG.base_currency,
        account_id           = CFG.account_id or None,
    )

    strat.plays = state.load(
        strat.account.snapshot().positions,
        account_id=strat.account.account_id,
    )
    if strat.restore_working_orders(strat.context()):
        state.save(strat.plays, account_id=strat.account.account_id)

    stop = threading.Event()
    q: queue.Queue[str] = queue.Queue()
    threading.Thread(
        target=_read_console, args=(q, stop), daemon=True, name="console",
    ).start()

    print(f"\nReady  (loop every {CFG.loop_interval}s).  Type 'help' for commands.\n")

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
                    if strat.restore_working_orders(strat.context()):
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
