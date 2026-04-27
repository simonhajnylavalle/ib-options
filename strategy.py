"""
strategy.py
─────────────────────────────────────────────────────────────────────────────
Strategy layer.  Owns: play specs, entry logic, exit logic, position
tracking, and the main strategy loop.

─────────────────────────────────────────────────────────────────────────────
EXIT MODEL  (unified for THESIS + APPROACH + SENTINEL)
─────────────────────────────────────────────────────────────────────────────

THESIS, APPROACH, and SENTINEL plays use the same three-layer exit model:

  Layer 1 — Hard exits (always active)
    stop_loss, trailing_stop, DTE_floor, max_hold_days, full_exit_target

  Layer 2 — Velocity spike (fires once)
    If P&L *gains* ≥ spike_pct within spike_window_hours, sell an
    aggressive fraction immediately.  This captures the intuition that
    +100% in 4 hours is likely to revert and should be harvested, while
    +100% over 2 weeks justifies holding per normal tranches.

  Layer 3 — Tranches (fires in order)
    Each (trigger, fraction) pair fires once as P&L climbs.  THESIS
    tranches are wider (+50%, +100%); APPROACH tranches are tighter
    (+15%, +30%) because theta drag makes overstaying expensive.

SNIPER plays use a simpler binary exit: target or stop, forced close
after max_hold_days.  No tranches, no spike.

─────────────────────────────────────────────────────────────────────────────
CONFIGURATION
─────────────────────────────────────────────────────────────────────────────

All tunable parameters are loaded by config.py from config.toml
(or built-in defaults when no file exists).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import state

from ib_core import IB, Account, AccountSnapshot, OptionChain, Right, Stock, connect
from portfolio import CashPolicy, PortfolioRisk
from execution import Executor, OrderResult, OrderSide, PriceMode, RetryProfile


_MARKET_TZ = ZoneInfo("America/New_York")


def _market_now() -> datetime:
    return datetime.now(_MARKET_TZ)


def _market_date():
    return _market_now().date()


def _as_market_dt(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=_MARKET_TZ)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class PlayType(str, Enum):
    THESIS   = "THESIS"
    APPROACH = "APPROACH"
    SENTINEL = "SENTINEL"
    SNIPER   = "SNIPER"


class ConvictionLevel(float, Enum):
    LOW    = 0.25
    MEDIUM = 0.50
    HIGH   = 1.00


class PlayStatus(str, Enum):
    PENDING  = "PENDING"
    OPEN     = "OPEN"
    SCALING  = "SCALING"
    CLOSED   = "CLOSED"


# ─────────────────────────────────────────────────────────────────────────────
# SIZING DEFAULTS  (used when Strategy is instantiated directly without the
# config loader)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_THESIS_MAX_NAV_PCT:   float = 0.06
_DEFAULT_APPROACH_MAX_NAV_PCT: float = 0.025
_DEFAULT_SENTINEL_MAX_NAV_PCT: float = 0.010
_DEFAULT_SNIPER_MAX_NAV_PCT:   float = 0.010


# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT SPEC
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContractSpec:
    """
    Per-play-type contract selection — what to fetch from IB and how to
    rank the results.  All concrete values come from config.toml.

    Usage:
        chain.select(**spec.to_kwargs())
    """
    delta_min:         Optional[float] = None
    delta_max:         Optional[float] = None
    dte_min:           Optional[int]   = None
    dte_max:           Optional[int]   = None
    strike_width:      float           = 0.25
    right:             Optional[Right] = Right.CALL
    expiry_count:      int             = 5
    max_spread_pct:    float           = 15.0
    min_open_interest: int             = 50
    min_volume:        int             = 5

    def to_kwargs(self) -> dict:
        """Dict suitable for OptionChain.select(**spec.to_kwargs())."""
        from dataclasses import asdict
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# EXIT PROFILE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitProfile:
    """
    Declarative exit rules for one play.  Set at entry, never changed.

    Fields shared by all play types
    ────────────────────────────────
    stop_loss_pct      close 100% if P&L ≤ this
    full_exit_pct      close 100% if P&L ≥ this
    dte_floor          force-close if DTE ≤ this
    trail_activate_pct / trail_drawdown_pct
                       activate trailing only after a profit threshold, then
                       close on a separate drawdown threshold. The legacy
                       trailing_stop_pct field remains a backward-compatible
                       shorthand when both new fields are omitted.
    max_hold_days      force-close after N days; 0 = no limit

    Velocity spike (THESIS + APPROACH + SENTINEL)
    ────────────────────────────────
    spike_pct            P&L *gain within window* that triggers  (e.g. 1.0)
    spike_window_hours   lookback window                         (e.g. 6)
    spike_sell_ratio     fraction of qty_open to sell            (e.g. 0.50)
    All three must be > 0 for the spike exit to be active.

    Tranches (THESIS + APPROACH + SENTINEL)
    ─────────────────────────
    tranches  list of (trigger_pnl_pct, fraction_of_initial_qty) pairs.
    Fires one per cycle in order.  After all tranches: full_exit_pct closes
    whatever remains.

    THESIS example: [(0.50, 0.25), (1.00, 0.35)]
      +50% → sell 25%,  +100% → sell 35%,  +150% → close all

    APPROACH example:  [(0.20, 0.30), (0.40, 0.40)]
      +20% → sell 30%,  +40% → sell 40%,   +60% → close all
      (tighter targets — don't overstay theta drag)

    """
    stop_loss_pct:        float
    full_exit_pct:        float
    dte_floor:            int
    # Backward-compatible legacy field. Prefer trail_activate_pct and
    # trail_drawdown_pct in config.toml so activation and drawdown are not
    # accidentally tied to the same number.
    trailing_stop_pct:    Optional[float] = None
    trail_activate_pct:   Optional[float] = None
    trail_drawdown_pct:   Optional[float] = None
    tranches:             list[tuple[float, float]] = field(default_factory=list)
    max_hold_days:        int             = 0
    spike_pct:            float = 0.0
    spike_window_hours:   float = 0.0
    spike_sell_ratio:     float = 0.0

    def trail_activate(self) -> Optional[float]:
        return self.trail_activate_pct if self.trail_activate_pct is not None else self.trailing_stop_pct

    def trail_drawdown(self) -> Optional[float]:
        return self.trail_drawdown_pct if self.trail_drawdown_pct is not None else self.trailing_stop_pct


@dataclass
class WorkingEntry:
    """
    Async entry state for a buy order that may still be live after a blocking
    retry cycle returns.  Kept separate from qty_open: qty_open is only live
    IB position, while this object represents possible future fills.
    """
    trade_result: Optional[OrderResult]
    requested_qty: int
    remaining_qty: int
    attempts_used: int
    submitted_at: datetime
    retry_kind: str = "patient"
    reason: str = "entry"
    status: str = "WORKING"  # WORKING, UNBOUND, EXHAUSTED
    accounted_fills: int = 0
    cancel_requested: bool = False
    order_id: Optional[int] = None
    perm_id: Optional[int] = None
    native_order_id: Optional[int] = None
    client_id: Optional[int] = None
    account_id: Optional[str] = None
    side: str = "BUY"
    limit_px: Optional[float] = None
    submitted_qty: Optional[int] = None


@dataclass
class WorkingOrder:
    """
    Async exit state.

    trade_result is None after a restart until Strategy.restore_working_orders()
    rebinds this metadata to a live broker order.  UNBOUND/EXHAUSTED states
    deliberately keep the play blocked to avoid duplicate SELL-to-close orders.
    """
    trade_result: Optional[OrderResult]
    remaining_qty: int
    attempts_used: int
    submitted_at: datetime
    retry_kind: str
    reason: str
    status: str = "WORKING"  # WORKING, UNBOUND, EXHAUSTED
    order_id: Optional[int] = None
    perm_id: Optional[int] = None
    native_order_id: Optional[int] = None
    client_id: Optional[int] = None
    account_id: Optional[str] = None
    side: str = "SELL"
    limit_px: Optional[float] = None
    submitted_qty: Optional[int] = None
    accounted_fills: int = 0
    cancel_requested: bool = False
    reserved_tranche_idx: Optional[int] = None
    reserve_spike_fired: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# PLAY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Play:
    """
    A single tracked position from entry to full close.

    spike_fired — set True after the velocity spike exit fires.  Prevents
                  double-firing.  Persisted to disk.

    tranche_idx — next tranche index.  After a spike, advanced past any
                  tranches already below current P&L.
    """
    play_id:     str
    account_id:  str
    play_type:   PlayType
    symbol:      str
    con_id:      int
    qty_initial: int
    qty_open:    int

    entry_time:  datetime
    entry_price: float
    entry_nav:   float

    exit_profile: ExitProfile
    entry_time_known: bool = True

    status:       PlayStatus = PlayStatus.OPEN
    peak_pnl_pct: float      = 0.0
    pnl_history:  list       = field(default_factory=list)
    tranche_idx:  int        = 0
    spike_fired:  bool       = False

    # Session-only order history. working_order metadata *is* serialized, but
    # its live Trade ref is rebound from IB on startup/reconnect.
    orders: list[OrderResult] = field(default_factory=list)
    working_order: Optional[WorkingOrder] = None
    working_entry: Optional[WorkingEntry] = None

    def current_pnl_pct(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price

    def velocity_pct_per_hour(self, lookback_hours: float = 4.0) -> Optional[float]:
        if len(self.pnl_history) < 2:
            return None
        now    = _market_now()
        cutoff = now.timestamp() - lookback_hours * 3600
        recent = [
            (_as_market_dt(t), p)
            for t, p in self.pnl_history
            if _as_market_dt(t).timestamp() >= cutoff
        ]
        if len(recent) < 2:
            return None
        hours_elapsed = (recent[-1][0] - recent[0][0]).total_seconds() / 3600
        if hours_elapsed < 0.1:
            return None
        return (recent[-1][1] - recent[0][1]) / hours_elapsed

    def pnl_gain_in_window(self, window_hours: float) -> Optional[float]:
        """
        How much P&L *changed* in the last window_hours.
        Returns current_pnl − pnl_at_window_start.  None if insufficient data.

        This is the core spike measurement: it captures rapid intraday jumps
        while ignoring gains accumulated gradually over prior days.

        If we have data points straddling the window boundary, we linearly
        interpolate to get the PnL at exactly the cutoff time.  This avoids
        under-counting the gain when recording is sparse (e.g. IB was
        disconnected for hours and the first in-window point is very recent).
        """
        if len(self.pnl_history) < 2 or window_hours <= 0:
            return None
        now    = _market_now()
        cutoff = now.timestamp() - window_hours * 3600
        current_pnl = self.pnl_history[-1][1]

        # Single pass: find boundary points and count in-window data.
        before = None
        first_in_idx = None
        in_window_count = 0
        first_in_window_time = None
        for i, (t, p) in enumerate(self.pnl_history):
            t = _as_market_dt(t)
            ts = t.timestamp()
            if ts < cutoff:
                before = (t, p)
            else:
                if first_in_idx is None:
                    first_in_idx = i
                    first_in_window_time = t
                in_window_count += 1

        if first_in_idx is None:
            return None

        # Require minimum data density to avoid false spikes.
        if in_window_count < 3:
            return None
        last_time = _as_market_dt(self.pnl_history[-1][0])
        span_minutes = (last_time - first_in_window_time).total_seconds() / 60
        if span_minutes < 15:
            return None

        if before is not None:
            t0, p0 = before
            t1, p1 = self.pnl_history[first_in_idx]
            t1 = _as_market_dt(t1)
            dt = t1.timestamp() - t0.timestamp()
            if dt > 0:
                frac = (cutoff - t0.timestamp()) / dt
                boundary_pnl = p0 + frac * (p1 - p0)
            else:
                boundary_pnl = p0
        else:
            boundary_pnl = self.pnl_history[0][1]

        return current_pnl - boundary_pnl

    def hours_since_entry(self) -> float:
        return (_market_now() - _as_market_dt(self.entry_time)).total_seconds() / 3600

    _MAX_PNL_HISTORY: int = 600

    def record_pnl(self, pnl_pct: float) -> None:
        self.pnl_history.append((_market_now(), pnl_pct))
        if len(self.pnl_history) > self._MAX_PNL_HISTORY:
            self.pnl_history = self.pnl_history[-self._MAX_PNL_HISTORY:]
        if pnl_pct > self.peak_pnl_pct:
            self.peak_pnl_pct = pnl_pct

    def __repr__(self) -> str:
        return (
            f"Play({self.play_id} {self.play_type.value} {self.symbol} "
            f"qty={self.qty_open}/{self.qty_initial} "
            f"status={self.status.value})"
        )


@dataclass
class StrategyContext:
    """
    One-cycle cache so strategy helpers can share the same account snapshot,
    risk view, live positions, resolved contracts, and latest prices.
    """
    snapshot:            AccountSnapshot
    risk:                PortfolioRisk
    positions_by_con_id: dict[int, dict]
    contract_cache:      dict[int, object] = field(default_factory=dict)
    price_cache:         dict[int, Optional[float]] = field(default_factory=dict)

    def position(self, con_id: int) -> Optional[dict]:
        return self.positions_by_con_id.get(con_id)


# ─────────────────────────────────────────────────────────────────────────────
# SNIPER SCANNER
# ─────────────────────────────────────────────────────────────────────────────

class SniperScanner:
    def __init__(
        self, ib: IB, watchlist: list[str],
        drop_threshold: float = 0.15, min_volume_ratio: float = 1.5,
    ):
        self.ib               = ib
        self.watchlist        = [s.upper() for s in watchlist]
        self.drop_threshold   = drop_threshold
        self.min_volume_ratio = min_volume_ratio
        self._stock_cache: dict[str, Stock] = {}
        self._adv_cache: dict[tuple[str, object], Optional[float]] = {}

    def scan(self) -> Optional[tuple[str, float]]:
        # Request market data for all watchlist symbols in parallel,
        # then evaluate — one sleep instead of one per symbol.
        stocks = []
        tickers = {}
        for sym in self.watchlist:
            stock = self._stock_cache.get(sym)
            if stock is None:
                stock = Stock(sym, "SMART", "USD")
                self.ib.qualifyContracts(stock)
                self._stock_cache[sym] = stock
            tickers[sym] = (stock, self.ib.reqMktData(stock, ""))
            stocks.append(stock)
        self.ib.sleep(1.5)  # single wait covers all symbols

        try:
            for sym in self.watchlist:
                stock, t = tickers[sym]
                result = self._check_ticker(sym, stock, t)
                if result:
                    print(f"[SCANNER] SNIPER hit: {sym} spot={result[1]:.2f}")
                    return result
            return None
        finally:
            for stock in stocks:
                self.ib.cancelMktData(stock)

    def _check_ticker(
        self, symbol: str, stock: Stock, t,
    ) -> Optional[tuple[str, float]]:
        open_ = t.open if t.open and t.open > 0 else None
        last  = t.last if t.last and t.last > 0 else None
        if not open_ or not last:
            return None
        drop = (last - open_) / open_
        if drop > -self.drop_threshold:
            return None
        today_vol = t.volume if t.volume and t.volume > 0 else None
        if today_vol is None:
            return (symbol, last)
        avg_vol = self._average_daily_volume(stock, days=20)
        if avg_vol and avg_vol > 0:
            session_progress = self._session_progress()
            # Compare against the portion of average daily volume that would be
            # reasonable *so far* in the regular session, with a small floor so
            # the scanner still demands meaningful participation near the open.
            required_vol = avg_vol * self.min_volume_ratio * max(0.10, session_progress)
            if today_vol < required_vol:
                print(
                    f"[SCANNER] {symbol}: drop={drop:.1%} but "
                    f"volume too light ({today_vol:,} vs "
                    f"{required_vol:,.0f} required so far)"
                )
                return None
        return (symbol, last)

    def _session_progress(self, now: Optional[datetime] = None) -> float:
        now = _as_market_dt(now or _market_now())
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end   = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now <= start:
            return 0.0
        if now >= end:
            return 1.0
        return (now - start).total_seconds() / (end - start).total_seconds()

    def _average_daily_volume(self, stock, days: int = 20) -> Optional[float]:
        cache_key = (stock.symbol, _market_date())
        if cache_key in self._adv_cache:
            return self._adv_cache[cache_key]
        try:
            bars = self.ib.reqHistoricalData(
                stock, endDateTime="", durationStr=f"{days} D",
                barSizeSetting="1 day", whatToShow="TRADES",
                useRTH=True, formatDate=1,
            )
            if not bars:
                self._adv_cache[cache_key] = None
                return None
            vols = [b.volume for b in bars if b.volume and b.volume > 0]
            avg = (sum(vols) / len(vols)) if vols else None
            self._adv_cache[cache_key] = avg
            return avg
        except Exception as exc:
            print(f"[SCANNER] avg volume failed {stock.symbol}: {exc}")
            self._adv_cache[cache_key] = None
            return None


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

class Strategy:
    """
    Top-level orchestrator.

    exit_profiles     — {PlayType.value: ExitProfile} from config.
    contract_specs    — {PlayType.value: ContractSpec} from config.
    Both are built by config.py from config.toml so strategy.py has no
    magic numbers.  Falls back to for_*() defaults when no config is given.
    """

    # Default retry profiles when none are provided (e.g. standalone usage).
    _DEFAULT_ENTRY = RetryProfile(
        fill_timeout_secs=30, max_retries=5, mode=PriceMode.IB_MODEL,
        fallback_mode=PriceMode.MID, fallback_after=2, last_resort_mode=None,
    )
    _DEFAULT_PATIENT = RetryProfile(
        fill_timeout_secs=60, max_retries=29, mode=PriceMode.IB_MODEL,
        fallback_mode=PriceMode.MID, fallback_after=10, last_resort_mode=None,
    )
    _DEFAULT_URGENT = RetryProfile(
        fill_timeout_secs=60, max_retries=7, mode=PriceMode.IB_MODEL,
        fallback_mode=PriceMode.MID, fallback_after=3,
        last_resort_mode=PriceMode.NATURAL,
    )

    def __init__(
        self,
        ib:                IB,
        policy:            CashPolicy,
        exit_profiles:     Optional[dict[str, ExitProfile]] = None,
        contract_specs:    Optional[dict[str, ContractSpec]] = None,
        sniper_scanner:       Optional[SniperScanner] = None,
        thesis_max_nav_pct:   float = _DEFAULT_THESIS_MAX_NAV_PCT,
        approach_max_nav_pct: float = _DEFAULT_APPROACH_MAX_NAV_PCT,
        sentinel_max_nav_pct: float = _DEFAULT_SENTINEL_MAX_NAV_PCT,
        sniper_max_nav_pct:   float = _DEFAULT_SNIPER_MAX_NAV_PCT,
        scanner_interval_secs: int  = 300,
        pending_max_hours:    float = 24.0,
        entry_retry:          Optional[RetryProfile] = None,
        patient_retry:        Optional[RetryProfile] = None,
        urgent_retry:         Optional[RetryProfile] = None,
        base_currency:        str   = "CHF",
        account_id:          Optional[str] = None,
    ):
        self.ib                   = ib
        self.policy               = policy
        self.account              = Account(
            ib,
            base_currency=base_currency,
            account_id=account_id,
        )
        self.executor             = Executor(
            ib,
            account_id=self.account.account_id,
        )
        self.exit_profiles        = exit_profiles or {}
        self.contract_specs       = contract_specs or {}
        self.scanner              = sniper_scanner
        self.thesis_max_nav_pct   = thesis_max_nav_pct
        self.approach_max_nav_pct = approach_max_nav_pct
        self.sentinel_max_nav_pct = sentinel_max_nav_pct
        self.sniper_max_nav_pct   = sniper_max_nav_pct
        self.scanner_interval_secs = max(1, int(scanner_interval_secs))
        self._last_scan_at: Optional[datetime] = None
        self.pending_max_hours    = pending_max_hours
        self.entry_retry          = entry_retry or self._DEFAULT_ENTRY
        self.patient_retry        = patient_retry or self._DEFAULT_PATIENT
        self.urgent_retry         = urgent_retry  or self._DEFAULT_URGENT
        self.plays: list[Play]    = []

    def _exit_profile(self, play_type: PlayType) -> ExitProfile:
        return self.exit_profiles[play_type.value]

    def _contract_spec(self, play_type: PlayType) -> ContractSpec:
        return self.contract_specs[play_type.value]

    def context(self, snapshot: Optional[AccountSnapshot] = None) -> StrategyContext:
        snapshot = snapshot or self.account.snapshot()
        positions_by_con_id: dict[int, dict] = {}
        if not snapshot.positions.empty and "con_id" in snapshot.positions.columns:
            for _, row in snapshot.positions.iterrows():
                positions_by_con_id[int(row["con_id"])] = row.to_dict()
        risk = PortfolioRisk.from_snapshot(snapshot, self.policy)
        return StrategyContext(
            snapshot=snapshot,
            risk=risk,
            positions_by_con_id=positions_by_con_id,
        )

    def _active_play_by_con_id(self, con_id: int) -> Optional[Play]:
        for play in self.plays:
            if (
                play.con_id == con_id
                and play.status in (PlayStatus.OPEN, PlayStatus.SCALING, PlayStatus.PENDING)
            ):
                return play
        return None

    def _reject_duplicate_contract(self, con_id: int, symbol: str) -> bool:
        existing = self._active_play_by_con_id(con_id)
        if not existing:
            return False
        print(
            f"[STRATEGY] {symbol.upper()} con_id={con_id} is already tracked "
            f"by play {existing.play_id} ({existing.play_type.value}, {existing.status.value})"
        )
        return True

    def _resolve_contract(self, con_id: int, ctx: Optional[StrategyContext] = None):
        if ctx is not None and con_id in ctx.contract_cache:
            return ctx.contract_cache[con_id]
        contract = self.executor.resolve_con_id(con_id)
        if ctx is not None:
            ctx.contract_cache[con_id] = contract
        return contract

    def _pending_entry_capital(self) -> float:
        """Premium budget reserved by live/unresolved BUY orders.

        IB portfolio risk only sees filled positions. Async entry orders can
        still fill later, so reserve their remaining premium from future
        headroom to avoid over-committing the option sleeve.
        """
        total = 0.0
        for play in self.plays:
            we = getattr(play, "working_entry", None)
            if we is None or we.remaining_qty <= 0:
                continue
            px = we.limit_px or play.entry_price
            if px and px > 0:
                total += float(px) * 100 * int(we.remaining_qty)
        return total

    def _available_headroom(self, ctx: StrategyContext) -> float:
        return max(0.0, ctx.risk.headroom() - self._pending_entry_capital())

    def _entry_budget(self, ctx: StrategyContext, desired_capital: float) -> float:
        if desired_capital <= 0:
            return 0.0
        caps = [desired_capital, self._available_headroom(ctx)]
        if ctx.snapshot.cash > 0:
            caps.append(ctx.snapshot.cash)
        if ctx.snapshot.buying_power > 0:
            caps.append(ctx.snapshot.buying_power)
        return min(caps) if caps else 0.0

    def _entry_guard(self, ctx: StrategyContext, contract_currency: Optional[str] = None) -> bool:
        """Fail-closed preflight for new option entries."""
        if not self._risk_guard(ctx):
            return False
        if self._option_market_values_missing(ctx.snapshot.positions):
            print(
                "[STRATEGY] Option position market values are missing from IB; "
                "blocking new entries until portfolio values are reliable."
            )
            return False
        if contract_currency:
            snapshot_currency = str(ctx.snapshot.currency or "").upper()
            contract_currency = str(contract_currency or "").upper()
            if snapshot_currency and contract_currency and snapshot_currency != contract_currency:
                print(
                    f"[STRATEGY] ⚠ Account snapshot currency is {snapshot_currency}, "
                    f"but selected option is {contract_currency}. Sizing assumes the "
                    "snapshot currency is already appropriate for this contract."
                )
        return True

    @staticmethod
    def _option_market_values_missing(positions: pd.DataFrame) -> bool:
        if positions is None or positions.empty or "sec_type" not in positions.columns:
            return False
        opt_rows = positions[positions["sec_type"] == "OPT"]
        if opt_rows.empty:
            return False
        for _, row in opt_rows.iterrows():
            qty = abs(float(row.get("position", 0) or 0))
            if qty <= 0:
                continue
            mv = row.get("market_value")
            if mv is None or pd.isna(mv):
                return True
        return False

    @staticmethod
    def _row_is_call(row) -> bool:
        return str(row.get("right", "")).upper() == Right.CALL.value

    def _contract_is_call(self, contract) -> bool:
        return str(getattr(contract, "right", "")).upper() == Right.CALL.value

    def _reject_non_call_row(self, row, symbol: str, source: str) -> bool:
        if self._row_is_call(row):
            return False
        print(f"[STRATEGY] {source} {symbol.upper()}: only CALL options are supported for now")
        return True

    def _soft_contract_warnings(self, row, spec: ContractSpec) -> list[str]:
        """Make soft filter violations visible without rejecting the contract."""
        warnings: list[str] = []
        delta = row.get("delta")
        if delta is not None and delta == delta:
            abs_delta = abs(float(delta))
            if spec.delta_min is not None and abs_delta < spec.delta_min:
                warnings.append(f"delta {abs_delta:.2f} below target {spec.delta_min:.2f}")
            if spec.delta_max is not None and abs_delta > spec.delta_max:
                warnings.append(f"delta {abs_delta:.2f} above target {spec.delta_max:.2f}")
        spread_pct = row.get("spread_pct")
        if spread_pct is not None and spread_pct == spread_pct and spread_pct > spec.max_spread_pct:
            warnings.append(f"spread {float(spread_pct):.1f}% above target {spec.max_spread_pct:.1f}%")
        volume = row.get("volume")
        if volume is not None and volume == volume and float(volume) < spec.min_volume:
            warnings.append(f"volume {float(volume):.0f} below target {spec.min_volume}")
        oi = row.get("open_interest")
        if oi is not None and oi == oi and float(oi) < spec.min_open_interest:
            warnings.append(f"open interest {float(oi):.0f} below target {spec.min_open_interest}")
        return warnings

    def _print_soft_contract_warnings(self, row, spec: ContractSpec, symbol: str) -> None:
        warnings = self._soft_contract_warnings(row, spec)
        if warnings:
            print(f"[STRATEGY] {symbol.upper()} selected least-bad contract: " + "; ".join(warnings))

    # ── entry methods ─────────────────────────────────────────────────────────

    def open_thesis(
        self, symbol: str, conviction: ConvictionLevel,
        right: Right = Right.CALL,
        spec: Optional[ContractSpec] = None,
        exit_profile: Optional[ExitProfile] = None,
    ) -> Optional[Play]:
        if right is not Right.CALL:
            print("[STRATEGY] THESIS: only CALL options are supported for now")
            return None
        spec = replace(spec or self._contract_spec(PlayType.THESIS), right=right)
        return self._open_directional(
            play_type    = PlayType.THESIS,
            symbol       = symbol,
            conviction   = conviction,
            spec         = spec,
            exit_profile = exit_profile or self._exit_profile(PlayType.THESIS),
        )

    def _open_capped(
        self, play_type: PlayType, symbol: str,
        max_nav_pct: float, right: Right = Right.CALL,
        spec: Optional[ContractSpec] = None,
        exit_profile: Optional[ExitProfile] = None,
    ) -> Optional[Play]:
        if right is not Right.CALL:
            print(f"[STRATEGY] {play_type.value}: only CALL options are supported for now")
            return None
        def _size(ctx: StrategyContext, ask: float) -> int:
            desired = min(ctx.snapshot.nav * max_nav_pct, ctx.risk.headroom())
            max_usd = self._entry_budget(ctx, desired)
            return int(max_usd / (ask * 100)) if (max_usd > 0 and ask > 0) else 0
        spec = replace(spec or self._contract_spec(play_type), right=right)
        return self._open_entry(
            play_type=play_type, symbol=symbol, spec=spec,
            size_fn=_size, exit_profile=exit_profile,
        )

    def open_approach(
        self, symbol: str, right: Right = Right.CALL,
        spec: Optional[ContractSpec] = None,
        exit_profile: Optional[ExitProfile] = None,
    ) -> Optional[Play]:
        return self._open_capped(
            PlayType.APPROACH, symbol, self.approach_max_nav_pct,
            right=right, spec=spec, exit_profile=exit_profile,
        )

    def open_sentinel(
        self, symbol: str, right: Right = Right.CALL,
        spec: Optional[ContractSpec] = None,
        exit_profile: Optional[ExitProfile] = None,
    ) -> Optional[Play]:
        return self._open_capped(
            PlayType.SENTINEL, symbol, self.sentinel_max_nav_pct,
            right=right, spec=spec, exit_profile=exit_profile,
        )

    def open_sniper(
        self,
        symbol: str,
        spot_price: Optional[float] = None,
        spec: Optional[ContractSpec] = None,
        ctx: Optional[StrategyContext] = None,
    ) -> Optional[Play]:
        ctx = ctx or self.context()
        spec  = spec or self._contract_spec(PlayType.SNIPER)
        chain = OptionChain(self.ib, symbol)
        picks = chain.select(spot_price=spot_price, **spec.to_kwargs())
        if picks.empty:
            print(f"[STRATEGY] SNIPER {symbol}: no qualifying contract")
            return None
        row = picks.iloc[0]
        if self._reject_non_call_row(row, symbol, "SNIPER"):
            return None
        contract_currency = str(row.get("currency", self.executor.currency) or self.executor.currency)
        if not self._entry_guard(ctx, contract_currency):
            return None
        self._print_soft_contract_warnings(row, spec, symbol)
        con_id = int(row["con_id"])
        if self._reject_duplicate_contract(con_id, symbol):
            return None
        ask = float(row["ask"])
        desired = ctx.snapshot.nav * self.sniper_max_nav_pct
        max_usd = self._entry_budget(ctx, desired)
        qty = int(max_usd / (ask * 100)) if (max_usd > 0 and ask > 0) else 0
        if qty < 1:
            print(f"[STRATEGY] SNIPER {symbol}: insufficient headroom")
            return None
        result = self._submit_entry_order(con_id, qty)
        return self._make_play(
            play_type    = PlayType.SNIPER,
            symbol       = symbol.upper(),
            con_id       = con_id,
            qty          = qty,
            entry_price  = float(row["ask"]),
            entry_nav    = ctx.snapshot.nav,
            exit_profile = self._exit_profile(PlayType.SNIPER),
            order_result = result,
        )

    def open_manual(
        self, con_id: int, qty: int, play_type: PlayType,
        conviction: Optional[ConvictionLevel] = None, symbol: str = "",
        exit_profile: Optional[ExitProfile] = None,
        ctx: Optional[StrategyContext] = None,
    ) -> Optional[Play]:
        ctx = ctx or self.context()
        if qty < 1:
            print("[STRATEGY] manual entry qty must be at least 1")
            return None
        profile  = exit_profile or self._exit_profile(play_type)
        contract = self._resolve_contract(con_id, ctx)
        sym      = symbol.upper() or contract.symbol
        if not self._contract_is_call(contract):
            print(f"[STRATEGY] manual {sym}: only CALL options are supported for now")
            return None
        if not self._entry_guard(ctx, getattr(contract, "currency", self.executor.currency)):
            return None
        if self._reject_duplicate_contract(con_id, sym):
            return None
        bid, ask, last = self._quote_option_contract(contract)
        mid = ((bid + ask) / 2) if (bid is not None and ask is not None) else ask
        if mid is None:
            mid = last
        ask = ask or 0.0
        ref_px = mid if mid > 0 else ask
        if ref_px <= 0:
            print(f"[STRATEGY] manual {sym}: no valid price (market data unavailable)")
            return None
        if play_type in (PlayType.APPROACH, PlayType.SENTINEL, PlayType.SNIPER):
            cap = {
                PlayType.APPROACH: self.approach_max_nav_pct,
                PlayType.SENTINEL: self.sentinel_max_nav_pct,
                PlayType.SNIPER: self.sniper_max_nav_pct,
            }[play_type]
            max_usd = self._entry_budget(ctx, ctx.snapshot.nav * cap)
            qty_algo = int(max_usd / (ref_px * 100)) if (max_usd > 0 and ref_px > 0) else 0
            qty_use = min(qty, qty_algo)
        else:
            conv = conviction or ConvictionLevel.MEDIUM
            desired = ctx.snapshot.nav * self.thesis_max_nav_pct * conv.value
            max_usd = self._entry_budget(ctx, desired)
            qty_algo = int(max_usd / (ref_px * 100)) if (max_usd > 0 and ref_px > 0) else 0
            qty_use = min(qty, qty_algo)
        if qty_use < 1:
            print(f"[STRATEGY] manual {sym}: insufficient headroom")
            return None
        result = self._submit_entry_order(con_id, qty_use)
        return self._make_play(
            play_type=play_type, symbol=sym, con_id=con_id,
            qty=qty_use, entry_price=ref_px, entry_nav=ctx.snapshot.nav,
            exit_profile=profile, order_result=result,
        )

    def track_position(
        self,
        con_id: int,
        play_type: PlayType,
        symbol: str = "",
        exit_profile: Optional[ExitProfile] = None,
        ctx: Optional[StrategyContext] = None,
    ) -> Optional[Play]:
        ctx = ctx or self.context()
        pos_row = ctx.position(con_id)
        if pos_row is None:
            print(f"[STRATEGY] track {con_id}: no live position found in account")
            return None
        if str(pos_row.get("sec_type", "")).upper() != "OPT":
            print(f"[STRATEGY] track {con_id}: only option positions can become plays")
            return None
        if str(pos_row.get("right", "")).upper() != Right.CALL.value:
            print(f"[STRATEGY] track {con_id}: only CALL option positions are supported for now")
            return None
        raw_qty = float(pos_row.get("position", 0) or 0)
        if raw_qty < 0:
            print(
                f"[STRATEGY] track {con_id}: short option positions are not supported; "
                "automatic exits are SELL-to-close only."
            )
            return None
        if self._reject_duplicate_contract(con_id, symbol or str(pos_row.get('symbol', ''))):
            return None
        live_qty = int(raw_qty)
        if live_qty < 1:
            print(f"[STRATEGY] track {con_id}: live quantity is zero")
            return None
        avg_cost = float(pos_row.get("avg_cost", 0) or 0)
        entry_price = (
            avg_cost / 100
            if avg_cost > 0
            else self._price_from_position(pos_row) or 0.0
        )
        if entry_price <= 0:
            contract = self._resolve_contract(con_id, ctx)
            bid, ask, last = self._quote_option_contract(contract)
            entry_price = ((bid + ask) / 2) if (bid is not None and ask is not None) else (ask or last or 0.0)
        if entry_price <= 0:
            print(f"[STRATEGY] track {con_id}: could not determine a usable entry price")
            return None
        play = Play(
            play_id      = uuid4().hex[:12],
            account_id   = self.account.account_id or "",
            play_type    = play_type,
            symbol       = symbol.upper() or str(pos_row.get("symbol", "UNKNOWN")).upper(),
            con_id       = con_id,
            qty_initial  = live_qty,
            qty_open     = live_qty,
            entry_time   = _market_now(),
            entry_price  = entry_price,
            entry_nav    = ctx.snapshot.nav,
            entry_time_known = False,
            exit_profile = exit_profile or self._exit_profile(play_type),
            status       = PlayStatus.OPEN,
        )
        self.plays.append(play)
        state.save(self.plays, account_id=self.account.account_id)
        print(f"[STRATEGY] Tracked existing position as {play} (time-based exits disabled)")
        return play

    # ── main loop ─────────────────────────────────────────────────────────────

    def step(self) -> None:
        ctx = self.context()
        self._monitor_plays(ctx)
        if self.scanner and self._scanner_due():
            self._last_scan_at = _market_now()
            hit = self.scanner.scan()
            if hit:
                sym, spot = hit
                if not self._has_open_play(sym, PlayType.SNIPER):
                    self.open_sniper(sym, spot_price=spot, ctx=self.context())

    def _scanner_due(self) -> bool:
        if self._last_scan_at is None:
            return True
        elapsed = (_market_now() - self._last_scan_at).total_seconds()
        return elapsed >= self.scanner_interval_secs

    # ── private: risk guard ───────────────────────────────────────────────────

    def _risk_guard(self, ctx: StrategyContext) -> bool:
        risk = ctx.risk
        if risk.risk_status == "ABOVE_CEILING":
            print(
                f"[STRATEGY] Ceiling breached "
                f"({risk.risk_pct:.1%} > {self.policy.risk_ceiling:.1%}) "
                f"— no new entries"
            )
            return False
        available = self._available_headroom(ctx)
        if available < 100:
            reserved = self._pending_entry_capital()
            suffix = f" after reserving ${reserved:,.0f} for pending entries" if reserved > 0 else ""
            print(f"[STRATEGY] Headroom too small (${available:.0f}{suffix})")
            return False
        return True

    def _retry_kind(self, profile: RetryProfile) -> str:
        if profile is self.urgent_retry:
            return "urgent"
        if profile is self.entry_retry:
            return "entry"
        return "patient"

    def _profile_for_kind(self, retry_kind: str) -> RetryProfile:
        if retry_kind == "urgent":
            return self.urgent_retry
        if retry_kind == "entry":
            return self.entry_retry
        return self.patient_retry

    def _live_qty(self, pos_row: Optional[dict]) -> int:
        return max(0, int(self._signed_position(pos_row)))

    def _working_exit_pending(self, play: Play) -> bool:
        return play.working_order is not None

    def _effective_tranche_idx(self, play: Play) -> int:
        wo = play.working_order
        if wo is not None and wo.reserved_tranche_idx is not None:
            return max(play.tranche_idx, wo.reserved_tranche_idx)
        return play.tranche_idx

    def _effective_spike_fired(self, play: Play) -> bool:
        wo = play.working_order
        return bool(play.spike_fired or (wo is not None and wo.reserve_spike_fired))

    def _apply_exit_reservations(
        self,
        play: Play,
        reserved_tranche_idx: Optional[int] = None,
        reserve_spike_fired: bool = False,
    ) -> None:
        if reserve_spike_fired:
            play.spike_fired = True
        if reserved_tranche_idx is not None:
            play.tranche_idx = max(play.tranche_idx, reserved_tranche_idx)

    def _clear_working_order(self, play: Play, commit_reservations: bool = False) -> None:
        wo = play.working_order
        if wo is None:
            return
        if commit_reservations:
            self._apply_exit_reservations(
                play,
                reserved_tranche_idx=wo.reserved_tranche_idx,
                reserve_spike_fired=wo.reserve_spike_fired,
            )
        play.working_order = None

    @staticmethod
    def _order_identity(result: OrderResult) -> dict:
        return {
            "order_id": result.order_id,
            "perm_id": result.perm_id,
            "native_order_id": result.native_order_id,
            "client_id": result.client_id,
            "account_id": result.account_id,
            "limit_px": result.limit_px,
            "submitted_qty": result.qty,
        }

    def _stamp_single_attempt_result(self, result: OrderResult, requested_qty: int) -> OrderResult:
        """Populate cross-attempt fields for a non-blocking single attempt."""
        filled = int(result.filled_qty() or 0)
        result.total_filled = filled
        result.total_cost = sum(
            f.execution.shares * f.execution.price
            for f in getattr(result.trade, "fills", [])
        )
        result.requested_qty = int(requested_qty)
        result.unfilled_qty = max(0, int(requested_qty) - filled)
        result.last_order_live = self.executor.is_live(result)
        result.cancel_unresolved = result.last_order_live
        return result

    def _submit_entry_order(self, con_id: int, qty: int, attempt: int = 0) -> OrderResult:
        profile = self.entry_retry
        total_attempts = profile.max_retries + 1
        mode = self.executor.mode_for_attempt(
            attempt=attempt,
            total_attempts=total_attempts,
            mode=profile.mode,
            fallback_mode=profile.fallback_mode,
            fallback_after=profile.fallback_after,
            last_resort_mode=profile.last_resort_mode,
        )
        result = self.executor.submit_option_order(
            side=OrderSide.BUY,
            con_id=con_id,
            qty=qty,
            mode=mode,
        )
        return self._stamp_single_attempt_result(result, requested_qty=qty)

    def _working_entry_from_result(
        self,
        result: OrderResult,
        requested_qty: int,
        remaining_qty: int,
        attempts_used: int,
        accounted_fills: int = 0,
    ) -> WorkingEntry:
        return WorkingEntry(
            trade_result=result,
            requested_qty=requested_qty,
            remaining_qty=remaining_qty,
            attempts_used=attempts_used,
            submitted_at=_market_now(),
            retry_kind="entry",
            reason="entry",
            accounted_fills=accounted_fills,
            side=result.side.value,
            **self._order_identity(result),
        )

    def _working_order_from_result(
        self,
        result: OrderResult,
        remaining_qty: int,
        attempts_used: int,
        retry_kind: str,
        reason: str,
        accounted_fills: int = 0,
        reserved_tranche_idx: Optional[int] = None,
        reserve_spike_fired: bool = False,
    ) -> WorkingOrder:
        return WorkingOrder(
            trade_result=result,
            remaining_qty=remaining_qty,
            attempts_used=attempts_used,
            submitted_at=_market_now(),
            retry_kind=retry_kind,
            reason=reason,
            accounted_fills=accounted_fills,
            reserved_tranche_idx=reserved_tranche_idx,
            reserve_spike_fired=reserve_spike_fired,
            side=result.side.value,
            **self._order_identity(result),
        )

    @staticmethod
    def _signed_position(pos_row: Optional[dict]) -> float:
        if pos_row is None:
            return 0.0
        return float(pos_row.get("position", 0) or 0)

    def _sell_to_close_allowed(self, play: Play, ctx: Optional[StrategyContext] = None) -> bool:
        """Last-line guard before submitting any SELL-to-close option order."""
        ctx = ctx or self.context()
        pos_row = ctx.position(play.con_id)
        if pos_row is None:
            print(f"[STRATEGY] {play.symbol}: no live IB position; SELL exit blocked")
            return False
        if str(pos_row.get("sec_type", "")).upper() != "OPT":
            print(f"[STRATEGY] {play.symbol}: live position is not an option; SELL exit blocked")
            return False
        if str(pos_row.get("right", "")).upper() != Right.CALL.value:
            print(f"[STRATEGY] {play.symbol}: only CALL option exits are supported; SELL exit blocked")
            return False
        signed_qty = self._signed_position(pos_row)
        if signed_qty <= 0:
            print(
                f"[STRATEGY] {play.symbol}: live option quantity is {signed_qty:g}; "
                "SELL-to-close blocked to avoid increasing a short position"
            )
            return False
        if qty_limit := int(signed_qty):
            if play.qty_open > qty_limit:
                play.qty_open = qty_limit
        return True

    def _manual_tranche_reservation(
        self, play: Play, ctx: StrategyContext
    ) -> Optional[int]:
        ep = play.exit_profile
        if not ep.tranches or play.qty_open <= 0:
            return None
        current_price = self.price_for_play(play, ctx)
        if current_price is None:
            return None
        pnl_pct = play.current_pnl_pct(current_price)
        reserved_idx = play.tranche_idx
        while (
            reserved_idx < len(ep.tranches)
            and pnl_pct >= ep.tranches[reserved_idx][0]
        ):
            reserved_idx += 1
        return reserved_idx if reserved_idx > play.tranche_idx else None

    def manual_close(
        self, play: Play, qty: int, ctx: Optional[StrategyContext] = None
    ) -> tuple[bool, bool]:
        ctx = ctx or self.context()
        reason = f"manual close (qty={qty})"
        if qty >= play.qty_open:
            ok = self._close_play(play, play.qty_open, reason, ctx=ctx)
        else:
            ok = self._partial_close(
                play,
                qty,
                reason,
                reserved_tranche_idx=self._manual_tranche_reservation(play, ctx),
                ctx=ctx,
            )
        submitted = self._working_exit_pending(play)
        if ok:
            state.save(self.plays, account_id=self.account.account_id)
        return ok, submitted

    def _select_live_trade(self, candidates: list, tracker) -> Optional[object]:
        if not candidates:
            return None

        # Strong IDs first.
        if getattr(tracker, "perm_id", None):
            for trade in candidates:
                if int(getattr(trade.order, "permId", 0) or 0) == int(tracker.perm_id):
                    return trade
        if getattr(tracker, "native_order_id", None):
            for trade in candidates:
                if int(getattr(trade.order, "orderId", 0) or 0) == int(tracker.native_order_id):
                    return trade

        # Fallback: account + remaining quantity + newest native order ID.
        def _sort_key(trade) -> tuple[int, int, int]:
            account = getattr(trade.order, "account", None)
            account_penalty = 0 if (not tracker.account_id or account == tracker.account_id) else 1
            remaining = self.executor.remaining_qty_from_trade(trade)
            order_id = int(getattr(trade.order, "orderId", 0) or 0)
            return (account_penalty, abs(remaining - tracker.remaining_qty), -order_id)

        return sorted(candidates, key=_sort_key)[0]

    def _copy_order_identity_to_tracker(self, tracker, result: OrderResult) -> None:
        tracker.trade_result = result
        tracker.status = "WORKING"
        tracker.order_id = result.order_id
        tracker.perm_id = result.perm_id
        tracker.native_order_id = result.native_order_id
        tracker.client_id = result.client_id
        tracker.account_id = result.account_id
        tracker.side = result.side.value
        tracker.limit_px = result.limit_px
        tracker.submitted_qty = result.qty

    def _bind_working_order(
        self,
        play: Play,
        wo: WorkingOrder,
        live_trades_by_con_id: dict[int, list],
    ) -> bool:
        candidates = live_trades_by_con_id.get(play.con_id, [])
        trade = self._select_live_trade(candidates, wo)
        if trade is None:
            return False
        candidates.remove(trade)
        result = self.executor.result_from_trade(trade)
        self._copy_order_identity_to_tracker(wo, result)
        print(
            f"[STRATEGY] Restored live exit for {play.symbol} "
            f"(remaining {wo.remaining_qty}, order_id={wo.order_id})"
        )
        return True

    def _bind_working_entry(
        self,
        play: Play,
        we: WorkingEntry,
        live_trades_by_con_id: dict[int, list],
    ) -> bool:
        candidates = live_trades_by_con_id.get(play.con_id, [])
        trade = self._select_live_trade(candidates, we)
        if trade is None:
            return False
        candidates.remove(trade)
        result = self.executor.result_from_trade(trade)
        self._copy_order_identity_to_tracker(we, result)
        print(
            f"[STRATEGY] Restored live entry for {play.symbol} "
            f"(remaining {we.remaining_qty}, order_id={we.order_id})"
        )
        return True

    def restore_working_entries(self, ctx: Optional[StrategyContext] = None) -> bool:
        ctx = ctx or self.context()
        dirty = False
        live_trades_by_con_id: dict[int, list] = {}
        for trade in self.executor.live_trades(side=OrderSide.BUY):
            con_id = int(getattr(getattr(trade, "contract", None), "conId", 0) or 0)
            if not con_id:
                continue
            live_trades_by_con_id.setdefault(con_id, []).append(trade)

        for play in [p for p in self.plays if p.working_entry is not None]:
            we = play.working_entry
            if we is None or we.trade_result is not None:
                continue
            if self._bind_working_entry(play, we, live_trades_by_con_id):
                dirty = True
                continue

            pos_row = ctx.position(play.con_id)
            live_qty = self._live_qty(pos_row)
            if live_qty > play.qty_open:
                delta = live_qty - play.qty_open
                play.qty_open = live_qty
                play.qty_initial = max(play.qty_initial, live_qty)
                play.status = PlayStatus.OPEN
                we.remaining_qty = max(0, we.remaining_qty - delta)
                dirty = True
                print(f"[STRATEGY] {play.symbol} restored pending entry via live-position reconciliation")

            if we.remaining_qty <= 0:
                play.working_entry = None
            else:
                we.status = "UNBOUND"
                dirty = True
                print(
                    f"[STRATEGY] ⚠ Could not rebind pending entry for {play.symbol}; "
                    "leaving it blocked to avoid duplicate BUY."
                )
        return dirty

    def restore_working_orders(self, ctx: Optional[StrategyContext] = None) -> bool:
        ctx = ctx or self.context()
        dirty = False
        live_trades_by_con_id: dict[int, list] = {}
        for trade in self.executor.live_trades(side=OrderSide.SELL):
            con_id = int(getattr(getattr(trade, "contract", None), "conId", 0) or 0)
            if not con_id:
                continue
            live_trades_by_con_id.setdefault(con_id, []).append(trade)

        for play in [p for p in self.plays if p.working_order is not None]:
            wo = play.working_order
            if wo is None or wo.trade_result is not None:
                continue
            if self._bind_working_order(play, wo, live_trades_by_con_id):
                dirty = True
                continue

            pos_row = ctx.position(play.con_id)
            live_qty = self._live_qty(pos_row)
            had_fill = False
            if live_qty < play.qty_open:
                delta = play.qty_open - live_qty
                play.qty_open = live_qty
                wo.remaining_qty = max(0, wo.remaining_qty - delta)
                play.status = PlayStatus.CLOSED if live_qty <= 0 else PlayStatus.SCALING
                had_fill = True

            if play.qty_open <= 0 or wo.remaining_qty <= 0:
                self._clear_working_order(
                    play,
                    commit_reservations=had_fill or wo.accounted_fills > 0 or wo.remaining_qty <= 0,
                )
                print(
                    f"[STRATEGY] {play.symbol} restored pending exit via live-position reconciliation"
                )
            else:
                wo.status = "UNBOUND"
                # Keep the working order attached. This blocks automatic duplicate SELLs
                # unless the operator explicitly clears the condition later.
                print(
                    f"[STRATEGY] ⚠ Could not rebind pending exit for {play.symbol}; "
                    "leaving it blocked to avoid duplicate SELL."
                )
            dirty = True
        return dirty

    def _advance_working_entries(self, ctx: StrategyContext) -> bool:
        dirty = False
        now = _market_now()

        def _apply_entry_fill(play: Play, we: WorkingEntry, delta: int, fill_px: Optional[float]) -> None:
            old_qty = max(0, play.qty_open)
            new_qty = old_qty + delta
            if new_qty > 0 and fill_px and fill_px > 0:
                play.entry_price = (
                    ((play.entry_price * old_qty) + (float(fill_px) * delta)) / new_qty
                    if old_qty > 0 else float(fill_px)
                )
            play.qty_open = new_qty
            play.qty_initial = max(play.qty_initial + delta, new_qty)
            play.status = PlayStatus.OPEN
            we.remaining_qty = max(0, we.remaining_qty - delta)

        for play in [p for p in self.plays if p.working_entry is not None]:
            we = play.working_entry
            if we is None:
                continue

            if we.trade_result is None:
                # UNBOUND/EXHAUSTED entries remain attached intentionally; this
                # blocks duplicate BUYs until the operator verifies the broker state.
                continue

            trade_result = we.trade_result
            trade_filled = trade_result.filled_qty()
            if trade_filled > we.accounted_fills:
                delta = trade_filled - we.accounted_fills
                fill_px = trade_result.avg_fill() or trade_result.limit_px or play.entry_price
                _apply_entry_fill(play, we, delta, fill_px)
                we.accounted_fills = trade_filled
                dirty = True

            pos_row = ctx.position(play.con_id)
            live_qty = self._live_qty(pos_row)
            if live_qty > play.qty_open:
                delta = live_qty - play.qty_open
                _apply_entry_fill(play, we, delta, trade_result.avg_fill() or play.entry_price)
                dirty = True

            if we.remaining_qty <= 0:
                play.working_entry = None
                if play.qty_open <= 0:
                    play.qty_open = 0
                    play.qty_initial = 0
                    play.status = PlayStatus.CLOSED
                    print(f"[STRATEGY] {play.symbol} PENDING entry ended with no fill → CLOSED")
                else:
                    play.status = PlayStatus.OPEN
                dirty = True
                continue

            profile = self._profile_for_kind(we.retry_kind)
            total_attempts = profile.max_retries + 1

            if self.executor.is_live(trade_result):
                elapsed = (now - we.submitted_at).total_seconds()
                if elapsed >= profile.fill_timeout_secs and not we.cancel_requested:
                    self.executor.cancel(trade_result)
                    we.cancel_requested = True
                    dirty = True
                continue

            if we.attempts_used >= total_attempts:
                if play.qty_open <= 0:
                    play.qty_open = 0
                    play.qty_initial = 0
                    play.status = PlayStatus.CLOSED
                    play.working_entry = None
                    print(
                        f"[STRATEGY] {play.symbol} entry exhausted "
                        f"after {total_attempts} attempts with no fill → CLOSED"
                    )
                else:
                    play.working_entry = None
                    play.status = PlayStatus.OPEN
                    print(
                        f"[STRATEGY] ⚠ {play.symbol} entry partially filled "
                        f"({play.qty_open}/{we.requested_qty}); remaining entry exhausted"
                    )
                dirty = True
                continue

            next_attempt = we.attempts_used
            try:
                result = self._submit_entry_order(
                    con_id=play.con_id,
                    qty=we.remaining_qty,
                    attempt=next_attempt,
                )
            except Exception as exc:
                print(
                    f"[STRATEGY] ⚠ ENTRY retry failed for {play.symbol}: {exc}. "
                    "Automatic re-submit blocked."
                )
                we.status = "EXHAUSTED"
                we.trade_result = None
                dirty = True
                continue

            play.orders.append(result)
            play.working_entry = self._working_entry_from_result(
                result=result,
                requested_qty=we.requested_qty,
                remaining_qty=we.remaining_qty,
                attempts_used=next_attempt + 1,
                accounted_fills=0,
            )
            dirty = True
            print(
                f"[STRATEGY] ENTRY retry {play.symbol} "
                f"{play.working_entry.attempts_used}/{total_attempts}  "
                f"qty={play.working_entry.remaining_qty}  status={trade_result.status()}"
            )
        return dirty

    def _advance_working_orders(self, ctx: StrategyContext) -> bool:
        dirty = False
        now = _market_now()
        for play in [p for p in self.plays if self._working_exit_pending(p)]:
            wo = play.working_order
            if wo is None or wo.trade_result is None:
                continue

            trade_result = wo.trade_result
            trade_filled = trade_result.filled_qty()
            if trade_filled > wo.accounted_fills:
                delta = min(trade_filled - wo.accounted_fills, play.qty_open)
                wo.accounted_fills = trade_filled
                if delta > 0:
                    play.qty_open -= delta
                    wo.remaining_qty = max(0, wo.remaining_qty - delta)
                    play.status = (
                        PlayStatus.CLOSED if play.qty_open <= 0 else PlayStatus.SCALING
                    )
                    self._apply_exit_reservations(
                        play,
                        reserved_tranche_idx=wo.reserved_tranche_idx,
                        reserve_spike_fired=wo.reserve_spike_fired,
                    )
                    dirty = True

            pos_row = ctx.position(play.con_id)
            live_qty = self._live_qty(pos_row)
            if live_qty < play.qty_open:
                delta = play.qty_open - live_qty
                play.qty_open = live_qty
                wo.remaining_qty = max(0, wo.remaining_qty - delta)
                play.status = PlayStatus.CLOSED if live_qty <= 0 else PlayStatus.SCALING
                self._apply_exit_reservations(
                    play,
                    reserved_tranche_idx=wo.reserved_tranche_idx,
                    reserve_spike_fired=wo.reserve_spike_fired,
                )
                dirty = True

            if play.qty_open <= 0 or wo.remaining_qty <= 0:
                play.qty_open = max(0, play.qty_open)
                play.status = PlayStatus.CLOSED if play.qty_open == 0 else PlayStatus.SCALING
                self._clear_working_order(
                    play,
                    commit_reservations=wo.accounted_fills > 0 or wo.remaining_qty <= 0,
                )
                dirty = True
                continue

            profile = self._profile_for_kind(wo.retry_kind)
            total_attempts = profile.max_retries + 1
            status = trade_result.status()

            if self.executor.is_live(trade_result):
                elapsed = (now - wo.submitted_at).total_seconds()
                if elapsed >= profile.fill_timeout_secs and not wo.cancel_requested:
                    self.executor.cancel(trade_result)
                    wo.cancel_requested = True
                    dirty = True
                continue

            if wo.attempts_used >= total_attempts:
                print(
                    f"[STRATEGY] ⚠  EXIT unresolved for {play.symbol}: "
                    f"{wo.remaining_qty} contract(s) still open after {total_attempts} attempts. "
                    "Automatic re-submit blocked; intervene manually."
                )
                wo.status = "EXHAUSTED"
                wo.trade_result = None
                dirty = True
                continue

            next_mode = self.executor.mode_for_attempt(
                attempt=wo.attempts_used,
                total_attempts=total_attempts,
                mode=profile.mode,
                fallback_mode=profile.fallback_mode,
                fallback_after=profile.fallback_after,
                last_resort_mode=profile.last_resort_mode,
            )
            try:
                result = self.executor.submit_option_order(
                    side=OrderSide.SELL,
                    con_id=play.con_id,
                    qty=wo.remaining_qty,
                    mode=next_mode,
                )
            except Exception as exc:
                print(
                    f"[STRATEGY] ⚠  EXIT retry failed for {play.symbol}: {exc}. "
                    "Automatic re-submit blocked."
                )
                wo.status = "EXHAUSTED"
                wo.trade_result = None
                dirty = True
                continue

            play.orders.append(result)
            play.working_order = self._working_order_from_result(
                result=result,
                remaining_qty=wo.remaining_qty,
                attempts_used=wo.attempts_used + 1,
                retry_kind=wo.retry_kind,
                reason=wo.reason,
                reserved_tranche_idx=wo.reserved_tranche_idx,
                reserve_spike_fired=wo.reserve_spike_fired,
            )
            dirty = True
            print(
                f"[STRATEGY] EXIT retry {play.symbol} "
                f"{play.working_order.attempts_used}/{total_attempts}  "
                f"qty={play.working_order.remaining_qty}  status={status}"
            )
        return dirty

    # ── private: monitoring ───────────────────────────────────────────────────

    def _monitor_plays(self, ctx: StrategyContext) -> None:
        dirty = self.restore_working_entries(ctx)
        dirty |= self.restore_working_orders(ctx)
        dirty |= self._advance_working_entries(ctx)
        dirty |= self._advance_working_orders(ctx)

        # promote PENDING or expire stale ones
        for play in [p for p in self.plays if p.status == PlayStatus.PENDING]:
            pos_row = ctx.position(play.con_id)
            if pos_row is not None:
                live_qty = self._live_qty(pos_row)
                avg_cost = float(pos_row.get("avg_cost", 0) or 0)
                if live_qty > 0:
                    play.qty_initial = live_qty
                    play.qty_open    = live_qty
                    play.status      = PlayStatus.OPEN
                    if avg_cost > 0:
                        # IB avgCost for options = price × multiplier (100);
                        # entry_price must be per-share for PnL calculation.
                        play.entry_price = avg_cost / 100
                    dirty = True
                    print(
                        f"[STRATEGY] {play.symbol} PENDING→OPEN "
                        f"(qty={live_qty} @ {play.entry_price:.2f})"
                    )
                    continue

            # No IB position — check if PENDING has exceeded max age.
            if play.hours_since_entry() >= self.pending_max_hours:
                if play.working_entry is not None:
                    if play.working_entry.status != "EXHAUSTED":
                        play.working_entry.status = "EXHAUSTED"
                        dirty = True
                        print(
                            f"[STRATEGY] ⚠  {play.symbol} pending entry unresolved "
                            f"after {self.pending_max_hours:.0f}h; keeping play blocked "
                            "until the operator verifies/clears the live order"
                        )
                    continue
                play.qty_open = 0
                play.status   = PlayStatus.CLOSED
                dirty = True
                print(
                    f"[STRATEGY] ⚠  {play.symbol} PENDING→CLOSED "
                    f"(no fill after {self.pending_max_hours:.0f}h)"
                )

        # evaluate exits
        for play in [
            p for p in self.plays
            if (
                p.status in (PlayStatus.OPEN, PlayStatus.SCALING)
                and not self._working_exit_pending(p)
            )
        ]:
            dirty |= self._evaluate_play(play, ctx)

        if dirty:
            state.save(self.plays, account_id=self.account.account_id)

    def _evaluate_play(self, play: Play, ctx: StrategyContext) -> bool:
        """
        Exit decision tree (priority order).  Returns True if any state was
        mutated (caller is responsible for a single state.save).

          1. DTE floor              → urgent full close
          2. Stop loss              → urgent full close
          3. Trailing stop          → urgent full close
          4. Max hold days          → full close
          5. VELOCITY SPIKE         → aggressive partial (once only)
          6. Full exit target       → full close
          7. Tranches (THESIS+APPROACH+SENTINEL)→ partial (one per cycle)
        """
        dirty = False
        pos_row = ctx.position(play.con_id)
        if pos_row is None:
            play.qty_open = 0
            play.status = PlayStatus.CLOSED
            self._clear_working_order(play)
            print(f"[STRATEGY] {play.symbol} vanished from IB → CLOSED")
            return True

        signed_qty = self._signed_position(pos_row)
        if signed_qty < 0:
            play.qty_open = 0
            play.status = PlayStatus.CLOSED
            self._clear_working_order(play)
            print(
                f"[STRATEGY] ⚠ {play.symbol} con_id={play.con_id} is short at IB. "
                "Automatic SELL exits disabled; play closed locally. Manage manually."
            )
            return True
        if signed_qty == 0:
            play.qty_open = 0
            play.status = PlayStatus.CLOSED
            self._clear_working_order(play)
            print(f"[STRATEGY] {play.symbol} position is zero at IB → CLOSED")
            return True
        if str(pos_row.get("right", "")).upper() != Right.CALL.value:
            play.qty_open = 0
            play.status = PlayStatus.CLOSED
            self._clear_working_order(play)
            print(
                f"[STRATEGY] ⚠ {play.symbol} con_id={play.con_id} is not a CALL. "
                "Automatic exits disabled; play closed locally."
            )
            return True

        live_qty = int(signed_qty)
        if live_qty != play.qty_open:
            print(f"[STRATEGY] {play.symbol} qty {play.qty_open}→{live_qty} (live sync)")
            play.qty_open = live_qty
            play.status = PlayStatus.OPEN if live_qty >= play.qty_initial else PlayStatus.SCALING
            dirty = True

        current_price = (
            self._price_from_position(pos_row)
            or self._price_from_market(play, ctx)
        )
        if current_price is None:
            # Position exists but no price data — still check time-based exits
            # such as DTE floor. Skip price-based exits until pricing recovers.
            ep = play.exit_profile
            qty = play.qty_open
            expiry_str = str(pos_row.get("expiry") or "")
            if expiry_str:
                try:
                    exp_date = datetime.strptime(expiry_str, "%Y%m%d").date()
                    dte = (exp_date - _market_date()).days
                    if dte <= ep.dte_floor:
                        return self._close_play(
                            play, qty,
                            f"DTE floor ({dte} ≤ {ep.dte_floor}), no price data",
                            retry=self.urgent_retry,
                            ctx=ctx,
                        ) or dirty
                except ValueError:
                    pass
            if (
                play.entry_time_known
                and ep.max_hold_days > 0
                and play.hours_since_entry() / 24 >= ep.max_hold_days
            ):
                return self._close_play(
                    play, qty, f"max hold ({ep.max_hold_days}d), no price data", ctx=ctx
                ) or dirty
            return dirty

        pnl_pct = play.current_pnl_pct(current_price)
        old_peak = play.peak_pnl_pct
        old_len = len(play.pnl_history)
        play.record_pnl(pnl_pct)
        dirty = dirty or play.peak_pnl_pct != old_peak or len(play.pnl_history) != old_len
        ep = play.exit_profile
        qty = play.qty_open

        # ── 1. DTE floor ─────────────────────────────────────────────────
        expiry_str = str(pos_row.get("expiry") or "")
        if expiry_str:
            try:
                exp_date = datetime.strptime(expiry_str, "%Y%m%d").date()
                dte = (exp_date - _market_date()).days
                if dte <= ep.dte_floor:
                    return self._close_play(
                        play, qty, f"DTE floor ({dte} ≤ {ep.dte_floor})",
                        retry=self.urgent_retry,
                        ctx=ctx,
                    ) or dirty
            except ValueError:
                pass

        # ── 2. Stop loss ─────────────────────────────────────────────────
        if pnl_pct <= ep.stop_loss_pct:
            return self._close_play(
                play, qty, f"stop loss ({pnl_pct:.1%})",
                retry=self.urgent_retry,
                ctx=ctx,
            ) or dirty

        # ── 3. Trailing stop ─────────────────────────────────────────────
        trail_activate = ep.trail_activate()
        trail_drawdown = ep.trail_drawdown()
        if (
            trail_activate is not None
            and trail_drawdown is not None
            and play.peak_pnl_pct >= trail_activate
        ):
            dd = play.peak_pnl_pct - pnl_pct
            if dd >= trail_drawdown:
                return self._close_play(
                    play, qty,
                    f"trailing stop (peak={play.peak_pnl_pct:.1%} "
                    f"now={pnl_pct:.1%} dd={dd:.1%})",
                    retry=self.urgent_retry,
                    ctx=ctx,
                ) or dirty

        # ── 4. Max hold days ─────────────────────────────────────────────
        if play.entry_time_known and ep.max_hold_days > 0:
            if play.hours_since_entry() / 24 >= ep.max_hold_days:
                return self._close_play(play, qty, f"max hold ({ep.max_hold_days}d)", ctx=ctx) or dirty

        effective_tranche_idx = self._effective_tranche_idx(play)
        effective_spike_fired = self._effective_spike_fired(play)

        # ── 5. VELOCITY SPIKE EXIT ───────────────────────────────────────
        if (
            play.entry_time_known
            and not effective_spike_fired
            and ep.spike_pct > 0
            and ep.spike_window_hours > 0
            and ep.spike_sell_ratio > 0
            and play.qty_open > 1
            and len(play.pnl_history) >= 3
        ):
            gain = play.pnl_gain_in_window(ep.spike_window_hours)
            if gain is not None and gain >= ep.spike_pct:
                sell_qty = min(
                    max(1, round(play.qty_open * ep.spike_sell_ratio)),
                    play.qty_open - 1,
                )
                if sell_qty >= 1:
                    reserved_tranche_idx = None
                    if ep.tranches:
                        reserved_tranche_idx = effective_tranche_idx
                        while (
                            reserved_tranche_idx < len(ep.tranches)
                            and pnl_pct >= ep.tranches[reserved_tranche_idx][0]
                        ):
                            reserved_tranche_idx += 1
                        if reserved_tranche_idx <= play.tranche_idx:
                            reserved_tranche_idx = None
                    return self._partial_close(
                        play,
                        sell_qty,
                        f"SPIKE (+{gain:.0%} in <{ep.spike_window_hours:.0f}h, "
                        f"sell {ep.spike_sell_ratio:.0%} = {sell_qty}/{play.qty_open})",
                        reserved_tranche_idx=reserved_tranche_idx,
                        reserve_spike_fired=True,
                        ctx=ctx,
                    ) or dirty

        # ── 6. Full exit target ──────────────────────────────────────────
        if pnl_pct >= ep.full_exit_pct:
            return self._close_play(play, qty, f"full target ({pnl_pct:.1%})", ctx=ctx) or dirty

        # ── 7. Tranche-based scale-out ─────────────────────────────────
        if (
            ep.tranches
            and effective_tranche_idx < len(ep.tranches)
            and play.qty_open > 1
        ):
            trigger, fraction = ep.tranches[effective_tranche_idx]
            if pnl_pct >= trigger:
                sell_qty = min(
                    max(1, round(play.qty_initial * fraction)),
                    play.qty_open - 1,
                )
                return self._partial_close(
                    play,
                    sell_qty,
                    f"tranche {effective_tranche_idx+1}/{len(ep.tranches)} "
                    f"({pnl_pct:.1%} ≥ {trigger:.0%}, "
                    f"sell {fraction:.0%} of initial)",
                    reserved_tranche_idx=effective_tranche_idx + 1,
                    ctx=ctx,
                ) or dirty

        return dirty

    # ── private: order helpers ────────────────────────────────────────────────

    def _execute_exit(
        self,
        play: Play,
        qty: int,
        reason: str,
        retry: Optional[RetryProfile] = None,
        label: str = "EXIT",
        reserved_tranche_idx: Optional[int] = None,
        reserve_spike_fired: bool = False,
        ctx: Optional[StrategyContext] = None,
    ) -> bool:
        profile = retry or self.patient_retry
        if qty < 1:
            print(f"[STRATEGY] {label} {play.symbol}: qty must be at least 1")
            return False
        if self._working_exit_pending(play):
            print(f"[STRATEGY] {label} already working for {play.symbol}; skipping duplicate request")
            return True
        if not self._sell_to_close_allowed(play, ctx):
            return False
        qty = min(qty, play.qty_open)
        if qty < 1:
            print(f"[STRATEGY] {label} {play.symbol}: no live long quantity left to sell")
            return False

        total_attempts = profile.max_retries + 1
        first_mode = self.executor.mode_for_attempt(
            attempt=0,
            total_attempts=total_attempts,
            mode=profile.mode,
            fallback_mode=profile.fallback_mode,
            fallback_after=profile.fallback_after,
            last_resort_mode=profile.last_resort_mode,
        )
        print(f"[STRATEGY] {label} {play}  qty={qty}  reason={reason}")
        try:
            result = self.executor.submit_option_order(
                side=OrderSide.SELL,
                con_id=play.con_id,
                qty=qty,
                mode=first_mode,
            )
        except Exception as e:
            print(f"[STRATEGY] ⚠  {label} failed: {e}")
            return False

        play.orders.append(result)
        filled = min(result.filled_qty(), play.qty_open)
        play.qty_open -= filled
        if play.qty_open <= 0:
            play.qty_open = 0
            play.status = PlayStatus.CLOSED
        elif filled > 0:
            play.status = PlayStatus.SCALING

        if filled > 0:
            self._apply_exit_reservations(
                play,
                reserved_tranche_idx=reserved_tranche_idx,
                reserve_spike_fired=reserve_spike_fired,
            )

        remaining = max(0, qty - filled)
        if remaining > 0 and play.qty_open > 0:
            play.working_order = self._working_order_from_result(
                result=result,
                remaining_qty=remaining,
                attempts_used=1,
                retry_kind=self._retry_kind(profile),
                reason=reason,
                accounted_fills=filled,
                reserved_tranche_idx=reserved_tranche_idx,
                reserve_spike_fired=reserve_spike_fired,
            )
            print(
                f"[STRATEGY] {label} submitted asynchronously for {play.symbol} "
                f"({remaining} contract(s) still working)"
            )
        return filled > 0 or play.working_order is not None

    def _close_play(
        self, play: Play, qty: int, reason: str,
        retry: Optional[RetryProfile] = None,
        ctx: Optional[StrategyContext] = None,
    ) -> bool:
        return self._execute_exit(play, qty, reason, retry=retry, label="CLOSE", ctx=ctx)

    def _partial_close(
        self,
        play: Play,
        qty: int,
        reason: str,
        retry: Optional[RetryProfile] = None,
        reserved_tranche_idx: Optional[int] = None,
        reserve_spike_fired: bool = False,
        ctx: Optional[StrategyContext] = None,
    ) -> bool:
        return self._execute_exit(
            play,
            qty,
            reason,
            retry=retry,
            label="PARTIAL",
            reserved_tranche_idx=reserved_tranche_idx,
            reserve_spike_fired=reserve_spike_fired,
            ctx=ctx,
        )

    # ── private: sizing & helpers ─────────────────────────────────────────────

    def _size_qty(
        self, ctx: StrategyContext,
        conviction: ConvictionLevel, ask_price: float,
    ) -> int:
        desired = ctx.snapshot.nav * self.thesis_max_nav_pct * conviction.value
        max_usd = self._entry_budget(ctx, desired)
        if max_usd <= 0 or ask_price <= 0:
            return 0
        return max(0, int(max_usd / (ask_price * 100)))

    def _open_directional(
        self, play_type: PlayType, symbol: str,
        conviction: ConvictionLevel,
        spec: ContractSpec,
        exit_profile: ExitProfile,
        ctx: Optional[StrategyContext] = None,
    ) -> Optional[Play]:
        def _size(ctx_: StrategyContext, ask: float) -> int:
            return self._size_qty(ctx_, conviction, ask)
        return self._open_entry(
            play_type=play_type, symbol=symbol, spec=spec,
            size_fn=_size, exit_profile=exit_profile, ctx=ctx,
        )

    def _open_entry(
        self, play_type: PlayType, symbol: str, spec: ContractSpec,
        size_fn, exit_profile: Optional[ExitProfile] = None,
        ctx: Optional[StrategyContext] = None,
    ) -> Optional[Play]:
        """Shared entry flow: snapshot → risk → chain → size → buy → make_play."""
        ctx = ctx or self.context()
        chain = OptionChain(self.ib, symbol)
        picks = chain.select(**spec.to_kwargs())
        if picks.empty:
            print(f"[STRATEGY] {play_type.value} {symbol.upper()}: no qualifying contract")
            return None
        row = picks.iloc[0]
        if self._reject_non_call_row(row, symbol, play_type.value):
            return None
        contract_currency = str(row.get("currency", self.executor.currency) or self.executor.currency)
        if not self._entry_guard(ctx, contract_currency):
            return None
        self._print_soft_contract_warnings(row, spec, symbol)
        con_id = int(row["con_id"])
        if self._reject_duplicate_contract(con_id, symbol):
            return None
        ask = float(row["ask"])
        qty = size_fn(ctx, ask)
        if qty < 1:
            print(f"[STRATEGY] {play_type.value} {symbol.upper()}: insufficient headroom")
            return None
        result = self._submit_entry_order(con_id, qty)
        return self._make_play(
            play_type=play_type, symbol=symbol.upper(),
            con_id=con_id, qty=qty,
            entry_price=ask, entry_nav=ctx.snapshot.nav,
            exit_profile=exit_profile or self._exit_profile(play_type),
            order_result=result,
        )

    def _make_play(
        self, play_type: PlayType, symbol: str, con_id: int,
        qty: int, entry_price: float, entry_nav: float,
        exit_profile: ExitProfile, order_result: OrderResult,
    ) -> Optional[Play]:
        filled = int(order_result.total_filled or 0)
        actual_price = order_result.total_avg_fill() or order_result.limit_px or entry_price
        remaining = max(0, int(order_result.unfilled_qty or (qty - filled)))

        if filled <= 0 and not order_result.last_order_live:
            print(f"[STRATEGY] {symbol}: entry unfilled and no live order remains — no play created")
            return None

        if filled <= 0:
            play = Play(
                play_id      = uuid4().hex[:12],
                account_id   = self.account.account_id or "",
                play_type    = play_type,
                symbol       = symbol,
                con_id       = con_id,
                qty_initial  = 0,
                qty_open     = 0,
                entry_time   = _market_now(),
                entry_price  = actual_price,
                entry_nav    = entry_nav,
                exit_profile = exit_profile,
                status       = PlayStatus.PENDING,
                working_entry = self._working_entry_from_result(
                    result=order_result,
                    requested_qty=qty,
                    remaining_qty=remaining or qty,
                    attempts_used=1,
                    accounted_fills=0,
                ),
            )
            play.orders.append(order_result)
            self.plays.append(play)
            state.save(self.plays, account_id=self.account.account_id)
            print(f"[STRATEGY] ⚠ {symbol}: entry still live/unresolved — saved as PENDING")
            return play

        play = Play(
            play_id      = uuid4().hex[:12],
            account_id   = self.account.account_id or "",
            play_type    = play_type,
            symbol       = symbol,
            con_id       = con_id,
            qty_initial  = filled,
            qty_open     = filled,
            entry_time   = _market_now(),
            entry_price  = actual_price,
            entry_nav    = entry_nav,
            exit_profile = exit_profile,
            status       = PlayStatus.OPEN,
        )
        play.orders.append(order_result)
        if remaining > 0 and order_result.last_order_live:
            play.working_entry = self._working_entry_from_result(
                result=order_result,
                requested_qty=qty,
                remaining_qty=remaining,
                attempts_used=1,
                accounted_fills=order_result.filled_qty(),
            )
            print(f"[STRATEGY] ⚠ Partial fill {filled}/{qty} @ {actual_price:.2f}; remaining entry still working")
        elif filled < qty:
            print(f"[STRATEGY] ⚠ Partial fill {filled}/{qty} @ {actual_price:.2f}; no live entry remains")

        self.plays.append(play)
        state.save(self.plays, account_id=self.account.account_id)
        print(f"[STRATEGY] Opened: {play}")
        return play

    def _price_from_position(self, pos_row: Optional[dict]) -> Optional[float]:
        """Derive per-share option price from IB portfolio row (no API call)."""
        if pos_row is None:
            return None
        market_value = pos_row.get("market_value")
        position     = pos_row.get("position")
        if market_value is None or not position or abs(float(position)) == 0:
            return None
        # market_value == 0 is ambiguous: could be stale (pre-market) or a
        # genuinely worthless deep-OTM option.  Fall through to live snapshot
        # to disambiguate.
        if float(market_value) == 0.0:
            return None
        return abs(float(market_value)) / (abs(float(position)) * 100)

    def _quote_option_contract(
        self,
        contract,
        generic_ticks: str = "106",
        wait_secs: float = 1.5,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        ticker = self.ib.reqMktData(contract, generic_ticks)
        try:
            self.ib.sleep(wait_secs)
            bid  = ticker.bid  if ticker.bid  and ticker.bid  > 0 else None
            ask  = ticker.ask  if ticker.ask  and ticker.ask  > 0 else None
            last = ticker.last if ticker.last and ticker.last > 0 else None
            return bid, ask, last
        finally:
            self.ib.cancelMktData(contract)

    def _price_from_market(self, play: Play, ctx: StrategyContext) -> Optional[float]:
        """
        Fall back to a live market-data snapshot: mid → ask → last.
        The caller decides how to treat a missing live price (for example,
        using 0.0 for a truly worthless option after time-based checks).
        """
        if play.con_id in ctx.price_cache:
            return ctx.price_cache[play.con_id]
        contract = self._resolve_contract(play.con_id, ctx)
        bid, ask, last = self._quote_option_contract(contract)
        price = ((bid + ask) / 2) if (bid is not None and ask is not None) else (ask or last or None)
        ctx.price_cache[play.con_id] = price
        return price

    def price_for_play(self, play: Play, ctx: StrategyContext) -> Optional[float]:
        """
        Single source of truth for a play's current per-share option price.

        Priority:
          1. Derive from IB portfolio market_value / position (no extra
             market-data request, always consistent with account view).
          2. Fall back to a live snapshot: mid → ask → last.

        Returns None only when both paths fail (contract illiquid / market
        closed / position vanished).
        """
        pos_row = ctx.position(play.con_id)
        return self._price_from_position(pos_row) or self._price_from_market(play, ctx)

    def _has_open_play(self, symbol: str, play_type: PlayType) -> bool:
        return any(
            p.symbol == symbol.upper()
            and p.play_type == play_type
            and p.status in (PlayStatus.OPEN, PlayStatus.SCALING, PlayStatus.PENDING)
            for p in self.plays
        )
