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
    stop_loss, full_exit_target, trailing_stop, DTE_floor, max_hold_days

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

All tunable parameters live in main.py's StrategyConfig.  ExitProfile
defaults accept **kw so any field can be overridden from config.
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
# SIZING DEFAULTS  (single source of truth — config.py falls back to these
# via config.toml; do NOT duplicate elsewhere)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_APPROACH_MAX_NAV_PCT: float = 0.03
_DEFAULT_SENTINEL_MAX_NAV_PCT: float = 0.03
_DEFAULT_SNIPER_MAX_NAV_PCT:   float = 0.05


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
    trailing_stop_pct  trail from peak P&L; None = disabled
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

    APPROACH example:  [(0.15, 0.30), (0.30, 0.40)]
      +15% → sell 30%,  +30% → sell 40%,   +60% → close all
      (tighter targets — don't overstay theta drag)

    """
    stop_loss_pct:        float
    full_exit_pct:        float
    dte_floor:            int
    trailing_stop_pct:    Optional[float] = None
    tranches:             list[tuple[float, float]] = field(default_factory=list)
    max_hold_days:        int             = 0
    spike_pct:            float = 0.0
    spike_window_hours:   float = 0.0
    spike_sell_ratio:     float = 0.0


@dataclass
class WorkingOrder:
    """Session-only async order state so exits do not block the strategy loop."""
    trade_result:   OrderResult
    remaining_qty:  int
    attempts_used:  int
    submitted_at:   datetime
    retry_kind:     str
    reason:         str
    accounted_fills: int = 0
    cancel_requested: bool = False


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

    # Session-only: not serialized to plays.json (contains live Trade refs).
    # After a restart this list is always empty.
    orders: list[OrderResult] = field(default_factory=list)
    working_order: Optional[WorkingOrder] = None

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
        last_time = self.pnl_history[-1][0]
        span_minutes = (last_time - first_in_window_time).total_seconds() / 60
        if span_minutes < 15:
            return None

        if before is not None:
            t0, p0 = before
            t1, p1 = self.pnl_history[first_in_idx]
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
            if today_vol < avg_vol * self.min_volume_ratio:
                print(
                    f"[SCANNER] {symbol}: drop={drop:.1%} but "
                    f"volume too light ({today_vol:,} vs "
                    f"{avg_vol * self.min_volume_ratio:,.0f})"
                )
                return None
        return (symbol, last)

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
        approach_max_nav_pct: float = _DEFAULT_APPROACH_MAX_NAV_PCT,
        sentinel_max_nav_pct: float = _DEFAULT_SENTINEL_MAX_NAV_PCT,
        sniper_max_nav_pct:   float = 0.05,
        pending_max_hours:    float = 24.0,
        patient_retry:        Optional[RetryProfile] = None,
        urgent_retry:         Optional[RetryProfile] = None,
        base_currency:        str   = "CHF",
        account_id:          Optional[str] = None,
    ):
        self.ib                   = ib
        self.policy               = policy
        self.executor             = Executor(ib)
        self.account              = Account(
            ib,
            base_currency=base_currency,
            account_id=account_id,
        )
        self.exit_profiles        = exit_profiles or {}
        self.contract_specs       = contract_specs or {}
        self.scanner              = sniper_scanner
        self.approach_max_nav_pct = approach_max_nav_pct
        self.sentinel_max_nav_pct = sentinel_max_nav_pct
        self.sniper_max_nav_pct   = sniper_max_nav_pct
        self.pending_max_hours    = pending_max_hours
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

    # ── entry methods ─────────────────────────────────────────────────────────

    def open_thesis(
        self, symbol: str, conviction: ConvictionLevel,
        right: Right = Right.CALL,
        spec: Optional[ContractSpec] = None,
        exit_profile: Optional[ExitProfile] = None,
    ) -> Optional[Play]:
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
        def _size(risk, nav, ask):
            max_usd = min(nav * max_nav_pct, risk.headroom())
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
        if not self._risk_guard(ctx.risk):
            return None
        spec  = spec or self._contract_spec(PlayType.SNIPER)
        chain = OptionChain(self.ib, symbol)
        picks = chain.select(spot_price=spot_price, **spec.to_kwargs())
        if picks.empty:
            print(f"[STRATEGY] SNIPER {symbol}: no qualifying contract")
            return None
        row = picks.iloc[0]
        con_id = int(row["con_id"])
        if self._reject_duplicate_contract(con_id, symbol):
            return None
        ask = float(row["ask"])
        # Cap sniper sizing by sniper_max_nav_pct (analogous to approach/sentinel)
        # instead of using 100% of headroom unconditionally.
        max_usd = min(ctx.snapshot.nav * self.sniper_max_nav_pct, ctx.risk.headroom())
        qty = int(max_usd / (ask * 100)) if (max_usd > 0 and ask > 0) else 0
        if qty < 1:
            print(f"[STRATEGY] SNIPER {symbol}: insufficient headroom")
            return None
        result = self.executor.buy_option(
            con_id, qty, **self.patient_retry.as_kwargs()
        )
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
        if not self._risk_guard(ctx.risk):
            return None
        if qty < 1:
            print("[STRATEGY] manual entry qty must be at least 1")
            return None
        profile  = exit_profile or self._exit_profile(play_type)
        contract = self._resolve_contract(con_id, ctx)
        sym      = symbol.upper() or contract.symbol
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
        if play_type in (PlayType.APPROACH, PlayType.SENTINEL):
            cap = (self.approach_max_nav_pct if play_type is PlayType.APPROACH
                   else self.sentinel_max_nav_pct)
            max_usd  = min(ctx.snapshot.nav * cap, ctx.risk.headroom())
            qty_algo = int(max_usd / (ref_px * 100)) if (max_usd > 0 and ref_px > 0) else 0
            qty_use  = min(qty, qty_algo)
        else:
            conv    = conviction or ConvictionLevel.MEDIUM
            max_usd = ctx.risk.headroom() * conv.value
            qty_algo = int(max_usd / (ref_px * 100)) if (max_usd > 0 and ref_px > 0) else 0
            qty_use  = min(qty, qty_algo)
        if qty_use < 1:
            print(f"[STRATEGY] manual {sym}: insufficient headroom")
            return None
        result = self.executor.buy_option(con_id, qty_use, **self.patient_retry.as_kwargs())
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
        if self._reject_duplicate_contract(con_id, symbol or str(pos_row.get('symbol', ''))):
            return None
        live_qty = int(abs(float(pos_row.get("position", 0) or 0)))
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
        if self.scanner:
            hit = self.scanner.scan()
            if hit:
                sym, spot = hit
                if not self._has_open_play(sym, PlayType.SNIPER):
                    self.open_sniper(sym, spot_price=spot, ctx=ctx)

    # ── private: risk guard ───────────────────────────────────────────────────

    def _risk_guard(self, risk: PortfolioRisk) -> bool:
        if risk.risk_status == "ABOVE_CEILING":
            print(
                f"[STRATEGY] Ceiling breached "
                f"({risk.risk_pct:.1%} > {self.policy.risk_ceiling:.1%}) "
                f"— no new entries"
            )
            return False
        if risk.headroom() < 100:
            print(f"[STRATEGY] Headroom too small (${risk.headroom():.0f})")
            return False
        return True

    def _retry_kind(self, profile: RetryProfile) -> str:
        return "urgent" if profile is self.urgent_retry else "patient"

    def _profile_for_kind(self, retry_kind: str) -> RetryProfile:
        return self.urgent_retry if retry_kind == "urgent" else self.patient_retry

    def _live_qty(self, pos_row: Optional[dict]) -> int:
        if pos_row is None:
            return 0
        return int(abs(float(pos_row.get("position", 0) or 0)))

    def _working_exit_pending(self, play: Play) -> bool:
        return bool(play.working_order and play.working_order.trade_result.side is OrderSide.SELL)

    def _advance_working_orders(self, ctx: StrategyContext) -> bool:
        dirty = False
        now = _market_now()
        for play in [p for p in self.plays if self._working_exit_pending(p)]:
            wo = play.working_order
            if wo is None:
                continue

            trade_result = wo.trade_result
            trade_filled = trade_result.filled_qty()
            if trade_filled > wo.accounted_fills:
                delta = min(trade_filled - wo.accounted_fills, play.qty_open)
                wo.accounted_fills = trade_filled
                if delta > 0:
                    play.qty_open -= delta
                    wo.remaining_qty = max(0, wo.remaining_qty - delta)
                    play.status = PlayStatus.CLOSED if play.qty_open <= 0 else PlayStatus.SCALING
                    dirty = True

            pos_row = ctx.position(play.con_id)
            live_qty = self._live_qty(pos_row)
            if live_qty < play.qty_open:
                delta = play.qty_open - live_qty
                play.qty_open = live_qty
                wo.remaining_qty = max(0, wo.remaining_qty - delta)
                play.status = PlayStatus.CLOSED if live_qty <= 0 else PlayStatus.SCALING
                dirty = True

            if play.qty_open <= 0 or wo.remaining_qty <= 0:
                play.qty_open = max(0, play.qty_open)
                play.status = PlayStatus.CLOSED if play.qty_open == 0 else PlayStatus.SCALING
                play.working_order = None
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
                continue

            if wo.attempts_used >= total_attempts:
                print(
                    f"[STRATEGY] ⚠  EXIT unresolved for {play.symbol}: "
                    f"{wo.remaining_qty} contract(s) still open after {total_attempts} attempts"
                )
                play.working_order = None
                dirty = True
                continue

            next_mode = self.executor.mode_for_attempt(
                attempt          = wo.attempts_used,
                total_attempts   = total_attempts,
                mode             = profile.mode,
                fallback_mode    = profile.fallback_mode,
                fallback_after   = profile.fallback_after,
                last_resort_mode = profile.last_resort_mode,
            )
            try:
                result = self.executor.submit_option_order(
                    side  = OrderSide.SELL,
                    con_id= play.con_id,
                    qty   = wo.remaining_qty,
                    mode  = next_mode,
                )
            except Exception as exc:
                print(f"[STRATEGY] ⚠  EXIT retry failed for {play.symbol}: {exc}")
                play.working_order = None
                dirty = True
                continue

            play.orders.append(result)
            play.working_order = WorkingOrder(
                trade_result   = result,
                remaining_qty  = wo.remaining_qty,
                attempts_used  = wo.attempts_used + 1,
                submitted_at   = _market_now(),
                retry_kind     = wo.retry_kind,
                reason         = wo.reason,
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
        dirty = self._advance_working_orders(ctx)

        # promote PENDING or expire stale ones
        for play in [p for p in self.plays if p.status == PlayStatus.PENDING]:
            pos_row = ctx.position(play.con_id)
            if pos_row is not None:
                live_qty = int(abs(float(pos_row.get("position", 0))))
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

          1. DTE floor              → full close
          2. Stop loss              → full close
          3. Full exit target       → full close
          4. Trailing stop          → full close
          5. Max hold days          → full close
          6. VELOCITY SPIKE         → aggressive partial (once only)
          7. Tranches (THESIS+APPROACH+SENTINEL)→ partial (one per cycle)
        """
        pos_row = ctx.position(play.con_id)
        current_price = (
            self._price_from_position(pos_row)
            or self._price_from_market(play, ctx)
        )
        if current_price is None:
            if pos_row is None:
                play.qty_open = 0
                play.status   = PlayStatus.CLOSED
                print(f"[STRATEGY] {play.symbol} vanished from IB → CLOSED")
                return True
            # Position exists but no price data — still check time-based exits
            # such as DTE floor. Skip price-based exits until pricing recovers.
            ep  = play.exit_profile
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
                        )
                except ValueError:
                    pass
            if (
                play.entry_time_known
                and ep.max_hold_days > 0
                and play.hours_since_entry() / 24 >= ep.max_hold_days
            ):
                return self._close_play(
                    play, qty, f"max hold ({ep.max_hold_days}d), no price data"
                )
            return False

        pnl_pct = play.current_pnl_pct(current_price)
        play.record_pnl(pnl_pct)
        ep  = play.exit_profile
        qty = play.qty_open

        # ── 1. DTE floor ─────────────────────────────────────────────────
        expiry_str = str(pos_row.get("expiry") or "") if pos_row is not None else ""
        if expiry_str:
            try:
                exp_date = datetime.strptime(expiry_str, "%Y%m%d").date()
                dte = (exp_date - _market_date()).days
                if dte <= ep.dte_floor:
                    return self._close_play(
                        play, qty, f"DTE floor ({dte} ≤ {ep.dte_floor})"
                    )
            except ValueError:
                pass

        # ── 2. Stop loss ─────────────────────────────────────────────────
        if pnl_pct <= ep.stop_loss_pct:
            return self._close_play(
                play, qty, f"stop loss ({pnl_pct:.1%})",
                retry=self.urgent_retry,
            )

        # ── 3. Full exit target ──────────────────────────────────────────
        if pnl_pct >= ep.full_exit_pct:
            return self._close_play(play, qty, f"full target ({pnl_pct:.1%})")

        # ── 4. Trailing stop ─────────────────────────────────────────────
        # Only activate once the peak has reached at least the trailing
        # distance itself.  A tiny +1% peak with a 30% trail would never
        # fire meaningfully (the stop loss catches it first), but with a
        # tight custom trail it could fire at a worse level than intended.
        if (
            ep.trailing_stop_pct is not None
            and play.peak_pnl_pct >= ep.trailing_stop_pct
        ):
            dd = play.peak_pnl_pct - pnl_pct
            if dd >= ep.trailing_stop_pct:
                return self._close_play(
                    play, qty,
                    f"trailing stop (peak={play.peak_pnl_pct:.1%} "
                    f"now={pnl_pct:.1%} dd={dd:.1%})",
                    retry=self.urgent_retry,
                )

        # ── 5. Max hold days ─────────────────────────────────────────────
        if play.entry_time_known and ep.max_hold_days > 0:
            if play.hours_since_entry() / 24 >= ep.max_hold_days:
                return self._close_play(play, qty, f"max hold ({ep.max_hold_days}d)")

        # ── 6. VELOCITY SPIKE EXIT ───────────────────────────────────────
        #
        # Measures P&L *change* within the lookback window, not absolute
        # level.  +30% → +130% in 4h triggers (gain=100%).
        # +130% reached over 2 weeks does NOT.
        #
        # After firing: advance tranche_idx past covered tranches
        # so the tranche logic doesn't double-sell.
        if (
            play.entry_time_known
            and not play.spike_fired
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
                    ok = self._partial_close(
                        play, sell_qty,
                        f"SPIKE (+{gain:.0%} in <{ep.spike_window_hours:.0f}h, "
                        f"sell {ep.spike_sell_ratio:.0%} = {sell_qty}/{play.qty_open})",
                    )
                    if ok:
                        play.spike_fired = True
                        if ep.tranches:
                            while (
                                play.tranche_idx < len(ep.tranches)
                                and pnl_pct >= ep.tranches[play.tranche_idx][0]
                            ):
                                play.tranche_idx += 1
                        return True
                    return False

        # ── 7. Tranche-based scale-out ─────────────────────────────────
        # Skip tranches when only 1 contract remains — the last contract
        # exits via hard exits only (stop, full target, trail, DTE, max hold).
        if (
            ep.tranches
            and play.tranche_idx < len(ep.tranches)
            and play.qty_open > 1
        ):
            trigger, fraction = ep.tranches[play.tranche_idx]
            if pnl_pct >= trigger:
                sell_qty = min(
                    max(1, round(play.qty_initial * fraction)),
                    play.qty_open - 1,
                )
                if self._partial_close(
                    play, sell_qty,
                    f"tranche {play.tranche_idx+1}/{len(ep.tranches)} "
                    f"({pnl_pct:.1%} ≥ {trigger:.0%}, "
                    f"sell {fraction:.0%} of initial)",
                ):
                    play.tranche_idx += 1
                    return True

        return False

    # ── private: order helpers ────────────────────────────────────────────────

    def _execute_exit(
        self, play: Play, qty: int, reason: str,
        retry: Optional[RetryProfile] = None, label: str = "EXIT",
    ) -> bool:
        profile = retry or self.patient_retry
        if self._working_exit_pending(play):
            print(f"[STRATEGY] {label} already working for {play.symbol}; skipping duplicate request")
            return True

        total_attempts = profile.max_retries + 1
        first_mode = self.executor.mode_for_attempt(
            attempt          = 0,
            total_attempts   = total_attempts,
            mode             = profile.mode,
            fallback_mode    = profile.fallback_mode,
            fallback_after   = profile.fallback_after,
            last_resort_mode = profile.last_resort_mode,
        )
        print(f"[STRATEGY] {label} {play}  qty={qty}  reason={reason}")
        try:
            result = self.executor.submit_option_order(
                side  = OrderSide.SELL,
                con_id= play.con_id,
                qty   = qty,
                mode  = first_mode,
            )
        except Exception as e:
            print(f"[STRATEGY] ⚠  {label} failed: {e}")
            return False

        play.orders.append(result)
        filled = min(result.filled_qty(), play.qty_open)
        play.qty_open -= filled
        if play.qty_open <= 0:
            play.qty_open = 0
            play.status   = PlayStatus.CLOSED
        elif filled > 0:
            play.status = PlayStatus.SCALING
        remaining = max(0, qty - filled)
        if remaining > 0 and play.qty_open > 0:
            play.working_order = WorkingOrder(
                trade_result   = result,
                remaining_qty  = remaining,
                attempts_used  = 1,
                submitted_at   = _market_now(),
                retry_kind     = self._retry_kind(profile),
                reason         = reason,
                accounted_fills= filled,
            )
            print(
                f"[STRATEGY] {label} submitted asynchronously for {play.symbol} "
                f"({remaining} contract(s) still working)"
            )
        return filled > 0

    def _close_play(
        self, play: Play, qty: int, reason: str,
        retry: Optional[RetryProfile] = None,
    ) -> bool:
        return self._execute_exit(play, qty, reason, retry=retry, label="CLOSE")

    def _partial_close(
        self, play: Play, qty: int, reason: str,
        retry: Optional[RetryProfile] = None,
    ) -> bool:
        return self._execute_exit(play, qty, reason, retry=retry, label="PARTIAL")

    # ── private: sizing & helpers ─────────────────────────────────────────────

    def _size_qty(
        self, risk: PortfolioRisk,
        conviction: ConvictionLevel, ask_price: float,
    ) -> int:
        max_usd = risk.headroom() * conviction.value
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
        def _size(risk, _nav, ask):
            return self._size_qty(risk, conviction, ask)
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
        if not self._risk_guard(ctx.risk):
            return None
        chain = OptionChain(self.ib, symbol)
        picks = chain.select(**spec.to_kwargs())
        if picks.empty:
            print(f"[STRATEGY] {play_type.value} {symbol.upper()}: no qualifying contract")
            return None
        row = picks.iloc[0]
        con_id = int(row["con_id"])
        if self._reject_duplicate_contract(con_id, symbol):
            return None
        ask = float(row["ask"])
        qty = size_fn(ctx.risk, ctx.snapshot.nav, ask)
        if qty < 1:
            print(f"[STRATEGY] {play_type.value} {symbol.upper()}: insufficient headroom")
            return None
        result = self.executor.buy_option(
            con_id, qty, **self.patient_retry.as_kwargs()
        )
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
    ) -> Play:
        filled = order_result.total_filled
        if filled <= 0:
            play = Play(
                play_id      = uuid4().hex[:12],
                account_id   = self.account.account_id or "",
                play_type=play_type, symbol=symbol, con_id=con_id,
                qty_initial=qty, qty_open=qty, entry_time=_market_now(),
                entry_price=entry_price, entry_nav=entry_nav,
                exit_profile=exit_profile, status=PlayStatus.PENDING,
            )
            play.orders.append(order_result)
            self.plays.append(play)
            state.save(self.plays, account_id=self.account.account_id)
            print(f"[STRATEGY] ⚠ Unfilled — {symbol} saved as PENDING")
            return play

        actual_price = order_result.total_avg_fill() or order_result.limit_px or entry_price
        play = Play(
            play_id      = uuid4().hex[:12],
            account_id   = self.account.account_id or "",
            play_type=play_type, symbol=symbol, con_id=con_id,
            qty_initial=filled, qty_open=filled, entry_time=_market_now(),
            entry_price=actual_price, entry_nav=entry_nav,
            exit_profile=exit_profile, status=PlayStatus.OPEN,
        )
        play.orders.append(order_result)
        self.plays.append(play)
        state.save(self.plays, account_id=self.account.account_id)
        if filled < qty:
            print(f"[STRATEGY] ⚠ Partial fill {filled}/{qty} @ {actual_price:.2f}")
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
