"""
execution.py
─────────────────────────────────────────────────────────────────────────────
Order execution layer.  No strategy logic — only placement, tracking, and
cancellation of individual orders.

Composes with the rest of the framework via con_id (int):
  TrimLeg.con_id          (portfolio.py)
  positions["con_id"]     (ib_core.Account)
  OptionChain df["con_id"] (ib_core.OptionChain)

IB option pricing note
──────────────────────
The default limit price is MID — (bid + ask) / 2.  For biotech names with
wide spreads this gives the best edge while still being a reasonable ask of
the market.  IB_MODEL (IB's theoretical optPrice) is a solid alternative
when spreads are very wide or liquidity is thin — it often sits a few cents
away from raw mid in a more defensible direction.  NATURAL (ask/bid) is
available but should be a deliberate last resort, not a default.

  MID       — (bid + ask) / 2                               (default)
  IB_MODEL  — IB's theoretical value clamped to [bid, ask]
  NATURAL   — ask (BUY) or bid (SELL); combine with offset for slight edge

Fill retry
──────────
All option orders are LMT.  If a limit order is not filled within
`fill_timeout_secs`, the order is cancelled, fresh bid/ask is fetched,
a new limit price is computed, and the order is resubmitted.

Mode ladder (configurable via config.toml [execution.patient/urgent]):
  1. First `fallback_after` attempts use `mode` (default: IB_MODEL)
  2. Remaining attempts use `fallback_mode` (default: MID)
  3. Very last attempt uses `last_resort_mode` if set (e.g. NATURAL
     for stop losses; disabled for patient orders)

Partial fills are handled correctly: each retry only submits the unfilled
remainder.

An optional `offset` (dollars, only used with NATURAL) shifts the price one
step toward mid, e.g. offset=0.05 means "5 cents inside the natural price".

Options are always placed as LMT.  Stocks default to MKT; pass limit_price
for LMT.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
from ib_insync import IB, Contract, LimitOrder, MarketOrder, Stock, Trade


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# How often (seconds) the fill-wait loop polls trade status.
_POLL_INTERVAL_SECS = 5

# Statuses IB uses while an order is still live on the exchange.
_PENDING_STATUSES = frozenset({
    "Submitted", "PreSubmitted", "PendingSubmit", "PendingCancel",
    "PartiallyFilled",
})

# Terminal states returned by IB once an order is no longer working.
_TERMINAL_STATUSES = frozenset({
    "Filled", "Cancelled", "ApiCancelled", "Inactive",
})


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class PriceMode(str, Enum):
    """
    How the limit price is derived from live market data for option orders.

    MID       — (bid + ask) / 2                              (default)
                Best edge for liquid enough markets.  Repriced on each retry
                so the limit tracks the market if it moves during the wait.

    IB_MODEL  — IB's theoretical option value (modelGreeks.optPrice),
                clamped to [bid, ask].  Falls back to MID if unavailable.
                Useful when the raw mid is noticeably off-model (thin books,
                fast-moving IV) — IB's pricer often sits in a better spot.

    NATURAL   — ask when buying, bid when selling.  Guarantees fill but
                surrenders edge.  Combine with offset for a small improvement
                while keeping fill probability high.  Use deliberately, not
                as a default.
    """
    MID      = "MID"
    IB_MODEL = "IB_MODEL"
    NATURAL  = "NATURAL"


@dataclass
class RetryProfile:
    """
    Retry / pricing settings for one urgency class (patient or urgent).

    Used by Strategy to pass config-driven execution params to the Executor
    without hardcoding timeouts or mode ladders in strategy logic.
    """
    fill_timeout_secs: int
    max_retries:       int
    mode:              PriceMode
    fallback_mode:     Optional[PriceMode]
    fallback_after:    Optional[int]
    last_resort_mode:  Optional[PriceMode]    # None = disabled

    def as_kwargs(self) -> dict:
        """Dict suitable for Executor.buy_option / sell_option(**profile.as_kwargs())."""
        return {
            "mode":              self.mode,
            "fill_timeout_secs": self.fill_timeout_secs,
            "max_retries":       self.max_retries,
            "fallback_mode":     self.fallback_mode,
            "fallback_after":    self.fallback_after,
            "last_resort_mode":  self.last_resort_mode,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ORDER RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    """
    Thin wrapper around an ib_insync Trade.

    Fields
    ──────
    order_id  — IB permId (stable; use for cancellation logging)
    symbol    — underlying ticker
    con_id    — IB contract conId
    side      — OrderSide.BUY / SELL
    qty       — quantity submitted for THIS order (not the original full qty
                when retries have reduced it due to partial fills)
    limit_px  — the limit price submitted; None for market stock orders
    trade     — raw ib_insync Trade object
                → trade.fills        : list of Execution records
                → trade.orderStatus  : live IB order status
                → trade.log          : audit trail

    Convenience methods
    ───────────────────
    result.status()      → str   e.g. "Submitted", "Filled", "Cancelled"
    result.is_filled()   → bool
    result.filled_qty()  → int
    result.avg_fill()    → float | None  (volume-weighted average fill price)
    """
    order_id: int
    symbol:   str
    con_id:   int
    side:     OrderSide
    qty:      int
    limit_px: Optional[float]
    trade:    Trade  # raw ib_insync Trade; full access to fills, log, status

    # Broker identity / routing. order_id is kept as the display/stable ID
    # used by existing logs; native_order_id and perm_id preserve the raw IB
    # fields for restart rebinding.
    perm_id:         Optional[int] = None
    native_order_id: Optional[int] = None
    client_id:       Optional[int] = None
    account_id:      Optional[str] = None

    # ── cross-retry fill/accounting metadata (set by _place_with_retry) ───
    total_filled: int   = 0    # contracts filled across ALL retry attempts
    total_cost:   float = 0.0  # Σ(shares × price) — for true avg fill price
    requested_qty: int  = 0    # original requested qty for the whole entry/exit
    unfilled_qty:  int  = 0    # requested_qty - total_filled after return
    last_order_live: bool = False
    cancel_unresolved: bool = False

    def status(self) -> str:
        return self.trade.orderStatus.status

    def is_filled(self) -> bool:
        return self.trade.orderStatus.status == "Filled"

    def filled_qty(self) -> int:
        fill_qty = int(sum(f.execution.shares for f in self.trade.fills))
        status_qty = int(getattr(self.trade.orderStatus, "filled", 0) or 0)
        return max(fill_qty, status_qty)

    def avg_fill(self) -> Optional[float]:
        """Volume-weighted average fill price. None if no fills yet."""
        fills = self.trade.fills
        if fills:
            total_qty = sum(f.execution.shares for f in fills)
            total_val = sum(f.execution.shares * f.execution.price for f in fills)
            return round(total_val / total_qty, 4) if total_qty else None
        status_avg = float(getattr(self.trade.orderStatus, "avgFillPrice", 0) or 0)
        return round(status_avg, 4) if status_avg > 0 else None

    def total_avg_fill(self) -> Optional[float]:
        """Volume-weighted avg fill price across ALL retry attempts.
        Falls back to single-attempt avg_fill() when total_cost was not set
        (e.g. stock orders that bypass the retry loop)."""
        if self.total_filled > 0 and self.total_cost > 0:
            return round(self.total_cost / self.total_filled, 4)
        return self.avg_fill()


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class Executor:
    """
    Place and manage individual option and stock orders.

    All option orders are LMT — never MKT.  The limit price is computed
    from a fresh single-snapshot market data request immediately before
    each submission (including retries, so the price always reflects the
    current market).

    Constructor parameters
    ──────────────────────
    fill_timeout_secs  — seconds to wait for a fill before repricing
                         and resubmitting.  Default: 300 (5 minutes).
    max_retries        — how many times to reprice and resubmit after the
                         initial attempt.  Total attempts = max_retries + 1.
                         Default: 3  (so up to 4 attempts / ~20 min total).

    Both can be overridden per-call on buy_option / sell_option for cases
    where tighter or looser timing is appropriate (e.g. a SNIPER play on a
    fast-moving name might want a shorter timeout).

    Usage — option:
        ex = Executor(ib)

        # open: buy 2 contracts at mid (default)
        result = ex.buy_option(con_id=123456, qty=2)

        # open: IB model price — good alternative for wide-spread biotech
        result = ex.buy_option(con_id=123456, qty=2, mode=PriceMode.IB_MODEL)

        # close: faster timeout for time-sensitive exits
        result = ex.sell_option(con_id=123456, qty=2, fill_timeout_secs=60)

    Usage — stock:
        result = ex.buy_stock("MRNA", qty=10)                      # MKT
        result = ex.sell_stock("MRNA", qty=10, limit_price=45.50)  # LMT

    Usage — cancel:
        ex.cancel(result)       # cancel one order by its OrderResult
        ex.cancel_all()         # reqGlobalCancel on this client

    Usage — inspect open orders:
        df = ex.pending_orders()
    """

    def __init__(
        self,
        ib:                IB,
        exchange:          str = "SMART",
        currency:          str = "USD",
        account_id:        Optional[str] = None,
        fill_timeout_secs: int = 300,   # 5 minutes between repricing
        max_retries:       int = 3,     # total attempts = max_retries + 1
    ):
        self.ib                = ib
        self.exchange          = exchange
        self.currency          = currency
        self.account_id        = account_id or None
        self.fill_timeout_secs = fill_timeout_secs
        self.max_retries       = max_retries
        self._contract_details_cache: dict[int, object] = {}
        self._market_rule_cache: dict[int, list] = {}

    # ── private: routing / identity ──────────────────────────────────────────

    def _route_order(self, order):
        """Attach the resolved IB account to every outbound order."""
        if self.account_id:
            setattr(order, "account", self.account_id)
        return order

    @staticmethod
    def _int_or_none(value) -> Optional[int]:
        try:
            ivalue = int(value or 0)
        except (TypeError, ValueError):
            return None
        return ivalue or None

    def _order_result(
        self,
        contract: Contract,
        trade: Trade,
        side: OrderSide,
        qty: int,
        limit_px: Optional[float],
        requested_qty: Optional[int] = None,
    ) -> OrderResult:
        order = trade.order
        perm_id = self._int_or_none(getattr(order, "permId", None))
        native_order_id = self._int_or_none(getattr(order, "orderId", None))
        client_id = self._int_or_none(getattr(order, "clientId", None))
        account_id = getattr(order, "account", None) or self.account_id
        return OrderResult(
            order_id=perm_id or native_order_id or 0,
            symbol=contract.symbol,
            con_id=int(contract.conId or 0),
            side=side,
            qty=int(qty),
            limit_px=limit_px,
            trade=trade,
            perm_id=perm_id,
            native_order_id=native_order_id,
            client_id=client_id,
            account_id=account_id,
            requested_qty=int(requested_qty if requested_qty is not None else qty),
        )

    # ── option orders ────────────────────────────────────────────────────────

    def buy_option(
        self,
        con_id:            int,
        qty:               int,
        mode:              PriceMode         = PriceMode.IB_MODEL,
        offset:            float             = 0.0,
        fill_timeout_secs: Optional[int]     = None,
        max_retries:       Optional[int]     = None,
        fallback_mode:     Optional[PriceMode] = None,
        fallback_after:    Optional[int]     = None,
        last_resort_mode:  Optional[PriceMode] = None,
    ) -> OrderResult:
        """
        Buy `qty` contracts of the option identified by `con_id`.

        con_id            — IB conId; comes directly from OptionChain df
        qty               — number of contracts (positive integer)
        mode              — starting price mode (default: IB_MODEL)
        offset            — dollars toward mid from NATURAL (NATURAL mode only)
        fill_timeout_secs — override instance default for this call
        max_retries       — override instance default for this call
        fallback_mode     — switch to this mode after fallback_after attempts
        fallback_after    — attempt count before switching to fallback_mode
        last_resort_mode  — mode for the very last attempt only (None = disabled)
        """
        return self._place_with_retry(
            side              = OrderSide.BUY,
            con_id            = con_id,
            qty               = qty,
            mode              = mode,
            offset            = offset,
            fill_timeout_secs = fill_timeout_secs if fill_timeout_secs is not None
                                else self.fill_timeout_secs,
            max_retries       = max_retries if max_retries is not None
                                else self.max_retries,
            fallback_mode     = fallback_mode,
            fallback_after    = fallback_after,
            last_resort_mode  = last_resort_mode,
        )

    def sell_option(
        self,
        con_id:            int,
        qty:               int,
        mode:              PriceMode         = PriceMode.IB_MODEL,
        offset:            float             = 0.0,
        fill_timeout_secs: Optional[int]     = None,
        max_retries:       Optional[int]     = None,
        fallback_mode:     Optional[PriceMode] = None,
        fallback_after:    Optional[int]     = None,
        last_resort_mode:  Optional[PriceMode] = None,
    ) -> OrderResult:
        """
        Sell `qty` contracts of the option identified by `con_id`.
        Use to open a short position or close an existing long.

        con_id            — IB conId
        qty               — number of contracts (positive integer)
        mode              — starting price mode (default: IB_MODEL)
        offset            — dollars toward mid from NATURAL (NATURAL mode only)
        fill_timeout_secs — override instance default for this call
        max_retries       — override instance default for this call
        fallback_mode     — switch to this mode after fallback_after attempts
        fallback_after    — attempt count before switching to fallback_mode
        last_resort_mode  — mode for the very last attempt only (None = disabled)
        """
        return self._place_with_retry(
            side              = OrderSide.SELL,
            con_id            = con_id,
            qty               = qty,
            mode              = mode,
            offset            = offset,
            fill_timeout_secs = fill_timeout_secs if fill_timeout_secs is not None
                                else self.fill_timeout_secs,
            max_retries       = max_retries if max_retries is not None
                                else self.max_retries,
            fallback_mode     = fallback_mode,
            fallback_after    = fallback_after,
            last_resort_mode  = last_resort_mode,
        )

    # ── stock orders ─────────────────────────────────────────────────────────

    def buy_stock(
        self,
        symbol:      str,
        qty:         int,
        limit_price: Optional[float] = None,
    ) -> OrderResult:
        """
        Buy `qty` shares of `symbol`.
        Submits MKT unless limit_price is provided (then LMT).
        """
        return self._stock_order(OrderSide.BUY, symbol, qty, limit_price)

    def sell_stock(
        self,
        symbol:      str,
        qty:         int,
        limit_price: Optional[float] = None,
    ) -> OrderResult:
        """
        Sell `qty` shares of `symbol`.
        Submits MKT unless limit_price is provided (then LMT).
        """
        return self._stock_order(OrderSide.SELL, symbol, qty, limit_price)

    # ── order management ─────────────────────────────────────────────────────

    def cancel(self, result: OrderResult) -> None:
        """Cancel the order associated with an OrderResult."""
        self.ib.cancelOrder(result.trade.order)
        acct = f" account={result.account_id}" if result.account_id else ""
        print(f"[EXEC] CANCEL order_id={result.order_id}{acct} "
              f"({result.side.value} {result.qty} con_id={result.con_id})")

    def cancel_all(self) -> None:
        """Cancel every open order on this IB client connection."""
        self.ib.reqGlobalCancel()
        print("[EXEC] CANCEL ALL — reqGlobalCancel sent")

    def is_live(self, result: OrderResult) -> bool:
        """Whether the order is still working at the exchange/IB."""
        return result.status() in _PENDING_STATUSES

    def result_from_trade(self, trade: Trade) -> OrderResult:
        """Wrap a live ib_insync Trade in OrderResult form."""
        contract = trade.contract
        order = trade.order
        side = OrderSide(str(order.action).upper())
        limit_px = order.lmtPrice if getattr(order, "orderType", "") == "LMT" else None
        return self._order_result(
            contract=contract,
            trade=trade,
            side=side,
            qty=int(order.totalQuantity),
            limit_px=limit_px,
            requested_qty=int(order.totalQuantity),
        )

    def remaining_qty_from_trade(self, trade: Trade) -> int:
        remaining = getattr(trade.orderStatus, "remaining", None)
        if remaining is not None:
            return int(remaining)
        filled = sum(f.execution.shares for f in getattr(trade, "fills", []))
        return max(0, int(getattr(trade.order, "totalQuantity", 0) - filled))

    def live_trades(
        self,
        con_id: Optional[int] = None,
        side: Optional[OrderSide] = None,
    ) -> list[Trade]:
        """Return currently live trades, optionally filtered by con_id and side."""
        if hasattr(self.ib, "reqOpenOrders"):
            try:
                self.ib.reqOpenOrders()
            except Exception:
                pass
        source = self.ib.openTrades() if hasattr(self.ib, "openTrades") else self.ib.trades()
        out: list[Trade] = []
        for trade in source:
            status = str(getattr(trade.orderStatus, "status", ""))
            if status not in _PENDING_STATUSES:
                continue
            if con_id is not None and int(getattr(trade.contract, "conId", 0) or 0) != con_id:
                continue
            action = str(getattr(trade.order, "action", "")).upper()
            if side is not None and action != side.value:
                continue
            out.append(trade)
        return out

    def wait_until_not_live(
        self,
        result: OrderResult,
        timeout_secs: float = 10.0,
        poll_secs: float = 0.5,
    ) -> str:
        """Wait until an order leaves a live IB state, then return the final status."""
        deadline = time.monotonic() + timeout_secs
        while time.monotonic() < deadline:
            status = result.status()
            if status not in _PENDING_STATUSES:
                return status
            self.ib.sleep(poll_secs)
        return result.status()

    def pending_orders(self) -> pd.DataFrame:
        """
        DataFrame of currently open / pending orders on this IB connection.

        Columns: order_id, symbol, con_id, sec_type, side,
                 qty, filled, remaining, limit_px, status
        """
        rows = []
        for trade in self.live_trades():
            c = trade.contract
            o = trade.order
            lmt = o.lmtPrice if (o.orderType == "LMT" and o.lmtPrice) else None
            rows.append({
                "order_id":  o.permId or o.orderId,
                "perm_id":   getattr(o, "permId", None),
                "native_id": getattr(o, "orderId", None),
                "account":   getattr(o, "account", None) or self.account_id,
                "symbol":    c.symbol,
                "con_id":    c.conId,
                "sec_type":  c.secType,
                "side":      o.action,
                "qty":       o.totalQuantity,
                "filled":    trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "limit_px":  lmt,
                "status":    trade.orderStatus.status,
            })
        return pd.DataFrame(rows)

    # ── private: fill-retry orchestration ────────────────────────────────────

    def _place_with_retry(
        self,
        side:              OrderSide,
        con_id:            int,
        qty:               int,
        mode:              PriceMode,
        offset:            float,
        fill_timeout_secs: int,
        max_retries:       int,
        fallback_mode:     Optional[PriceMode] = None,
        fallback_after:    Optional[int]       = None,
        last_resort_mode:  Optional[PriceMode] = None,
    ) -> OrderResult:
        """
        Submit a limit option order and reprice/resubmit if unfilled.

        Mode ladder
        ───────────
        Attempts 0 .. fallback_after-1   → `mode`           (e.g. IB_MODEL)
        Attempts fallback_after .. N-2   → `fallback_mode`  (e.g. MID)
        Attempt  N-1 (final)             → `last_resort_mode` if set
                                           (e.g. NATURAL for stop losses)

        If fallback_mode / fallback_after are not provided, every attempt
        uses `mode`.

        Partial fill handling
        ─────────────────────
        `total_filled` accumulates across all attempts. Each new attempt
        only submits `qty - total_filled` contracts, so there is no risk of
        accidentally doubling up on already-filled contracts.

        Return value
        ────────────
        The OrderResult for the *last* attempt submitted.
        """
        contract = self.resolve_con_id(con_id)
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        total_filled = 0
        total_cost   = 0.0
        last_result: Optional[OrderResult] = None
        last_bid: Optional[float] = None
        last_ask: Optional[float] = None

        total_attempts = max_retries + 1

        for attempt in range(total_attempts):
            remaining_qty = qty - total_filled
            if remaining_qty <= 0:
                break

            # ── select mode for this attempt ─────────────────────────────
            attempt_mode = self.mode_for_attempt(
                attempt          = attempt,
                total_attempts   = total_attempts,
                mode             = mode,
                fallback_mode    = fallback_mode,
                fallback_after   = fallback_after,
                last_resort_mode = last_resort_mode,
            )

            # ── price and submit ──────────────────────────────────────────
            try:
                limit_px, last_bid, last_ask = self._option_limit_price(contract, side, attempt_mode, offset)
            except RuntimeError as e:
                print(
                    f"[EXEC] ⚠  attempt {attempt + 1}/{total_attempts}: {e}"
                )
                if attempt < max_retries:
                    self.ib.sleep(3)
                    continue
                print("[EXEC] ⚠  All attempts failed — no valid market data")
                break
            order = self._route_order(LimitOrder(
                action        = side.value,
                totalQuantity = remaining_qty,
                lmtPrice      = limit_px,
            ))
            trade = self.ib.placeOrder(contract, order)

            attempt_label = f"attempt {attempt + 1}/{total_attempts}"
            print(
                f"[EXEC] {attempt_label}  "
                f"{side.value} {remaining_qty}x {contract.symbol} "
                f"opt con_id={con_id}  lmt={limit_px:.2f}  mode={attempt_mode.value}"
                f"{('  account=' + self.account_id) if self.account_id else ''}"
            )

            last_result = self._order_result(
                contract=contract,
                trade=trade,
                side=side,
                qty=remaining_qty,
                limit_px=limit_px,
                requested_qty=qty,
            )

            # ── wait for fill ─────────────────────────────────────────────
            deadline = time.monotonic() + fill_timeout_secs
            while time.monotonic() < deadline:
                self.ib.sleep(_POLL_INTERVAL_SECS)
                if trade.orderStatus.status == "Filled":
                    break

            was_fully_filled = trade.orderStatus.status == "Filled"
            retry_allowed = True

            # ── cancel if not fully filled ────────────────────────────────
            if not was_fully_filled:
                still_live = trade.orderStatus.status in _PENDING_STATUSES
                if still_live:
                    self.ib.cancelOrder(trade.order)
                    cancel_status = self.wait_until_not_live(
                        last_result,
                        timeout_secs=max(5.0, min(float(fill_timeout_secs), 15.0)),
                    )
                    if cancel_status in _PENDING_STATUSES:
                        retry_allowed = False
                        print(
                            f"[EXEC] ⚠  Cancel not yet acknowledged for order_id={last_result.order_id} "
                            f"(status={cancel_status}). Not resubmitting to avoid overlapping live orders."
                        )

            # ── account for fills AFTER any cancel to capture late
            #    fill notifications delivered during the cancel wait ────
            filled_this_attempt = last_result.filled_qty()
            total_filled += filled_this_attempt
            for f in trade.fills:
                total_cost += f.execution.shares * f.execution.price

            if last_result.status() == "Filled" or total_filled >= qty:
                if filled_this_attempt > 0:
                    print(
                        f"[EXEC] ✓ Filled {filled_this_attempt} contracts "
                        f"@ avg {last_result.avg_fill():.2f}  "
                        f"(total filled: {total_filled}/{qty})"
                    )
                break

            if not retry_allowed:
                break

            if attempt < max_retries:
                # Log using bid/ask from the pricing call that will happen
                # at the top of the next iteration — no separate request needed.
                mid_str = (
                    f"last mid={((last_bid + last_ask) / 2):.2f}"
                    if (last_bid and last_ask) else "mid unavailable"
                )
                print(
                    f"[EXEC] ✗ Not filled after {fill_timeout_secs}s  "
                    f"(filled {filled_this_attempt}/{remaining_qty} this attempt, "
                    f"{total_filled}/{qty} total)  "
                    f"{mid_str}  — repricing and retrying"
                )
            else:
                unfilled = qty - total_filled
                print(
                    f"[EXEC] ⚠  {unfilled}/{qty} contracts UNFILLED after "
                    f"{total_attempts} attempts.  "
                    f"Returning last result.  "
                    f"Review with 'plays' or retry manually."
                )

        if last_result is None:
            raise RuntimeError(
                f"Could not submit any order for con_id={con_id} — "
                f"market data unavailable across all {total_attempts} attempts"
            )
        last_result.total_filled = total_filled
        last_result.total_cost   = total_cost
        last_result.requested_qty = qty
        last_result.unfilled_qty = max(0, qty - total_filled)
        last_result.last_order_live = self.is_live(last_result)
        last_result.cancel_unresolved = last_result.last_order_live
        return last_result

    # ── private: stock order ─────────────────────────────────────────────────

    def _stock_order(
        self,
        side:        OrderSide,
        symbol:      str,
        qty:         int,
        limit_price: Optional[float],
    ) -> OrderResult:
        contract = Stock(symbol.upper(), self.exchange, self.currency)
        self.ib.qualifyContracts(contract)

        if limit_price is not None:
            order = self._route_order(LimitOrder(
                action        = side.value,
                totalQuantity = qty,
                lmtPrice      = round(limit_price, 2),
            ))
            order_desc = f"LMT {limit_price:.2f}"
        else:
            order = self._route_order(MarketOrder(
                action        = side.value,
                totalQuantity = qty,
            ))
            order_desc = "MKT"

        trade = self.ib.placeOrder(contract, order)

        print(
            f"[EXEC] {side.value} {qty}x {symbol.upper()} "
            f"stock  {order_desc}"
            f"{('  account=' + self.account_id) if self.account_id else ''}"
        )
        return self._order_result(
            contract=contract,
            trade=trade,
            side=side,
            qty=qty,
            limit_px=limit_price,
            requested_qty=qty,
        )

    # ── contract resolution ───────────────────────────────────────────────────

    def resolve_con_id(self, con_id: int) -> Contract:
        """
        Reconstruct a fully qualified Contract from a con_id alone.

        IB requires the full contract object for order placement.
        reqContractDetails with only conId is the standard server-side
        lookup — it works for any sec_type including OPT.
        """
        detail = self._contract_details(con_id)
        if detail is None:
            raise ValueError(
                f"IB returned no contract details for con_id={con_id}. "
                "Contract may be expired or the con_id is invalid."
            )
        return detail.contract

    def submit_option_order(
        self,
        side:   OrderSide,
        con_id: int,
        qty:    int,
        mode:   PriceMode = PriceMode.IB_MODEL,
        offset: float     = 0.0,
    ) -> OrderResult:
        """
        Submit one option limit order attempt and return immediately.

        Strategy uses this for non-blocking exits so the main risk loop can
        keep running while the order works at the exchange.
        """
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        contract = self.resolve_con_id(con_id)
        limit_px, _, _ = self._option_limit_price(contract, side, mode, offset)
        order = self._route_order(LimitOrder(
            action        = side.value,
            totalQuantity = qty,
            lmtPrice      = limit_px,
        ))
        trade = self.ib.placeOrder(contract, order)
        print(
            f"[EXEC] submit  {side.value} {qty}x {contract.symbol} "
            f"opt con_id={con_id}  lmt={limit_px:.2f}  mode={mode.value}"
            f"{('  account=' + self.account_id) if self.account_id else ''}"
        )
        return self._order_result(
            contract=contract,
            trade=trade,
            side=side,
            qty=qty,
            limit_px=limit_px,
            requested_qty=qty,
        )

    @staticmethod
    def mode_for_attempt(
        attempt:          int,
        total_attempts:   int,
        mode:             PriceMode,
        fallback_mode:    Optional[PriceMode],
        fallback_after:   Optional[int],
        last_resort_mode: Optional[PriceMode],
    ) -> PriceMode:
        """Pricing mode selection shared by blocking and non-blocking flows."""
        is_last = attempt == total_attempts - 1
        if is_last and last_resort_mode is not None:
            return last_resort_mode
        if (
            fallback_mode is not None
            and fallback_after is not None
            and attempt >= fallback_after
        ):
            return fallback_mode
        return mode

    # ── private: price computation ────────────────────────────────────────────

    def _quote_option_contract(
        self, contract: Contract, wait_secs: float = 2.0
    ) -> tuple[Optional[float], Optional[float], object]:
        """
        Request option market data long enough to populate bid/ask/model greeks,
        then immediately cancel the subscription to avoid ticker buildup.
        """
        ticker = self.ib.reqMktData(contract, "106")
        try:
            self.ib.sleep(wait_secs)
            bid = ticker.bid if (ticker.bid and ticker.bid > 0) else None
            ask = ticker.ask if (ticker.ask and ticker.ask > 0) else None
            return bid, ask, ticker.modelGreeks
        finally:
            self.ib.cancelMktData(contract)

    def _contract_details(self, con_id: int):
        detail = self._contract_details_cache.get(con_id)
        if detail is not None:
            return detail
        stub    = Contract(conId=con_id)
        details = self.ib.reqContractDetails(stub)
        if not details:
            return None
        detail = details[0]
        self._contract_details_cache[con_id] = detail
        return detail

    def _price_increment(self, contract: Contract, raw_price: float) -> float:
        detail = self._contract_details(contract.conId)
        if detail is None:
            return 0.01

        increment = float(getattr(detail, "minTick", 0.01) or 0.01)
        market_rule_ids = str(getattr(detail, "marketRuleIds", "") or "")
        for raw_rule_id in market_rule_ids.split(","):
            raw_rule_id = raw_rule_id.strip()
            if not raw_rule_id:
                continue
            rule_id = int(raw_rule_id)
            rules = self._market_rule_cache.get(rule_id)
            if rules is None:
                rules = list(self.ib.reqMarketRule(rule_id))
                self._market_rule_cache[rule_id] = rules
            if not rules:
                continue
            for rule in rules:
                if raw_price >= rule.lowEdge:
                    increment = float(rule.increment)
                else:
                    break
            break
        return increment if increment > 0 else 0.01

    def _snap_option_price(
        self,
        contract: Contract,
        side:     OrderSide,
        raw_price: float,
    ) -> float:
        increment = self._price_increment(contract, raw_price)
        steps = raw_price / increment
        if side is OrderSide.BUY:
            snapped_steps = math.ceil(steps - 1e-12)
        else:
            snapped_steps = math.floor(steps + 1e-12)
        snapped = max(increment, snapped_steps * increment)
        return round(snapped, 6)

    def _option_limit_price(
        self,
        contract: Contract,
        side:     OrderSide,
        mode:     PriceMode,
        offset:   float,
    ) -> tuple[float, Optional[float], Optional[float]]:
        """
        Request a single market-data snapshot for `contract` and compute a
        limit price according to `mode`.

        Called before every submission attempt (initial + retries), so the
        price always reflects current market conditions.

        MID (default)
        ─────────────
        Simple (bid + ask) / 2.  Good edge without asking for too much.
        Repriced on every retry so stale mid is never the reason for
        repeated non-fills.

        IB_MODEL clamping
        ─────────────────
        modelGreeks.optPrice is IB's own theoretical fair value. It can
        occasionally sit outside [bid, ask] due to stale greeks near open /
        close or during fast markets.  We clamp it to [bid, ask] so we never
        submit a clearly outside-market order, which IB would reject anyway.
        Falls back to MID if the model price is unavailable.

        NATURAL + offset
        ────────────────
        offset shifts the natural price by `offset` dollars toward mid,
        bounded so we never cross mid:
          BUY  → ask − offset  (clamped: no lower than mid)
          SELL → bid + offset  (clamped: no higher than mid)
        """
        bid, ask, g = self._quote_option_contract(contract)

        if bid is None or ask is None:
            raise RuntimeError(
                f"No valid bid/ask for con_id={contract.conId} "
                f"({contract.symbol}).  "
                "Market may be closed or contract is illiquid."
            )

        mid = (bid + ask) / 2

        if mode is PriceMode.MID:
            return self._snap_option_price(contract, side, mid), bid, ask

        if mode is PriceMode.IB_MODEL:
            model_px = (
                g.optPrice
                if (g and g.optPrice and g.optPrice > 0)
                else None
            )
            if model_px is not None:
                clamped = max(bid, min(ask, model_px))
                return self._snap_option_price(contract, side, clamped), bid, ask
            # Model price unavailable (outside hours, illiquid) — fall back to mid.
            print(
                f"[EXEC] IB model price unavailable for con_id={contract.conId} "
                "— falling back to MID"
            )
            return self._snap_option_price(contract, side, mid), bid, ask

        if mode is PriceMode.NATURAL:
            natural = ask if side is OrderSide.BUY else bid
            if offset > 0:
                if side is OrderSide.BUY:
                    shifted = natural - offset
                    shifted = max(mid, min(ask, shifted))   # stay in [mid, ask]
                else:
                    shifted = natural + offset
                    shifted = min(mid, max(bid, shifted))   # stay in [bid, mid]
                return self._snap_option_price(contract, side, shifted), bid, ask
            return self._snap_option_price(contract, side, natural), bid, ask

        raise ValueError(f"Unknown PriceMode: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK  (python execution.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ib_core import OptionChain, Account, connect, Right

    ib = connect()
    ex = Executor(ib)

    # ── inspect pending orders ───────────────────────────────────────────────
    pending = ex.pending_orders()
    print("\nPending orders:")
    print(pending if not pending.empty else "  (none)")

    # ── option example (read-only price check, no order placed) ─────────────
    chain = OptionChain(ib, "MRNA")
    df    = chain.fetch(expiry_count=1, strike_width=0.15)
    atm   = chain.filter(df, right=Right.CALL, delta_min=0.40, delta_max=0.60)

    if not atm.empty:
        row    = atm.iloc[0]
        con_id = int(row["con_id"])
        print(
            f"\nTop ATM call: con_id={con_id}  "
            f"strike={row['strike']}  expiry={row['expiry']}  "
            f"bid={row['bid']}  ask={row['ask']}  mid={row['mid']}"
        )
        print("To place at mid (default):      ex.buy_option(con_id, qty=1)")
        print("To place at IB model:           ex.buy_option(con_id, qty=1, mode=PriceMode.IB_MODEL)")
        print("To place with 60s timeout:      ex.buy_option(con_id, qty=1, fill_timeout_secs=60)")

    ib.disconnect()
