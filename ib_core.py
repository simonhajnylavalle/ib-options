"""
ib_core.py
──────────────────────────────────────────────────────────────────────────────
Interactive Brokers foundation: connection, option chain, account snapshot.

Two responsibilities:
  1. OptionChain   — fetch and filter an option chain for any symbol
  2. Account       — pull account values and current positions

The AccountSnapshot dataclass is the clean handoff to portfolio.py —
portfolio never imports IB types directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

import logging

import pandas as pd
from ib_insync import IB, Option, Stock


_MARKET_TZ = ZoneInfo("America/New_York")


def _market_date():
    return datetime.now(_MARKET_TZ).date()


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class Right(str, Enum):
    """Option right — prevents magic-string bugs downstream."""
    CALL = "C"
    PUT  = "P"


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def connect(
    host:      str = "127.0.0.1",
    port:      int = 4001,
    client_id: int = 1,
) -> IB:
    """Connect to TWS / IB Gateway. port 4001 = Gateway, 7497 = TWS paper."""
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    print(f"Connected  |  accounts: {ib.managedAccounts()}")
    return ib


# ─────────────────────────────────────────────────────────────────────────────
# OPTION CHAIN
# ─────────────────────────────────────────────────────────────────────────────

class OptionChain:
    """
    Fetches a raw option chain for a single symbol and exposes
    fetch / filter / select methods to find tradeable contracts.

    Usage (select — preferred for strategy entries):
        chain = OptionChain(ib, "MRNA")
        picks = chain.select(
            right=Right.CALL, delta_min=0.40, delta_max=0.55,
            dte_min=25, dte_max=50, strike_width=0.25,
        )

    Usage (manual fetch + filter — for raw exploration):
        chain = OptionChain(ib, "MRNA")
        df    = chain.fetch(expiry_count=2, strike_width=0.20)
        atm   = chain.filter(df, right=Right.CALL, delta_min=0.35, delta_max=0.65)
    """

    _STOCK_CACHE: dict[tuple[str, str], Stock] = {}
    _SECDEF_CACHE: dict[tuple[str, str], list] = {}
    _MAX_CONTRACTS: int = 240

    def __init__(self, ib: IB, symbol: str, exchange: str = "SMART"):
        self.ib       = ib
        self.symbol   = symbol.upper()
        self.exchange = exchange
        self.spot: Optional[float] = None  # populated after first fetch

    # ── public ──────────────────────────────────────────────────────────────

    def fetch(
        self,
        expiry_count: int            = 5,     # how many nearest expiries to load
        strike_width: float          = 0.3,   # ±% around spot  (0.20 = ±20%)
        batch_size:   int            = 50,    # IB snapshot rate limit
        rights:       list[str] | None = None,  # ["C"], ["P"], or None for both
        dte_min:      Optional[int]  = None,  # only load expiries ≥ dte_min
        dte_max:      Optional[int]  = None,  # only load expiries ≤ dte_max
        spot_price:   Optional[float] = None,  # reuse a known spot when available
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per (expiry, strike, right).

        When dte_min / dte_max are given, only expiries within that DTE
        window are loaded (ignoring expiry_count).  This avoids fetching
        hundreds of near-term weeklies when you only want 30-50 DTE.

        When neither is given, falls back to loading the nearest
        expiry_count expiries (original behaviour).

        Columns:
          expiry, dte, strike, right,
          bid, ask, mid, spread, spread_pct,
          volume, open_interest,
          iv, delta, gamma, theta, vega, con_id
        """
        stock = self._stock_contract()
        chains = self._option_params(stock)
        if not chains:
            raise ValueError(f"No option chain found for {self.symbol}")

        chain       = next((c for c in chains if c.exchange == "SMART"), chains[0])
        all_expiries = sorted(chain.expirations)

        today = _market_date()
        if dte_min is not None or dte_max is not None:
            # select only expiries within the DTE window
            expiries = []
            for exp_str in all_expiries:
                dte = (datetime.strptime(exp_str, "%Y%m%d").date() - today).days
                if dte_min is not None and dte < dte_min:
                    continue
                if dte_max is not None and dte > dte_max:
                    continue
                expiries.append(exp_str)
        else:
            expiries = all_expiries[:expiry_count]

        if not expiries:
            raise ValueError(
                f"No expiries for {self.symbol} in DTE range "
                f"[{dte_min or 0}, {dte_max or '∞'}]"
            )

        self.spot = float(spot_price) if spot_price and spot_price > 0 else self._spot(stock)
        lo = self.spot * (1 - strike_width)
        hi = self.spot * (1 + strike_width)
        strikes = [s for s in sorted(chain.strikes) if lo <= s <= hi]

        print(
            f"{self.symbol}  spot=${self.spot:.2f}  "
            f"expiries={expiries}  strikes={len(strikes)}"
        )

        rights_to_fetch = rights if rights else [Right.CALL.value, Right.PUT.value]
        contracts = [
            Option(self.symbol, exp, strike, right, "SMART")
            for exp in expiries
            for strike in strikes
            for right in rights_to_fetch
        ]
        if len(contracts) > self._MAX_CONTRACTS:
            contracts = sorted(
                contracts,
                key=lambda c: (
                    abs(float(c.strike) - self.spot),
                    c.lastTradeDateOrContractMonth,
                    c.right,
                ),
            )[: self._MAX_CONTRACTS]
            print(f"  trimmed to {len(contracts)} contracts nearest spot for pacing safety")

        # Suppress IB "no security definition" errors (code 200) during
        # qualification — expected for strike/expiry combos without contracts
        _ib_log = logging.getLogger('ib_insync')
        _prev_level = _ib_log.level
        _ib_log.setLevel(logging.CRITICAL)
        self.ib.qualifyContracts(*contracts)
        _ib_log.setLevel(_prev_level)

        valid   = [c for c in contracts if c.conId]
        skipped = len(contracts) - len(valid)
        contracts = valid
        if skipped:
            print(f"  {len(contracts)} contracts qualified  ({skipped} invalid combos skipped)")

        # 101 = open interest, 106 = implied vol + model greeks
        # streaming mode required — snapshot rejects generic ticks
        #
        # Subscribe in batches (IB rate limit), but keep all subscriptions
        # alive until a single longer wait at the end.  This gives IB time
        # to populate less-liquid contracts (30+ DTE biotech options can be
        # slow) instead of cancelling after just 2 seconds per batch.
        tickers = []
        for i in range(0, len(contracts), batch_size):
            batch = contracts[i : i + batch_size]
            batch_tickers = [self.ib.reqMktData(c, "101,106") for c in batch]
            tickers.extend(batch_tickers)
            self.ib.sleep(1)  # brief pause between batches for rate limit

        # one wait for all subscriptions to populate
        self.ib.sleep(5)

        for c in contracts:
            self.ib.cancelMktData(c)

        return self._to_dataframe(tickers)

    def filter(
        self,
        df:                 pd.DataFrame,
        right:              Optional[Right | str] = None,   # None = both
        max_spread_pct:     float                 = 15.0,
        min_open_interest:  int                   = 50,
        min_volume:         int                   = 5,
        delta_min:          Optional[float]       = None,   # abs(delta)
        delta_max:          Optional[float]       = None,   # abs(delta)
        dte_min:            Optional[int]         = None,
        dte_max:            Optional[int]         = None,
    ) -> pd.DataFrame:
        """
        Filter option rows, always returning the best available contract.

        Hard filters (define what you want — rows are always dropped):
          right, dte_min/dte_max

        Soft filters (used for ranking, relaxed if nothing passes):
          max_spread_pct, min_open_interest, min_volume, delta_min/delta_max

        If no rows survive the soft liquidity filters, they are relaxed to
        keep every row that passed right + DTE, so you always get the
        least-bad contract for illiquid biotech names.

        Sort order (4 tiers):
          1. liquidity_ok   — meets spread/OI/volume thresholds (preferred)
          2. delta_in_range — valid delta in [min, max]         (preferred)
          3. spread_pct     — tightest spread first
          4. open_interest  — most liquid first
        """
        out = df.copy()

        # ── hard filters: right + DTE (these define the contract type) ─────
        if right is not None:
            right_val = right.value if isinstance(right, Right) else right
            out = out[out["right"] == right_val]

        if dte_min is not None:
            out = out[out["dte"] >= dte_min]
        if dte_max is not None:
            out = out[out["dte"] <= dte_max]

        # must have a valid bid/ask to be tradeable at all
        out = out[out["spread_pct"].notna()]

        if out.empty:
            return out

        # ── soft filters: liquidity thresholds ─────────────────────────────
        liq_mask = (
            (out["spread_pct"] <= max_spread_pct)
            & (out["open_interest"].fillna(0) >= min_open_interest)
            & (out["volume"].fillna(0) >= min_volume)
        )

        if liq_mask.any():
            # tag rows: 0 = passes liquidity, 1 = fallback
            out["_liq_rank"] = (~liq_mask).astype(int)
        else:
            # nothing passes — keep everything, rank by spread + OI
            out["_liq_rank"] = 0

        # ── soft filter: delta preference ──────────────────────────────────
        has_delta_filter = delta_min is not None or delta_max is not None
        if has_delta_filter:
            abs_delta = out["delta"].abs()
            in_range = pd.Series(True, index=out.index)
            if delta_min is not None:
                in_range = in_range & (abs_delta >= delta_min)
            if delta_max is not None:
                in_range = in_range & (abs_delta <= delta_max)
            # NaN delta → not in_range but not dropped
            out["_delta_rank"] = (~in_range).astype(int)
        else:
            out["_delta_rank"] = 0

        out = out.sort_values(
            ["_liq_rank", "_delta_rank", "spread_pct", "open_interest"],
            ascending=[True, True, True, False],
        ).drop(columns=["_liq_rank", "_delta_rank"]).reset_index(drop=True)

        return out

    # ── select: fetch + filter in one call ───────────────────────────────

    def select(
        self,
        right:             Optional[Right | str] = None,
        delta_min:         Optional[float] = None,
        delta_max:         Optional[float] = None,
        dte_min:           Optional[int]   = None,
        dte_max:           Optional[int]   = None,
        strike_width:      float           = 0.25,
        expiry_count:      int             = 5,
        max_spread_pct:    float           = 15.0,
        min_open_interest: int             = 50,
        min_volume:        int             = 5,
        spot_price:        Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Fetch + filter + rank in one call.

        All parameters are explicit — no external spec object needed.
        Returns a ranked DataFrame of tradeable contracts (best first),
        or empty if nothing qualifies.
        """
        rights = None
        if right is not None:
            rv = right.value if isinstance(right, Right) else right
            rights = [rv]

        df = self.fetch(
            strike_width = strike_width,
            dte_min      = dte_min,
            dte_max      = dte_max,
            expiry_count = expiry_count,
            rights       = rights,
            spot_price   = spot_price,
        )
        if df.empty:
            return df

        return self.filter(
            df,
            right             = right,
            delta_min         = delta_min,
            delta_max         = delta_max,
            max_spread_pct    = max_spread_pct,
            min_open_interest = min_open_interest,
            min_volume        = min_volume,
            dte_min           = dte_min,
            dte_max           = dte_max,
        )

    # ── private ─────────────────────────────────────────────────────────────

    def _spot(self, stock: Stock) -> float:
        t = self.ib.reqMktData(stock, "")
        try:
            self.ib.sleep(4)
        finally:
            self.ib.cancelMktData(stock)
        price = next(
            (v for v in (t.last, t.close, t.bid, t.marketPrice())
             if v and v == v and v > 0),
            None,
        )
        if price is None:
            raise RuntimeError(f"Could not get spot price for {self.symbol}")
        return float(price)

    def _stock_contract(self) -> Stock:
        key = (self.symbol, self.exchange)
        stock = self._STOCK_CACHE.get(key)
        if stock is not None:
            return stock
        stock = Stock(self.symbol, self.exchange, "USD")
        self.ib.qualifyContracts(stock)
        self._STOCK_CACHE[key] = stock
        return stock

    def _option_params(self, stock: Stock):
        key = (self.symbol, self.exchange)
        cached = self._SECDEF_CACHE.get(key)
        if cached is not None:
            return cached
        chains = self.ib.reqSecDefOptParams(
            stock.symbol, "", stock.secType, stock.conId
        )
        self._SECDEF_CACHE[key] = chains
        return chains

    def _to_dataframe(self, tickers: list) -> pd.DataFrame:
        today = _market_date()
        rows  = []
        for t in tickers:
            c          = t.contract
            bid        = t.bid  if t.bid  and t.bid  > 0 else None
            ask        = t.ask  if t.ask  and t.ask  > 0 else None
            mid        = round((bid + ask) / 2, 4) if bid and ask else None
            spread     = round(ask - bid, 4)        if bid and ask else None
            spread_pct = round(spread / ask * 100, 2) if (spread is not None and ask) else None
            exp_date   = datetime.strptime(
                c.lastTradeDateOrContractMonth, "%Y%m%d"
            ).date()
            g = t.modelGreeks
            rows.append({
                "expiry":        c.lastTradeDateOrContractMonth,
                "dte":           (exp_date - today).days,
                "strike":        c.strike,
                "right":         c.right,
                "bid":           bid,
                "ask":           ask,
                "mid":           mid,
                "spread":        spread,
                "spread_pct":    spread_pct,
                "volume":        t.volume,
                "open_interest": (t.callOpenInterest if c.right == "C"
                                  else t.putOpenInterest),
                "iv":            round(t.impliedVolatility * 100, 2)
                                 if t.impliedVolatility else None,
                "delta":         g.delta if g else None,
                "gamma":         g.gamma if g else None,
                "theta":         g.theta if g else None,
                "vega":          g.vega  if g else None,
                "con_id":        c.conId,
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ACCOUNT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AccountSnapshot:
    """
    Point-in-time account view. All monetary values in USD.
    This is the only object portfolio.py ever receives from ib_core —
    the clean boundary between data-fetching and risk-measurement.

    account_id     — selected IB account for this snapshot
    nav            — total equity (Net Asset Value)
    cash           — available cash
    buying_power   — marginable buying power
    unrealized_pnl — open position P&L
    realized_pnl   — closed P&L this session
    positions      — DataFrame of current holdings (empty if flat)

    Positions DataFrame columns:
      account, symbol, con_id, sec_type, expiry, strike, right,
      position, avg_cost, market_value, unrealized_pnl
    """
    account_id:     Optional[str]
    nav:            float
    cash:           float
    buying_power:   float
    unrealized_pnl: float
    realized_pnl:   float
    positions:      pd.DataFrame


class Account:
    """
    Pulls a fresh AccountSnapshot from IB on demand.

    Usage:
        acc      = Account(ib)
        snapshot = acc.snapshot()
        print(snapshot.nav, snapshot.unrealized_pnl)
        print(snapshot.positions)
    """

    _TAG_MAP = {
        "NetLiquidation":  "nav",
        "TotalCashValue":  "cash",
        "BuyingPower":     "buying_power",
        "UnrealizedPnL":   "unrealized_pnl",
        "RealizedPnL":     "realized_pnl",
    }

    def __init__(
        self,
        ib: IB,
        base_currency: str = "CHF",
        account_id: Optional[str] = None,
    ):
        self.ib = ib
        self._base_currency = base_currency
        self.account_id = self._resolve_account_id(account_id)

    def snapshot(self) -> AccountSnapshot:
        values    = self._account_values()
        positions = self._positions()
        return AccountSnapshot(
            account_id     = self.account_id,
            nav            = values.get("nav",            0.0),
            cash           = values.get("cash",           0.0),
            buying_power   = values.get("buying_power",   0.0),
            unrealized_pnl = values.get("unrealized_pnl", 0.0),
            realized_pnl   = values.get("realized_pnl",   0.0),
            positions      = positions,
        )

    def print_snapshot(self) -> None:
        s = self.snapshot()
        print(f"\n{'═'*45}")
        print(f"  Account Snapshot")
        print(f"{'═'*45}")
        if s.account_id:
            print(f"  Account        : {s.account_id}")
        print(f"  NAV            : ${s.nav:>12,.2f}")
        print(f"  Cash           : ${s.cash:>12,.2f}")
        print(f"  Buying Power   : ${s.buying_power:>12,.2f}")
        print(f"  Unrealized P&L : ${s.unrealized_pnl:>12,.2f}")
        print(f"  Realized P&L   : ${s.realized_pnl:>12,.2f}")
        print(f"{'─'*45}")
        if s.positions.empty:
            print("  No open positions")
        else:
            print(s.positions.to_string(index=False))
        print(f"{'═'*45}\n")

    # ── private ─────────────────────────────────────────────────────────────

    def _resolve_account_id(self, account_id: Optional[str]) -> Optional[str]:
        managed = [acct for acct in self.ib.managedAccounts() if acct]
        if account_id:
            if managed and account_id not in managed:
                raise ValueError(
                    f"Configured account_id={account_id!r} is not in "
                    f"managed accounts {managed}"
                )
            return account_id
        if len(managed) == 1:
            return managed[0]
        if len(managed) > 1:
            chosen = managed[0]
            print(
                f"[ACCOUNT] Multiple managed accounts detected {managed}. "
                f"Defaulting to {chosen}. Set general.account_id in config.toml "
                f"to pin a specific account."
            )
            return chosen
        return None

    def _account_values(self) -> dict:
        """
        Pull account values via accountSummary() — synchronous, no sleep
        or subscription needed.  Filter by currency to avoid multi-currency
        accounts returning wrong values (last-write-wins).

        Priority: BASE (IB's base-currency denomination) is preferred.
        Falls back to the configured base currency (e.g. CHF) if BASE
        rows are unavailable for a given tag.
        """
        base_results: dict[str, float] = {}
        fallback_results: dict[str, float] = {}
        for item in self.ib.accountSummary(self.account_id or ""):
            field = self._TAG_MAP.get(item.tag)
            if not field:
                continue
            try:
                val = float(item.value)
            except ValueError:
                continue
            if item.currency == "BASE":
                base_results[field] = val
            elif item.currency == self._base_currency:
                fallback_results[field] = val
        # Prefer BASE rows; fall back to configured base currency.
        return {**fallback_results, **base_results}

    def _positions(self) -> pd.DataFrame:
        rows = []
        for p in self.ib.positions(self.account_id or ""):
            c = p.contract
            rows.append({
                "account":        getattr(p, "account", self.account_id),
                "symbol":         c.symbol,
                "con_id":         c.conId,
                "sec_type":       c.secType,
                "expiry":         getattr(c, "lastTradeDateOrContractMonth", None),
                "strike":         getattr(c, "strike", None),
                "right":          getattr(c, "right",  None),
                "position":       p.position,
                "avg_cost":       round(p.avgCost, 4),
                "market_value":   None,
                "unrealized_pnl": None,
            })

        if not rows:
            return pd.DataFrame()

        portfolio_map = {
            (getattr(item, "account", self.account_id), item.contract.conId): item
            for item in self.ib.portfolio(self.account_id or "")
        }
        for row in rows:
            item = portfolio_map.get((row["account"], row["con_id"]))
            if item:
                row["market_value"]   = round(item.marketValue, 2)
                row["unrealized_pnl"] = round(item.unrealizedPNL, 2)

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK  (python ib_core.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ib = connect()
    try:
        symbol = "CRSP"
        chain  = OptionChain(ib, symbol)

        cols = ["expiry", "dte", "strike", "right", "bid", "ask",
                "mid", "spread_pct", "delta", "open_interest", "con_id"]

        # ── raw chain (3 nearest expiries, unfiltered) ────────────────────
        df   = chain.fetch(expiry_count=3, strike_width=0.20)
        spot = chain.spot
        print(f"\n{len(df)} contracts fetched  |  spot=${spot:.2f}")
        if not df.empty:
            for exp, grp in df.groupby("expiry"):
                dte = grp["dte"].iloc[0]
                print(f"\n── {exp}  (DTE {dte}) ──")
                print(grp[cols].to_string(index=False))

        # ── spec-based selection via select() kwargs ──────────────────────
        specs = {
            "THESIS":   dict(right=Right.CALL, delta_min=0.40, delta_max=0.55,
                             dte_min=25, dte_max=50, strike_width=0.25),
            "APPROACH":  dict(right=Right.CALL, delta_min=0.30, delta_max=0.45,
                             dte_min=25, dte_max=45, strike_width=0.20),
            "SENTINEL":  dict(right=Right.CALL, delta_min=0.15, delta_max=0.30,
                             dte_min=60, dte_max=120, strike_width=0.30,
                             max_spread_pct=25.0, min_open_interest=10, min_volume=1),
            "SNIPER":    dict(right=Right.CALL, delta_min=0.35, delta_max=0.55,
                             dte_min=2, dte_max=14, strike_width=0.20),
            "CUSTOM":    dict(right=Right.CALL, delta_min=0.15, delta_max=0.40,
                             dte_min=30, dte_max=50, strike_width=0.25,
                             max_spread_pct=25.0, min_open_interest=10, min_volume=1),
        }

        for label, kw in specs.items():
            picks = chain.select(**kw)  # type: ignore[arg-type]
            print(f"\n{'═' * 60}")
            if picks.empty:
                print(f"  {label}: no matches")
                continue
            top = picks.iloc[0]
            delta_s = f"{top['delta']:.2f}" if pd.notna(top["delta"]) else "N/A"
            print(f"  {label}  ({len(picks)} candidates)")
            show = [c for c in cols if c in picks.columns]
            print(picks[show].head(5).to_string(index=False))
            print(f"\n  Top: con_id={int(top['con_id'])}  "
                  f"strike={top['strike']}  DTE={top['dte']}  delta={delta_s}")

        Account(ib).print_snapshot()
    finally:
        ib.disconnect()
