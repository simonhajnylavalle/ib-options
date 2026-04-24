"""
Interactive Brokers foundation: connection, option chain, and account snapshots.

The rest of the application depends on this module for a small boundary around
ib_insync types:

- OptionChain fetches, filters, and ranks option chains.
- Account pulls account values and current positions into AccountSnapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from ib_insync import IB, Option, Stock


_MARKET_TZ = ZoneInfo("America/New_York")


def _market_date():
    return datetime.now(_MARKET_TZ).date()


class Right(str, Enum):
    """Option right."""

    CALL = "C"
    PUT = "P"


def connect(
    host: str = "127.0.0.1",
    port: int = 4001,
    client_id: int = 1,
) -> IB:
    """Connect to TWS or IB Gateway. Port 4001 is Gateway; 7497 is TWS paper."""

    ib = IB()
    ib.connect(host, port, clientId=client_id)
    print(f"Connected  |  accounts: {ib.managedAccounts()}")
    return ib


class OptionChain:
    """
    Fetch and rank option contracts for a single symbol.

    The public select() method is the strategy-facing entrypoint: it fetches a
    chain, applies contract constraints, and returns a ranked DataFrame.
    """

    _STOCK_CACHE: dict[tuple[str, str], Stock] = {}
    _SECDEF_CACHE: dict[tuple[str, str], list] = {}
    _MAX_CONTRACTS: int = 240

    def __init__(self, ib: IB, symbol: str, exchange: str = "SMART"):
        self.ib = ib
        self.symbol = symbol.upper()
        self.exchange = exchange
        self.spot: Optional[float] = None

    def fetch(
        self,
        expiry_count: int = 5,
        strike_width: float = 0.3,
        batch_size: int = 50,
        rights: list[str] | None = None,
        dte_min: Optional[int] = None,
        dte_max: Optional[int] = None,
        spot_price: Optional[float] = None,
    ) -> pd.DataFrame:
        """Return one row per qualified option contract."""

        stock = self._stock_contract()
        chains = self._option_params(stock)
        if not chains:
            raise ValueError(f"No option chain found for {self.symbol}")

        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
        all_expiries = sorted(chain.expirations)

        today = _market_date()
        if dte_min is not None or dte_max is not None:
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
                f"[{dte_min or 0}, {dte_max or 'inf'}]"
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

        ib_log = logging.getLogger("ib_insync")
        previous_level = ib_log.level
        ib_log.setLevel(logging.CRITICAL)
        try:
            self.ib.qualifyContracts(*contracts)
        finally:
            ib_log.setLevel(previous_level)

        valid = [c for c in contracts if c.conId]
        skipped = len(contracts) - len(valid)
        contracts = valid
        if skipped:
            print(f"  {len(contracts)} contracts qualified  ({skipped} invalid combos skipped)")

        tickers = []
        for i in range(0, len(contracts), batch_size):
            batch = contracts[i : i + batch_size]
            tickers.extend(self.ib.reqMktData(c, "101,106") for c in batch)
            self.ib.sleep(1)

        self.ib.sleep(5)

        for c in contracts:
            self.ib.cancelMktData(c)

        return self._to_dataframe(tickers)

    def filter(
        self,
        df: pd.DataFrame,
        right: Optional[Right | str] = None,
        max_spread_pct: float = 15.0,
        min_open_interest: int = 50,
        min_volume: int = 5,
        delta_min: Optional[float] = None,
        delta_max: Optional[float] = None,
        dte_min: Optional[int] = None,
        dte_max: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Apply hard contract filters, then rank by liquidity and delta fit.

        Liquidity and delta checks are soft preferences, so illiquid symbols can
        still return the least-bad eligible contract.
        """

        out = df.copy()

        if right is not None:
            right_val = right.value if isinstance(right, Right) else right
            out = out[out["right"] == right_val]

        if dte_min is not None:
            out = out[out["dte"] >= dte_min]
        if dte_max is not None:
            out = out[out["dte"] <= dte_max]

        out = out[out["spread_pct"].notna()]
        if out.empty:
            return out

        liq_mask = (
            (out["spread_pct"] <= max_spread_pct)
            & (out["open_interest"].fillna(0) >= min_open_interest)
            & (out["volume"].fillna(0) >= min_volume)
        )
        out["_liq_rank"] = (~liq_mask).astype(int) if liq_mask.any() else 0

        has_delta_filter = delta_min is not None or delta_max is not None
        if has_delta_filter:
            abs_delta = out["delta"].abs()
            in_range = pd.Series(True, index=out.index)
            if delta_min is not None:
                in_range = in_range & (abs_delta >= delta_min)
            if delta_max is not None:
                in_range = in_range & (abs_delta <= delta_max)
            out["_delta_rank"] = (~in_range).astype(int)
        else:
            out["_delta_rank"] = 0

        return (
            out.sort_values(
                ["_liq_rank", "_delta_rank", "spread_pct", "open_interest"],
                ascending=[True, True, True, False],
            )
            .drop(columns=["_liq_rank", "_delta_rank"])
            .reset_index(drop=True)
        )

    def select(
        self,
        right: Optional[Right | str] = None,
        delta_min: Optional[float] = None,
        delta_max: Optional[float] = None,
        dte_min: Optional[int] = None,
        dte_max: Optional[int] = None,
        strike_width: float = 0.25,
        expiry_count: int = 5,
        max_spread_pct: float = 15.0,
        min_open_interest: int = 50,
        min_volume: int = 5,
        spot_price: Optional[float] = None,
    ) -> pd.DataFrame:
        """Fetch, filter, and rank contracts in one call."""

        rights = None
        if right is not None:
            right_value = right.value if isinstance(right, Right) else right
            rights = [right_value]

        df = self.fetch(
            strike_width=strike_width,
            dte_min=dte_min,
            dte_max=dte_max,
            expiry_count=expiry_count,
            rights=rights,
            spot_price=spot_price,
        )
        if df.empty:
            return df

        return self.filter(
            df,
            right=right,
            delta_min=delta_min,
            delta_max=delta_max,
            max_spread_pct=max_spread_pct,
            min_open_interest=min_open_interest,
            min_volume=min_volume,
            dte_min=dte_min,
            dte_max=dte_max,
        )

    def _spot(self, stock: Stock) -> float:
        ticker = self.ib.reqMktData(stock, "")
        try:
            self.ib.sleep(4)
        finally:
            self.ib.cancelMktData(stock)
        price = next(
            (
                v
                for v in (ticker.last, ticker.close, ticker.bid, ticker.marketPrice())
                if v and v == v and v > 0
            ),
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
        chains = self.ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        self._SECDEF_CACHE[key] = chains
        return chains

    def _to_dataframe(self, tickers: list) -> pd.DataFrame:
        today = _market_date()
        rows = []
        for ticker in tickers:
            contract = ticker.contract
            bid = ticker.bid if ticker.bid and ticker.bid > 0 else None
            ask = ticker.ask if ticker.ask and ticker.ask > 0 else None
            mid = round((bid + ask) / 2, 4) if bid and ask else None
            spread = round(ask - bid, 4) if bid and ask else None
            spread_pct = round(spread / ask * 100, 2) if (spread is not None and ask) else None
            exp_date = datetime.strptime(
                contract.lastTradeDateOrContractMonth, "%Y%m%d"
            ).date()
            greeks = ticker.modelGreeks
            rows.append(
                {
                    "expiry": contract.lastTradeDateOrContractMonth,
                    "dte": (exp_date - today).days,
                    "strike": contract.strike,
                    "right": contract.right,
                    "currency": getattr(contract, "currency", "USD") or "USD",
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread": spread,
                    "spread_pct": spread_pct,
                    "volume": ticker.volume,
                    "open_interest": (
                        ticker.callOpenInterest
                        if contract.right == "C"
                        else ticker.putOpenInterest
                    ),
                    "iv": round(ticker.impliedVolatility * 100, 2)
                    if ticker.impliedVolatility
                    else None,
                    "delta": greeks.delta if greeks else None,
                    "gamma": greeks.gamma if greeks else None,
                    "theta": greeks.theta if greeks else None,
                    "vega": greeks.vega if greeks else None,
                    "con_id": contract.conId,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class AccountSnapshot:
    """Point-in-time account view. Monetary values are in `currency`."""

    account_id: Optional[str]
    nav: float
    cash: float
    buying_power: float
    unrealized_pnl: float
    realized_pnl: float
    positions: pd.DataFrame
    currency: str = "USD"


class Account:
    """Pull a fresh AccountSnapshot from IB on demand."""

    _TAG_MAP = {
        "NetLiquidation": "nav",
        "TotalCashValue": "cash",
        "BuyingPower": "buying_power",
        "UnrealizedPnL": "unrealized_pnl",
        "RealizedPnL": "realized_pnl",
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
        values = self._account_values()
        positions = self._positions()
        return AccountSnapshot(
            account_id=self.account_id,
            nav=values.get("nav", 0.0),
            cash=values.get("cash", 0.0),
            buying_power=values.get("buying_power", 0.0),
            unrealized_pnl=values.get("unrealized_pnl", 0.0),
            realized_pnl=values.get("realized_pnl", 0.0),
            positions=positions,
            currency=self._base_currency,
        )

    def print_snapshot(self) -> None:
        snapshot = self.snapshot()
        print(f"\n{'=' * 45}")
        print("  Account Snapshot")
        print(f"{'=' * 45}")
        if snapshot.account_id:
            print(f"  Account        : {snapshot.account_id}")
        print(f"  Currency       : {snapshot.currency}")
        print(f"  NAV            : ${snapshot.nav:>12,.2f}")
        print(f"  Cash           : ${snapshot.cash:>12,.2f}")
        print(f"  Buying Power   : ${snapshot.buying_power:>12,.2f}")
        print(f"  Unrealized P&L : ${snapshot.unrealized_pnl:>12,.2f}")
        print(f"  Realized P&L   : ${snapshot.realized_pnl:>12,.2f}")
        print(f"{'-' * 45}")
        if snapshot.positions.empty:
            print("  No open positions")
        else:
            print(snapshot.positions.to_string(index=False))
        print(f"{'=' * 45}\n")

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
            raise ValueError(
                f"Multiple managed accounts detected {managed}. "
                "Set general.account_id in config.toml so every order is routed explicitly."
            )
        raise ValueError(
            "IB returned no managed accounts. Set general.account_id in config.toml "
            "or verify the IB connection before placing routed orders."
        )

    def _account_values(self) -> dict:
        """
        Read account summary values, preferring IB's BASE currency rows and
        falling back to the configured base currency.
        """

        base_results: dict[str, float] = {}
        fallback_results: dict[str, float] = {}
        for item in self.ib.accountSummary(self.account_id or ""):
            field = self._TAG_MAP.get(item.tag)
            if not field:
                continue
            try:
                value = float(item.value)
            except ValueError:
                continue
            if item.currency == "BASE":
                base_results[field] = value
            elif item.currency == self._base_currency:
                fallback_results[field] = value
        return {**fallback_results, **base_results}

    def _positions(self) -> pd.DataFrame:
        rows = []
        for position in self.ib.positions(self.account_id or ""):
            contract = position.contract
            rows.append(
                {
                    "account": getattr(position, "account", self.account_id),
                    "symbol": contract.symbol,
                    "con_id": contract.conId,
                    "sec_type": contract.secType,
                    "expiry": getattr(contract, "lastTradeDateOrContractMonth", None),
                    "strike": getattr(contract, "strike", None),
                    "right": getattr(contract, "right", None),
                    "position": position.position,
                    "avg_cost": round(position.avgCost, 4),
                    "market_value": None,
                    "unrealized_pnl": None,
                }
            )

        if not rows:
            return pd.DataFrame()

        portfolio_map = {
            (getattr(item, "account", self.account_id), item.contract.conId): item
            for item in self.ib.portfolio(self.account_id or "")
        }
        for row in rows:
            item = portfolio_map.get((row["account"], row["con_id"]))
            if item:
                row["market_value"] = round(item.marketValue, 2)
                row["unrealized_pnl"] = round(item.unrealizedPNL, 2)

        return pd.DataFrame(rows)


if __name__ == "__main__":
    ib = connect()
    try:
        Account(ib).print_snapshot()
    finally:
        ib.disconnect()
