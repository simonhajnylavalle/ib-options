"""
Portfolio statistics and NAV-based risk measurement.

This module contains no trading logic. It converts an AccountSnapshot into a
PortfolioRisk object that strategy.py can use for sizing and risk checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ib_core import Account, AccountSnapshot, connect


@dataclass
class CashPolicy:
    """Single portfolio-level options exposure limit."""

    risk_ceiling: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.risk_ceiling <= 1.0:
            raise ValueError("risk_ceiling must be in (0, 1]")


@dataclass
class StockExposure:
    """Aggregated exposure for one underlying symbol."""

    symbol: str
    spot_value: float
    option_notional: float
    notional: float
    net_value: float
    nav_pct: float
    cost_basis: float
    unrealized_pnl: float
    legs: pd.DataFrame


@dataclass
class PortfolioRisk:
    """Point-in-time portfolio risk snapshot."""

    nav: float
    cash: float
    spot_value: float
    risk_capital: float
    risk_pct: float
    risk_status: str
    exposures: dict[str, StockExposure]
    policy: CashPolicy

    @classmethod
    def from_snapshot(
        cls,
        snapshot: AccountSnapshot,
        policy: CashPolicy,
    ) -> PortfolioRisk:
        """Build a PortfolioRisk from an AccountSnapshot and CashPolicy."""

        nav = snapshot.nav
        positions = snapshot.positions

        if positions is None or positions.empty:
            spot_value = 0.0
            risk_capital = 0.0
            signed_positions = 0.0
        else:
            spot_rows = _filter_sectype(positions, "STK")
            option_rows = _filter_sectype(positions, "OPT")
            spot_value = float(spot_rows["market_value"].fillna(0).sum())
            risk_capital = float(option_rows["market_value"].fillna(0).abs().sum())
            signed_positions = float(positions["market_value"].fillna(0).sum())

        cash = nav - signed_positions
        risk_pct = risk_capital / nav if nav else 0.0
        risk_status = "ABOVE_CEILING" if risk_pct > policy.risk_ceiling else "OK"

        return cls(
            nav=nav,
            cash=cash,
            spot_value=spot_value,
            risk_capital=risk_capital,
            risk_pct=risk_pct,
            risk_status=risk_status,
            exposures=cls._build_exposures(positions, nav),
            policy=policy,
        )

    def headroom(self) -> float:
        """Additional options premium that can be deployed before the ceiling."""

        return self.nav * self.policy.risk_ceiling - self.risk_capital

    def option_exposure(self, symbol: str) -> float:
        """Options notional for one symbol, or 0 if absent."""

        exposure = self.exposures.get(symbol.upper())
        return exposure.option_notional if exposure else 0.0

    def spot_exposure(self, symbol: str) -> float:
        """Stock market value for one symbol, or 0 if absent."""

        exposure = self.exposures.get(symbol.upper())
        return exposure.spot_value if exposure else 0.0

    def print(self) -> None:
        width = 55
        risk_marker = "OK" if self.risk_status == "OK" else "BREACH"
        pct = lambda v: f"{v / self.nav:.1%}" if self.nav else "n/a"
        print(f"\n{'=' * width}")
        print("  Portfolio Risk Snapshot")
        print(f"{'=' * width}")
        if not self.nav:
            print("  NAV = 0; account data not yet received from IB")
            print(f"{'=' * width}\n")
            return
        print(f"  NAV (liq. value)     : ${self.nav:>12,.2f}")
        print(f"{'-' * width}")
        print(f"  Cash                 : ${self.cash:>12,.2f}  ({pct(self.cash)})")
        print(f"  Spot                 : ${self.spot_value:>12,.2f}  ({pct(self.spot_value)})")
        print(f"{'-' * width}")
        print(f"  Options (risk)       : ${self.risk_capital:>12,.2f}  ({self.risk_pct:.1%})")
        print(f"  Risk ceiling         :              {self.policy.risk_ceiling:.1%}  {risk_marker}")
        print(f"  Headroom             : ${self.headroom():>+12,.2f}")
        print(f"{'-' * width}")
        if not self.exposures:
            print("  No open positions")
        else:
            print(
                f"\n  {'SYM':<6} {'TYPE':<6} {'NOTIONAL':>10} "
                f"{'NAV%':>6} {'COST':>10} {'UNREAL PnL':>12} {'LEGS':>5}"
            )
            print(f"  {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 6} {'-' * 10} {'-' * 12} {'-' * 5}")
            for symbol, exposure in sorted(self.exposures.items()):
                if exposure.spot_value != 0:
                    print(f"  {symbol:<6} {'spot':<6} ${exposure.spot_value:>9,.2f}")
                if exposure.option_notional != 0:
                    print(f"  {symbol:<6} {'option':<6} ${exposure.option_notional:>9,.2f}")
                pnl = f"${exposure.unrealized_pnl:>+,.2f}"
                print(
                    f"  {symbol:<6} {'TOTAL':<6} ${exposure.notional:>9,.2f} "
                    f"{exposure.nav_pct:>5.1%} ${exposure.cost_basis:>9,.2f} "
                    f"{pnl:>12}  {len(exposure.legs):>4}"
                )
                print()
        print(f"{'=' * width}\n")

    def detail(self, symbol: str) -> None:
        """Print individual legs for one stock."""

        exposure = self.exposures.get(symbol.upper())
        if exposure is None:
            print(f"  No position in {symbol}")
            return
        print(f"\n  {symbol}  -  {len(exposure.legs)} leg(s)")
        print(
            f"  Notional  : ${exposure.notional:,.2f}"
            f"  (spot ${exposure.spot_value:,.2f} + options ${exposure.option_notional:,.2f})"
        )
        print(f"  Cost basis: ${exposure.cost_basis:,.2f}")
        print(f"  Unrealized: ${exposure.unrealized_pnl:+,.2f}")
        cols = [
            c
            for c in [
                "sec_type",
                "expiry",
                "strike",
                "right",
                "position",
                "avg_cost",
                "market_value",
                "unrealized_pnl",
            ]
            if c in exposure.legs.columns
        ]
        print(exposure.legs[cols].to_string(index=False))
        print()

    @staticmethod
    def _build_exposures(
        positions: pd.DataFrame,
        nav: float,
    ) -> dict[str, StockExposure]:
        if positions is None or positions.empty:
            return {}

        exposures = {}
        for symbol, group in positions.groupby("symbol"):
            spot_rows = group[group["sec_type"] == "STK"]
            option_rows = group[group["sec_type"] == "OPT"]
            spot_value = float(spot_rows["market_value"].fillna(0).sum())
            option_value = float(option_rows["market_value"].fillna(0).sum())
            gross_notional = abs(spot_value) + abs(option_value)
            net_value = spot_value + option_value
            pnl = float(group["unrealized_pnl"].fillna(0).sum())
            cost = float(
                (group["avg_cost"].fillna(0) * group["position"].fillna(0).abs()).sum()
            )
            exposures[symbol] = StockExposure(
                symbol=symbol,
                spot_value=spot_value,
                option_notional=option_value,
                notional=gross_notional,
                net_value=net_value,
                nav_pct=gross_notional / nav if nav else 0.0,
                cost_basis=cost,
                unrealized_pnl=pnl,
                legs=group.reset_index(drop=True),
            )
        return exposures


def _filter_sectype(positions: pd.DataFrame, sec_type: str) -> pd.DataFrame:
    if positions is None or positions.empty:
        return pd.DataFrame()
    return positions[positions["sec_type"] == sec_type]


if __name__ == "__main__":
    ib = connect()
    try:
        snapshot = Account(ib).snapshot()
        risk = PortfolioRisk.from_snapshot(snapshot, CashPolicy(risk_ceiling=0.40))
        risk.print()
    finally:
        ib.disconnect()
