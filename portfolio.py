"""
portfolio.py
─────────────────────────────────────────────────────────────────────────────
Portfolio statistics and risk snapshot. Contains NO trading logic — only
measurement.

All capital is measured against NAV (IB "Net Liquidation Value"), which is
the single denominator for every percentage in this module.

  NAV  =  cash  +  spot positions  +  options market value

Building blocks (in dependency order):
  CashPolicy       — single parameter: risk_ceiling
  StockExposure    — all capital deployed in one underlying (options + shares)
  PortfolioRisk    — full picture assembled from an AccountSnapshot

Risk ceiling enforcement:
  The ceiling is enforced passively: no new entry is allowed while options
  notional exceeds risk_ceiling × NAV (see Strategy._risk_guard).
  Sizing uses headroom() so conviction always scales to available room.
  Existing positions are never automatically liquidated — that is handled
  entirely by the per-play ExitProfile in strategy.py.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from broker import Account, AccountSnapshot, connect


# ─────────────────────────────────────────────────────────────────────────────
# CASH POLICY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CashPolicy:
    """
    Single constraint on the portfolio:

      risk_ceiling  : options_market_value / NAV ≤ risk_ceiling

    Breach blocks all new entries until existing plays bring notional back
    under the ceiling through their normal ExitProfile exits.

    Example:
        CashPolicy(risk_ceiling=0.30)
    """
    risk_ceiling: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.risk_ceiling <= 1.0:
            raise ValueError("risk_ceiling must be in (0, 1]")


# ─────────────────────────────────────────────────────────────────────────────
# STOCK EXPOSURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StockExposure:
    """
    Aggregated view of all capital at work in one underlying.

    spot_value      — market value of share position (0 if none)
    option_notional — signed market value of all option legs (0 if none)
    notional        — gross deployed capital = |spot| + |options|
    net_value       — signed spot + signed options
    nav_pct         — gross notional / NAV
    cost_basis      — what you paid across all legs
    unrealized_pnl  — combined open P&L
    legs            — raw position rows for this symbol
    """
    symbol:          str
    spot_value:      float
    option_notional: float
    notional:        float
    net_value:       float
    nav_pct:         float
    cost_basis:      float
    unrealized_pnl:  float
    legs:            pd.DataFrame


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK  —  the main object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioRisk:
    """
    Point-in-time risk snapshot assembled from an AccountSnapshot.

    nav           — IB Net Liquidation Value; denominator for all percentages
    cash          — nav − signed market value of all positions
    spot_value    — signed market value of all share positions
    risk_capital  — gross option market value (used for the ceiling)
    risk_pct      — risk_capital / nav
    risk_status   — "OK" | "ABOVE_CEILING"  ← gates new entries
    exposures     — {symbol: StockExposure}
    policy        — CashPolicy used to compute this snapshot
    """
    nav:          float
    cash:         float
    spot_value:   float
    risk_capital: float
    risk_pct:     float
    risk_status:  str          # "OK" | "ABOVE_CEILING"  ← gates new entries
    exposures:    dict[str, StockExposure]
    policy:       CashPolicy

    # ── constructor ───────────────────────────────────────────────────────────

    @classmethod
    def from_snapshot(
        cls,
        snapshot: AccountSnapshot,
        policy:   CashPolicy,
    ) -> PortfolioRisk:
        """Build a PortfolioRisk from a live AccountSnapshot and a CashPolicy."""
        nav = snapshot.nav

        positions = snapshot.positions

        if positions is None or positions.empty:
            spot_value   = 0.0
            risk_capital = 0.0
            signed_positions = 0.0
        else:
            spot_rows    = _filter_sectype(positions, "STK")
            option_rows  = _filter_sectype(positions, "OPT")
            spot_value   = float(spot_rows["market_value"].fillna(0).sum())
            risk_capital = float(option_rows["market_value"].fillna(0).abs().sum())
            signed_positions = float(positions["market_value"].fillna(0).sum())

        # Cash derived from NAV so it is consistent with IB's liquidation value.
        # Use signed market values so short stock / short options do not
        # distort the residual cash figure.
        cash = nav - signed_positions

        risk_pct    = risk_capital / nav if nav else 0.0
        risk_status = "ABOVE_CEILING" if risk_pct > policy.risk_ceiling else "OK"

        return cls(
            nav          = nav,
            cash         = cash,
            spot_value   = spot_value,
            risk_capital = risk_capital,
            risk_pct     = risk_pct,
            risk_status  = risk_status,
            exposures    = cls._build_exposures(positions, nav),
            policy       = policy,
        )

    # ── convenience ───────────────────────────────────────────────────────────

    def headroom(self) -> float:
        """
        How much additional options premium can be deployed before hitting
        the risk ceiling.
        Positive → room to add risk.  Negative → already above ceiling.
        """
        return self.nav * self.policy.risk_ceiling - self.risk_capital

    def option_exposure(self, symbol: str) -> float:
        """Options notional for one stock, 0 if not held."""
        exp = self.exposures.get(symbol.upper())
        return exp.option_notional if exp else 0.0

    def spot_exposure(self, symbol: str) -> float:
        """Spot market value for one stock, 0 if not held."""
        exp = self.exposures.get(symbol.upper())
        return exp.spot_value if exp else 0.0

    # ── printing ──────────────────────────────────────────────────────────────

    def print(self) -> None:
        w = 55
        risk_marker = "✓" if self.risk_status == "OK" else "✗"
        pct = lambda v: f"{v / self.nav:.1%}" if self.nav else "n/a"
        print(f"\n{'═' * w}")
        print(f"  Portfolio Risk Snapshot")
        print(f"{'═' * w}")
        if not self.nav:
            print(f"  ⚠  NAV = 0 — account data not yet received from IB")
            print(f"{'═' * w}\n")
            return
        print(f"  NAV (liq. value)     : ${self.nav:>12,.2f}")
        print(f"{'─' * w}")
        print(f"  Cash                 : ${self.cash:>12,.2f}"
              f"  ({pct(self.cash)})")
        print(f"  Spot                 : ${self.spot_value:>12,.2f}"
              f"  ({pct(self.spot_value)})")
        print(f"{'─' * w}")
        print(f"  Options (risk)       : ${self.risk_capital:>12,.2f}"
              f"  ({self.risk_pct:.1%})")
        print(f"  Risk ceiling         :              {self.policy.risk_ceiling:.1%}"
              f"  {risk_marker} {self.risk_status}")
        print(f"  Headroom             : ${self.headroom():>+12,.2f}")
        print(f"{'─' * w}")
        if not self.exposures:
            print("  No open positions")
        else:
            print(f"\n  {'SYM':<6} {'TYPE':<6} {'NOTIONAL':>10} "
                  f"{'NAV%':>6} {'COST':>10} {'UNREAL PnL':>12} {'LEGS':>5}")
            print(f"  {'─'*6} {'─'*6} {'─'*10} "
                  f"{'─'*6} {'─'*10} {'─'*12} {'─'*5}")
            for sym, exp in sorted(self.exposures.items()):
                if exp.spot_value != 0:
                    print(f"  {sym:<6} {'spot':<6} ${exp.spot_value:>9,.2f}")
                if exp.option_notional != 0:
                    print(f"  {sym:<6} {'option':<6} ${exp.option_notional:>9,.2f}")
                pnl_str = f"${exp.unrealized_pnl:>+,.2f}"
                print(f"  {sym:<6} {'TOTAL':<6} ${exp.notional:>9,.2f} "
                      f"{exp.nav_pct:>5.1%} ${exp.cost_basis:>9,.2f} "
                      f"{pnl_str:>12}  {len(exp.legs):>4}")
                print()
        print(f"{'═' * w}\n")

    def detail(self, symbol: str) -> None:
        """Print every individual leg for one stock."""
        exp = self.exposures.get(symbol.upper())
        if exp is None:
            print(f"  No position in {symbol}")
            return
        print(f"\n  {symbol}  —  {len(exp.legs)} leg(s)")
        print(f"  Notional  : ${exp.notional:,.2f}"
              f"  (spot ${exp.spot_value:,.2f} + options ${exp.option_notional:,.2f})")
        print(f"  Cost basis: ${exp.cost_basis:,.2f}")
        print(f"  Unrealized: ${exp.unrealized_pnl:+,.2f}")
        cols = [c for c in [
            "sec_type", "expiry", "strike", "right",
            "position", "avg_cost", "market_value", "unrealized_pnl",
        ] if c in exp.legs.columns]
        print(exp.legs[cols].to_string(index=False))
        print()

    # ── private static ────────────────────────────────────────────────────────

    @staticmethod
    def _build_exposures(
        positions: pd.DataFrame,
        nav: float,
    ) -> dict[str, StockExposure]:
        if positions is None or positions.empty:
            return {}
        exposures = {}
        for symbol, group in positions.groupby("symbol"):
            spot_rows   = group[group["sec_type"] == "STK"]
            option_rows = group[group["sec_type"] == "OPT"]
            spot_val    = float(spot_rows["market_value"].fillna(0).sum())
            opt_val     = float(option_rows["market_value"].fillna(0).sum())
            gross_notional = abs(spot_val) + abs(opt_val)
            net_value   = spot_val + opt_val
            pnl         = float(group["unrealized_pnl"].fillna(0).sum())
            cost        = float(
                (group["avg_cost"].fillna(0) * group["position"].fillna(0).abs()).sum()
            )
            exposures[symbol] = StockExposure(
                symbol          = symbol,
                spot_value      = spot_val,
                option_notional = opt_val,
                notional        = gross_notional,
                net_value       = net_value,
                nav_pct         = gross_notional / nav if nav else 0.0,
                cost_basis      = cost,
                unrealized_pnl  = pnl,
                legs            = group.reset_index(drop=True),
            )
        return exposures


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _filter_sectype(positions: pd.DataFrame, sec_type: str) -> pd.DataFrame:
    if positions is None or positions.empty:
        return pd.DataFrame()
    return positions[positions["sec_type"] == sec_type]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK  (python portfolio.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ib       = connect()
    snapshot = Account(ib).snapshot()

    policy = CashPolicy(risk_ceiling=0.40)

    risk = PortfolioRisk.from_snapshot(snapshot, policy)
    risk.print()

    ib.disconnect()
