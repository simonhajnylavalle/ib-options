"""
Read-only live IB connectivity smoke check.

This command submits no orders. It verifies that the configured IB endpoint,
account summary, positions, and open-order read paths are usable.
"""

from __future__ import annotations

import config
from execution import Executor
from ib_core import Account, connect


def main() -> None:
    cfg = config.load()
    print("Connecting to IB Gateway...")
    ib = connect(cfg.ib_host, cfg.ib_port, cfg.ib_client_id)
    try:
        account = Account(
            ib,
            base_currency=cfg.base_currency,
            account_id=cfg.account_id or None,
        )
        snapshot = account.snapshot()
        print(
            f"[SMOKE] account={snapshot.account_id} "
            f"currency={snapshot.currency} "
            f"NAV={snapshot.nav:,.2f} "
            f"cash={snapshot.cash:,.2f} "
            f"buying_power={snapshot.buying_power:,.2f}"
        )
        if snapshot.nav <= 0:
            raise RuntimeError("IB account summary returned NAV <= 0")

        positions = 0 if snapshot.positions.empty else len(snapshot.positions)
        pending = Executor(ib, account_id=account.account_id).pending_orders()
        pending_count = 0 if pending.empty else len(pending)
        print(f"[SMOKE] positions={positions} open_orders={pending_count}")
        print("[SMOKE] OK - read-only live IB paths are usable")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
