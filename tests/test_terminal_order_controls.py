from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace


def _install_ib_insync_stub() -> None:
    """Allow these broker-independent tests to import modules without IB installed."""
    if "ib_insync" in sys.modules:
        return

    fake = types.ModuleType("ib_insync")

    class Dummy:
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class IB(Dummy):
        pass

    class Contract(Dummy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.conId = kwargs.get("conId", 0)

    class LimitOrder(Dummy):
        def __init__(self, action=None, totalQuantity=None, lmtPrice=None, **kwargs):
            super().__init__(**kwargs)
            self.action = action
            self.totalQuantity = totalQuantity
            self.lmtPrice = lmtPrice
            self.orderType = "LMT"

    class MarketOrder(Dummy):
        def __init__(self, action=None, totalQuantity=None, **kwargs):
            super().__init__(**kwargs)
            self.action = action
            self.totalQuantity = totalQuantity
            self.orderType = "MKT"

    class Stock(Dummy):
        pass

    class Option(Dummy):
        pass

    class Trade(Dummy):
        pass

    fake.IB = IB
    fake.Contract = Contract
    fake.LimitOrder = LimitOrder
    fake.MarketOrder = MarketOrder
    fake.Stock = Stock
    fake.Option = Option
    fake.Trade = Trade
    sys.modules["ib_insync"] = fake


def _install_rich_stub() -> None:
    if "rich.console" in sys.modules:
        return

    rich = types.ModuleType("rich")
    console_mod = types.ModuleType("rich.console")
    panel_mod = types.ModuleType("rich.panel")
    table_mod = types.ModuleType("rich.table")

    class Console:
        def print(self, *args, **kwargs):
            pass

        def print_exception(self, *args, **kwargs):
            pass

    class Panel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Table:
        def __init__(self, *args, **kwargs):
            self.rows = []

        @classmethod
        def grid(cls, *args, **kwargs):
            return cls(*args, **kwargs)

        def add_column(self, *args, **kwargs):
            pass

        def add_row(self, *args, **kwargs):
            self.rows.append(args)

    console_mod.Console = Console
    panel_mod.Panel = Panel
    table_mod.Table = Table
    rich.console = console_mod
    rich.panel = panel_mod
    rich.table = table_mod
    sys.modules["rich"] = rich
    sys.modules["rich.console"] = console_mod
    sys.modules["rich.panel"] = panel_mod
    sys.modules["rich.table"] = table_mod


def _main_module():
    _install_ib_insync_stub()
    _install_rich_stub()
    return importlib.import_module("main")


class FakeExecutor:
    def __init__(self, trades):
        self._trades = trades

    def live_trades(self, con_id=None, side=None):
        out = []
        for trade in self._trades:
            if con_id is not None and int(trade.contract.conId) != int(con_id):
                continue
            if side is not None and trade.order.action != side.value:
                continue
            out.append(trade)
        return out

    @staticmethod
    def remaining_qty_from_trade(trade):
        return int(trade.orderStatus.remaining)


def _trade(*, con_id=123, perm=111, native=222, action="BUY", remaining=2):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol="MRNA", conId=con_id, secType="OPT"),
        order=SimpleNamespace(
            action=action,
            permId=perm,
            orderId=native,
            account="DU123",
            totalQuantity=2,
            orderType="LMT",
            lmtPrice=1.23,
        ),
        orderStatus=SimpleNamespace(status="Submitted", filled=0, remaining=remaining),
    )


def _strategy(plays, trades):
    return SimpleNamespace(
        plays=plays,
        executor=FakeExecutor(trades),
        account=SimpleNamespace(account_id="DU123"),
    )


def test_order_rows_match_live_order_to_entry_tracker_by_perm_id():
    main = _main_module()
    tracker = SimpleNamespace(
        status="WORKING",
        perm_id=111,
        native_order_id=222,
        order_id=111,
        account_id="DU123",
        reason="entry",
        side="BUY",
        submitted_qty=2,
        remaining_qty=2,
        limit_px=1.23,
        accounted_fills=0,
        trade_result=None,
    )
    play = SimpleNamespace(
        symbol="MRNA",
        con_id=123,
        working_entry=tracker,
        working_order=None,
    )

    rows = main._order_rows(_strategy([play], [_trade()]))

    assert len(rows) == 1
    assert rows[0]["live"] is True
    assert rows[0]["play_row"] == "0"
    assert rows[0]["tracker"] == "ENTRY"
    assert rows[0]["tstate"] == "WORKING"


def test_order_rows_include_blocked_tracker_without_live_order():
    main = _main_module()
    tracker = SimpleNamespace(
        status="UNBOUND",
        perm_id=111,
        native_order_id=222,
        order_id=111,
        account_id="DU123",
        reason="manual close",
        side="SELL",
        submitted_qty=1,
        remaining_qty=1,
        limit_px=2.34,
        accounted_fills=0,
        trade_result=None,
    )
    play = SimpleNamespace(
        symbol="MRNA",
        con_id=123,
        working_entry=None,
        working_order=tracker,
    )

    rows = main._order_rows(_strategy([play], []))

    assert len(rows) == 1
    assert rows[0]["live"] is False
    assert rows[0]["play_row"] == "0"
    assert rows[0]["tracker"] == "EXIT"
    assert rows[0]["tstate"] == "UNBOUND"
    assert rows[0]["reason"] == "manual close"


def test_bare_numeric_order_selector_prefers_broker_id_over_row_number():
    main = _main_module()
    trades = [
        _trade(con_id=123, perm=1, native=11),
        _trade(con_id=456, perm=999, native=222),
    ]
    row, kind, err = main._resolve_order_selector(_strategy([], trades), "1", allow_row=True)

    assert err is None
    assert kind == "broker"
    assert row["perm"] == "1"
    assert row["row"] == 0


def test_confirmed_cancel_rejects_row_selector():
    main = _main_module()
    row, kind, err = main._resolve_order_selector(_strategy([], [_trade()]), "row:0", allow_row=False)

    assert row is None
    assert kind == "row"
    assert "preview-only" in err


def test_recovery_matching_live_order_uses_broad_open_order_query():
    _install_ib_insync_stub()
    from execution import OrderSide
    from strategy import Strategy

    class BroadExecutor:
        def __init__(self):
            self.calls = []

        def live_trades(self, con_id=None, side=None, broad=False):
            self.calls.append({"con_id": con_id, "side": side, "broad": broad})
            return [_trade(con_id=con_id, action=side.value if side else "BUY")]

    class FakeStrategy:
        _order_field_int = staticmethod(Strategy._order_field_int)
        _recovery_live_trades = Strategy._recovery_live_trades
        _matching_live_order_exists = Strategy._matching_live_order_exists

    fake = FakeStrategy()
    fake.executor = BroadExecutor()
    fake.account = SimpleNamespace(account_id="DU123")
    play = SimpleNamespace(con_id=123)
    tracker = SimpleNamespace(account_id="DU123", perm_id=None, native_order_id=None)

    assert fake._matching_live_order_exists(play, tracker, OrderSide.BUY) is True
    assert fake.executor.calls[0]["broad"] is True


def test_clear_working_refuses_when_broad_open_order_query_fails():
    _install_ib_insync_stub()
    from strategy import PlayStatus, Strategy

    class FailingExecutor:
        def live_trades(self, con_id=None, side=None, broad=False):
            raise RuntimeError("IB unavailable")

    class FakeStrategy:
        _order_field_int = staticmethod(Strategy._order_field_int)
        _recovery_live_trades = Strategy._recovery_live_trades
        _matching_live_order_exists = Strategy._matching_live_order_exists
        clear_working_entry_verified = Strategy.clear_working_entry_verified

    fake = FakeStrategy()
    fake.executor = FailingExecutor()
    fake.account = SimpleNamespace(account_id="DU123")
    tracker = SimpleNamespace(account_id="DU123", perm_id=111, native_order_id=222)
    play = SimpleNamespace(
        symbol="MRNA",
        con_id=123,
        working_entry=tracker,
        qty_open=0,
        qty_initial=0,
        status=PlayStatus.PENDING,
    )

    ok, msg = fake.clear_working_entry_verified(play, ctx=None)

    assert ok is False
    assert "broad IB open-order verification failed" in msg
    assert play.working_entry is tracker


def test_chain_research_argument_helpers_parse_dates_and_ladder_expiries():
    import pandas as pd

    main = _main_module()

    assert main._parse_chain_dte("dte=45-90") == (45, 90)
    assert main._parse_chain_dte("90+") == (90, None)
    assert main._parse_chain_dte("dte<=120") == (None, 120)
    assert main._parse_chain_dte("dte=60") == (60, 60)
    assert main._parse_chain_expiries("exp=2026-06-19,20260918") == [
        "20260619",
        "20260918",
    ]

    available = pd.DataFrame(
        [
            {"expiry": "A", "dte": 25},
            {"expiry": "B", "dte": 43},
            {"expiry": "C", "dte": 58},
            {"expiry": "D", "dte": 92},
            {"expiry": "E", "dte": 121},
        ]
    )

    targets = main._chain_targets(3, 45, 120)
    assert main._chain_pick_ladder_expiries(available, 3, targets) == ["B", "D", "E"]
