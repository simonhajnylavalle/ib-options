from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import importlib

import pandas as pd


MARKET_TZ = ZoneInfo("America/New_York")


def _positions_df(*rows: dict) -> pd.DataFrame:
    cols = [
        "account", "symbol", "con_id", "sec_type", "expiry", "strike", "right",
        "position", "avg_cost", "market_value", "unrealized_pnl",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    for col in cols:
        if col not in out.columns:
            out[col] = None
    if "sec_type" in out.columns and "right" in out.columns:
        mask = (out["sec_type"] == "OPT") & (out["right"].isna())
        out.loc[mask, "right"] = "C"
    return out[cols]


def _snapshot(ib_core, positions: pd.DataFrame | None = None):
    positions = positions if positions is not None else _positions_df()
    return ib_core.AccountSnapshot(
        account_id="ACC",
        nav=10_000.0,
        cash=10_000.0,
        buying_power=10_000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        positions=positions,
    )


def _play(strategy, exit_profile, *, qty_open=2, qty_initial=2, con_id=1):
    return strategy.Play(
        play_id="play1",
        account_id="ACC",
        play_type=strategy.PlayType.THESIS,
        symbol="XYZ",
        con_id=con_id,
        qty_initial=qty_initial,
        qty_open=qty_open,
        entry_time=datetime.now(MARKET_TZ),
        entry_price=1.0,
        entry_nav=10_000.0,
        exit_profile=exit_profile,
        status=strategy.PlayStatus.OPEN,
    )


def test_config_load_merges_defaults_and_allows_blank_fallback_mode(tmp_path):
    import config

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[execution.patient]
fallback_mode = ""
fallback_after = ""

[execution.urgent]
fallback_mode = ""
last_resort_mode = ""

[sniper.scanner]
drop_pct = 0.20
""".strip()
    )

    cfg = config.load(cfg_path)

    assert cfg.patient.fallback_mode is None
    assert cfg.patient.fallback_after is None
    assert cfg.urgent.fallback_mode is None
    assert cfg.urgent.last_resort_mode is None
    assert cfg.exit_profiles["THESIS"].dte_floor == 5
    assert cfg.contract_specs["SENTINEL"].min_volume == 1
    assert cfg.sniper_drop_pct == 0.20


def test_vanished_position_closes_without_submitting_exit(monkeypatch):
    import ib_core
    import portfolio
    import strategy

    snapshot = _snapshot(ib_core)

    class FakeAccount:
        def __init__(self, *args, **kwargs):
            self.account_id = "ACC"

        def snapshot(self):
            return snapshot

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            self.ib = args[0] if args else kwargs.get("ib")

    monkeypatch.setattr(strategy, "Account", FakeAccount)
    monkeypatch.setattr(strategy, "Executor", DummyExecutor)
    monkeypatch.setattr(strategy.state, "save", lambda *args, **kwargs: None)

    strat = strategy.Strategy(ib=object(), policy=portfolio.CashPolicy())
    ep = strategy.ExitProfile(stop_loss_pct=-0.5, full_exit_pct=1.0, dte_floor=5)
    play = _play(strategy, ep, qty_open=2, qty_initial=2, con_id=123)
    ctx = strat.context(snapshot)

    close_calls: list[bool] = []
    monkeypatch.setattr(strat, "_close_play", lambda *a, **k: close_calls.append(True) or True)
    monkeypatch.setattr(strat, "_price_from_market", lambda *a, **k: 9.99)

    changed = strat._evaluate_play(play, ctx)

    assert changed is True
    assert play.status is strategy.PlayStatus.CLOSED
    assert play.qty_open == 0
    assert close_calls == []


def test_executor_does_not_resubmit_before_cancel_ack(monkeypatch):
    import execution

    class FakeIB:
        def __init__(self):
            self.place_calls = 0
            self.cancel_calls = 0
            self._trades = []

        def placeOrder(self, contract, order):
            self.place_calls += 1
            order.orderId = self.place_calls
            order.permId = self.place_calls
            trade = SimpleNamespace(
                contract=contract,
                order=order,
                orderStatus=SimpleNamespace(status="Submitted", filled=0, remaining=order.totalQuantity),
                fills=[],
            )
            self._trades = [trade]
            return trade

        def cancelOrder(self, order):
            self.cancel_calls += 1

        def sleep(self, secs):
            return None

        def reqOpenOrders(self):
            return None

        def trades(self):
            return list(self._trades)

    ib = FakeIB()
    ex = execution.Executor(ib)
    contract = SimpleNamespace(conId=1, symbol="XYZ", secType="OPT")

    monkeypatch.setattr(ex, "resolve_con_id", lambda con_id: contract)
    monkeypatch.setattr(ex, "_option_limit_price", lambda *a, **k: (1.25, 1.20, 1.30))
    monkeypatch.setattr(ex, "wait_until_not_live", lambda *a, **k: "Submitted")

    result = ex.buy_option(
        con_id=1,
        qty=2,
        mode=execution.PriceMode.MID,
        fill_timeout_secs=0,
        max_retries=2,
    )

    assert ib.place_calls == 1
    assert ib.cancel_calls == 1
    assert result.qty == 2


def test_state_roundtrip_preserves_pending_exit_metadata(monkeypatch, tmp_path):
    import strategy
    import state

    monkeypatch.setenv("OPTION_ALGORITHM_STATE_FILE", str(tmp_path / "plays.json"))
    importlib.reload(state)
    monkeypatch.setattr(strategy.state, "save", state.save)

    ep = strategy.ExitProfile(stop_loss_pct=-0.5, full_exit_pct=1.0, dte_floor=5)
    play = _play(strategy, ep, qty_open=2, qty_initial=2, con_id=321)
    play.working_order = strategy.WorkingOrder(
        trade_result=None,
        remaining_qty=1,
        attempts_used=2,
        submitted_at=datetime.now(MARKET_TZ),
        retry_kind="patient",
        reason="test",
        reserved_tranche_idx=1,
        reserve_spike_fired=True,
    )

    state.save([play], account_id="ACC")
    loaded = state.load(account_id="ACC")

    assert len(loaded) == 1
    wo = loaded[0].working_order
    assert wo is not None
    assert wo.trade_result is None
    assert wo.remaining_qty == 1
    assert wo.attempts_used == 2
    assert wo.reserved_tranche_idx == 1
    assert wo.reserve_spike_fired is True


def test_restore_working_orders_rebinds_live_exit(monkeypatch):
    import execution
    import ib_core
    import portfolio
    import strategy

    positions = _positions_df(
        {
            "account": "ACC",
            "symbol": "XYZ",
            "con_id": 1,
            "sec_type": "OPT",
            "position": 2,
            "market_value": 320.0,
            "avg_cost": 100.0,
            "unrealized_pnl": 0.0,
        }
    )
    snapshot = _snapshot(ib_core, positions)

    class FakeAccount:
        def __init__(self, *args, **kwargs):
            self.account_id = "ACC"

        def snapshot(self):
            return snapshot

    live_trade = SimpleNamespace(
        contract=SimpleNamespace(conId=1, symbol="XYZ", secType="OPT"),
        order=SimpleNamespace(action="SELL", totalQuantity=1, lmtPrice=1.6, orderType="LMT", orderId=11, permId=11),
        orderStatus=SimpleNamespace(status="Submitted", filled=0, remaining=1),
        fills=[],
    )

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            self.ib = args[0] if args else kwargs.get("ib")

        def live_trades(self, side=None):
            return [live_trade]

        def remaining_qty_from_trade(self, trade):
            return int(trade.orderStatus.remaining)

        def result_from_trade(self, trade):
            return execution.OrderResult(
                order_id=trade.order.permId or trade.order.orderId,
                symbol=trade.contract.symbol,
                con_id=trade.contract.conId,
                side=execution.OrderSide.SELL,
                qty=int(trade.order.totalQuantity),
                limit_px=trade.order.lmtPrice,
                trade=trade,
            )

    monkeypatch.setattr(strategy, "Account", FakeAccount)
    monkeypatch.setattr(strategy, "Executor", FakeExecutor)
    monkeypatch.setattr(strategy.state, "save", lambda *args, **kwargs: None)

    strat = strategy.Strategy(ib=object(), policy=portfolio.CashPolicy())
    ep = strategy.ExitProfile(stop_loss_pct=-0.5, full_exit_pct=1.0, dte_floor=5)
    play = _play(strategy, ep, qty_open=2, qty_initial=2, con_id=1)
    play.working_order = strategy.WorkingOrder(
        trade_result=None,
        remaining_qty=1,
        attempts_used=1,
        submitted_at=datetime.now(MARKET_TZ),
        retry_kind="patient",
        reason="pending exit",
    )
    strat.plays = [play]

    dirty = strat.restore_working_orders(strat.context(snapshot))

    assert dirty is True
    assert play.working_order is not None
    assert play.working_order.trade_result is not None
    assert play.working_order.trade_result.order_id == 11


def test_manual_async_partial_close_commits_tranche_reservation(monkeypatch):
    import execution
    import ib_core
    import portfolio
    import strategy

    initial_positions = _positions_df(
        {
            "account": "ACC",
            "symbol": "XYZ",
            "con_id": 1,
            "sec_type": "OPT",
            "position": 4,
            "market_value": 640.0,
            "avg_cost": 100.0,
            "unrealized_pnl": 0.0,
        }
    )
    filled_positions = _positions_df(
        {
            "account": "ACC",
            "symbol": "XYZ",
            "con_id": 1,
            "sec_type": "OPT",
            "position": 3,
            "market_value": 480.0,
            "avg_cost": 100.0,
            "unrealized_pnl": 0.0,
        }
    )
    snapshot_initial = _snapshot(ib_core, initial_positions)
    snapshot_after_fill = _snapshot(ib_core, filled_positions)

    class FakeAccount:
        def __init__(self, *args, **kwargs):
            self.account_id = "ACC"
            self.snapshot_obj = snapshot_initial

        def snapshot(self):
            return self.snapshot_obj

    class DummyExecutorCtor:
        def __init__(self, *args, **kwargs):
            self.ib = args[0] if args else kwargs.get("ib")

    monkeypatch.setattr(strategy, "Account", FakeAccount)
    monkeypatch.setattr(strategy, "Executor", DummyExecutorCtor)
    monkeypatch.setattr(strategy.state, "save", lambda *args, **kwargs: None)

    strat = strategy.Strategy(ib=object(), policy=portfolio.CashPolicy())

    trade = SimpleNamespace(
        contract=SimpleNamespace(conId=1, symbol="XYZ", secType="OPT"),
        order=SimpleNamespace(action="SELL", totalQuantity=1, lmtPrice=1.6, orderType="LMT", orderId=22, permId=22),
        orderStatus=SimpleNamespace(status="Submitted", filled=0, remaining=1),
        fills=[],
    )

    class ManualExecutor:
        mode_for_attempt = staticmethod(execution.Executor.mode_for_attempt)

        def submit_option_order(self, side, con_id, qty, mode):
            return execution.OrderResult(
                order_id=trade.order.permId or trade.order.orderId,
                symbol=trade.contract.symbol,
                con_id=con_id,
                side=side,
                qty=qty,
                limit_px=trade.order.lmtPrice,
                trade=trade,
            )

        def is_live(self, result):
            return result.status() in {"Submitted", "PreSubmitted", "PendingSubmit", "PendingCancel", "PartiallyFilled"}

        def cancel(self, result):
            return None

    strat.executor = ManualExecutor()

    ep = strategy.ExitProfile(
        stop_loss_pct=-0.9,
        full_exit_pct=10.0,
        dte_floor=0,
        tranches=[(0.50, 0.25)],
    )
    play = _play(strategy, ep, qty_open=4, qty_initial=4, con_id=1)
    strat.plays = [play]

    ok, submitted = strat.manual_close(play, 1, ctx=strat.context(snapshot_initial))

    assert ok is True
    assert submitted is True
    assert play.working_order is not None
    assert play.working_order.reserved_tranche_idx == 1
    assert play.tranche_idx == 0  # reserved, not yet committed until a fill lands

    trade.fills = [SimpleNamespace(execution=SimpleNamespace(shares=1, price=1.6))]
    trade.orderStatus.status = "Filled"
    trade.orderStatus.filled = 1
    trade.orderStatus.remaining = 0

    strat._advance_working_orders(strat.context(snapshot_after_fill))

    assert play.working_order is None
    assert play.qty_open == 3
    assert play.tranche_idx == 1

    partial_calls: list[tuple] = []
    monkeypatch.setattr(strat, "_partial_close", lambda *a, **k: partial_calls.append((a, k)) or True)

    changed = strat._evaluate_play(play, strat.context(snapshot_after_fill))

    assert changed is True  # PnL history/peak update is now persisted as dirty state
    assert partial_calls == []


def test_executor_routes_account_on_option_and_stock_orders(monkeypatch):
    import execution

    placed = []

    class FakeIB:
        def qualifyContracts(self, *contracts):
            for c in contracts:
                if not getattr(c, "conId", 0):
                    c.conId = 99
            return contracts

        def placeOrder(self, contract, order):
            order.orderId = len(placed) + 1
            order.permId = 1000 + len(placed) + 1
            trade = SimpleNamespace(
                contract=contract,
                order=order,
                orderStatus=SimpleNamespace(status="Submitted", filled=0, remaining=order.totalQuantity),
                fills=[],
            )
            placed.append((contract, order, trade))
            return trade

    ex = execution.Executor(FakeIB(), account_id="ACC123")
    option_contract = SimpleNamespace(conId=1, symbol="XYZ", secType="OPT")
    monkeypatch.setattr(ex, "resolve_con_id", lambda con_id: option_contract)
    monkeypatch.setattr(ex, "_option_limit_price", lambda *a, **k: (1.25, 1.20, 1.30))

    ex.submit_option_order(execution.OrderSide.BUY, con_id=1, qty=1, mode=execution.PriceMode.MID)
    ex.buy_stock("XYZ", qty=3)

    assert placed[0][1].account == "ACC123"
    assert placed[1][1].account == "ACC123"


def test_track_position_rejects_short_option(monkeypatch):
    import ib_core
    import portfolio
    import strategy

    positions = _positions_df(
        {
            "account": "ACC",
            "symbol": "XYZ",
            "con_id": 55,
            "sec_type": "OPT",
            "right": "C",
            "position": -1,
            "market_value": -120.0,
            "avg_cost": 100.0,
        }
    )
    snapshot = _snapshot(ib_core, positions)

    class FakeAccount:
        def __init__(self, *args, **kwargs):
            self.account_id = "ACC"

        def snapshot(self):
            return snapshot

    class DummyExecutor:
        currency = "USD"
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(strategy, "Account", FakeAccount)
    monkeypatch.setattr(strategy, "Executor", DummyExecutor)
    monkeypatch.setattr(strategy.state, "save", lambda *args, **kwargs: None)

    strat = strategy.Strategy(ib=object(), policy=portfolio.CashPolicy(), base_currency="USD")
    play = strat.track_position(55, strategy.PlayType.THESIS, symbol="XYZ")

    assert play is None
    assert strat.plays == []


def test_unfilled_cancelled_entry_does_not_create_play(monkeypatch):
    import execution
    import portfolio
    import strategy

    class FakeAccount:
        def __init__(self, *args, **kwargs):
            self.account_id = "ACC"

    class DummyExecutor:
        currency = "USD"
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(strategy, "Account", FakeAccount)
    monkeypatch.setattr(strategy, "Executor", DummyExecutor)
    monkeypatch.setattr(strategy.state, "save", lambda *args, **kwargs: None)

    strat = strategy.Strategy(ib=object(), policy=portfolio.CashPolicy(), base_currency="USD")
    ep = strategy.ExitProfile(stop_loss_pct=-0.5, full_exit_pct=1.0, dte_floor=5)
    trade = SimpleNamespace(
        contract=SimpleNamespace(conId=1, symbol="XYZ", secType="OPT"),
        order=SimpleNamespace(action="BUY", totalQuantity=2, lmtPrice=1.2, orderType="LMT", orderId=1, permId=1),
        orderStatus=SimpleNamespace(status="Cancelled", filled=0, remaining=2),
        fills=[],
    )
    result = execution.OrderResult(
        order_id=1,
        symbol="XYZ",
        con_id=1,
        side=execution.OrderSide.BUY,
        qty=2,
        limit_px=1.2,
        trade=trade,
        total_filled=0,
        unfilled_qty=2,
        last_order_live=False,
    )

    play = strat._make_play(
        strategy.PlayType.THESIS, "XYZ", 1, 2, 1.2, 10_000.0, ep, result
    )

    assert play is None
    assert strat.plays == []


def test_unbound_working_exit_blocks_resubmit(monkeypatch):
    import ib_core
    import portfolio
    import strategy

    positions = _positions_df(
        {
            "account": "ACC",
            "symbol": "XYZ",
            "con_id": 1,
            "sec_type": "OPT",
            "right": "C",
            "position": 1,
            "market_value": 300.0,
            "avg_cost": 100.0,
        }
    )
    snapshot = _snapshot(ib_core, positions)

    class FakeAccount:
        def __init__(self, *args, **kwargs):
            self.account_id = "ACC"

        def snapshot(self):
            return snapshot

    class FakeExecutor:
        currency = "USD"
        def __init__(self, *args, **kwargs):
            self.submit_calls = 0

        def live_trades(self, side=None):
            return []

        def is_live(self, result):
            return False

        def submit_option_order(self, *args, **kwargs):
            self.submit_calls += 1
            raise AssertionError("duplicate exit submitted")

    monkeypatch.setattr(strategy, "Account", FakeAccount)
    monkeypatch.setattr(strategy, "Executor", FakeExecutor)
    monkeypatch.setattr(strategy.state, "save", lambda *args, **kwargs: None)

    strat = strategy.Strategy(ib=object(), policy=portfolio.CashPolicy(), base_currency="USD")
    ep = strategy.ExitProfile(stop_loss_pct=-0.9, full_exit_pct=0.1, dte_floor=0)
    play = _play(strategy, ep, qty_open=1, qty_initial=1, con_id=1)
    play.working_order = strategy.WorkingOrder(
        trade_result=None,
        remaining_qty=1,
        attempts_used=1,
        submitted_at=datetime.now(MARKET_TZ),
        retry_kind="patient",
        reason="full target",
    )
    strat.plays = [play]

    strat._monitor_plays(strat.context(snapshot))

    assert play.working_order is not None
    assert play.working_order.status == "UNBOUND"
    assert strat.executor.submit_calls == 0


def test_account_snapshot_uses_usd_summary_when_chf_rows_are_missing():
    import ib_core

    class FakeIB:
        def managedAccounts(self):
            return ["ACC"]

        def accountSummary(self, account):
            assert account == "ACC"
            return [
                SimpleNamespace(tag="NetLiquidation", value="10000", currency="USD"),
                SimpleNamespace(tag="TotalCashValue", value="7500", currency="USD"),
                SimpleNamespace(tag="BuyingPower", value="30000", currency="USD"),
                SimpleNamespace(tag="UnrealizedPnL", value="12.5", currency="USD"),
                SimpleNamespace(tag="RealizedPnL", value="4.5", currency="USD"),
            ]

        def positions(self, account):
            return []

    snapshot = ib_core.Account(FakeIB(), base_currency="CHF").snapshot()

    assert snapshot.currency == "USD"
    assert snapshot.nav == 10_000.0
    assert snapshot.cash == 7_500.0
    assert snapshot.buying_power == 30_000.0


def test_account_snapshot_prefers_complete_summary_over_configured_partial_rows():
    import ib_core

    class FakeIB:
        def managedAccounts(self):
            return ["ACC"]

        def accountSummary(self, account):
            return [
                SimpleNamespace(tag="NetLiquidation", value="9000", currency="CHF"),
                SimpleNamespace(tag="NetLiquidation", value="10000", currency="USD"),
                SimpleNamespace(tag="TotalCashValue", value="7500", currency="USD"),
                SimpleNamespace(tag="BuyingPower", value="30000", currency="USD"),
            ]

        def positions(self, account):
            return []

    snapshot = ib_core.Account(FakeIB(), base_currency="CHF").snapshot()

    assert snapshot.currency == "USD"
    assert snapshot.nav == 10_000.0
    assert snapshot.cash == 7_500.0
    assert snapshot.buying_power == 30_000.0


def test_currency_mismatch_does_not_block_entry_guard(monkeypatch):
    import ib_core
    import portfolio
    import strategy

    snapshot = ib_core.AccountSnapshot(
        account_id="ACC",
        nav=10_000.0,
        cash=10_000.0,
        buying_power=10_000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        positions=_positions_df(),
        currency="CHF",
    )

    class FakeAccount:
        def __init__(self, *args, **kwargs):
            self.account_id = "ACC"

        def snapshot(self):
            return snapshot

    class DummyExecutor:
        currency = "USD"
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(strategy, "Account", FakeAccount)
    monkeypatch.setattr(strategy, "Executor", DummyExecutor)

    strat = strategy.Strategy(ib=object(), policy=portfolio.CashPolicy(), base_currency="CHF")
    assert strat._entry_guard(strat.context(snapshot), "USD") is True


def test_empty_startup_positions_does_not_close_active_play():
    import state
    import strategy

    ep = strategy.ExitProfile(stop_loss_pct=-0.5, full_exit_pct=1.0, dte_floor=5)
    play = _play(strategy, ep, qty_open=2, qty_initial=2, con_id=123)

    state._reconcile([play], _positions_df())

    assert play.status is strategy.PlayStatus.OPEN
    assert play.qty_open == 2


def test_pending_entry_accounts_late_fill_from_live_last_order(monkeypatch):
    import execution
    import ib_core
    import portfolio
    import strategy

    snapshot = _snapshot(ib_core, _positions_df())

    class FakeAccount:
        def __init__(self, *args, **kwargs):
            self.account_id = "ACC"

        def snapshot(self):
            return snapshot

    class DummyExecutor:
        currency = "USD"
        def __init__(self, *args, **kwargs):
            pass

        def is_live(self, result):
            return result.status() in {"Submitted", "PartiallyFilled"}

    monkeypatch.setattr(strategy, "Account", FakeAccount)
    monkeypatch.setattr(strategy, "Executor", DummyExecutor)
    monkeypatch.setattr(strategy.state, "save", lambda *args, **kwargs: None)

    strat = strategy.Strategy(ib=object(), policy=portfolio.CashPolicy(), base_currency="USD")
    ep = strategy.ExitProfile(stop_loss_pct=-0.5, full_exit_pct=1.0, dte_floor=5)
    trade = SimpleNamespace(
        contract=SimpleNamespace(conId=1, symbol="XYZ", secType="OPT"),
        order=SimpleNamespace(action="BUY", totalQuantity=1, lmtPrice=1.2, orderType="LMT", orderId=1, permId=1),
        orderStatus=SimpleNamespace(status="Submitted", filled=0, remaining=1),
        fills=[],
    )
    result = execution.OrderResult(
        order_id=1,
        symbol="XYZ",
        con_id=1,
        side=execution.OrderSide.BUY,
        qty=1,
        limit_px=1.2,
        trade=trade,
        total_filled=1,       # one contract filled on an earlier retry attempt
        total_cost=1.1,
        unfilled_qty=1,
        last_order_live=True, # the last retry order is still live for one more
    )

    play = strat._make_play(strategy.PlayType.THESIS, "XYZ", 1, 2, 1.2, 10_000.0, ep, result)
    assert play is not None
    assert play.qty_open == 1
    assert play.working_entry is not None
    assert play.working_entry.accounted_fills == 0

    trade.fills = [SimpleNamespace(execution=SimpleNamespace(shares=1, price=1.2))]
    trade.orderStatus.status = "PartiallyFilled"
    trade.orderStatus.filled = 1
    trade.orderStatus.remaining = 0

    dirty = strat._advance_working_entries(strat.context(snapshot))

    assert dirty is True
    assert play.qty_open == 2
    assert play.working_entry.remaining_qty == 0
