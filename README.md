# Option Algorithm

Interactive Brokers options trading console with a stateful strategy loop, NAV-based risk controls, and contract selection logic tuned for directional biotech trades.

## What is here

- `main.py`: console entrypoint and timed strategy loop
- `strategy.py`: play lifecycle, entry sizing, exits, scanner integration
- `execution.py`: option and stock order submission, pricing modes, retry ladder
- `ib_core.py`: IB connection, account snapshots, option-chain queries
- `portfolio.py`: NAV-based exposure and headroom calculations
- `state.py`: `plays.json` persistence and startup reconciliation
- `config.toml`: sample runtime configuration

## Requirements

- Python 3.10+
- IB Gateway or TWS for live trading

## Install

```bash
python -m pip install -e ".[test]"
```

For a repeatable install using the versions validated in this repo:

```bash
python -m pip install -c requirements.lock -e ".[test]"
```

Installed wheel/venv usage also exposes:

```bash
option-algorithm
```

## Runtime files

Config and state now live in user-writable locations by default:

- config: `~/.config/option_algorithm/config.toml`
- state: `~/.local/state/option_algorithm/plays.json`

Overrides:

- `OPTION_ALGORITHM_CONFIG`
- `OPTION_ALGORITHM_STATE_FILE`
- `OPTION_ALGORITHM_HOME`

A default config file is bootstrapped automatically if none exists.

## Run

```bash
python main.py
```

or:

```bash
option-algorithm
```

The app connects to IB using the host/port/client ID configured in `config.toml`, resolves the managed account, routes outbound orders to that account, loads any persisted plays, rebinds pending live entry/exit orders, runs an immediate strategy tick, and then continues on the configured risk loop interval. SNIPER scanning is throttled separately by `scanner_interval`, so exits can be monitored more frequently than new scanner entries.

## Operator Console

Order-submitting commands use an explicit confirmation flag by default. A first command prints a compact preview and does not submit; re-run with `--yes` to execute using fresh broker state and a fresh quote. For cancels, table row selectors are preview-only because live-order rows can change. The preview prints a stable `perm:` or `native:` confirmation command, optionally guarded by `conid:`.

Useful command groups:

```text
orders | pending
  Show live IB orders plus matched ENTRY/EXIT trackers.

cancel row:<n>
  Preview one live order cancel and print the stable confirmation command.

cancel perm:<id> [conid:<con_id>] --yes
  Cancel one live order and block automatic retry/resubmission.

cancel perm:<id> --retry --yes
cancel-retry native:<id> [conid:<con_id>] --yes
  Cancel only this attempt and let the retry ladder continue.

cancel-all [--yes]
  Send IB global cancel and block working trackers.

rebind
  Re-run working entry/exit rebinding from live IB open orders.

clear-working <play-row> entry|exit [--yes]
  Clear a local blocked tracker only after a broad IB open-order query verifies no matching live order remains.
```

Scanner behavior is controlled separately:

```toml
[sniper.scanner]
auto_open = false
```

With `auto_open = false`, `scan` reports hits only. Use `scan --open --yes` to submit a SNIPER entry from a scan hit.

### Option chain research

The console chain view is CALL-focused by default. The underlying `OptionChain`
boundary still supports both option rights, but the operator-facing research
command is intentionally kept on the call side until the strategy expands.

```text
chain <SYM>
  Show CALL contracts using a ladder of expiries across roughly 21-180 DTE,
  rather than only the nearest/front expiries.

chain <SYM> expiries [p<N>]
  List available expiries and DTEs without requesting option quotes.

chain <SYM> dte=45-90
chain <SYM> 45-90
  Research calls in a specific DTE window.

chain <SYM> exp=YYYYMMDD[,YYYYMMDD]
  Research exact expiry dates.

chain <SYM> front 4
  Explicitly request the old nearest-expiry style.

chain <SYM> thesis|approach|sentinel|sniper
  Show contracts matching a play type's configured delta/DTE/liquidity shape.
```

## Live Smoke Check

```bash
option-algorithm-smoke
```

This is read-only: it connects to the configured live IB endpoint, resolves the account, reads account summary, positions, and open orders, then disconnects without submitting orders.

## Test

```bash
pytest
```

The test suite is broker-independent. It covers:

- config loading and default fallback behaviour
- state save/load including pending entry/exit metadata
- execution retry cancellation safety and explicit account routing
- play evaluation when positions vanish from the account
- startup rebinding or safe blocking of pending exits
- manual scale-out reservation logic that prevents double-sells
- long-call-only guards, pending-entry handling, and Swiss-account USD summary fallback
- read-only live smoke command packaging

## Current Limits

- Live trading still requires IB Gateway or TWS.
- The strategy is intentionally long-CALL-only for now. PUT and short-option tracking are rejected.
- Account summary values prefer the configured currency first, then IB `BASE`, then USD. The sample config uses USD because the implemented option strategies trade US-listed calls.
- Entry orders and exit orders are tracked asynchronously. Remaining premium from pending BUY orders is reserved against future option headroom so unresolved entries cannot silently over-commit the sleeve.
- Unbound or exhausted pending orders are kept blocked until the operator verifies them, rather than auto-clearing and risking duplicate orders.
- The live smoke check is read-only; this repository intentionally does not automate live order placement as a test.
- `plays.json` is local process state, not a shared or durable order journal.
