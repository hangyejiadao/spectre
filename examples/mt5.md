# MetaTrader 5: read-only market data

The fork adds `spectre.data.MT5Loader`. The live example takes a snapshot from a running,
signed-in terminal. It does not select symbols, change trading settings or place orders.
Use an already open demo or read-only terminal with the symbols available in Market Watch.
Use the exact broker symbols, including suffixes. Do not put credentials in scripts or Git.

From the repository root, for Windows x64 with CPython 3.14:

```powershell
uv sync --python 3.14 --locked --no-group cuda --group cpu --extra mt5
.\.venv\Scripts\python.exe examples\mt5_factors.py --terminal 'D:\mt5\terminal64.exe' --symbols XAUUSDm DXYm --timeframe D1 --bars 250
```

Pass `--portable` only if the running terminal was started with `/portable`. The example checks
the executable path before calling the SDK. If a symbol has insufficient history, open its
chart, load history or lower `--bars` (at least 21 for the example). No symbol is substituted.
The terminal's **Max. bars in chart** setting limits available history. The count is per symbol.

No terminal or SDK is needed for offline replay. Use one comma-separated UTF-8 file per symbol:

```powershell
.\.venv\Scripts\python.exe examples\mt5_factors.py --csv 'XAUUSDm=C:\data\XAUUSDm_D1.csv' 'DXYm=C:\data\DXYm_D1.csv'
```

- `time` must be Unix seconds or ISO date/time. Naive strings are interpreted as UTC; convert
  local/server times before import. Prices use `open,high,low,close`. `tick_volume` (or the CSV
  alias `volume`) is the default, not traded units. `--volume real_volume` requires that column
  and never falls back if it is absent or zero. `spread` stays in broker points, not price units.
- Terminal reads begin at MT5 position 1, skipping the latest bar conservatively. CSV loading
  drops the newest row **per symbol**. For archives known to contain only completed bars use
  `MT5Loader.from_csv(files, drop_last=False)`. No bars are filled across sessions or gaps.
- Results are indexed by bar **open** in UTC. The example reports each symbol's own last close
  and `close / close.shift(20) - 1` after that bar has closed. Different symbols can have different
  last dates. It is not a signal available at the opening timestamp. Keep the default factor
  delay when simulating open-time decisions. Bars are unadjusted for corporate actions.
- Python callers with an already initialized `MetaTrader5` session can call
  `MT5Loader.from_terminal(mt5, ["XAUUSDm"], "H4", 250)` directly. The caller owns the session
  lifecycle. Each call is a snapshot, not a streaming subscription or an execution adapter.

## GPU and M15 intraday research

For a supported NVIDIA GPU, install the project's locked CUDA group instead of the CPU group:

```powershell
uv sync --locked --extra mt5
.\.venv\Scripts\python.exe -m examples.mt5_intraday --terminal D:\mt5\terminal64.exe --portable --symbol XAUUSD --bars 60000 --device cuda --check-cpu
```

`--device cuda` requires CUDA; it never silently uses CPU. In the simpler `mt5_factors.py`
example `--device auto` is the default. Only factor calculation runs on the GPU; terminal I/O,
evaluation and output use the CPU. `--check-cpu` compares all five factor columns on the same
bars and fails on a mismatch. Timings in run.json are single calls, not a warmed benchmark.

The M15 screen uses momentum (4 bars), distance from SMA(16), signed candle body/range,
range/SMA(range,16), and tick volume/SMA(tick volume,16). Train/validation/test use chronological
60/20/20 splits. Only training data determines the sign (Spearman IC) and the 20/80 percentiles.
After a break in the M15 grid, wait 16 bars. Enter at the next open; exit 1, 4 or 8 bars after
entry. Purge labels reaching another split or crossing missing bars. Each rule holds at most one
position, but the IC uses overlapping labels; the trades are not 15 independent portfolios.

On bid-based charts use the bar spread at buy entry or short exit as an *approximation* to ask.
Both endpoint spreads must be positive. `--extra-roundtrip-bps` (default 1.0) is a cost scenario
for additional slippage and fees, not measured execution. Results in `summary.csv` are average
basis points per trade, not cumulative performance. Do not treat the largest test value from a
multi-rule screen as confirmed alpha; test it on new data and realistic quotes.

Each run prints its private output directory (`.cache/mt5_intraday/...` by default). It contains
the bar snapshot, run settings and per-split results. Reproduce without MT5 using `--replay`:

```powershell
.\.venv\Scripts\python.exe -m examples.mt5_intraday --replay PATH_TO_PRIOR_RUN --device cuda --check-cpu
```
