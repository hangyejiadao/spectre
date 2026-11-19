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
