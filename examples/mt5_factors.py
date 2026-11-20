"""Print a close-of-bar factor snapshot from MT5 or its CSV exports (no trading)."""

import argparse
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import torch

from spectre.data import MT5Loader
from spectre.factors import OHLCV, FactorEngine, Returns


def configure_device(engine, requested):
    if requested == "cpu" or (requested == "auto" and not torch.cuda.is_available()):
        engine.to_cpu()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: install the CUDA torch group or use --device cpu")
        engine.to_cuda()
    return str(engine.device)


@contextmanager
def terminal_session(args):
    if os.name != "nt":
        raise ValueError("The terminal requires Windows; use --csv on other platforms.")
    path = Path(args.terminal).resolve(strict=True)
    # initialize() may start a stopped terminal, including its EAs. Check the exact executable.
    check = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "if (Get-Process -Name terminal64 -ErrorAction SilentlyContinue | "
                "Where-Object { $_.Path -eq $env:SPECTRE_TERMINAL }) { exit 0 } else { exit 1 }"
            ),
        ],
        env={**os.environ, "SPECTRE_TERMINAL": str(path)},
        check=False,
    )
    if check.returncode:
        raise RuntimeError(f"Terminal not running at {path}. Start and sign in manually first.")
    import MetaTrader5 as mt5

    if not mt5.initialize(str(path), timeout=15000, portable=args.portable):
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        info = mt5.terminal_info()
        if info is None or not info.connected:
            raise RuntimeError("Terminal is not connected to the broker.")
        if Path(info.path).resolve() != path.parent:
            raise RuntimeError("SDK connected to a different terminal installation.")
        yield mt5
    finally:
        mt5.shutdown()


def terminal_snapshot(args):
    with terminal_session(args) as mt5:
        return MT5Loader.from_terminal(
            mt5, args.symbols, args.timeframe, args.bars, volume=args.volume
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--terminal", help="full path to an already running terminal64.exe")
    source.add_argument("--csv", nargs="+", metavar="SYMBOL=FILE")
    parser.add_argument("--symbols", nargs="+", help="exact broker names (terminal mode)")
    parser.add_argument("--timeframe", default="D1", help="MT5 timeframe (terminal mode)")
    parser.add_argument("--bars", type=int, default=250, help="closed bars per symbol (terminal)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--portable", action="store_true", help="attach using MT5 portable mode")
    parser.add_argument("--volume", choices=["tick_volume", "real_volume"], default="tick_volume")
    args = parser.parse_args()
    try:
        if args.terminal:
            if not args.symbols or args.bars < 21:
                parser.error("terminal mode requires --symbols and at least 21 bars")
            loader = terminal_snapshot(args)
        else:
            files = {}
            for item in args.csv:
                symbol, sep, path = item.partition("=")
                if not symbol or not sep or not path or symbol in files:
                    parser.error(f"expected a distinct SYMBOL=FILE, got {item!r}")
                files[symbol] = path
            loader = MT5Loader.from_csv(files, volume=args.volume)
        bars = loader.load()
        if bars.groupby(level="asset", observed=True).size().min() < 21:
            raise ValueError("20-bar returns require at least 21 completed bars per symbol")
        engine = FactorEngine(loader)
        print(f"Spectre device: {configure_device(engine, args.device)}")
        engine.add(OHLCV.close, "close")
        engine.add(Returns(21), "return_20_bars")
        # Snapshot after close, not a signal tradable at this row's opening timestamp.
        start = bars.groupby(level="asset", observed=True).nth(20).index.get_level_values(0).min()
        result = engine.run(start, None, delay_factor=False)
        print(result.groupby(level="asset", observed=True).tail(1).to_string())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
