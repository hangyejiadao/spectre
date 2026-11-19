"""Read-only snapshots of MetaTrader 5 bars. No terminal SDK is needed to read CSVs."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from .memory import MemoryLoader


class MT5Loader(MemoryLoader):
    """Raw, unadjusted OHLCV indexed by UTC bar-open time and exact broker symbol."""

    def __init__(self, frames: Mapping, *, volume="tick_volume", drop_last=True):
        if volume not in ("tick_volume", "real_volume"):
            raise ValueError("volume must be tick_volume or real_volume")
        if not frames:
            raise ValueError("at least one symbol is required")
        prices = ["open", "high", "low", "close"]
        bars = []
        for symbol, rates in frames.items():
            df = pd.DataFrame(rates).copy()
            # Some exporters rename tick_volume to volume; never infer real volume.
            if volume == "tick_volume" and volume not in df and "volume" in df:
                df = df.rename(columns={"volume": volume})
            missing = {"time", *prices, volume} - set(df)
            if missing:
                raise ValueError(f"{symbol}: missing columns {sorted(missing)}")
            unit = "s" if pd.api.types.is_numeric_dtype(df.time) else None
            df["date"] = pd.to_datetime(df.time, unit=unit, utc=True)
            if df.date.isna().any() or df.date.duplicated().any():
                raise ValueError(f"{symbol}: missing or duplicate bar time")
            df = df.sort_values("date")
            if drop_last:
                df = df.iloc[:-1]
            if df.empty:
                raise ValueError(f"{symbol}: no completed bars")
            fields = [*prices, *(f for f in ("tick_volume", "real_volume", "spread") if f in df)]
            df = df.set_index("date")[fields].astype(np.float64)
            if not np.isfinite(df[[*prices, volume]].to_numpy()).all():
                raise ValueError(f"{symbol}: non-finite OHLCV")
            df["volume"] = df[volume]
            df["asset"] = symbol
            bars.append(df.set_index("asset", append=True))
        self._snapshot_time = time()
        super().__init__(pd.concat(bars), ohlcv=(*prices, "volume"), adjustments=None)

    @property
    def last_modified(self) -> float:
        return self._snapshot_time

    @classmethod
    def from_csv(cls, files: Mapping[str, str | Path], *, volume="tick_volume", drop_last=True):
        """Read one comma-separated MT5 export per symbol, dropping its newest row by default."""
        return cls(
            {s: pd.read_csv(p, encoding="utf-8-sig") for s, p in files.items()},
            volume=volume,
            drop_last=drop_last,
        )

    @classmethod
    def from_terminal(
        cls, mt5, symbols: Sequence[str], timeframe="D1", count=250, *, volume="tick_volume"
    ):
        """Fetch a snapshot using a caller-owned, initialized MetaTrader5 Python session."""
        if isinstance(symbols, str) or not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("pass a nonempty sequence of distinct broker symbols")
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        tf = getattr(mt5, f"TIMEFRAME_{timeframe.upper()}", None)
        if tf is None:
            raise ValueError(f"unsupported MT5 timeframe: {timeframe}")
        frames = {}
        for symbol in symbols:
            # Position zero is the current bar, even if the market is not ticking.
            rates = mt5.copy_rates_from_pos(symbol, tf, 1, count)
            if rates is None or len(rates) == 0:
                raise RuntimeError(f"{symbol} {timeframe}: no MT5 history ({mt5.last_error()})")
            if len(rates) < count:
                raise ValueError(f"{symbol} {timeframe}: requested {count} bars, got {len(rates)}")
            frames[symbol] = rates
        return cls(frames, volume=volume, drop_last=False)
