"""Gold M15 factor screen. Run from the checkout: python -m examples.mt5_intraday --help."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from examples.mt5_factors import configure_device, terminal_session
from spectre.data import MT5Loader
from spectre.factors import OHLCV, SMA, FactorEngine, Returns

HORIZONS = (1, 4, 8)
WINDOW = 16
STEP = pd.Timedelta(minutes=15)


def features(loader, symbol, device):
    engine = FactorEngine(loader)
    actual = configure_device(engine, device)
    c, o, h, low, v = OHLCV.close, OHLCV.open, OHLCV.high, OHLCV.low, OHLCV.volume
    width = h - low
    for name, factor in {
        "momentum_4": Returns(5),
        "ma_distance_16": c / SMA(WINDOW) - 1,
        "body_fraction": (c - o) / width,
        "relative_range_16": width / SMA(WINDOW, inputs=[width]),
        "relative_ticks_16": v / SMA(WINDOW, inputs=[v]),
    }.items():
        engine.add(factor, name)
    dates = loader.load().index.levels[0]
    start = perf_counter()
    # A row describes the closed bar; labels enter one open later.
    df = engine.run(dates[WINDOW - 1], None, delay_factor=False).xs(symbol, level="asset")
    if actual.startswith("cuda"):
        torch.cuda.synchronize(engine.device)
    return df.replace([np.inf, -np.inf], np.nan), actual, perf_counter() - start


def targets(bars, point, horizon, extra_bps):
    """Bid-chart open-to-open returns in basis points, with directional spread charges."""
    entry, exit_ = bars.open.shift(-1), bars.open.shift(-1 - horizon)
    a, b = bars.spread.shift(-1), bars.spread.shift(-1 - horizon)
    clock = bars.index.to_series()
    seg = clock.diff().ne(STEP).cumsum()
    valid = seg.eq(seg.shift(-1 - horizon)) & a.gt(0) & b.gt(0) & entry.gt(0) & exit_.gt(0)
    gross = (exit_ - entry) / entry * 10000
    return pd.DataFrame(
        {
            "gross": gross,
            "long": (exit_ - (entry + a * point)) / entry * 10000 - extra_bps,
            "short": (entry - (exit_ + b * point)) / entry * 10000 - extra_bps,
            "exit_time": clock.shift(-1 - horizon),
        }
    ).where(valid)


def evaluate(bars, matrix, point, extra_bps, *, frozen=None):
    dates = bars.index
    cut1, cut2 = dates[len(dates) * 3 // 5], dates[len(dates) * 4 // 5]
    seg = dates.to_series().diff().ne(STEP).cumsum()
    warm = seg.groupby(seg).cumcount().ge(WINDOW - 1)
    rows = []
    for h in HORIZONS:
        lab = targets(bars, point, h, extra_bps)
        for name in matrix:
            data = lab.join(matrix[name]).loc[warm].dropna()
            train = data.loc[data.exit_time < cut1]
            if len(train) < 100:
                raise ValueError(f"not enough training data for {name}, horizon {h}")
            if frozen is None:
                lo, hi = train[name].quantile([0.2, 0.8])
                ic = train[name].corr(train.gross, method="spearman")
                if not np.isfinite(ic) or lo >= hi:
                    raise ValueError(f"uninformative training distribution: {name}, horizon {h}")
                direction = 1 if ic >= 0 else -1
            else:
                rule = frozen.loc[(name, h)]
                lo, hi, direction = rule.low, rule.high, int(rule.direction)
                if not np.isfinite([lo, hi]).all() or lo >= hi or direction not in (-1, 1):
                    raise ValueError(f"invalid frozen rule: {name}, horizon {h}")
            for fold, part in [
                ("train", train),
                ("validation", data.loc[(data.index >= cut1) & (data.exit_time < cut2)]),
                ("test", data.loc[data.index >= cut2]),
            ]:
                # Only one position per factor/horizon. A new entry can reuse an exit open.
                side = (part[name].gt(hi).astype(int) - part[name].lt(lo).astype(int)) * direction
                free = None
                trades = []
                for t, s in side[side.ne(0)].items():
                    if free is None or t + STEP >= free:
                        trades.append((t, s))
                        free = part.at[t, "exit_time"]
                ts = [t for t, _ in trades]
                signs = np.array([s for _, s in trades])
                taken = part.loc[ts]
                net = np.where(signs == 1, taken.long, taken.short)
                rows.append(
                    dict(
                        factor=name,
                        horizon=h,
                        fold=fold,
                        n=len(part),
                        low=lo,
                        high=hi,
                        direction=direction,
                        rank_ic=part[name].corr(part.gross, method="spearman"),
                        trades=len(trades),
                        gross_bps=(signs * taken.gross).mean(),
                        net_bps=net.mean() if len(net) else np.nan,
                    )
                )
    return pd.DataFrame(rows), cut1, cut2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--terminal", help="full path to an already running MT5 terminal")
    source.add_argument("--replay", type=Path, help="prior run directory with bars and run.json")
    ap.add_argument("--portable", action="store_true")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--bars", type=int, default=60000)
    ap.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    ap.add_argument("--check-cpu", action="store_true", help="compare the same GPU snapshot on CPU")
    ap.add_argument("--extra-roundtrip-bps", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path(".cache/mt5_intraday"))
    args = ap.parse_args()
    extra = args.extra_roundtrip_bps
    if not np.isfinite(extra) or extra < 0:
        ap.error("extra cost must be finite and nonnegative")
    if args.replay:
        prior = json.loads((args.replay / "run.json").read_text())
        symbol, point = prior["symbol"], prior["point"]
        records = pd.read_parquet(args.replay / "bars.parquet")
        loader = MT5Loader({symbol: records}, drop_last=False)
    else:
        symbol = args.symbol
        with terminal_session(args) as mt5:
            info = mt5.symbol_info(symbol)
            if info is None or info.point <= 0 or info.chart_mode != mt5.SYMBOL_CHART_MODE_BID:
                ap.error("research needs a known symbol with bid bars and a positive point size")
            point = info.point
            loader = MT5Loader.from_terminal(mt5, [symbol], "M15", args.bars)
    bars = loader.load().xs(symbol, level="asset")
    if len(bars) < 1000 or "spread" not in bars:
        ap.error("need at least 1000 closed M15 bars with historical spread")
    matrix, device, seconds = features(loader, symbol, args.device)
    gpu = torch.cuda.get_device_name() if device.startswith("cuda") else None
    cpu_seconds = None
    if args.check_cpu:
        if gpu is None:
            ap.error("--check-cpu needs a CUDA run to compare")
        reference, _, cpu_seconds = features(loader, symbol, "cpu")
        pd.testing.assert_frame_equal(matrix, reference, check_exact=False, atol=1e-10, rtol=1e-8)
    result, cut1, cut2 = evaluate(bars, matrix, point, extra)
    folder = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    bars.drop(columns=[loader.time_category]).rename_axis("time").reset_index().to_parquet(
        folder / "bars.parquet", index=False
    )
    result.to_csv(folder / "summary.csv", index=False)
    meta = dict(
        symbol=symbol,
        timeframe="M15",
        bars=len(bars),
        start=str(bars.index[0]),
        end=str(bars.index[-1]),
        validation_start=str(cut1),
        test_start=str(cut2),
        point=point,
        extra_roundtrip_bps=extra,
        zero_spreads=int((bars.spread == 0).sum()),
        device=device,
        gpu=gpu,
        cuda_peak_mib=torch.cuda.max_memory_allocated() / 2**20 if gpu else None,
        torch=torch.__version__,
        gpu_cpu_parity=args.check_cpu,
        factor_seconds=seconds,
        cpu_check_seconds=cpu_seconds,
    )
    (folder / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(
        result.loc[result.fold.eq("test"), ["factor", "horizon", "trades", "net_bps"]].to_string(
            index=False
        )
    )
    print(folder.resolve())


if __name__ == "__main__":
    main()
