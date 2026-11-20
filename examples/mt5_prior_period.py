"""Compare the prior M15 screen with strictly earlier, non-overlapping broker history."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from examples.mt5_factors import terminal_session
from examples.mt5_intraday import evaluate, features
from spectre.data import MT5Loader


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--terminal", required=True)
    ap.add_argument("--portable", action="store_true")
    ap.add_argument("--bars", type=int, default=40000)
    ap.add_argument("--out", type=Path, default=Path(".cache/mt5_prior_period"))
    args = ap.parse_args()
    prior = json.loads((args.reference / "run.json").read_text())
    symbol, point = prior["symbol"], prior["point"]
    if prior["timeframe"] != "M15" or prior["extra_roundtrip_bps"] != 1.0:
        ap.error("reference must be M15 with the baseline extra cost of 1 bp")
    cutoff = pd.Timestamp(prior["start"])
    ref_dates = pd.DatetimeIndex(
        pd.read_parquet(args.reference / "bars.parquet", columns=["time"]).time
    )
    assert cutoff == ref_dates.min()
    ref = pd.read_csv(args.reference / "summary.csv", converters={"low": float, "high": float})
    keys = ["factor", "horizon"]
    rules = ref.loc[ref.fold == "train"].set_index(keys)
    if not rules.index.is_unique:
        ap.error("reference contains duplicate rules")
    with terminal_session(args) as mt5:
        info = mt5.symbol_info(symbol)
        if info is None or info.chart_mode != mt5.SYMBOL_CHART_MODE_BID or point != info.point:
            ap.error("reference and terminal symbol specifications differ")
        rates = mt5.copy_rates_from(
            symbol, mt5.TIMEFRAME_M15, (cutoff - pd.Timedelta(seconds=1)).to_pydatetime(), args.bars
        )
        if rates is None or len(rates) < 1000:
            ap.error(f"insufficient earlier bars: {mt5.last_error()}")
    loader = MT5Loader({symbol: rates}, drop_last=False)
    bars = loader.load().xs(symbol, level="asset")
    assert bars.index.max() < cutoff and bars.index.intersection(ref_dates).empty
    x, device, seconds = features(loader, symbol, "cuda")
    frozen, cut1, cut2 = evaluate(bars, x, point, 1.0, frozen=rules)
    refit, _, _ = evaluate(bars, x, point, 1.0)
    # Frozen rules use all three disjoint blocks; restart flat and purge at each boundary.
    frozen["total_bps"] = frozen.trades * frozen.net_bps
    grouped = frozen.groupby(keys).agg(trades=("trades", "sum"), total=("total_bps", "sum"))
    summary = (
        ref[ref.fold == "test"]
        .set_index(keys)[["net_bps"]]
        .rename(columns={"net_bps": "round1_test_bps"})
        .join(grouped)
    )
    summary["earlier_frozen_bps"] = summary.pop("total") / summary.trades
    summary["earlier_frozen_stress_bps"] = summary.earlier_frozen_bps - 1.0
    for split in ["validation", "test"]:
        frame = refit[refit.fold == split].set_index(keys)
        summary[f"earlier_refit_{split}_bps"] = frame.net_bps
    summary["same_direction_after_refit"] = (
        rules.direction == refit[refit.fold == "train"].set_index(keys).direction
    )
    if not np.isfinite(summary.select_dtypes("number")).all().all():
        raise ValueError("incomplete comparison")
    folder = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    bars.drop(columns=[loader.time_category]).rename_axis("time").reset_index().to_parquet(
        folder / "bars.parquet", index=False
    )
    frozen.drop(columns="total_bps").to_csv(folder / "frozen_splits.csv", index=False)
    refit.to_csv(folder / "refit_splits.csv", index=False)
    summary.to_csv(folder / "comparison.csv")
    meta = dict(
        symbol=symbol,
        timeframe="M15",
        point=point,
        extra_roundtrip_bps=1.0,
        bars=len(bars),
        start=str(bars.index[0]),
        end=str(bars.index[-1]),
        reference=str(args.reference.resolve()),
        overlap=0,
        device=device,
        factor_seconds=seconds,
        validation_start=str(cut1),
        test_start=str(cut2),
        zero_spreads=int(bars.spread.eq(0).sum()),
        zero_spreads_by_split={
            k: int(g.spread.eq(0).sum())
            for k, g in [
                ("train", bars.loc[bars.index < cut1]),
                ("validation", bars.loc[(bars.index >= cut1) & (bars.index < cut2)]),
                ("test", bars.loc[bars.index >= cut2]),
            ]
        },
    )
    (folder / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(summary.round(3).to_string())
    print(folder.resolve())


if __name__ == "__main__":
    main()
