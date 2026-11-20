"""Test a pre-specified gold M15 breakout: overlap vs other sessions, 1-hour hold."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from examples.mt5_factors import configure_device
from examples.mt5_intraday import STEP, targets
from spectre.data import MT5Loader
from spectre.factors import OHLCV, FactorEngine

LOOKBACK = 16
HOLD = 4
EXTRA_BPS = 1.0
LONDON = "Europe/London"
NEW_YORK = "America/New_York"


def session_flags(index):
    """Bar-open membership. London 08:00-16:30, New York 08:00-17:00, local civil time."""
    london = index.tz_convert(LONDON)
    ny = index.tz_convert(NEW_YORK)
    l_min = np.asarray(london.hour) * 60 + np.asarray(london.minute)
    n_min = np.asarray(ny.hour) * 60 + np.asarray(ny.minute)
    in_london = (l_min >= 8 * 60) & (l_min < 16 * 60 + 30)
    in_ny = (n_min >= 8 * 60) & (n_min < 17 * 60)
    return pd.DataFrame(
        {"london": in_london, "ny": in_ny, "overlap": in_london & in_ny},
        index=index,
    )


def breakout_features(loader, symbol, device):
    engine = FactorEngine(loader)
    actual = configure_device(engine, device)
    prior_high = OHLCV.high.ts_max(LOOKBACK).shift(1)
    prior_low = OHLCV.low.ts_min(LOOKBACK).shift(1)
    engine.add(OHLCV.high > prior_high, "up")
    engine.add(OHLCV.low < prior_low, "down")
    dates = loader.load().index.levels[0]
    start = perf_counter()
    df = engine.run(dates[LOOKBACK], None, delay_factor=False).xs(symbol, level="asset")
    if actual.startswith("cuda"):
        torch.cuda.synchronize(engine.device)
    return df.astype("float64"), actual, perf_counter() - start


def events(bars, matrix, point):
    lab = targets(bars, point, HOLD, EXTRA_BPS)
    sess = session_flags(bars.index)
    seg = bars.index.to_series().diff().ne(STEP).cumsum()
    ready = seg.groupby(seg).cumcount().ge(LOOKBACK)
    data = lab.join(matrix).join(sess).loc[ready].dropna()
    up = data.up.eq(1) & data.down.eq(0)
    down = data.down.eq(1) & data.up.eq(0)
    side = up.astype(int) - down.astype(int)
    data = data.loc[side.ne(0)].copy()
    data["side"] = side.loc[data.index]
    data["gross_signed"] = data.side * data.gross
    data["net"] = np.where(data.side.eq(1), data.long, data.short)
    data["bucket"] = np.where(data.overlap, "overlap", "other")
    return data


def thin(part):
    free = None
    keep = []
    for t, exit_at in part.exit_time.items():
        if free is None or t + STEP >= free:
            keep.append(t)
            free = exit_at
    return part.loc[keep]


def summarize(part, how):
    used = thin(part) if how == "nonoverlap" else part
    return dict(
        how=how,
        n=int(len(used)),
        longs=int(used.side.eq(1).sum()),
        shorts=int(used.side.eq(-1).sum()),
        win_rate=float(used.net.gt(0).mean()) if len(used) else np.nan,
        gross_bps=float(used.gross_signed.mean()) if len(used) else np.nan,
        net_bps=float(used.net.mean()) if len(used) else np.nan,
    )


def period_table(name, data):
    rows = []
    for bucket, part in data.groupby("bucket", sort=False):
        for how in ("events", "nonoverlap"):
            rows.append(dict(period=name, bucket=bucket, **summarize(part, how)))
    return pd.DataFrame(rows)


def load_snapshot(folder):
    meta = json.loads((folder / "run.json").read_text())
    records = pd.read_parquet(folder / "bars.parquet")
    loader = MT5Loader({meta["symbol"]: records}, drop_last=False)
    bars = loader.load().xs(meta["symbol"], level="asset")
    return meta, loader, bars


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discovery", type=Path, required=True)
    ap.add_argument("--confirmation", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--out", type=Path, default=Path(".cache/mt5_session_breakout"))
    args = ap.parse_args()
    disc_meta, disc_loader, disc_bars = load_snapshot(args.discovery)
    conf_meta, conf_loader, conf_bars = load_snapshot(args.confirmation)
    if disc_meta["symbol"] != conf_meta["symbol"] or disc_meta["point"] != conf_meta["point"]:
        ap.error("discovery and confirmation snapshots are not the same contract")
    if not disc_bars.index.intersection(conf_bars.index).empty:
        ap.error("discovery and confirmation bars overlap")
    symbol, point = disc_meta["symbol"], disc_meta["point"]
    disc_x, device, disc_sec = breakout_features(disc_loader, symbol, args.device)
    conf_x, _, conf_sec = breakout_features(conf_loader, symbol, args.device)
    disc = events(disc_bars, disc_x, point)
    conf = events(conf_bars, conf_x, point)
    table = pd.concat(
        [period_table("discovery", disc), period_table("confirmation", conf)],
        ignore_index=True,
    )
    folder = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    table.to_csv(folder / "summary.csv", index=False)
    disc.assign(period="discovery").to_csv(folder / "discovery_events.csv")
    conf.assign(period="confirmation").to_csv(folder / "confirmation_events.csv")
    meta = dict(
        hypothesis=(
            "XAUUSD M15 close-of-bar break of the prior 16-bar high or low, follow for 4 bars, "
            "compare London/NY overlap with all other sessions"
        ),
        symbol=symbol,
        point=point,
        extra_roundtrip_bps=EXTRA_BPS,
        lookback_bars=LOOKBACK,
        hold_bars=HOLD,
        london=LONDON,
        new_york=NEW_YORK,
        device=device,
        gpu=torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
        discovery=str(args.discovery.resolve()),
        confirmation=str(args.confirmation.resolve()),
        discovery_range=[str(disc_bars.index[0]), str(disc_bars.index[-1])],
        confirmation_range=[str(conf_bars.index[0]), str(conf_bars.index[-1])],
        factor_seconds={"discovery": disc_sec, "confirmation": conf_sec},
        overlap_bars={
            "discovery": int(session_flags(disc_bars.index).overlap.sum()),
            "confirmation": int(session_flags(conf_bars.index).overlap.sum()),
        },
    )
    (folder / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(table.to_string(index=False))
    print(folder.resolve())


if __name__ == "__main__":
    main()
