import numpy as np
import pandas as pd
import pytest

from examples.mt5_intraday import STEP, evaluate, features, targets
from spectre.data import MT5Loader


def sample(n):
    t = pd.date_range("2024-01-01", periods=n, freq=STEP, tz="UTC")
    rng = np.random.default_rng(23)
    close = 2000 + rng.normal(size=n).cumsum()
    return pd.DataFrame(
        dict(
            open=close + 0.1,
            high=close + 1,
            low=close - 1,
            close=close,
            spread=2.0,
            tick_volume=11.0,
        ),
        index=t,
    )


def test_bid_cost_and_gap():
    df = sample(8)
    df.iloc[2, df.columns.get_loc("spread")] = 5
    result = targets(df, 0.01, 1, 1)
    first = result.iloc[0]
    a, b = df.open.iloc[1:3]
    assert first.gross == pytest.approx((b - a) / a * 10000)
    assert first.long == pytest.approx((b - a - 0.02) / a * 10000 - 1)
    assert first.short == pytest.approx((a - b - 0.05) / a * 10000 - 1)
    assert targets(df.drop(df.index[1]), 0.01, 1, 1).iloc[0].isna().all()
    df.iloc[2, df.columns.get_loc("spread")] = 0
    assert targets(df, 0.01, 1, 1).iloc[0].isna().all()


def test_future_changes_do_not_train_thresholds_or_move_validation():
    bars = sample(1000)
    x = pd.DataFrame({"f": np.sin(np.arange(len(bars)))}, index=bars.index)
    initial, cut1, cut2 = evaluate(bars, x, 0.01, 1.0)
    changed = bars.copy()
    changed.loc[cut2:, "open"] *= 3
    x.loc[cut2:, "f"] *= -2
    other, _, _ = evaluate(changed, x, 0.01, 1.0)
    pd.testing.assert_frame_equal(initial[initial.fold != "test"], other[other.fold != "test"])
    for h in (1, 4, 8):
        row = initial[(initial.horizon == h) & (initial.fold == "validation")].iloc[0]
        assert row.n == len(bars.loc[cut1:cut2]) - 1 - h - 1
        assert row.trades <= np.ceil(row.n / h)


def test_factor_parity(device):
    data = sample(125).rename_axis("time").reset_index()
    loader = MT5Loader({"gold": data}, drop_last=False)
    frame, actual, _ = features(loader, "gold", device)
    assert actual.startswith(device)
    reference = data.close.pct_change(4).iloc[15:]
    np.testing.assert_allclose(frame.momentum_4, reference, atol=1e-10)
