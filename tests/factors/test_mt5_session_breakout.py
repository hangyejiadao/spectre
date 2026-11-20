import numpy as np
import pandas as pd

from examples.mt5_intraday import STEP
from examples.mt5_session_breakout import LOOKBACK, breakout_features, session_flags
from spectre.data import MT5Loader


def test_dst_overlap_windows():
    winter = pd.DatetimeIndex(["2024-01-10 12:00", "2024-01-10 13:00"], tz="UTC")
    summer = pd.DatetimeIndex(["2024-07-10 11:00", "2024-07-10 12:00"], tz="UTC")
    w = session_flags(winter)
    s = session_flags(summer)
    assert list(w.overlap) == [False, True]
    assert list(s.overlap) == [False, True]
    assert w.loc[winter[1], ["london", "ny"]].all()
    assert s.loc[summer[1], ["london", "ny"]].all()


def test_breakout_uses_prior_window_only(device):
    n = LOOKBACK + 4
    t = pd.date_range("2024-01-02", periods=n, freq=STEP, tz="UTC")
    high = np.full(n, 10.0)
    low = np.full(n, 9.0)
    high[LOOKBACK] = 12.0
    high[-1] = 20.0
    low[-1] = 8.0
    df = (
        pd.DataFrame(
            dict(open=9.5, high=high, low=low, close=9.5, spread=2.0, tick_volume=5.0),
            index=t,
        )
        .rename_axis("time")
        .reset_index()
    )
    loader = MT5Loader({"gold": df}, drop_last=False)
    x, actual, _ = breakout_features(loader, "gold", device)
    assert actual.startswith(device)
    first = t[LOOKBACK]
    assert x.loc[first, "up"] == 1 and x.loc[first, "down"] == 0
    assert x.loc[t[-2], ["up", "down"]].eq(0).all()
    assert x.loc[t[-1], "up"] == 1 and x.loc[t[-1], "down"] == 1
