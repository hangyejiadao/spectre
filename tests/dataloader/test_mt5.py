from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from spectre.data import MT5Loader
from spectre.factors import OHLCV, FactorEngine, Returns


def rates(offset=0):
    closes = np.arange(22.0) + 100 + offset
    return pd.DataFrame(
        {
            "time": 1704067200 + np.arange(22) * 86400,
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "tick_volume": 100,
            "real_volume": 0,
            "spread": 12,
        }
    )


def test_terminal_and_csv_match_engine(tmp_path):
    files, records = {}, {}
    for symbol, offset in [("XAUm", 300), ("DXY.a", 1)]:
        df = rates(offset)
        records[symbol] = df.iloc[:-1].to_records(index=False)
        df["time"] = pd.to_datetime(df.time, unit="s", utc=True)
        files[symbol] = tmp_path / f"{symbol}.csv"
        df.rename(columns={"tick_volume": "volume"}).iloc[::-1].to_csv(files[symbol], index=False)
    sdk = Mock(spec_set=["TIMEFRAME_D1", "copy_rates_from_pos"])
    sdk.TIMEFRAME_D1 = 16408

    def fetch(symbol, timeframe, pos, count):
        assert (timeframe, pos, count) == (16408, 1, 21)
        return records[symbol]

    sdk.copy_rates_from_pos.side_effect = fetch
    live = MT5Loader.from_terminal(sdk, list(files), "d1", 21)
    saved = MT5Loader.from_csv(files)
    pd.testing.assert_frame_equal(live.load(), saved.load(), check_index_type=False)
    df = live.load()
    assert df.index.names == ["date", "asset"]
    assert str(df.index.levels[0].tz) == "UTC"
    assert len(df) == 42 and (df.volume == 100).all() and (df.spread == 12).all()
    assert (MT5Loader.from_csv(files, volume="real_volume").load().volume == 0).all()
    engine = FactorEngine(live)
    engine.add(Returns(21), "ret")
    engine.add(OHLCV.volume, "vol")
    latest = engine.run(df.index.levels[0][-1], None, delay_factor=False)
    for (_, symbol), row in latest.iterrows():
        closes = records[symbol]["close"]
        assert row.ret == pytest.approx(closes[-1] / closes[-21] - 1)
        assert row.vol == 100


@pytest.mark.parametrize("result", [None, [], rates()[:4].to_records(index=False)])
def test_terminal_rejects_no_or_short_history(result):
    sdk = Mock(TIMEFRAME_H4=1)
    sdk.copy_rates_from_pos.return_value = result
    with pytest.raises((RuntimeError, ValueError), match="EURUSD H4"):
        MT5Loader.from_terminal(sdk, ["EURUSD"], "H4", 10)


def test_bad_data_is_not_silently_repaired():
    df = rates()
    with pytest.raises(ValueError, match="duplicate"):
        MT5Loader({"x": pd.concat([df, df.iloc[-1:]])})
    with pytest.raises(ValueError, match="missing columns"):
        MT5Loader({"x": df.drop(columns="real_volume")}, volume="real_volume")
    with pytest.raises(ValueError, match="no completed bars"):
        MT5Loader({"x": df[:1]})
