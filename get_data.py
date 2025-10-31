from tinkoff.invest import Client, CandleInterval
from datetime import datetime, timedelta, timezone
import pandas as pd

TOKEN = "t.sWXWh2h48nFyH3cFr886QxrA9xNOHh2Sy6ULpJydAb0f_7_HbQqfUaRbQ6BmGI6cMRNT6fcC4VmRYW7NmzOseg"
FIGI = "FUTGLDRUBF00"

interval = CandleInterval.CANDLE_INTERVAL_15_MIN  # ✅ для твоей версии SDK

now = datetime.now(timezone.utc)
start = now - timedelta(days=1600)

candles_data = []

with Client(TOKEN) as client:
    for candle in client.get_all_candles(
        figi=FIGI,
        from_=start,
        to=now,
        interval=interval,
    ):
        candles_data.append({
            "time": candle.time,
            "open": candle.open.units + candle.open.nano / 1e9,
            "high": candle.high.units + candle.high.nano / 1e9,
            "low": candle.low.units + candle.low.nano / 1e9,
            "close": candle.close.units + candle.close.nano / 1e9,
            "volume": candle.volume,
        })

df = pd.DataFrame(candles_data)
df.to_csv("SILV_futures_15m.csv", index=False)

print(df.head())