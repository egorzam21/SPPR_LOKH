from tinkoff.invest import Client, CandleInterval
from datetime import datetime, timedelta
import pandas as pd

TOKEN = ""   

FIGI_LUKOIL = "BBG004731032"   
INTERVAL = CandleInterval.CANDLE_INTERVAL_1_MIN
NEEDED = 40000                 
CSV_FILE = "lukoil_ohlc.csv"   

def get_minutes_data(token, figi, needed):
    all_candles = []
    
    end = datetime.utcnow()
    start = end - timedelta(days=1)

    with Client(token) as client:
        while len(all_candles) < needed:
            print(f"Загрузка: {start} → {end}")

            resp = client.market_data.get_candles(
                figi=figi,
                from_=start,
                to=end,
                interval=INTERVAL
            )

            candles = resp.candles

            if not candles:
                print("Нет больше данных.")
                break

            all_candles.extend(candles)

            end = start
            start = end - timedelta(days=1)

        print(f"Загружено свечей: {len(all_candles)}")
        return all_candles[:needed]


def candles_to_df(candles):
    rows = []
    for c in candles:
        rows.append({
            "time": c.time,
            "open": c.open.units + c.open.nano / 1e9,
            "high": c.high.units + c.high.nano / 1e9,
            "low": c.low.units + c.low.nano / 1e9,
            "close": c.close.units + c.close.nano / 1e9,
            "volume": c.volume
        })

    return pd.DataFrame(rows)


candles = get_minutes_data(TOKEN, FIGI_LUKOIL, NEEDED)

df = candles_to_df(candles)

df = df.sort_values("time")

df.to_csv(CSV_FILE, index=False, encoding="utf-8")

print(f"Данные сохранены в {CSV_FILE}")

