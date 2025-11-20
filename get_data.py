from tinkoff.invest import Client, CandleInterval
from datetime import datetime, timedelta
import pandas as pd

TOKEN = "t.sWXWh2h48nFyH3cFr886QxrA9xNOHh2Sy6ULpJydAb0f_7_HbQqfUaRbQ6BmGI6cMRNT6fcC4VmRYW7NmzOseg"   # <-- поставьте ваш API-ключ

FIGI_LUKOIL = "BBG004731032"   # FIGI Лукойл
INTERVAL = CandleInterval.CANDLE_INTERVAL_1_MIN
NEEDED = 40000                 # сколько свечей нужно
CSV_FILE = "lukoil_ohlc.csv"   # имя выходного файла

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


# --- ЗАГРУЗКА ---
candles = get_minutes_data(TOKEN, FIGI_LUKOIL, NEEDED)

# --- ПЕРЕВОД В DATAFRAME ---
df = candles_to_df(candles)

# --- СОРТИРОВКА ПО ВРЕМЕНИ ---
df = df.sort_values("time")

# --- СОХРАНЕНИЕ В CSV ---
df.to_csv(CSV_FILE, index=False, encoding="utf-8")

print(f"Данные сохранены в {CSV_FILE}")
