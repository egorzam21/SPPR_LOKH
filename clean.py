import pandas as pd
import numpy as np

df = pd.read_csv("GOLD_futures_15m.csv")

df.columns = [c.strip().lower() for c in df.columns]

df['time'] = pd.to_datetime(df['time'], errors='coerce')
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['time', 'open', 'high', 'low', 'close'])
df = df.drop_duplicates(subset=['time']).sort_values('time')

df = df[(df['high'] >= df['low']) & 
        (df['open'] >= df['low']) & (df['open'] <= df['high']) &
        (df['close'] >= df['low']) & (df['close'] <= df['high'])]

df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
df = df[df['volume'] >= 0]

def remove_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.between(lower, upper)

price_cols = ['open', 'high', 'low', 'close']
mask = np.logical_and.reduce([remove_outliers_iqr(df[c]) for c in price_cols])
df = df[mask].copy()

returns = df['close'].pct_change().abs()
df = df[returns < 0.05]  

print(f"Очищено: осталось {len(df)} строк из исходных")
df.to_csv("gold_futures_clean.csv", index=False)
