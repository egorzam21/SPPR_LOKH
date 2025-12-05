import os
import sys
import pandas as pd
import numpy as np

def load_data(path):
    if not os.path.exists(path):
        print(f"Ошибка: файл {path} не найден.")
        sys.exit(1)

    df = pd.read_csv(path)

    if 'time' not in df.columns:
        raise ValueError("CSV должен содержать колонку 'time'")

    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)

    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)

    return df
