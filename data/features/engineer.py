import numpy as np
import pandas as pd

def robust_feature_engineering(df):
    df = df.copy()

    df['ret'] = df['close'].pct_change()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    for lag in [1, 2, 3, 5, 10]:
        df[f'ret_lag_{lag}'] = df['ret'].shift(lag)

    for w in [5, 10, 20]:
        df[f'sma_{w}'] = df['close'].rolling(w).mean()
        df[f'std_{w}'] = df['close'].rolling(w).std()
        df[f'close_vs_sma_{w}'] = (df['close'] / df[f'sma_{w}'] - 1) * 100
        df[f'volatility_{w}'] = df['ret'].rolling(w).std() * 100
        df[f'volume_sma_{w}'] = df['volume'].rolling(w).mean()
        df[f'volume_ratio_{w}'] = df['volume'] / df[f'volume_sma_{w}']

    for w in [6, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(w).mean()
        avg_loss = loss.rolling(w).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'rsi_{w}'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    for w in [5, 14]:
        df[f'atr_{w}'] = df['true_range'].rolling(w).mean()

    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']

    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday
    df['is_russian_market_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 18)).astype(int)

    df['volatility_volume'] = df['volatility_5'] * np.log1p(df['volume'])

    for horizon in [1, 3, 5, 10]:
        df[f'target_ret_{horizon}'] = df['close'].shift(-horizon) / df['close'] - 1

    df['target_main'] = df['target_ret_5']
    df['target_direction'] = (df['target_ret_5'] > 0).astype(int)

    threshold = df['target_ret_5'].abs().quantile(0.6)
    df['target_conditional'] = df['target_ret_5'].where(
        df['target_ret_5'].abs() > threshold, 0
    )

    df = df.dropna().reset_index(drop=True)
    return df
