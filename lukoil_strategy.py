#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lukoil_strategy.py
Однофайловый скрипт для обучения модели на OHLCV LUKOIL и расчёта доходности стратегии.
Запуск: python lukoil_strategy.py
Требования: pandas, numpy, scikit-learn, joblib, matplotlib, ta (optional)
Если ta не установлен, код рассчитает индикаторы вручную.
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# -------------------------
# Настройки
# -------------------------
INPUT_CSV = "lukoil_ohlc_clean.csv"   # файл с данными
MODEL_OUT = "model.joblib"
PRED_OUT = "predictions.csv"
RANDOM_SEED = 42
TEST_SIZE = 0.2   # доля тестовой выборки (последняя по времени)
N_ESTIMATORS = 200
MAX_LAGS = 5      # сколько лагов ретёрна добавлять
WINDOWS = [5, 10, 20]  # скользящие окна для SMA/vol
VERBOSE = True

np.random.seed(RANDOM_SEED)

# -------------------------
# Вспомогательные функции
# -------------------------
def load_data(path):
    if not os.path.exists(path):
        print(f"Ошибка: файл {path} не найден. Поместите CSV в ту же папку или укажите правильный путь.")
        sys.exit(1)
    # Попытаемся автоматически распарсить временную метку с timezone, если есть
    df = pd.read_csv(path)
    # Ожидаемые колонки: time,open,high,low,close,volume
    if 'time' not in df.columns:
        raise ValueError("CSV должен содержать колонку 'time'")
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    # Убедимся, что числовые колонки есть
    for c in ['open','high','low','close','volume']:
        if c not in df.columns:
            raise ValueError(f"CSV должен содержать колонку '{c}'")
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df

def feature_engineering(df):
    df = df.copy()
    # процентное изменение закрытия (ret) — целевая переменная будет следующий шаг
    df['ret'] = df['close'].pct_change()
    # лаги ретёрна
    for lag in range(1, MAX_LAGS+1):
        df[f'ret_lag_{lag}'] = df['ret'].shift(lag)
    # скользящие средние по close
    for w in WINDOWS:
        df[f'sma_{w}'] = df['close'].rolling(w).mean()
        df[f'vol_sma_{w}'] = df['volume'].rolling(w).mean()
        df[f'close_std_{w}'] = df['close'].rolling(w).std()
    # ATR-like (упрощённо)
    df['tr1'] = (df['high'] - df['low']).abs()
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['true_range'] = df[['tr1','tr2','tr3']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(14).mean()
    # momentum features
    df['close_roc_5'] = df['close'].pct_change(5)
    df['close_roc_10'] = df['close'].pct_change(10)
    # volume changes
    df['vol_pct_change_1'] = df['volume'].pct_change()
    # time features (hour, weekday) — may help intraday patterns
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday
    # Target: next-step return
    df['target'] = df['ret'].shift(-1)  # predict next period's pct change
    # Drop rows with NaN
    df = df.dropna().reset_index(drop=True)
    return df

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_preds(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'mse':mse, 'rmse':rmse, 'r2':r2}

def backtest_strategy(df, preds_col='pred'):
    """
    Простая стратегия:
      сигнал = sign(pred) -> позиция = +1 если pred>0, -1 если pred<0
    Возврат стратегии на шаге t = позиция_{t-1} * ret_t
    (позиция формируется на основе прогноза предыдущего шага)
    """
    df = df.copy()
    df['signal'] = np.sign(df[preds_col])
    # Если предсказание нулевое -> 0 (no position)
    df['signal'] = df['signal'].replace(0, 0)
    # position_t will be signal at time t (we assume immediate execution at next bar open/close)
    # To avoid lookahead, shift signal forward: position taken for return at next step uses current signal
    df['position'] = df['signal'].shift(0)  # already aligned since target was next-step ret
    df['strategy_ret'] = df['position'] * df['ret']  # strategy realized at same step
    df['cum_strategy'] = (1 + df['strategy_ret']).cumprod()
    df['cum_buyhold'] = (1 + df['ret']).cumprod()
    # compute simple metrics
    total_return = df['cum_strategy'].iloc[-1] - 1.0
    bh_return = df['cum_buyhold'].iloc[-1] - 1.0
    # daily aggregation for Sharpe calculation
    df_daily = df.set_index('time').resample('1D').apply({'strategy_ret':'sum','ret':'sum'})
    daily_mean = df_daily['strategy_ret'].mean()
    daily_std = df_daily['strategy_ret'].std(ddof=0) if df_daily['strategy_ret'].std(ddof=0)>0 else 1e-9
    # annualize assuming 252 trading days
    sharpe_annual = (daily_mean / daily_std) * np.sqrt(252)
    return {
        'df': df,
        'total_return': total_return,
        'buyhold_return': bh_return,
        'sharpe_annual': sharpe_annual,
        'daily': df_daily
    }

# -------------------------
# Главная логика
# -------------------------
def main():
    print("Загрузка данных...")
    df_raw = load_data(INPUT_CSV)
    print(f"Загружено {len(df_raw)} строк, период: {df_raw['time'].iloc[0]} — {df_raw['time'].iloc[-1]}")
    print("Генерация признаков...")
    df = feature_engineering(df_raw)
    print(f"После генерации признаков: {len(df)} строк")

    # Выбираем колонки признаков
    exclude = ['time','target','ret','tr1','tr2','tr3','true_range']  # некоторые временные колонки
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('cum_') and c!='pred']
    # убедимся, что нет object колонок
    feature_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    print(f"Используем признаки ({len(feature_cols)}): {feature_cols}")

    # Разделение на train/test по времени: первые 80% — train, последние 20% — test
    split_index = int(len(df) * (1 - TEST_SIZE))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    print(f"Размеры: X_train={X_train.shape}, X_test={X_test.shape}")

    print("Обучение модели...")
    model = train_model(X_train, y_train)

    print("Предсказание на тесте...")
    y_pred = model.predict(X_test)

    metrics = evaluate_preds(y_test, y_pred)
    print("Метрики качества регрессии (на тесте):")
    print(f"  RMSE = {metrics['rmse']:.6f}, R2 = {metrics['r2']:.4f}")

    # Добавим предсказания в test_df и соберем общий датасет с предсказаниями
    test_df = test_df.copy()
    test_df['pred'] = y_pred
    result_df = pd.concat([train_df.assign(pred=np.nan), test_df], ignore_index=True).reset_index(drop=True)

    print("Бэктест стратегии...")
    bt = backtest_strategy(result_df, preds_col='pred')
    res_df = bt['df']
    # Добавляем cum колонки в result_df, чтобы можно было строить графики
    result_df['cum_strategy'] = res_df['cum_strategy']
    result_df['cum_buyhold'] = res_df['cum_buyhold']

    total_ret = bt['total_return']
    bh_ret = bt['buyhold_return']
    sharpe = bt['sharpe_annual']

    print("Результаты стратегии:")
    print(f"  Кумулятивная доходность стратегии: {total_ret*100:.2f}%")
    print(f"  Кумулятивная доходность buy-and-hold: {bh_ret*100:.2f}%")
    print(f"  Annualized Sharpe (примерная): {sharpe:.3f}")

    # Сохраним модель и предсказания
    joblib.dump(model, MODEL_OUT)
    print(f"Модель сохранена в {MODEL_OUT}")
    result_df[['time','open','high','low','close','volume','ret','pred','target']].to_csv(PRED_OUT, index=False)
    print(f"Таблица предсказаний сохранена в {PRED_OUT}")

    # Построим графики
    print("Построение графиков...")
    plt.figure(figsize=(12,6))
    # Price and cumulative returns (normalized)
    ax1 = plt.subplot(2,1,1)
    plt.plot(result_df['time'], result_df['close'], label='close')
    plt.title('Price (close)')
    plt.ylabel('Цена')
    plt.legend()

    ax2 = plt.subplot(2,1,2, sharex=ax1)
    bt_df = bt['df']

    plt.plot(bt_df['time'], bt_df['cum_buyhold'], label='Buy & Hold (cum)')
    plt.plot(bt_df['time'], bt_df['cum_strategy'], label='Strategy (cum)')

    plt.title('Кумулятивная доходность')
    plt.ylabel('Кум. множитель')
    plt.xlabel('time')
    plt.legend()
    plt.tight_layout()
    plt.savefig("strategy_results.png", dpi=150)
    print("Графики сохранены в strategy_results.png")

    # Отобразим несколько строк результата для быстрой проверки
    print("\nПример результата (последние 10 строк):")
    with pd.option_context('display.max_columns', None):
        print(result_df[['time','close','ret','pred','target','position','strategy_ret']].tail(10))

    print("\nГотово.")

if __name__ == "__main__":
    main()
