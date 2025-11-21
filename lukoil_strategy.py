
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

INPUT_CSV = "lukoil_ohlc_clean.csv"   
MODEL_OUT = "model.joblib"
PRED_OUT = "predictions.csv"
RANDOM_SEED = 42
TEST_SIZE = 0.2   
N_ESTIMATORS = 200
MAX_LAGS = 5      
WINDOWS = [5, 10, 20]  
VERBOSE = True

np.random.seed(RANDOM_SEED)

def load_data(path):
    if not os.path.exists(path):
        print(f"Ошибка: файл {path} не найден. Поместите CSV в ту же папку или укажите правильный путь.")
        sys.exit(1)
    df = pd.read_csv(path)
    if 'time' not in df.columns:
        raise ValueError("CSV должен содержать колонку 'time'")
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    for c in ['open','high','low','close','volume']:
        if c not in df.columns:
            raise ValueError(f"CSV должен содержать колонку '{c}'")
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df

def feature_engineering(df):
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    for lag in range(1, MAX_LAGS+1):
        df[f'ret_lag_{lag}'] = df['ret'].shift(lag)
    for w in WINDOWS:
        df[f'sma_{w}'] = df['close'].rolling(w).mean()
        df[f'vol_sma_{w}'] = df['volume'].rolling(w).mean()
        df[f'close_std_{w}'] = df['close'].rolling(w).std()
    df['tr1'] = (df['high'] - df['low']).abs()
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['true_range'] = df[['tr1','tr2','tr3']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(14).mean()
    df['close_roc_5'] = df['close'].pct_change(5)
    df['close_roc_10'] = df['close'].pct_change(10)
    df['vol_pct_change_1'] = df['volume'].pct_change()
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday

    df['target'] = df['ret'].shift(-1)  
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
    df = df.copy()
    df['signal'] = np.sign(df[preds_col])
    df['signal'] = df['signal'].replace(0, 0)
    df['position'] = df['signal'].shift(0)  
    df['strategy_ret'] = df['position'] * df['ret']  
    df['cum_strategy'] = (1 + df['strategy_ret']).cumprod()
    df['cum_buyhold'] = (1 + df['ret']).cumprod()
    total_return = df['cum_strategy'].iloc[-1] - 1.0
    bh_return = df['cum_buyhold'].iloc[-1] - 1.0
    df_daily = df.set_index('time').resample('1D').apply({'strategy_ret':'sum','ret':'sum'})
    daily_mean = df_daily['strategy_ret'].mean()
    daily_std = df_daily['strategy_ret'].std(ddof=0) if df_daily['strategy_ret'].std(ddof=0)>0 else 1e-9
    sharpe_annual = (daily_mean / daily_std) * np.sqrt(252)
    return {
        'df': df,
        'total_return': total_return,
        'buyhold_return': bh_return,
        'sharpe_annual': sharpe_annual,
        'daily': df_daily
    }

def main():
    print("Загрузка данных...")
    df_raw = load_data(INPUT_CSV)
    print(f"Загружено {len(df_raw)} строк, период: {df_raw['time'].iloc[0]} — {df_raw['time'].iloc[-1]}")
    print("Генерация признаков...")
    df = feature_engineering(df_raw)
    print(f"После генерации признаков: {len(df)} строк")
    exclude = ['time','target','ret','tr1','tr2','tr3','true_range']  
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('cum_') and c!='pred']
    feature_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    print(f"Используем признаки ({len(feature_cols)}): {feature_cols}")

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

    test_df = test_df.copy()
    test_df['pred'] = y_pred
    result_df = pd.concat([train_df.assign(pred=np.nan), test_df], ignore_index=True).reset_index(drop=True)

    print("Бэктест стратегии...")
    bt = backtest_strategy(result_df, preds_col='pred')
    res_df = bt['df']
    result_df['cum_strategy'] = res_df['cum_strategy']
    result_df['cum_buyhold'] = res_df['cum_buyhold']

    total_ret = bt['total_return']
    bh_ret = bt['buyhold_return']
    sharpe = bt['sharpe_annual']

    print("Результаты стратегии:")
    print(f"  Кумулятивная доходность стратегии: {total_ret*100:.2f}%")
    print(f"  Кумулятивная доходность buy-and-hold: {bh_ret*100:.2f}%")
    print(f"  Annualized Sharpe (примерная): {sharpe:.3f}")

    joblib.dump(model, MODEL_OUT)
    print(f"Модель сохранена в {MODEL_OUT}")
    result_df[['time','open','high','low','close','volume','ret','pred','target']].to_csv(PRED_OUT, index=False)
    print(f"Таблица предсказаний сохранена в {PRED_OUT}")

    print("Построение графиков...")
    plt.figure(figsize=(12,6))
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

    print("\nПример результата (последние 10 строк):")
    with pd.option_context('display.max_columns', None):
        print(result_df[['time','close','ret','pred','target','position','strategy_ret']].tail(10))

    print("\nГотово.")

if __name__ == "__main__":
    main()
