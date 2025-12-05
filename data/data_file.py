import os
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from tinkoff.invest import Client, CandleInterval
from features.engineer import robust_feature_engineering
from sklearn.preprocessing import StandardScaler
from backtest.backtester import Backtester
from config.params import EXECUTION

TOKEN = ""  
FIGI_LUKOIL = "BBG004731032"   
INTERVAL = CandleInterval.CANDLE_INTERVAL_1_MIN
RAW_CSV_FILE = "lukoil_ohlc.csv"   
CLEAN_CSV_FILE = "lukoil_ohlc_clean.csv"
MODEL_FILE = "final_model.joblib"

TOKEN = ""  
FIGI_LUKOIL = "BBG004731032"   
INTERVAL = CandleInterval.CANDLE_INTERVAL_1_MIN
NEEDED = 40000                
RAW_CSV_FILE = "lukoil_ohlc.csv"   
CLEAN_CSV_FILE = "lukoil_ohlc_clean.csv"

def get_minutes_data(token, figi, needed):
    all_candles = []

    end = datetime.now(timezone.utc)
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

def get_new_data(token, figi, last_time):

    end = datetime.now(timezone.utc)

    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)
    
    if (end - last_time).total_seconds() < 60:
        return []
    
    print(f"Загрузка новых данных: {last_time} → {end}")
    
    all_candles = []
    start = last_time
    
    with Client(token) as client:
        while start < end:
            batch_end = min(start + timedelta(days=1), end)
            
            try:
                resp = client.market_data.get_candles(
                    figi=figi,
                    from_=start,
                    to=batch_end,
                    interval=INTERVAL
                )

                candles = resp.candles
                if candles:
                    all_candles.extend(candles)
                
                start = batch_end

                time.sleep(0.1)
                
            except Exception as e:
                print(f"Ошибка при загрузке батча: {e}")
                break
    
    return all_candles

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

def clean_data(df):
    """Очистка данных"""
    df = df.copy()
    

    df.columns = [c.strip().lower() for c in df.columns]


    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
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
    df = df[returns < 0.05].copy()

    print(f"Очищено: осталось {len(df)} строк")
    return df

def save_data(df, filename):
    """Сохранение данных с обработкой временных зон"""
    df_to_save = df.copy()

    df_to_save['time'] = df_to_save['time'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
    df_to_save.to_csv(filename, index=False, encoding="utf-8")

def load_data(filename):
    """Загрузка данных с обработкой временных зон"""
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
        return df
    return None

def initial_load():
    """Первоначальная загрузка данных"""
    print("Начальная загрузка данных...")
    candles = get_minutes_data(TOKEN, FIGI_LUKOIL, NEEDED)
    df = candles_to_df(candles)
    df = df.sort_values("time")
    

    clean_df = clean_data(df)
    

    save_data(df, RAW_CSV_FILE)
    save_data(clean_df, CLEAN_CSV_FILE)
    
    print(f"Первоначальные данные сохранены в {RAW_CSV_FILE}")
    print(f"Очищенные данные сохранены в {CLEAN_CSV_FILE}")
    return df, clean_df



def continuous_update_with_signals():

    import time
    from datetime import datetime, timezone
    import joblib
    from data_file import load_data, RAW_CSV_FILE, CLEAN_CSV_FILE, get_new_data, candles_to_df, clean_data, TOKEN, FIGI_LUKOIL
    from features.engineer import robust_feature_engineering
    from models.ensemble import ensemble_predict


    raw_df = load_data(RAW_CSV_FILE)
    clean_df = load_data(CLEAN_CSV_FILE)

    if raw_df is None or clean_df is None:
        from data_file import initial_load
        raw_df, clean_df = initial_load()
    else:
        print(f"Загружено существующих данных: {len(raw_df)} строк")
        print(f"Загружено очищенных данных: {len(clean_df)} строк")


    MODEL_FILE = "final_model.joblib"
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"{MODEL_FILE} не найден. Сначала запустите main.py для обучения модели.")

    model_data = joblib.load(MODEL_FILE)
    models = model_data['models']
    scaler = model_data['scaler']
    features = model_data['features']
    best_params = model_data['best_params']

    if not models or len(models) == 0:
        raise ValueError("Ансамбль моделей пуст. Проверьте, что main.py обучил модели и сохранил их.")

    print(f"Ансамбль моделей загружен. Количество моделей: {len(models)}")

    while True:
        try:
            last_time = raw_df['time'].max()
            new_candles = get_new_data(TOKEN, FIGI_LUKOIL, last_time)

            if new_candles:

                new_df = candles_to_df(new_candles)
                raw_df = pd.concat([raw_df, new_df], ignore_index=True)
                raw_df = raw_df.drop_duplicates(subset=['time']).sort_values('time')

                clean_df = clean_data(raw_df)

                clean_df.to_csv(CLEAN_CSV_FILE, index=False, encoding="utf-8")
                raw_df.to_csv(RAW_CSV_FILE, index=False, encoding="utf-8")

                df_feat = robust_feature_engineering(clean_df)
                df_feat = df_feat.sort_values('time').reset_index(drop=True)

                X_new = df_feat[features].iloc[-1:]

                if X_new.isna().any().any():
                    print("Последняя свеча содержит NaN признаки. Пропускаем предсказание.")
                    time.sleep(60)
                    continue

                X_scaled = scaler.transform(X_new)

                pred = ensemble_predict(models, X_scaled)[0]

                threshold = 0.0000005
                if pred > threshold:
                    signal = "BUY"
                elif pred < -threshold:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                print(f"{datetime.now(timezone.utc)} | Новая свеча: close={df_feat['close'].iloc[-1]:.2f} "
                      f"| Pred={pred:.6f} | Signal: {signal}")

            else:
                print(f"{datetime.now(timezone.utc)} | Новых данных нет")

            time.sleep(60)  

        except Exception as e:
            print(f"Ошибка при обновлении: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":

    continuous_update_with_signals()
