
import os
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from tinkoff.invest import Client, CandleInterval
from features.engineer import robust_feature_engineering
from sklearn.preprocessing import StandardScaler
from models.ensemble import ensemble_predict
from explanations import generate_detailed_explanation
import threading
import traceback
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

TOKEN = ""
FIGI_LUKOIL = "BBG004731032"
INTERVAL = CandleInterval.CANDLE_INTERVAL_1_MIN
RAW_CSV_FILE = "lukoil_ohlc.csv"
CLEAN_CSV_FILE = "lukoil_ohlc_clean.csv"
MODEL_FILE = "final_model.joblib"
NEEDED = 40000
UPDATE_SECONDS = 60
SIGNALS_LOG_CSV = "signals_log.csv"

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

    if last_time is None:
        return []

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
    df_to_save = df.copy()
    df_to_save['time'] = df_to_save['time'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
    df_to_save.to_csv(filename, index=False, encoding="utf-8")

def load_data(filename):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
        return df
    return None

class TradingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Market Signals Dashboard")
        self.geometry("1100x700")

        self.raw_df = None
        self.clean_df = None
        self.model_data = None
        self.models = None
        self.scaler = None
        self.features = None

        
        self.signals = []  
        self.counts = {"BUY":0, "SELL":0, "HOLD":0}

        self._create_widgets()
        self._load_model_and_data()

        self.after(1000, self.periodic_update)

    def _create_widgets(self):
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, (self.ax_price, self.ax_vol) = plt.subplots(2, 1, figsize=(7,5), sharex=True, gridspec_kw={'height_ratios':[3,1]})
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(self, width=350)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        lbl = ttk.Label(right, text="Последнее объяснение сигнала", font=("TkDefaultFont", 12, "bold"))
        lbl.pack(pady=(8,0))
        self.txt_expl = scrolledtext.ScrolledText(right, width=40, height=12, wrap=tk.WORD)
        self.txt_expl.pack(padx=8, pady=(4,8))

        cnt_frame = ttk.Frame(right)
        cnt_frame.pack(pady=(4,8), padx=8, fill=tk.X)
        ttk.Label(cnt_frame, text="Counters:", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
        self.lbl_buy = ttk.Label(cnt_frame, text="BUY: 0")
        self.lbl_buy.pack(anchor=tk.W)
        self.lbl_sell = ttk.Label(cnt_frame, text="SELL: 0")
        self.lbl_sell.pack(anchor=tk.W)
        self.lbl_hold = ttk.Label(cnt_frame, text="HOLD: 0")
        self.lbl_hold.pack(anchor=tk.W)

        ttk.Label(right, text="Журнал сигналов (последние):", font=("TkDefaultFont", 10, "bold")).pack(pady=(8,0), padx=8, anchor=tk.W)
        self.lst_signals = tk.Listbox(right, height=12, width=50)
        self.lst_signals.pack(padx=8, pady=(4,8), fill=tk.BOTH, expand=False)

        ctrl_frame = ttk.Frame(right)
        ctrl_frame.pack(padx=8, pady=(4,8), fill=tk.X)
        ttk.Button(ctrl_frame, text="Force update", command=self.force_update).pack(side=tk.LEFT)
        ttk.Button(ctrl_frame, text="Export signals CSV", command=self.export_signals).pack(side=tk.LEFT, padx=(8,0))
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, left)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _load_model_and_data(self):
        if not os.path.exists(MODEL_FILE):
            messagebox.showerror("Error", f"{MODEL_FILE} not found. Train model first (run main.py).")
            self.destroy()
            return

        print("Loading model...")
        self.model_data = joblib.load(MODEL_FILE)
        self.models = self.model_data['models']
        self.scaler = self.model_data['scaler']
        self.features = self.model_data['features']
        print(f"Loaded models: {list(self.models.keys())}")

        raw = load_data(RAW_CSV_FILE)
        clean = load_data(CLEAN_CSV_FILE)
        if raw is None or clean is None:
            print("No saved CSVs found. Doing initial load from API (this may take a while)...")
            df_raw, df_clean = initial_load_local()
            raw = df_raw
            clean = df_clean

        self.raw_df = raw.sort_values('time').reset_index(drop=True)
        self.clean_df = clean.sort_values('time').reset_index(drop=True)

        if os.path.exists(SIGNALS_LOG_CSV):
            try:
                df_sig = pd.read_csv(SIGNALS_LOG_CSV, parse_dates=['time'])
                for _, r in df_sig.iterrows():
                    rec = {
                        'time': r['time'],
                        'signal': r['signal'],
                        'pred': r['pred'],
                        'close': r['close'],
                        'explanation': r.get('explanation', '')
                    }
                    self.signals.append(rec)
                    self.counts[ r['signal'] ] = self.counts.get(r['signal'], 0) + 1
                self._refresh_signals_list()
                self._refresh_counters()
            except Exception as e:
                print("Не удалось загрузить signals_log.csv:", e)

        self._draw_chart()

    def initial_load_local(self):
        print("Initial load (may be slow)...")
        candles = get_minutes_data(TOKEN, FIGI_LUKOIL, NEEDED)
        df = candles_to_df(candles)
        df = df.sort_values("time").reset_index(drop=True)
        clean_df = clean_data(df)
        save_data(df, RAW_CSV_FILE)
        save_data(clean_df, CLEAN_CSV_FILE)
        return df, clean_df

    def _draw_chart(self):
        if self.clean_df is None or len(self.clean_df) == 0:
            return

        df = self.clean_df.tail(600)
        xs = pd.to_datetime(df['time'])
        ys = df['close']

        self.ax_price.clear()
        self.ax_vol.clear()

        self.ax_price.plot(xs, ys, color='blue', label='Close')
        self.ax_price.set_ylabel("Price")
        self.ax_price.grid(True)

        self.ax_vol.bar(xs, df['volume'], width=0.0005, color='gray')
        self.ax_vol.set_ylabel("Volume")
        self.ax_vol.grid(True)

        for s in self.signals[-50:]:
            ts = pd.to_datetime(s['time'])
            close = s['close']
            if s['signal'] == 'BUY':
                self.ax_price.scatter(ts, close, color='green', marker='^', s=100, zorder=5)
            elif s['signal'] == 'SELL':
                self.ax_price.scatter(ts, close, color='red', marker='v', s=100, zorder=5)

        self.fig.autofmt_xdate()
        self.canvas.draw()

    def force_update(self):
        self._perform_update()

    def export_signals(self):
        if not self.signals:
            messagebox.showinfo("Export", "Нет сигналов для экспорта.")
            return
        df = pd.DataFrame(self.signals)
        df.to_csv(SIGNALS_LOG_CSV, index=False, encoding="utf-8")
        messagebox.showinfo("Export", f"signals saved to {SIGNALS_LOG_CSV}")

    def _refresh_signals_list(self):
        self.lst_signals.delete(0, tk.END)
        for s in self.signals[-200:]:
            ts = pd.to_datetime(s['time'])
            text = f"{ts} | {s['signal']} | close={s['close']:.2f} | pred={s['pred']:.6f}"
            self.lst_signals.insert(tk.END, text)

    def _refresh_counters(self):
        self.lbl_buy.config(text=f"BUY: {self.counts.get('BUY',0)}")
        self.lbl_sell.config(text=f"SELL: {self.counts.get('SELL',0)}")
        self.lbl_hold.config(text=f"HOLD: {self.counts.get('HOLD',0)}")

    def periodic_update(self):
        try:
            t = threading.Thread(target=self._perform_update, daemon=True)
            t.start()
        except Exception as e:
            print("Error scheduling update:", e)
        self.after(UPDATE_SECONDS * 1000, self.periodic_update)

    def _perform_update(self):
        try:
            last_time = None
            if self.raw_df is not None and len(self.raw_df) > 0:
                last_time = self.raw_df['time'].max()

            new_candles = get_new_data(TOKEN, FIGI_LUKOIL, last_time)

            if new_candles:
                new_df = candles_to_df(new_candles)
                self.raw_df = pd.concat([self.raw_df, new_df], ignore_index=True)
                self.raw_df = self.raw_df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)

                self.clean_df = clean_data(self.raw_df)
                save_data(self.raw_df, RAW_CSV_FILE)
                save_data(self.clean_df, CLEAN_CSV_FILE)

                df_feat = robust_feature_engineering(self.clean_df)
                df_feat = df_feat.sort_values('time').reset_index(drop=True)

                X_new = df_feat[self.features].iloc[-1:]
                if X_new.isna().any().any():
                    print("Последняя свеча содержит NaN признаки. Пропускаем предсказание.")
                    self._draw_chart()
                    return

                X_scaled = self.scaler.transform(X_new)

                pred = ensemble_predict(self.models, X_scaled)[0]

                threshold = 0.0000005
                if pred > threshold:
                    signal = "BUY"
                elif pred < -threshold:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                explanation = generate_detailed_explanation(signal, pred, df_feat, self.features, self.models)

                tnow = datetime.now(timezone.utc)
                close_val = float(df_feat['close'].iloc[-1])
                rec = {
                    'time': tnow,
                    'signal': signal,
                    'pred': float(pred),
                    'close': close_val,
                    'explanation': explanation.replace("\n", " || ")
                }
                self.signals.append(rec)
                self.counts[signal] = self.counts.get(signal, 0) + 1

                self.after(0, lambda: self._update_ui_with_signal(explanation, rec))

            else:
                self.after(0, self._draw_chart)
                print(f"{datetime.now(timezone.utc)} | Новых данных нет")

        except Exception as e:
            print("Ошибка при обновлении в _perform_update:", e)
            traceback.print_exc()

    def _update_ui_with_signal(self, explanation, rec):
        self._draw_chart()

        self.txt_expl.configure(state='normal')
        self.txt_expl.delete('1.0', tk.END)
        self.txt_expl.insert(tk.END, explanation)
        self.txt_expl.configure(state='disabled')

        self._refresh_signals_list()
        self._refresh_counters()

        try:
            df = pd.DataFrame([rec])
            if os.path.exists(SIGNALS_LOG_CSV):
                df.to_csv(SIGNALS_LOG_CSV, mode='a', header=False, index=False, encoding='utf-8')
            else:
                df.to_csv(SIGNALS_LOG_CSV, index=False, encoding='utf-8')
        except Exception as e:
            print("Ошибка записи логов сигналов:", e)

if __name__ == "__main__":
    app = TradingApp()
    app.mainloop()

