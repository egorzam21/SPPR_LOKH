import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from features.engineer import robust_feature_engineering
from features.selector import select_important_features
from models.ensemble import create_final_model, train_ensemble, ensemble_predict
from models.optimizer import optimize_hyperparams
from backtest.backtester import Backtester
from config.params import INPUT_CSV, EXECUTION, OPT_GRID, RANDOM_SEED, TEST_SIZE

def main():
    # --- Загрузка данных ---
    df_raw = pd.read_csv(INPUT_CSV, parse_dates=['time'])
    df_raw = df_raw.sort_values('time').reset_index(drop=True)

    # --- Признаки ---
    df = robust_feature_engineering(df_raw)
    exclude_cols = ['time', 'target_ret_1','target_ret_3','target_ret_5','target_ret_10',
                    'target_direction','target_conditional','target_main',
                    'ret','log_ret','tr1','tr2','tr3','true_range']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64,np.float32,np.int64,np.int32]]
    target_col = 'target_main'

    # --- Разделение на train/test ---
    split_index = int(len(df)*(1-TEST_SIZE))
    train_df = df.iloc[:split_index].copy().reset_index(drop=True)
    test_df = df.iloc[split_index:].copy().reset_index(drop=True)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # --- Отбор признаков ---
    X_train_sel, selected_features = select_important_features(X_train, y_train, k=25)
    X_test_sel = X_test[selected_features]

    # --- Масштабирование ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # --- Обучение модели ---
    models = create_final_model()
    models = train_ensemble(models, X_train_scaled, y_train)

    # --- Предсказания ---
    train_pred = ensemble_predict(models, X_train_scaled)
    test_pred = ensemble_predict(models, X_test_scaled)
    train_df['pred'] = train_pred
    test_df['pred'] = test_pred

    # --- Оптимизация hyperparams ---
    best = optimize_hyperparams(train_df, OPT_GRID, preds_col='pred', exec_params=EXECUTION)
    best_params = best['params']
    print("Лучшие параметры:", best_params)

    # --- Бэктест на полном наборе ---
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df['pred'] = full_df['pred'].fillna(0.0)

    bt = Backtester(full_df, preds_col='pred', exec_params=EXECUTION)
    res = bt.run(
        stop_loss=best_params['stop_loss'],
        take_profit=best_params['take_profit'],
        pos_mult=best_params['pos_mult'],
        cash=1.0
    )

    print("Total return:", res['total_return'])
    print("Buy&Hold return:", res['buyhold_return'])
    print("Win rate:", res['win_rate'])
    print("Profit factor:", res['profit_factor'])

    # --- Сохранение модели ---
    joblib.dump({
        'models': models,
        'scaler': scaler,
        'features': selected_features,
        'best_params': best_params
    }, 'final_model.joblib')

if __name__ == "__main__":
    main()
