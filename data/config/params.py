INPUT_CSV = "data/lukoil_ohlc_clean.csv"
MODEL_OUT = "final_model_updated.joblib"
PRED_OUT = "final_predictions_updated.csv"
STRAT_OUT = "final_strategy_results_updated.png"

RANDOM_SEED = 42
TEST_SIZE = 0.2
VERBOSE = True

EXECUTION = {
    'commission_pct': 0.0005,
    'slippage_pct': 0.0007,
    'limit_fill_prob': 0.85,
    'use_limit_orders': True
}

OPT_GRID = {
    'stop_loss': [0.002, 0.004, 0.006],
    'take_profit': [0.002, 0.006, 0.01],
    'pos_mult': [0.6, 1.0, 1.6]
}
