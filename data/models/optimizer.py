import itertools
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from backtest.backtester import Backtester

def optimize_hyperparams(df, grid, preds_col='pred', exec_params=None):

    keys = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    best = None
    tscv = TimeSeriesSplit(n_splits=3)

    for combo in combos:
        params = dict(zip(keys, combo))
        scores = []

        idx = np.arange(len(df))
        for tr_idx, val_idx in tscv.split(idx):
            train_df = df.iloc[tr_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            bt = Backtester(val_df, preds_col=preds_col, exec_params=exec_params)
            res = bt.run(
                stop_loss=params['stop_loss'],
                take_profit=params['take_profit'],
                pos_mult=params['pos_mult'],
                cash=1.0
            )
            scores.append(res['total_return'])

        avg_score = np.mean(scores)
        if best is None or avg_score > best['score']:
            best = {'params': params, 'score': avg_score}

    return best

