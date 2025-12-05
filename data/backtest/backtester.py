"""
Backtester module
Provides Backtester class which runs a trade-by-trade simulation using bar OHLCV data
and a column of predictions/signals.

Compatible return format with previous `backtest_with_execution` function:
{ 'total_return', 'buyhold_return', 'portfolio', 'equity_curve', 'trades_df', 'win_rate', 'profit_factor', 'sharpe' }

Usage:
from backtest.backtester import Backtester
bt = Backtester(df, preds_col='pred', exec_params=EXECUTION_DICT)
res = bt.run(stop_loss=0.004, take_profit=0.006, pos_mult=1.0, cash=1.0)
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# import simulate_execution from executor (should be implemented in backtest/executor.py)
try:
    from .executor import simulate_execution
except Exception:
    # fallback if package import style differs
    from backtest.executor import simulate_execution


class Backtester:
    def __init__(self, df: pd.DataFrame, preds_col: str = 'pred', exec_params: Optional[Dict] = None):
        """
        df: dataframe with columns ['open','high','low','close', ...] and a column with predictions/signals
        preds_col: name of column in df containing numeric predictions (regression forecasts). Zero/NaN means no signal.
        exec_params: dict with execution parameters (commission_pct, slippage_pct, limit_fill_prob, use_limit_orders)
        """
        self.df = df.copy().reset_index(drop=True)
        self.preds_col = preds_col
        self.exec_params = exec_params

    def _prepare_signals(self) -> None:
        df = self.df
        df['signal_raw'] = 0

        # если предсказания есть, ставим знак
        if self.preds_col in df.columns:
            mask = df[self.preds_col].notna()
            df.loc[mask, 'signal_raw'] = np.sign(df.loc[mask, self.preds_col])

        # Shift на 1 бар вперед
        df['signal'] = df['signal_raw'].shift(1)
        df['signal'] = df['signal'].fillna(0).astype(int)

        # Сохраняем обратно
        self.df = df

    def run(self, stop_loss=0.004, take_profit=0.006, pos_mult=1.0, cash=1.0):
        # подготовка сигналов
        self._prepare_signals()
        df = self.df.copy().reset_index(drop=True)
        self._prepare_signals()

        trades = []
        portfolio = cash
        equity_curve = np.ones(len(df)) * portfolio

        position_open = False
        entry_price = None
        entry_index = None
        position_direction = 0
        current_trade_idx = None

        for i in range(1, len(df)):
            price_open = df.loc[i, 'open']

            # Check exit first if position is open
            if position_open:
                if position_direction == 1:
                    hit_tp = (df.loc[i, 'high'] >= entry_price * (1 + take_profit))
                    hit_sl = (df.loc[i, 'low'] <= entry_price * (1 - stop_loss))
                else:
                    hit_tp = (df.loc[i, 'low'] <= entry_price * (1 - take_profit))
                    hit_sl = (df.loc[i, 'high'] >= entry_price * (1 + stop_loss))

                closed = False
                exit_price = None
                reason = None

                if hit_tp and not hit_sl:
                    # TP
                    if position_direction == 1:
                        exit_price = entry_price * (1 + take_profit)
                    else:
                        exit_price = entry_price * (1 - take_profit)
                    reason = 'tp'
                    closed = True
                elif hit_sl and not hit_tp:
                    # SL
                    if position_direction == 1:
                        exit_price = entry_price * (1 - stop_loss)
                    else:
                        exit_price = entry_price * (1 + stop_loss)
                    reason = 'sl'
                    closed = True
                elif hit_sl and hit_tp:
                    # both occurred in same bar -> conservative: assume stop-loss first
                    if position_direction == 1:
                        exit_price = entry_price * (1 - stop_loss)
                    else:
                        exit_price = entry_price * (1 + stop_loss)
                    reason = 'both'
                    closed = True

                if closed:
                    # apply execution model for exit and recompute effective entry for P&L calc
                    slippage = self.exec_params.get('slippage_pct') if self.exec_params else None
                    commission = self.exec_params.get('commission_pct') if self.exec_params else None

                    # If exec_params is None, simulate_execution will use its own defaults
                    # For consistency compute effective prices relative to stored entry_price
                    if position_direction == 1:
                        eff_exit = exit_price * (1 - (slippage or 0))
                        eff_entry = entry_price * (1 + (slippage or 0))
                    else:
                        eff_exit = exit_price * (1 + (slippage or 0))
                        eff_entry = entry_price * (1 - (slippage or 0))

                    # position_size stored in the trade dict when entry was created
                    position_size = trades[current_trade_idx]['position_size'] if current_trade_idx is not None else (0.05 * pos_mult)

                    gross_return = (eff_exit / eff_entry - 1) * position_direction
                    pnl = portfolio * position_size * gross_return

                    comm = portfolio * position_size * ((commission or 0) * 2)
                    net_pnl = pnl - comm
                    portfolio += net_pnl

                    # update trade record
                    trades[current_trade_idx].update({
                        'exit_index': i,
                        'exit_price': eff_exit,
                        'gross_return': gross_return,
                        'net_pnl': net_pnl,
                        'reason': reason
                    })

                    # reset position state
                    position_open = False
                    entry_price = None
                    entry_index = None
                    position_direction = 0
                    current_trade_idx = None

            # ENTRY logic
            if (not position_open) and (df.loc[i, 'signal'] != 0):
                direction = int(df.loc[i, 'signal'])
                pred_strength = abs(df.loc[i-1, self.preds_col]) if (i-1) >= 0 and self.preds_col in df.columns else 0.01

                position_size = min(0.5, 0.02 * (pred_strength * 100) * pos_mult + 0.01)
                position_size = max(0.005, position_size)

                # simulate execution for entry (could return None if limit not filled)
                eff_entry = simulate_execution(price_open, direction, exec_params=self.exec_params)
                if eff_entry is not None:
                    # open position
                    position_open = True
                    entry_price = eff_entry
                    entry_index = i
                    position_direction = direction
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': None,
                        'entry_price': entry_price,
                        'exit_price': None,
                        'direction': direction,
                        'position_size': position_size,
                        'gross_return': None,
                        'net_pnl': None,
                        'reason': None
                    })
                    current_trade_idx = len(trades) - 1

            equity_curve[i] = portfolio

        # If position still open at the end -> close at last close price
        if position_open:
            last_price = df.iloc[-1]['close']
            slippage = self.exec_params.get('slippage_pct') if self.exec_params else None
            commission = self.exec_params.get('commission_pct') if self.exec_params else None

            if position_direction == 1:
                eff_exit = last_price * (1 - (slippage or 0))
                eff_entry = entry_price
                gross_return = (eff_exit / eff_entry - 1) * position_direction
            else:
                eff_exit = last_price * (1 + (slippage or 0))
                eff_entry = entry_price
                gross_return = (eff_exit / eff_entry - 1) * position_direction

            if trades and trades[-1]['exit_index'] is None:
                position_size = trades[-1]['position_size']
                pnl = portfolio * position_size * gross_return
                comm = portfolio * position_size * (commission or 0) * 2
                net_pnl = pnl - comm
                portfolio += net_pnl
                trades[-1].update({
                    'exit_index': len(df) - 1,
                    'exit_price': eff_exit,
                    'gross_return': gross_return,
                    'net_pnl': net_pnl,
                    'reason': 'close_end'
                })

        # Metrics
        total_return = portfolio - cash
        buyhold_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * cash

        net_pnls = [t['net_pnl'] for t in trades if t['net_pnl'] is not None]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p < 0]
        win_rate = (len(wins) / len(net_pnls)) if len(net_pnls) > 0 else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if sum(losses) != 0 else float('inf')

        if len(net_pnls) > 1 and np.std(net_pnls) > 0:
            sharpe = (np.mean(net_pnls) / np.std(net_pnls)) * np.sqrt(252)
        else:
            sharpe = 0.0

        trades_df = pd.DataFrame(trades)

        result = {
            'total_return': total_return,
            'buyhold_return': buyhold_return,
            'portfolio': portfolio,
            'equity_curve': equity_curve,
            'trades_df': trades_df,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe': sharpe
        }

        return result
