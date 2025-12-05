
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

try:
    from .executor import simulate_execution
except Exception:
   
    from backtest.executor import simulate_execution


class Backtester:
    def __init__(self, df: pd.DataFrame, preds_col: str = 'pred', exec_params: Optional[Dict] = None):
        self.df = df.copy().reset_index(drop=True)
        self.preds_col = preds_col
        self.exec_params = exec_params

    def _prepare_signals(self) -> None:
        df = self.df
        df['signal_raw'] = 0

        if self.preds_col in df.columns:
            mask = df[self.preds_col].notna()
            df.loc[mask, 'signal_raw'] = np.sign(df.loc[mask, self.preds_col])

        df['signal'] = df['signal_raw'].shift(1)
        df['signal'] = df['signal'].fillna(0).astype(int)

        self.df = df

    def run(self, stop_loss=0.004, take_profit=0.006, pos_mult=1.0, cash=1.0):
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
                    if position_direction == 1:
                        exit_price = entry_price * (1 + take_profit)
                    else:
                        exit_price = entry_price * (1 - take_profit)
                    reason = 'tp'
                    closed = True
                elif hit_sl and not hit_tp:
                    if position_direction == 1:
                        exit_price = entry_price * (1 - stop_loss)
                    else:
                        exit_price = entry_price * (1 + stop_loss)
                    reason = 'sl'
                    closed = True
                elif hit_sl and hit_tp:
                    if position_direction == 1:
                        exit_price = entry_price * (1 - stop_loss)
                    else:
                        exit_price = entry_price * (1 + stop_loss)
                    reason = 'both'
                    closed = True

                if closed:
                    slippage = self.exec_params.get('slippage_pct') if self.exec_params else None
                    commission = self.exec_params.get('commission_pct') if self.exec_params else None

                    if position_direction == 1:
                        eff_exit = exit_price * (1 - (slippage or 0))
                        eff_entry = entry_price * (1 + (slippage or 0))
                    else:
                        eff_exit = exit_price * (1 + (slippage or 0))
                        eff_entry = entry_price * (1 - (slippage or 0))


                    position_size = trades[current_trade_idx]['position_size'] if current_trade_idx is not None else (0.05 * pos_mult)

                    gross_return = (eff_exit / eff_entry - 1) * position_direction
                    pnl = portfolio * position_size * gross_return

                    comm = portfolio * position_size * ((commission or 0) * 2)
                    net_pnl = pnl - comm
                    portfolio += net_pnl

                    trades[current_trade_idx].update({
                        'exit_index': i,
                        'exit_price': eff_exit,
                        'gross_return': gross_return,
                        'net_pnl': net_pnl,
                        'reason': reason
                    })

                    position_open = False
                    entry_price = None
                    entry_index = None
                    position_direction = 0
                    current_trade_idx = None

            if (not position_open) and (df.loc[i, 'signal'] != 0):
                direction = int(df.loc[i, 'signal'])
                pred_strength = abs(df.loc[i-1, self.preds_col]) if (i-1) >= 0 and self.preds_col in df.columns else 0.01

                position_size = min(0.5, 0.02 * (pred_strength * 100) * pos_mult + 0.01)
                position_size = max(0.005, position_size)

                eff_entry = simulate_execution(price_open, direction, exec_params=self.exec_params)
                if eff_entry is not None:

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

