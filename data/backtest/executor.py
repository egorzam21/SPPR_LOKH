import numpy as np
from config.params import EXECUTION

def simulate_execution(price, direction, exec_params=None):
    if exec_params is None:
        exec_params = EXECUTION

    slippage = exec_params['slippage_pct']
    limit_fill_prob = exec_params['limit_fill_prob']
    use_limit = exec_params['use_limit_orders']

    filled = True
    if use_limit:
        filled = np.random.rand() < limit_fill_prob

    if not filled:
        return None

    if direction == 1:
        price_eff = price * (1 + slippage)
    else:
        price_eff = price * (1 - slippage)

    return price_eff
