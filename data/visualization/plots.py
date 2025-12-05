import matplotlib.pyplot as plt
from config.params import STRAT_OUT

def plot_final_results(df, equity, trades):
    plt.figure(figsize=(14, 10))

    plt.subplot(2,1,1)
    plt.plot(df['time'], df['close'], label='close')
    plt.title('Price + Trades')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(df['time'], equity, linewidth=2)
    plt.title('Equity curve')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(STRAT_OUT, dpi=150)
    plt.close()
