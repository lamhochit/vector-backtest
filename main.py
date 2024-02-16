import matplotlib.pyplot as plt
from backtest_utils import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')


if __name__ == '__main__':
    start_date = "2020-09-20"
    end_date = "2024-02-16"
    basket = ["ETH", "BNB", "SOL", "AVAX", "MATIC", "TRX", "XRP", "LINK", "ADA", "LTC"]
    n = 2
    lookback = 14
    fee = 0.00055

    data_df = generate_df(lookback=lookback, symbols=basket, start_date=start_date, end_date=end_date)
    backtest_df, position_df, return_df = backtest(df=data_df, n=n, factor_suffix=f'_return_{lookback}d', fee=fee)
