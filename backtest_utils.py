import pandas as pd
from binance.client import Client
import numpy as np


def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    client = Client()
    df = pd.DataFrame(client.get_historical_klines(f"{symbol}USDT", Client.KLINE_INTERVAL_1DAY, start_date, end_date))
    df = df[[0, 1, 2, 3, 4]]
    df = df.rename({0: "date", 1: f"{symbol}_open", 2: f"{symbol}_high", 3: f"{symbol}_low", 4: f"{symbol}_close"}, axis=1)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index, unit="ms")

    return df.astype(float)


def generate_df(lookback: int, symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    processed_df = pd.DataFrame()
    for symbol in symbols:
        temp_df = load_data(symbol, start_date, end_date)
        processed_df = temp_df if processed_df.empty else processed_df.join(temp_df)
        for i in range(1, lookback+1):
            processed_df[f"{symbol}_return_{i}d"] = ((processed_df[f"{symbol}_close"] -
                                                      processed_df[f"{symbol}_close"].shift(i)) /
                                                     processed_df[f"{symbol}_close"].shift(i))

    return processed_df.dropna()


def backtest(df: pd.DataFrame, n: int, factor_suffix: str, fee: float = 0.00055) -> (pd.DataFrame, pd.DataFrame, pd.Series):
    factor_cols = [col for col in df.columns if col.endswith(factor_suffix)]
    return_cols = [col for col in df.columns if col.endswith('_return_1d')]
    result_cols = [col.replace('_return_1d', '') for col in df.columns if '_return_1d' in col]

    partitions = np.partition(df[factor_cols].values, [-n, n])
    mask_smallest = df[factor_cols].values <= partitions[:, [n - 1]]
    mask_largest = df[factor_cols].values >= partitions[:, [-n]]
    df[factor_cols] = np.where(mask_smallest, -1, np.where(mask_largest, 1, 0))

    n_net = abs(df[factor_cols]).sum(axis=1).mean()  # Number of open positions per day
    fee_df = 1 - (abs(df[factor_cols].diff()).sum(axis=1) / 8).shift(1).iloc[1:] * fee
    position_df = df[factor_cols].shift(1).iloc[1:]
    return_df = df[return_cols].iloc[1:] / n_net  # Divide the portfolio by number of positions
    final_df = pd.DataFrame(position_df.values * return_df.values, columns=result_cols)
    final_df.index = position_df.index

    return_sr = (final_df.sum(axis=1) + 1).cumprod() * fee_df

    return final_df, df[factor_cols], return_sr