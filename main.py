import os
import sys
import time
import random
import numpy as np
import pandas as pd
import akshare as ak
import pandas_ta as ta
from loguru import logger
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class FutureBot:
    def __init__(self) -> None:
        self.m_symbols = []
        self.m_macd_config = {"fast": 12, "slow": 26, "signal": 9}
        self.m_exchanges = [
            "dce",
            "shfe",
            "czce",
            "gfex",
        ]  # 大商所, 上期所, 郑商所，广期所

    def get_symbols(self):
        with HiddenPrints():
            for exchange in self.m_exchanges:
                self.m_symbols += ak.match_main_contract(symbol=exchange).split(",")

    def __clear_log():
        if os.path.exists("result.log"):
            with open("result.log", "r+") as f:
                f.seek(0)
                f.truncate()

    @staticmethod
    def __dkx_cross_strategy(symbols: list, period: str, macd_config: dict):
        for _, symbol in enumerate(symbols):
            try:
                logger.info("check {0}".format(symbol))
                time.sleep(1)

                df = pd.DataFrame()
                if period in ["1", "5", "15", "30", "60"]:
                    df = ak.futures_zh_minute_sina(symbol, period)
                else:
                    df = ak.futures_zh_daily_sina(symbol)

                if df.shape[0] < 100:
                    continue

                if period == "3d":
                    df = df.drop(columns=["datetime", "volume", "hold"])

                    # transform to 3d period
                    blocks = -1
                    if df.shape[0] % 3 == 0:
                        blocks = df.shape[0] / 3
                    else:
                        blocks = df.shape[0] / 3 + 1
                    split_dfs = np.array_split(df.copy(), blocks)
                    df = pd.DataFrame(columns=["open", "close", "high", "low"])
                    for split_df in split_dfs:
                        df.loc[len(df)] = [
                            split_df[:1]["open"].to_list()[0],
                            split_df[-1:]["close"].to_list()[0],
                            split_df["high"].max(),
                            split_df["low"].min(),
                        ]

                df = df.iloc[::-1]
                df.reset_index(drop=True, inplace=True)

                # calculate dkx
                dkx = np.ones(df.shape[0])
                for idx in range(0, df.shape[0]):
                    sum = 0
                    count = 0
                    if df.iloc[idx : idx + 20].shape[0] == 20:
                        for _, row in df.iloc[idx : idx + 20].iterrows():
                            sum += (20 - count) * (
                                (
                                    3 * row["close"]
                                    + row["low"]
                                    + row["open"]
                                    + row["high"]
                                )
                                / 6
                            )
                            count += 1
                    sum /= 210
                    dkx[idx] = sum

                df["dkx"] = dkx.tolist()

                # calculate dkx_ma
                dkx_sma = np.ones(df.shape[0])
                for idx in range(0, df.shape[0]):
                    sum = 0
                    if df.iloc[idx : idx + 10].shape[0] == 10:
                        for _, row in df.iloc[idx : idx + 10].iterrows():
                            sum += row["dkx"]
                    sum /= 10
                    dkx_sma[idx] = sum
                df["dkx_sma"] = dkx_sma.tolist()

                df = df.iloc[::-1]
                df.reset_index(drop=True, inplace=True)

                # check dkx crossing up dkx_sma
                if (
                    df.iloc[-2]["dkx"] < df.iloc[-2]["dkx_sma"]
                    and df.iloc[-1]["dkx"] > df.iloc[-1]["dkx_sma"]
                ):
                    # check macd and signal
                    macd_df = ta.macd(
                        df["close"],
                        macd_config["fast"],
                        macd_config["slow"],
                        macd_config["signal"],
                    )
                    if (
                        macd_df.iloc[-1]["MACD_12_26_9"] > 0
                        and macd_df.iloc[-1]["MACDs_12_26_9"] > 0
                    ) and (
                        macd_df.iloc[-1]["MACD_12_26_9"]
                        > macd_df.iloc[-1]["MACDs_12_26_9"]
                    ):
                        logger.success("{0} Cross Up !".format(symbol))
            except Exception as error:
                logger.error("process {0} error: {1}!".format(symbol, error))

    def dkx_cross_strategy(self, period: str):
        FutureBot.__clear_log()
        logger.add("result.log")
        logger.info("start checking with period: {0}...".format(period))
        max_workers = min(32, os.cpu_count() + 4)
        split_symbols = [
            self.m_symbols[i : i + int(len(self.m_symbols) / max_workers)]
            for i in range(
                0, len(self.m_symbols), int(len(self.m_symbols) / max_workers)
            )
        ]
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="stock_bot_"
        ) as pool:
            all_task = [
                pool.submit(
                    FutureBot.__dkx_cross_strategy,
                    split_symbols[i],
                    period,
                    self.m_macd_config,
                )
                for i in range(0, len(split_symbols))
            ]
            wait(all_task, return_when=ALL_COMPLETED)
            logger.info("check finished")


if __name__ == "__main__":
    future_bot = FutureBot()
    future_bot.get_symbols()
    future_bot.dkx_cross_strategy(
        period="60"
    )  # choice of {"1": "1分钟", "5": "5分钟", "15": "15分钟", "30": "30分钟", "60": "60分钟", '1d': "1日", '3d': '3日'}
