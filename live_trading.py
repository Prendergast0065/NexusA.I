import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client

from my_backtester_logic import (
    calculate_rsi,
    calculate_atr,
    get_gpt_action_for_web,
)

logger = logging.getLogger("LiveTrading")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [LiveTrading] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


SYMBOL = os.getenv("BINANCE_SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("BINANCE_INTERVAL", "1m")
LOOKBACK_MINUTES = int(os.getenv("LOOKBACK_MINUTES", "60"))
TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT_BTC", "0.001"))
USER_PROMPT = os.getenv("LIVE_TRADING_PROMPT", "")


class BinanceTrader:
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)

    def get_recent_data(self):
        klines = self.client.get_klines(
            symbol=SYMBOL,
            interval=INTERVAL,
            limit=LOOKBACK_MINUTES,
        )
        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "num_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        df = df.astype(
            {
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
            }
        )
        df["sma_30"] = df["close"].rolling(window=30, min_periods=1).mean()
        df["sma_60"] = df["close"].rolling(window=60, min_periods=1).mean()
        df["volatility"] = df["close"].rolling(window=30, min_periods=1).std()
        df["rsi"] = calculate_rsi(df["close"])
        df["atr"] = calculate_atr(df["high"], df["low"], df["close"])
        df.dropna(inplace=True)
        return df

    def get_balances(self):
        btc_balance = float(self.client.get_asset_balance(asset="BTC")["free"])
        usdt_balance = float(self.client.get_asset_balance(asset="USDT")["free"])
        return btc_balance, usdt_balance

    def market_buy(self, qty):
        logger.info(f"Placing BUY order for {qty} {SYMBOL}")
        self.client.create_order(
            symbol=SYMBOL,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=qty,
        )

    def market_sell(self, qty):
        logger.info(f"Placing SELL order for {qty} {SYMBOL}")
        self.client.create_order(
            symbol=SYMBOL,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=qty,
        )


def run_live_trading():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not api_key or not api_secret or not openai_key or not USER_PROMPT:
        logger.error("Missing required environment variables for live trading.")
        return

    trader = BinanceTrader(api_key, api_secret)

    while True:
        try:
            df = trader.get_recent_data()
            btc_balance, usdt_balance = trader.get_balances()
            entry_price = df["close"].iloc[-1]

            action, confidence, reasoning, _ = get_gpt_action_for_web(
                df.tail(10),
                usdt_balance,
                btc_balance,
                TRADE_AMOUNT,
                entry_price,
                USER_PROMPT,
                openai_key,
                0,
                "gpt-4o",
            )
            logger.info(f"GPT action: {action} ({confidence:.1f}%) - {reasoning}")

            if action == "BUY" and usdt_balance >= TRADE_AMOUNT * entry_price:
                trader.market_buy(TRADE_AMOUNT)
            elif action == "SELL" and btc_balance >= TRADE_AMOUNT:
                trader.market_sell(TRADE_AMOUNT)
        except Exception as e:
            logger.error(f"Live trading loop error: {e}")

        logger.info("Sleeping for 30 minutes...")
        time.sleep(30 * 60)


if __name__ == "__main__":
    run_live_trading()
