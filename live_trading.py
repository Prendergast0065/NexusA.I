import os
import time
import logging
from datetime import datetime
import pandas as pd
import requests

from my_backtester_logic import calculate_rsi, calculate_atr, get_gpt_action_for_web

logger = logging.getLogger("LiveTrading")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [LiveTrading] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
LIVE_REPORT_INTERVAL_SECONDS = int(os.getenv("LIVE_REPORT_INTERVAL_SECONDS", "15"))
FLASK_STATUS_ENDPOINT = os.getenv("FLASK_STATUS_ENDPOINT")
if not FLASK_STATUS_ENDPOINT:
    logger.error(
        "FLASK_STATUS_ENDPOINT environment variable not set. Live trading will not report status."
    )
LOOKBACK_MINUTES = int(os.getenv("LOOKBACK_MINUTES", "60"))

class SimulatedTrader:
    def __init__(self, initial_usd_balance: float = 10000.0):
        self.initial_usd_balance = initial_usd_balance
        self.usdt_balance = initial_usd_balance
        self.btc_balance = 0.0
        self.trade_log = []

    def simulate_buy(self, qty_btc: float, price: float) -> None:
        cost = qty_btc * price
        if self.usdt_balance >= cost:
            self.usdt_balance -= cost
            self.btc_balance += qty_btc
            self.trade_log.append({"time": datetime.utcnow(), "action": "BUY", "qty": qty_btc, "price": price})

    def simulate_sell(self, qty_btc: float, price: float) -> None:
        if self.btc_balance >= qty_btc:
            self.btc_balance -= qty_btc
            self.usdt_balance += qty_btc * price
            self.trade_log.append({"time": datetime.utcnow(), "action": "SELL", "qty": qty_btc, "price": price})

    def get_net_pl(self, current_price: float) -> float:
        total_equity = self.usdt_balance + self.btc_balance * current_price
        return total_equity - self.initial_usd_balance

class CoinGeckoDataFetcher:
    def get_current_price(self) -> float | None:
        try:
            resp = requests.get(COINGECKO_API_URL)
            resp.raise_for_status()
            data = resp.json()
            return data.get("bitcoin", {}).get("usd")
        except requests.exceptions.RequestException as exc:
            logger.error(f"Error fetching price from CoinGecko: {exc}")
            return None

def run_live_trading():
    openai_key = os.getenv("OPENAI_API_KEY")
    user_prompt = os.getenv("LIVE_TRADING_PROMPT", "")
    initial_usd_balance = float(os.getenv("INITIAL_USD_BALANCE_LIVE", "10000"))
    trade_amount_btc = float(os.getenv("TRADE_AMOUNT_LIVE", "0.01"))
    session_id = os.getenv("LIVE_SESSION_ID", "N/A")

    if not openai_key or not user_prompt:
        logger.error("Missing required environment variables for live trading.")
        return

    fetcher = CoinGeckoDataFetcher()
    trader = SimulatedTrader(initial_usd_balance)
    last_report = 0.0

    while True:
        price = fetcher.get_current_price()
        if price is None:
            time.sleep(LIVE_REPORT_INTERVAL_SECONDS)
            continue

        df_slice = pd.DataFrame([{"open_time": datetime.now(), "open": price, "high": price, "low": price,
                                   "close": price, "volume": 0, "sma_30": price, "sma_60": price,
                                   "volatility": 0, "rsi": 50, "atr": 0}], index=[datetime.now()])

        action, confidence, reasoning, _ = get_gpt_action_for_web(
            df_slice.tail(LOOKBACK_MINUTES),
            trader.usdt_balance,
            trader.btc_balance,
            trade_amount_btc,
            price,
            user_prompt,
            openai_key,
            0,
            "gpt-4o",
        )
        logger.info(f"GPT Signal: {action} ({confidence:.1f}%) - {reasoning}")

        if action == "BUY":
            trader.simulate_buy(trade_amount_btc, price)
        elif action == "SELL":
            trader.simulate_sell(trade_amount_btc, price)

        if time.time() - last_report >= LIVE_REPORT_INTERVAL_SECONDS:
            payload = {
                "session_id": session_id,
                "signal": action,
                "current_price": price,
                "usd_balance": trader.usdt_balance,
                "btc_balance": trader.btc_balance,
                "estimated_pl": trader.get_net_pl(price),
                "reason": reasoning,
                "message": f"AI signaled {action}. Reason: {reasoning}",
            }
            if FLASK_STATUS_ENDPOINT:
                try:
                    requests.post(FLASK_STATUS_ENDPOINT, json=payload, timeout=5)
                    logger.info("Reported live status to Flask backend.")
                except Exception as exc:
                    logger.error(f"Error reporting status: {exc}")
            else:
                logger.error("FLASK_STATUS_ENDPOINT not configured, cannot report status to Flask.")
            last_report = time.time()

        time.sleep(LIVE_REPORT_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_live_trading()

