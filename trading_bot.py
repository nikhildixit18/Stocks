# trading_bot.py
import os
import time
import logging
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Config from environment (set these as GitHub Secrets / Actions env) ---
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
TICKER = os.environ.get("TICKER", "AAPL")
SHORT_WINDOW = int(os.environ.get("SHORT_WINDOW", "10"))
LONG_WINDOW = int(os.environ.get("LONG_WINDOW", "60"))
POSITION_SIZE_USD = float(os.environ.get("POSITION_SIZE_USD", "100"))  # $ per trade
TIMEFRAME = os.environ.get("TIMEFRAME", "1d")  # used by yfinance: '1d', '1h', etc.
DATA_LOOKBACK_DAYS = int(os.environ.get("DATA_LOOKBACK_DAYS", "90"))

if not API_KEY or not API_SECRET:
    logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables.")
    raise SystemExit("Missing Alpaca API credentials")

# Alpaca connection
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def fetch_data(ticker: str, lookback_days: int, interval: str = "1d") -> pd.DataFrame:
    period = f"{max(30, lookback_days)}d"
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df.dropna()
    return df

def calculate_signals(df: pd.DataFrame, short_w: int, long_w: int) -> pd.DataFrame:
    df = df.copy()
    df["short_ma"] = df["Close"].rolling(window=short_w).mean()
    df["long_ma"] = df["Close"].rolling(window=long_w).mean()
    df.dropna(inplace=True)
    return df

def get_latest_signal(df: pd.DataFrame) -> str:
    """
    Determines latest crossover signal using scalar values (safe vs ambiguous Series).
    Returns one of: "buy", "sell", "hold".
    """
    try:
        # require at least two rows to detect a crossover
        if len(df) < 2:
            return "hold"

        # explicitly get scalar floats (safe even if column indexing is weird)
        prev_short = float(df["short_ma"].iloc[-2])
        prev_long  = float(df["long_ma"].iloc[-2])
        curr_short = float(df["short_ma"].iloc[-1])
        curr_long  = float(df["long_ma"].iloc[-1])

        # guard against NaN values
        if any(np.isnan([prev_short, prev_long, curr_short, curr_long])):
            logger.warning("One or more MA values are NaN — returning hold")
            return "hold"

        # crossover logic (scalar comparisons)
        if prev_short <= prev_long and curr_short > curr_long:
            return "buy"
        if prev_short >= prev_long and curr_short < curr_long:
            return "sell"
        return "hold"

    except Exception as e:
        # defensive: log the types/contents if something unexpected happened
        logger.exception("Error computing latest signal — defaulting to hold")
        try:
            logger.debug("df.tail(5):\n%s", df.tail(5).to_string())
        except Exception:
            pass
        return "hold"

def get_cash_balance() -> float:
    acct = api.get_account()
    return float(acct.cash)

def get_position_qty(symbol: str) -> float:
    try:
        pos = api.get_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0

def place_market_order(symbol: str, qty: float, side: str):
    if qty <= 0:
        logger.info("Quantity <= 0, nothing to place.")
        return None
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc"
        )
        logger.info(f"Submitted {side} order for {qty} {symbol}. order_id={order.id}")
        return order
    except Exception as e:
        logger.exception(f"Order failed: {e}")
        return None

def size_position_by_cash(symbol: str, usd_amount: float) -> int:
    # Use latest price to estimate shares to buy (integer shares)
    ticker_data = yf.download(symbol, period="2d", interval="1d", progress=False)
    if ticker_data.empty:
        logger.error("No price data to size position.")
        return 0
    price = ticker_data["Close"].iloc[-1]
    qty = int(np.floor(usd_amount / price))
    return max(qty, 0)

def cancel_open_orders_for(symbol: str):
    try:
        open_orders = api.list_orders(status='open', symbols=[symbol])
        for o in open_orders:
            api.cancel_order(o.id)
            logger.info(f"Cancelled open order {o.id} for {symbol}")
    except Exception:
        logger.exception("Failed while cancelling open orders.")

def main():
    logger.info("=== Starting trading bot run ===")
    try:
        df = fetch_data(TICKER, DATA_LOOKBACK_DAYS, interval=TIMEFRAME)
        if df.empty:
            logger.error("No market data retrieved — aborting run.")
            return

        df = calculate_signals(df, SHORT_WINDOW, LONG_WINDOW)
        signal = get_latest_signal(df)
        logger.info(f"Latest signal for {TICKER}: {signal}")

        # Cancel any open orders first
        cancel_open_orders_for(TICKER)

        position_qty = get_position_qty(TICKER)
        logger.info(f"Current position qty: {position_qty}")

        if signal == "buy" and position_qty == 0:
            # position sizing by USD amount
            qty = size_position_by_cash(TICKER, POSITION_SIZE_USD)
            if qty > 0:
                place_market_order(TICKER, qty, "buy")
            else:
                logger.info("Calculated quantity is 0 (POSITION_SIZE_USD too small). No buy placed.")
        elif signal == "sell" and position_qty > 0:
            # sell all existing shares for the symbol
            place_market_order(TICKER, int(position_qty), "sell")
        else:
            logger.info("No trade executed this run.")

    except Exception:
        logger.exception("Unexpected error in main loop.")
    finally:
        logger.info("=== Run finished ===")

if __name__ == "__main__":
    main()