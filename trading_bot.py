# trading_bot.py
import os
import time
import logging
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import re

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def sanitize_ticker(raw_ticker: str, default: str = "AAPL") -> str:
    """
    Clean and validate the raw ticker string from env / secrets.
    Returns a valid ticker (or default) and logs details for debugging.
    """
    if raw_ticker is None:
        logger.error("TICKER is None (not set). Falling back to default: %s", default)
        return default

    # show repr for debugging hidden characters (will appear in Action logs)
    logger.info("Raw TICKER repr: %r (len=%d)", raw_ticker, len(raw_ticker))

    # Strip whitespace and surrounding quotes (single or double)
    t = raw_ticker.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
        logger.info("Removed surrounding quotes from TICKER. New repr: %r (len=%d)", t, len(t))

    # remove any stray control chars
    t = re.sub(r'[\r\n\t]+', '', t).strip()

    # Basic validation: allowed characters A-Z, a-z, 0-9, dot, dash
    if not re.fullmatch(r"[A-Za-z0-9\.\-]+", t):
        logger.error("TICKER contains invalid characters after cleaning: %r. Falling back to default: %s", t, default)
        return default

    if len(t) == 0:
        logger.error("TICKER is empty after cleaning. Falling back to default: %s", default)
        return default

    logger.info("Using sanitized TICKER: %s", t)
    return t

def sanitize_timeframe(raw_tf: str, default: str = "1d") -> str:
    """
    Ensure timeframe is a non-empty string that yfinance accepts (e.g. '1d', '1h', '5m').
    Falls back to default on invalid input.
    """
    if raw_tf is None:
        logger.warning("TIMEFRAME is not set; using default '%s'.", default)
        return default
    tf = raw_tf.strip()
    # remove accidental quotes/newlines
    if (tf.startswith('"') and tf.endswith('"')) or (tf.startswith("'") and tf.endswith("'")):
        tf = tf[1:-1].strip()

    tf = tf.replace("\n", "").replace("\r", "").replace("\t", "").strip()
    if tf == "":
        logger.warning("TIMEFRAME is empty after cleaning; using default '%s'.", default)
        return default

    # very permissive validation: must contain digit then a letter (e.g. '1d', '60m')
    import re
    if not re.fullmatch(r"\d+[mhdwM]", tf):
        # still accept common values like '1d', '1h', '5m'
        logger.warning("TIMEFRAME '%s' looks unusual; falling back to default '%s'.", tf, default)
        return default

    logger.info("Using sanitized TIMEFRAME: %s", tf)
    return tf

# --- Config from environment (set these as GitHub Secrets / Actions env) ---
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
TICKER = sanitize_ticker(os.environ.get("TICKER", "AAPL"))
SHORT_WINDOW = int(os.environ.get("SHORT_WINDOW", "10"))
LONG_WINDOW = int(os.environ.get("LONG_WINDOW", "60"))
POSITION_SIZE_USD = float(os.environ.get("POSITION_SIZE_USD", "100"))  # $ per trade
TIMEFRAME = sanitize_timeframe(os.environ.get("TIMEFRAME", "1d"))
DATA_LOOKBACK_DAYS = int(os.environ.get("DATA_LOOKBACK_DAYS", "90"))

if not API_KEY or not API_SECRET:
    logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables.")
    raise SystemExit("Missing Alpaca API credentials")

# Alpaca connection
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def fetch_data(ticker: str, lookback_days: int, interval: str = "1d") -> pd.DataFrame:
    """
    Robust data fetcher with interval defaulting to '1d' if invalid/empty,
    and clearer logging for debugging.
    """
    ticker = (ticker or "").strip()
    if ticker == "":
        logger.error("TICKER is empty in fetch_data().")
        return pd.DataFrame()

    # ensure we have a sane interval
    interval = (interval or "").strip()
    if interval == "":
        logger.warning("Interval empty; forcing default interval '1d'.")
        interval = "1d"

    # yfinance expects period such as '90d' for n days
    period = f"{max(30, lookback_days)}d"
    logger.info("Calling yfinance.download(ticker=%r, period=%r, interval=%r)", ticker, period, interval)

    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                logger.warning("yfinance returned empty or None for %s on attempt %d/%d.", ticker, attempt, max_attempts)
                if attempt < max_attempts:
                    time.sleep(2 ** attempt)
                    continue
                logger.error("All %d attempts exhausted — no data available for %s.", max_attempts, ticker)
                return pd.DataFrame()
            df = df.dropna()
            if df.empty:
                logger.warning("Data exists but dropped to empty after dropna() for %s.", ticker)
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.exception("Exception while fetching data from yfinance (attempt %d/%d): %s", attempt, max_attempts, e)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                logger.error("All attempts failed due to exceptions. Returning empty DataFrame.")
                return pd.DataFrame()

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
