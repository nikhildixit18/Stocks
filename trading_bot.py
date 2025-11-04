# trading_bot.py
import os
import time
import logging
import re
from datetime import datetime, timedelta, timezone

import yfinance as yf
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Helpers: sanitize inputs ---
def sanitize_ticker(raw_ticker: str, default: str = "AAPL") -> str:
    if raw_ticker is None:
        logger.error("TICKER is None (not set). Falling back to default: %s", default)
        return default

    logger.info("Raw TICKER repr: %r (len=%d)", raw_ticker, len(raw_ticker))
    t = raw_ticker.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
        logger.info("Removed surrounding quotes from TICKER. New repr: %r (len=%d)", t, len(t))
    t = re.sub(r'[\r\n\t]+', '', t).strip()
    if not re.fullmatch(r"[A-Za-z0-9\.\-]+", t):
        logger.error("TICKER contains invalid characters after cleaning: %r. Falling back to default: %s", t, default)
        return default
    if len(t) == 0:
        logger.error("TICKER is empty after cleaning. Falling back to default: %s", default)
        return default
    logger.info("Using sanitized TICKER: %s", t)
    return t

def sanitize_timeframe(raw_tf: str, default: str = "1d") -> str:
    if raw_tf is None:
        logger.warning("TIMEFRAME is not set; using default '%s'.", default)
        return default
    tf = raw_tf.strip()
    if (tf.startswith('"') and tf.endswith('"')) or (tf.startswith("'") and tf.endswith("'")):
        tf = tf[1:-1].strip()
    tf = tf.replace("\n", "").replace("\r", "").replace("\t", "").strip()
    if tf == "":
        logger.warning("TIMEFRAME is empty after cleaning; using default '%s'.", default)
        return default
    # Accept typical patterns like '1d', '1h', '5m', '60m'
    if not re.fullmatch(r"\d+[mhdwM]", tf):
        logger.warning("TIMEFRAME '%s' looks unusual; falling back to default '%s'.", tf, default)
        return default
    logger.info("Using sanitized TIMEFRAME: %s", tf)
    return tf

# --- Config from environment ---
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
TICKER = sanitize_ticker(os.environ.get("TICKER", "AAPL"))
SHORT_WINDOW = int(os.environ.get("SHORT_WINDOW", "10"))
LONG_WINDOW = int(os.environ.get("LONG_WINDOW", "60"))
POSITION_SIZE_USD = float(os.environ.get("POSITION_SIZE_USD", "100"))
TIMEFRAME = sanitize_timeframe(os.environ.get("TIMEFRAME", "1d"))
DATA_LOOKBACK_DAYS = int(os.environ.get("DATA_LOOKBACK_DAYS", "90"))

if not API_KEY or not API_SECRET:
    logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables.")
    # Do not raise here if you prefer graceful no-op; we'll stop later if missing.
    # raise SystemExit("Missing Alpaca API credentials")

# Alpaca client (constructed only if keys provided)
api = None
try:
    if API_KEY and API_SECRET:
        api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        logger.info("Alpaca REST initialized (base_url=%s)", BASE_URL)
    else:
        logger.warning("Alpaca credentials missing; Alpaca fallback will be unavailable.")
except Exception as e:
    logger.exception("Failed to initialize Alpaca REST client: %s", e)
    api = None

# --- Alpaca helper: normalize bars to DataFrame ---
def _alpaca_bars_to_df(bars):
    if bars is None:
        return pd.DataFrame()
    rows = []
    for b in bars:
        try:
            # support object-like (attributes) and dict-like
            t = getattr(b, "t", None) or (b.get("t") if isinstance(b, dict) else None)
            o = getattr(b, "o", None) or (b.get("o") if isinstance(b, dict) else None)
            h = getattr(b, "h", None) or (b.get("h") if isinstance(b, dict) else None)
            l = getattr(b, "l", None) or (b.get("l") if isinstance(b, dict) else None)
            c = getattr(b, "c", None) or (b.get("c") if isinstance(b, dict) else None)
            v = getattr(b, "v", None) or (b.get("v") if isinstance(b, dict) else None)
        except Exception:
            try:
                t, o, h, l, c, v = b
            except Exception:
                continue
        try:
            dt = pd.to_datetime(t)
        except Exception:
            dt = t
        rows.append({"Datetime": dt, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("Datetime").sort_index()
    return df

# --- Interval mapping for Alpaca fallback ---
_ALPACA_INTERVAL_MAP = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "60m": "1Hour",
    "1h": "1Hour",
    "1d": "1Day",
    "1w": "1Week",
}

# --- Robust fetch_data with yfinance + Alpaca fallback (limit-based) ---
def fetch_data(ticker: str, lookback_days: int, interval: str = "1d") -> pd.DataFrame:
    ticker = (ticker or "").strip()
    if ticker == "":
        logger.error("TICKER is empty in fetch_data().")
        return pd.DataFrame()

    interval = (interval or "").strip() or "1d"
    period = f"{max(30, lookback_days)}d"
    logger.info("fetch_data: yfinance.download(ticker=%r, period=%r, interval=%r)", ticker, period, interval)

    # Try yfinance
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                logger.warning("yfinance returned empty or None for %s (attempt %d/%d).", ticker, attempt, max_attempts)
                if attempt < max_attempts:
                    time.sleep(2 ** attempt)
                    continue
                logger.info("yfinance exhausted attempts, will try Alpaca fallback.")
                df = pd.DataFrame()
            else:
                df = df.dropna()
                if df.empty:
                    logger.warning("Data exists but dropped to empty after dropna() for %s.", ticker)
                    df = pd.DataFrame()
                else:
                    logger.info("yfinance succeeded for %s (%d rows).", ticker, len(df))
                    return df
        except Exception as e:
            logger.exception("yfinance exception for %s (attempt %d/%d): %s", ticker, attempt, max_attempts, e)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                logger.info("yfinance final exception, will try Alpaca fallback.")

    # Alpaca fallback (limit-based)
    if api is None:
        logger.error("Alpaca client not available; cannot use fallback.")
        return pd.DataFrame()

    try:
        logger.info("Attempting Alpaca fallback for %s (limit-based call)", ticker)
        alp_tf = _ALPACA_INTERVAL_MAP.get(interval, None)
        if alp_tf is None:
            alp_tf = _ALPACA_INTERVAL_MAP.get(interval.lower(), "1Day")

        # Determine limit (number of bars) conservatively
        if interval.endswith("d") or interval.lower() in ("1d", "1w"):
            limit = max(lookback_days, 30)
        else:
            if interval.endswith("m"):
                try:
                    mult = int(interval[:-1])
                except Exception:
                    mult = 1
                minutes_per_day = 390
                approximate_bars = (minutes_per_day // max(1, mult)) * lookback_days
                limit = min(max(approximate_bars, 100), 5000)
            elif interval.endswith("h") or interval.lower() == "1h":
                bars_per_day = 7
                limit = min(max(bars_per_day * lookback_days, 30), 2000)
            else:
                limit = max(lookback_days, 30)

        logger.info("Alpaca fallback: get_bars(symbol=%r, timeframe=%r, limit=%d)", ticker, alp_tf, limit)

        # Try modern get_bars signature
        try:
            bars = api.get_bars(ticker, alp_tf, limit=limit)
            df_bars = _alpaca_bars_to_df(bars)
            if not df_bars.empty:
                logger.info("Alpaca get_bars succeeded for %s (%d rows).", ticker, len(df_bars))
                return df_bars
            logger.warning("Alpaca get_bars returned empty for %s.", ticker)
        except Exception as e:
            logger.debug("Alpaca get_bars exception: %s", e)

        # Try older get_barset as fallback
        try:
            barset = api.get_barset(ticker, interval, limit=limit)
            bars = barset.get(ticker) if isinstance(barset, dict) else getattr(barset, ticker, barset)
            df_bars = _alpaca_bars_to_df(bars)
            if not df_bars.empty:
                logger.info("Alpaca get_barset succeeded for %s (%d rows).", ticker, len(df_bars))
                return df_bars
            logger.warning("Alpaca get_barset returned empty for %s.", ticker)
        except Exception as e:
            logger.debug("Alpaca get_barset failed: %s", e)

        logger.error("Alpaca fallback did not return data for %s.", ticker)
    except Exception as e:
        logger.exception("Unexpected exception during Alpaca fallback for %s: %s", ticker, e)

    logger.error("No market data available from yfinance or Alpaca for %s.", ticker)
    return pd.DataFrame()

# --- Strategy helpers ---
def calculate_signals(df: pd.DataFrame, short_w: int, long_w: int) -> pd.DataFrame:
    df = df.copy()
    df["short_ma"] = df["Close"].rolling(window=short_w).mean()
    df["long_ma"] = df["Close"].rolling(window=long_w).mean()
    df.dropna(inplace=True)
    return df

def get_latest_signal(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "hold"
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    if prev["short_ma"] <= prev["long_ma"] and curr["short_ma"] > curr["long_ma"]:
        return "buy"
    if prev["short_ma"] >= prev["long_ma"] and curr["short_ma"] < curr["long_ma"]:
        return "sell"
    return "hold"

# --- Alpaca trading helpers ---
def get_cash_balance() -> float:
    try:
        acct = api.get_account()
        return float(acct.cash)
    except Exception:
        logger.exception("Failed to get account cash balance.")
        return 0.0

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
        logger.info("Submitted %s order for %s %s. order_id=%s", side, qty, symbol, getattr(order, "id", "n/a"))
        return order
    except Exception:
        logger.exception("Order failed.")
        return None

def size_position_by_cash(symbol: str, usd_amount: float) -> int:
    try:
        ticker_data = yf.download(symbol, period="2d", interval="1d", progress=False)
        if ticker_data.empty:
            logger.error("No recent price data to size position for %s.", symbol)
            return 0
        price = ticker_data["Close"].iloc[-1]
        qty = int(np.floor(usd_amount / price))
        return max(qty, 0)
    except Exception:
        logger.exception("Failed to size position by cash.")
        return 0

def cancel_open_orders_for(symbol: str):
    try:
        open_orders = api.list_orders(status='open', symbols=[symbol])
        for o in open_orders:
            api.cancel_order(o.id)
            logger.info("Cancelled open order %s for %s", o.id, symbol)
    except Exception:
        logger.exception("Failed while cancelling open orders.")

# --- Main ---
def main():
    logger.info("=== Starting trading bot run ===")
    try:
        df = fetch_data(TICKER, DATA_LOOKBACK_DAYS, interval=TIMEFRAME)
        if df.empty:
            logger.error("No market data retrieved — aborting run.")
            return

        df = calculate_signals(df, SHORT_WINDOW, LONG_WINDOW)
        signal = get_latest_signal(df)
        logger.info("Latest signal for %s: %s", TICKER, signal)

        # Cancel any open orders first
        if api:
            cancel_open_orders_for(TICKER)
        else:
            logger.warning("Alpaca API client not available; skipping order cancellation.")

        position_qty = get_position_qty(TICKER) if api else 0
        logger.info("Current position qty: %s", position_qty)

        if signal == "buy" and (position_qty == 0 or position_qty == 0.0):
            qty = size_position_by_cash(TICKER, POSITION_SIZE_USD)
            if qty > 0 and api:
                place_market_order(TICKER, qty, "buy")
            else:
                logger.info("Calculated quantity is 0 or Alpaca not configured. No buy placed.")
        elif signal == "sell" and position_qty > 0:
            if api:
                place_market_order(TICKER, int(position_qty), "sell")
            else:
                logger.info("Alpaca not configured; would have sold %s shares.", position_qty)
        else:
            logger.info("No trade executed this run.")
    except Exception:
        logger.exception("Unexpected error in main loop.")
    finally:
        logger.info("=== Run finished ===")

if __name__ == "__main__":
    main()