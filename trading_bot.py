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

from datetime import datetime, timedelta

# map yfinance-style intervals to a reasonable Alpaca timeframe string
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

def _alpaca_bars_to_df(bars):
    """
    Convert Alpaca 'bars' (iterable of bar objects/dicts) into a pandas DataFrame
    with columns: Open, High, Low, Close, Volume and index as timestamp (UTC).
    Works with both v2 Bar objects and older bar dicts.
    """
    if bars is None:
        return pd.DataFrame()
    rows = []
    for b in bars:
        # support both object-like and dict-like bars
        try:
            t = getattr(b, "t", None) or b.get("t")  # time
            o = getattr(b, "o", None) or b.get("o")
            h = getattr(b, "h", None) or b.get("h")
            l = getattr(b, "l", None) or b.get("l")
            c = getattr(b, "c", None) or b.get("c")
            v = getattr(b, "v", None) or b.get("v")
        except Exception:
            # last-resort attempt: attempt indexing
            try:
                t, o, h, l, c, v = b
            except Exception:
                continue
        # normalize timestamp: alpaca may return ISO or datetime
        if isinstance(t, str):
            try:
                t = pd.to_datetime(t)
            except Exception:
                pass
        rows.append({"Datetime": pd.to_datetime(t), "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("Datetime").sort_index()
    # yfinance returns columns labelled like 'Open' 'Close' etc; keep consistent
    return df

def fetch_data(ticker: str, lookback_days: int, interval: str = "1d") -> pd.DataFrame:
    """
    Try yfinance, then fall back to Alpaca market data if yfinance returns empty.
    Returns an empty DataFrame if both methods fail.
    """
    ticker = (ticker or "").strip()
    if ticker == "":
        logger.error("TICKER is empty in fetch_data().")
        return pd.DataFrame()

    # sanitize interval to a sensible default
    interval = (interval or "").strip() or "1d"
    period = f"{max(30, lookback_days)}d"
    logger.info("fetch_data: yfinance.download(ticker=%r, period=%r, interval=%r)", ticker, period, interval)

    # 1) Try yfinance first (same logic as before)
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
                    # proceed to Alpaca fallback
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

    # 2) Alpaca fallback (simpler, use `limit` to avoid start/end date format issues)
    try:
        logger.info("Attempting Alpaca fallback for %s (limit-based call)", ticker)
        alp_tf = _ALPACA_INTERVAL_MAP.get(interval, None)
        if alp_tf is None:
            alp_tf = _ALPACA_INTERVAL_MAP.get(interval.lower(), "1Day")

        # Use 'limit' instead of start/end to avoid RFC3339 parsing problems
        # Choose a sensible bar count: lookback_days (daily) or convert days->bars for intraday
        if interval.endswith("d") or interval.lower() in ("1d", "1w"):
            limit = max(lookback_days, 30)
        else:
            # for intraday, choose more bars to cover lookback_days worth of trading minutes
            # e.g. for '1m' take 390 minutes per day * lookback_days (cap it reasonably)
            if interval.endswith("m"):
                minutes_per_day = 390
                try:
                    mult = int(interval[:-1])
                except Exception:
                    mult = 1
                approximate_bars = (minutes_per_day // max(1, mult)) * lookback_days
                limit = min(max(approximate_bars, 100), 5000)  # avoid huge limits
            elif interval.endswith("h") or interval.lower() == "1h":
                bars_per_day = 7  # rough trading hours
                limit = min(max(bars_per_day * lookback_days, 30), 2000)
            else:
                limit = max(lookback_days, 30)

        logger.info("Alpaca fallback: get_bars(symbol=%r, timeframe=%r, limit=%d)", ticker, alp_tf, limit)

        # Try modern get_bars signature: (symbol, timeframe, limit=...)
        try:
            bars = api.get_bars(ticker, alp_tf, limit=limit)
            df_bars = _alpaca_bars_to_df(bars)
            if not df_bars.empty:
                logger.info("Alpaca get_bars succeeded for %s (%d rows).", ticker, len(df_bars))
                return df_bars
            logger.warning("Alpaca get_bars returned empty for %s.", ticker)
        except Exception as e:
            logger.debug("Alpaca get_bars exception: %s", e)

        # Fallback: try get_barset (older API)
        try:
            barset = api.get_barset(ticker, interval, limit=limit)
            # barset[ticker] is a list of bar objects
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