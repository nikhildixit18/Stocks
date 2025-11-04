# GitHub Actions Trading Bot (Alpaca)

## Files
- `trading_bot.py` — main bot script
- `requirements.txt`
- `.github/workflows/run-bot.yml` — GitHub Actions workflow

## Setup Steps
1. Create a GitHub repo and push these files.
2. Add your Alpaca API keys as secrets in the repo:
   - ALPACA_API_KEY
   - ALPACA_SECRET_KEY
   - (Optional) ALPACA_BASE_URL, TICKER, SHORT_WINDOW, LONG_WINDOW, etc.
3. Trigger the workflow manually or let it run on schedule.
4. Check logs in Actions tab.
