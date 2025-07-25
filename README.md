# NexusA.I

This Flask application provides an interface for backtesting trading strategies and integrates Stripe for payments.

## Setup

1. Clone the repository and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   # edit .env and set STRIPE and OPENAI keys
   ```
3. Initialize the database using Flask-Migrate:
   ```bash
   flask db upgrade
   ```
4. Run the application locally:
   ```bash
   python app.py
   ```

The application reads configuration from environment variables using `python-dotenv`. The most important variables are:

- `STRIPE_SECRET_KEY` – your Stripe secret key
- `STRIPE_PUBLISHABLE_KEY` – your Stripe publishable key
- `STRIPE_PRICE_ID` – the price ID for the checkout session
- `STRIPE_PRICING_TABLE_ID` – pricing table ID used for the Stripe pricing table widget
- `OPENAI_API_KEY` – optional key used by the backtester logic
- `HOSTED_PROMPT_ID` – optional ID for an OpenAI hosted prompt
- `HOSTED_PROMPT_VERSION` – version number for the hosted prompt (default `1`)
- `ENABLE_API_CALL_BUFFER` – set to `1` to wait after LLM calls (default disables delay)
- `FLASK_DEBUG` – set to `1` to enable debug mode
- `ADMIN_EMAIL` – email address allowed to access the `/users` page (default `harry.prendergast307@gmail.com`)
- `GUNICORN_CMD_ARGS` – set to "--timeout 120" for gunicorn timeout

### Stripe pricing table

When configuring your pricing table in the Stripe Dashboard choose **Redirect to your site** after payment.

Use these routes for the final URLs:

```
Success URL: https://your-domain.com/payment/success?session_id={CHECKOUT_SESSION_ID}
Cancel URL:  https://your-domain.com/payment/cancel
```

During local development you can use `http://127.0.0.1:5000` in place of `https://your-domain.com`.

## Deployment

A simple `render.yaml` is included for deployment to Render. Adjust the environment variables there as needed.

## Viewing Users

The `/users` page lists all registered accounts, showing each user's ID, email
and payment status. Access to this page is restricted to the address specified
by `ADMIN_EMAIL`.

## Live Trading (Experimental)

An optional `live_trading.py` script connects to the Binance Spot API and
periodically asks ChatGPT for a trading action. Every 30 minutes the script
gathers recent price data, sends your custom prompt to OpenAI, and places a
market **BUY** or **SELL** order based on the response.

Environment variables required for live trading:

- `BINANCE_API_KEY` and `BINANCE_API_SECRET` – your Binance credentials
- `LIVE_TRADING_PROMPT` – the prompt describing your strategy
- `TRADE_AMOUNT_BTC` – amount of BTC to trade each cycle (default `0.001`)

Run it with:

```bash
python live_trading.py
```

**Use at your own risk.** This example is for educational purposes only and does
not guarantee profits or prevent losses.
