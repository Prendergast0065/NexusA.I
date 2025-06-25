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
3. Run the application locally:
   ```bash
   python app.py
   ```

The application reads configuration from environment variables using `python-dotenv`. The most important variables are:

- `STRIPE_SECRET_KEY` – your Stripe secret key
- `STRIPE_PUBLISHABLE_KEY` – your Stripe publishable key
- `STRIPE_PRICE_ID` – the price ID for the checkout session
- `STRIPE_PRICING_TABLE_ID` – pricing table ID used for the Stripe pricing table widget
- `OPENAI_API_KEY` – optional key used by the backtester logic
- `FLASK_DEBUG` – set to `1` to enable debug mode

## Deployment

A simple `render.yaml` is included for deployment to Render. Adjust the environment variables there as needed.
