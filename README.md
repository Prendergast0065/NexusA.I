# NexusTrade AI Lab

This Flask application provides a conceptual backtesting tool with optional Stripe checkout.

## Setup

1. **Clone the repository** and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables.** Copy `.env.example` to `.env` and fill in your Stripe keys and price ID. The example file provides placeholders:
   ```bash
   cp .env.example .env
   # edit .env with your credentials
   ```

   - `STRIPE_SECRET_KEY` – your Stripe secret key
   - `STRIPE_PUBLISHABLE_KEY` – your Stripe publishable key
   - `STRIPE_PRICE_ID` – the price ID for the product or subscription you wish to charge

3. **Run the application:**
   ```bash
   gunicorn app:app
   ```

   By default the Flask app will run with debugging disabled when using `gunicorn`.

## Deploying to Render

The provided `render.yaml` defines the service. Set the same Stripe environment variables in the Render dashboard or update `render.yaml` to sync them.
