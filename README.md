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
- `FLASK_DEBUG` – set to `1` to enable debug mode

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

Once logged in, you can visit `/users` to see a table of all registered accounts. The page displays each user's ID, email and whether they have completed payment.
