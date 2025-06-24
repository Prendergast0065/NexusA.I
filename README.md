# NexusA.I Sign-up Flow

This app demonstrates a minimal Flask 3 setup with Stripe subscriptions.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env with your Stripe keys
flask --app run.py run
```

Alembic is included for migrations. Configure your database via `DATABASE_URL` in `.env`.
