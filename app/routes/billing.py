from flask import Blueprint, current_app, redirect, url_for, request, jsonify
from flask_login import login_required, current_user
import stripe
from config import Config
from ..models import db, User

billing_bp = Blueprint('billing', __name__)


@billing_bp.route('/subscribe')
@login_required
def subscribe():
    stripe.api_key = Config.STRIPE_SECRET_KEY
    try:
        checkout_session = stripe.checkout.Session.create(
            customer=current_user.stripe_customer_id,
            mode='subscription',
            line_items=[{'price': Config.STRIPE_PRICE_ID, 'quantity': 1}],
            success_url=url_for('billing.subscribe_success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=url_for('billing.subscribe_cancel', _external=True),
        )
        return redirect(checkout_session.url)
    except Exception as e:
        current_app.logger.error(f'Stripe error: {e}')
        return 'Error creating checkout session', 500


@billing_bp.route('/subscribe/success')
@login_required
def subscribe_success():
    return 'Subscription started. Check your email.'


@billing_bp.route('/subscribe/cancel')
@login_required
def subscribe_cancel():
    return 'Subscription canceled.'


@billing_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    stripe.api_key = Config.STRIPE_SECRET_KEY
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('stripe-signature')
    endpoint_secret = request.args.get('secret')  # optional
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except Exception as e:
        return jsonify(success=False), 400

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        customer_id = session.get('customer')
        user = db.session.execute(db.select(User).filter_by(stripe_customer_id=customer_id)).scalar_one_or_none()
        if user:
            user.is_active = True
            db.session.commit()
    return jsonify(success=True)
