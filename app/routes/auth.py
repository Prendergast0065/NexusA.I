from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required
from passlib.hash import argon2
from ..models import User, db
import stripe
from config import Config

auth_bp = Blueprint('auth', __name__)


class SignupForm:
    def __init__(self, form):
        self.email = form.get('email', '').strip()
        self.password = form.get('password', '')


class LoginForm:
    def __init__(self, form):
        self.email = form.get('email', '').strip()
        self.password = form.get('password', '')


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        form = SignupForm(request.form)
        if not form.email or not form.password:
            flash('Email and password are required.', 'danger')
            return render_template('signup.html')
        if db.session.execute(db.select(User).filter_by(email=form.email)).scalar_one_or_none():
            flash('Email already registered.', 'danger')
            return render_template('signup.html')
        password_hash = argon2.hash(form.password)
        user = User(email=form.email, password_hash=password_hash)
        db.session.add(user)
        db.session.commit()
        stripe.api_key = Config.STRIPE_SECRET_KEY
        customer = stripe.Customer.create(email=user.email)
        user.stripe_customer_id = customer.id
        db.session.commit()
        login_user(user)
        return redirect(url_for('billing.subscribe'))
    return render_template('signup.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        form = LoginForm(request.form)
        user = db.session.execute(db.select(User).filter_by(email=form.email)).scalar_one_or_none()
        if not user or not argon2.verify(form.password, user.password_hash):
            flash('Invalid credentials.', 'danger')
            return render_template('login.html')
        login_user(user)
        return redirect(url_for('billing.subscribe'))
    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
