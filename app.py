from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    url_for,
    redirect,
    session,
    flash,
)
import os
import uuid  # For generating unique job IDs
import logging  # For logging within Flask app
import stripe  # Stripe payment processing
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate
from passlib.hash import bcrypt
from dotenv import load_dotenv

# --- Import your actual backtesting logic ---
from my_backtester_logic import execute_backtest_strategy  # Assuming my_backtester_logic.py is in the same directory

# Configure Flask's built-in logger
# You might want to configure this more extensively in a production app
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] [FlaskAPP] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Login manager setup
login_manager = LoginManager(app)
login_manager.login_view = 'login_page'


def hash_pw(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hash(password)


def verify_pw(password: str, pw_hash: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    return bcrypt.verify(password, pw_hash)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    pw_hash = db.Column(db.String(255), nullable=False)
    stripe_cus = db.Column(db.String(120))
    is_paid = db.Column(db.Boolean, default=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Database tables will be created via Flask-Migrate migrations

# Ensure directories exist
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = os.path.join('static', 'results')  # Ensure 'static' is at the root of your Flask app
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Stripe configuration
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
STRIPE_PRICE_ID = os.environ.get('STRIPE_PRICE_ID', '')
STRIPE_PRICING_TABLE_ID = os.environ.get('STRIPE_PRICING_TABLE_ID', '')


# --- Routes to serve your HTML pages ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            return render_template('signup.html', message='Email and password required')
        if User.query.filter_by(email=email).first():
            return render_template('signup.html', message='User already exists')
        try:
            customer = stripe.Customer.create(email=email)
            user = User(email=email,
                        pw_hash=hash_pw(password),
                        stripe_cus=customer['id'])
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for('checkout_page'))
        except Exception as e:
            app.logger.error(f"Signup failed: {e}")
            return render_template('signup.html', message='Signup failed')
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and verify_pw(password, user.pw_hash):
            login_user(user)
            return redirect(url_for('member_dashboard_page'))
        return render_template('login.html', message='Invalid credentials')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout_page():
    logout_user()
    return redirect(url_for('home'))


@app.route('/checkout')
def checkout_page():
    if not current_user.is_authenticated:
        return redirect(url_for('signup_page'))
    plan = request.args.get('plan', 'developer')
    return render_template('checkout.html',
                           selected_plan=plan,
                           publishable_key=STRIPE_PUBLISHABLE_KEY,
                           pricing_table_id=STRIPE_PRICING_TABLE_ID)


@app.route('/success')
@login_required
def success_page():
    return render_template('success.html')


@app.route('/payment/success')
@login_required
def payment_success():
    sess_id = request.args.get('session_id')
    if sess_id:
        try:
            session_obj = stripe.checkout.Session.retrieve(sess_id)
            if session_obj and session_obj.get('payment_status') == 'paid':
                current_user.is_paid = True
                db.session.commit()
        except Exception as e:
            app.logger.error(f"Failed to verify Stripe session: {e}")
    flash('Payment received – welcome!')
    return render_template('success.html')


@app.route('/payment/cancel')
@login_required
def payment_cancel():
    flash('Payment cancelled.')
    return redirect(url_for('checkout_page'))


@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    try:
        session = stripe.checkout.Session.create(
            customer=current_user.stripe_cus,
            client_reference_id=current_user.id,
            payment_method_types=['card'],
            line_items=[{
                'price': STRIPE_PRICE_ID,
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('payment_success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=url_for('payment_cancel', _external=True),
        )
        return jsonify({'url': session.url})
    except Exception as e:
        app.logger.error(f"Stripe checkout creation failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('stripe-signature')
    endpoint_secret = os.environ.get('STRIPE_WEBHOOK_SECRET', '')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except Exception as e:
        app.logger.error(f"Webhook error: {e}")
        return '', 400

    if event['type'] == 'checkout.session.completed':
        session_obj = event['data']['object']
        user_id = session_obj.get('client_reference_id')
        if user_id:
            user = User.query.get(int(user_id))
            if user:
                user.is_paid = True
                db.session.commit()
                app.logger.info(f"User {user.email} marked as paid")
    return '', 200


@app.route('/how-it-works')
def how_it_works_page():
    return render_template('how_it_works.html')


@app.route('/prompt-guide')
def prompt_guide_page():
    return render_template('prompt_guide.html')


@app.route('/dashboard')
@login_required
def member_dashboard_page():
    if not current_user.is_paid:
        return redirect(url_for('checkout_page'))
    return render_template('member_dashboard.html')


# --- API Endpoint for Backtesting ---
@app.route('/run-backtest', methods=['POST'])
@login_required
def handle_run_backtest():
    if not current_user.is_paid:
        return jsonify({"error": "Payment required"}), 402
    if request.method == 'POST':
        app.logger.info("Received /run-backtest request")

        try:
            openai_key = request.form.get('backtest_openai_api_key')
            strategy_prompt = request.form.get('backtest_strategy_prompt')
            # Default to '7' if not provided or empty, then convert to int
            num_days_str = request.form.get('backtest_days_csv', '7')
            num_days = int(num_days_str) if num_days_str else 7

            randomize_period_str = request.form.get('backtest_randomize_period_csv', 'yes')
            randomize = True if randomize_period_str.lower() == 'yes' else False

            data_source = request.form.get('dataSource', 'csv')
            app.logger.info(f"Data source selected: {data_source}")

            csv_file_path = None
            job_id = str(uuid.uuid4())
            app.logger.info(f"Generated Job ID: {job_id}")

            if not openai_key:
                app.logger.error("OpenAI API key was not provided.")
                return jsonify({"error": "OpenAI API Key is required."}), 400
            if not strategy_prompt:
                app.logger.error("Strategy prompt was not provided.")
                return jsonify({"error": "Strategy prompt is required."}), 400

            if data_source == 'csv':
                if 'backtest_csv_data' not in request.files:
                    app.logger.error("No CSV file part in request.")
                    return jsonify({"error": "No CSV file part"}), 400
                file = request.files['backtest_csv_data']
                if file.filename == '':
                    app.logger.error("No CSV file selected.")
                    return jsonify({"error": "No selected CSV file"}), 400
                if file:
                    filename = f"{job_id}_{file.filename}"
                    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(csv_file_path)
                    app.logger.info(f"Uploaded CSV saved to: {csv_file_path}")
                else:  # Should not happen if filename check passes, but as a safeguard
                    app.logger.error("File object is present but somehow invalid.")
                    return jsonify({"error": "CSV file is invalid."}), 400

            elif data_source == 'exchange':
                # This part remains conceptual as per previous discussions
                trading_pair = request.form.get('backtest_exchange_pair')
                start_date = request.form.get('backtest_start_date')
                end_date = request.form.get('backtest_end_date')
                app.logger.info(f"Conceptual: Fetch data for {trading_pair} from {start_date} to {end_date}")
                # For now, return not implemented
                return jsonify({"error": "Fetch from exchange not yet implemented in this application."}), 501

            if not csv_file_path and data_source == 'csv':
                app.logger.error("CSV file path was not set after upload attempt.")
                return jsonify({"error": "CSV file processing failed."}), 500

            # --- Call your actual backtesting logic ---
            app.logger.info(f"Calling execute_backtest_strategy for job {job_id}")

            # Parameters for execute_backtest_strategy:
            # openai_api_key_param, user_strategy_prompt_str, csv_data_path_param,
            # num_days_param, randomize_period_from_csv_param, job_id_param
            # Other params will use defaults defined in execute_backtest_strategy

            backtest_results_data = execute_backtest_strategy(
                openai_api_key_param=openai_key,
                user_strategy_prompt_str=strategy_prompt,
                csv_data_path_param=csv_file_path,
                num_days_param=num_days,
                randomize_period_from_csv_param=randomize,
                job_id_param=job_id,
                output_base_dir=RESULTS_FOLDER  # Ensure this is the 'static/results' path
                # api_call_buffer_seconds and gpt_model_to_use will use defaults from my_backtester_logic.py
            )

            if "error" in backtest_results_data:
                app.logger.error(f"Backtest for job {job_id} failed: {backtest_results_data['error']}")
                return jsonify({"error": backtest_results_data["error"]}), 500

            app.logger.info(f"Backtest for job {job_id} completed successfully. Results: {backtest_results_data}")
            return jsonify({
                "status": "completed",
                "job_id": job_id,
                "results": backtest_results_data
            })

        except Exception as e:
            app.logger.error(f"Critical error in /run-backtest: {e}", exc_info=True)
            return jsonify({"error": "An internal server error occurred during backtest.", "details": str(e)}), 500


# Route to serve result files from the dynamic job_id subdirectories
@app.route('/static/results/<path:job_id>/<path:filename>')
def serve_job_result_file(job_id, filename):
    # Construct the directory path carefully
    directory = os.path.join(app.root_path, RESULTS_FOLDER, job_id)  # app.root_path is the app's root
    app.logger.info(f"Attempting to serve file: {filename} from directory: {directory}")
    return send_from_directory(directory, filename)


if __name__ == '__main__':
    # Allow debug mode to be toggled via the FLASK_DEBUG environment variable
    debug_mode = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(debug=debug_mode)

