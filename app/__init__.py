from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf import CSRFProtect
from config import Config

# Initialize extensions
csrf = CSRFProtect()
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    csrf.init_app(app)
    db.init_app(app)
    login_manager.init_app(app)

    from .routes.auth import auth_bp
    from .routes.billing import billing_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(billing_bp)

    with app.app_context():
        db.create_all()

    return app
