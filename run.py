import os
import sys
from io import StringIO
from contextlib import redirect_stderr
from app import app, add_sample_users_for_leaderboard, update_leaderboard_in_background
import threading

def run_migrations_and_server():
    """
    Handles database migrations and then starts the Gunicorn server.
    This method ensures the database is correctly stamped and upgraded.
    """
    print("Running database migrations...")
    
    with app.app_context():
        # Capture stderr to check for specific error messages
        f = StringIO()
        with redirect_stderr(f):
            try:
                from flask_migrate import upgrade
                upgrade()
            except SystemExit:
                error_output = f.getvalue()
                if "Can't locate revision identified by" in error_output:
                    print("Migration history mismatch detected. Stamping database to head.")
                    from flask_migrate import stamp
                    stamp()
                    print("Database migration history stamped. Re-running upgrade.")
                    try:
                        upgrade()
                    except SystemExit:
                        print("Failed to upgrade after stamping. Exiting.")
                        sys.exit(1)
                else:
                    print("An unexpected migration error occurred. Exiting.")
                    print(error_output)
                    sys.exit(1)
    
    # Add demo users on every startup for a robust demo experience
    print("Adding sample users for leaderboard...")
    add_sample_users_for_leaderboard()
    
    # Start the background thread for updating the leaderboard
    print("Starting leaderboard background thread...")
    thread = threading.Thread(target=update_leaderboard_in_background, daemon=True)
    thread.start()
    
    # Now, start the Gunicorn server
    print("Starting Gunicorn server...")
    os.system(f"gunicorn app:app --timeout 300 --bind 0.0.0.0:{os.environ.get('PORT', 5000)} --workers 2")

if __name__ == "__main__":
    run_migrations_and_server()
