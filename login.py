from flask import Flask, render_template, request, redirect, url_for
from database import init_db, get_user, add_user
import subprocess
import threading
import os
import time
import requests

app = Flask(__name__)

# Initialize the database
init_db()

# Ports
FLASK_PORT = 5002       # Flask will run on port 5002
CHATBOT_PORT = 8001     # Chainlit will run on port 8001
chainlit_process = None  # Track the Chainlit process

def is_chainlit_running():
    """Check if Chainlit is running by sending a request to the server."""
    try:
        response = requests.get(f"http://localhost:{CHATBOT_PORT}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_chainlit(email):
    """
    Start Chainlit only if it's not already running.
    Pass the logged-in user email as an environment variable so 
    that Chainlit can fetch the role from the DB without needing URL params.
    """
    global chainlit_process
    if is_chainlit_running():
        print("✅ Chainlit is already running.")
        return

    print(f"🔵 Launching Chainlit on port {CHATBOT_PORT}...")

    chainlit_process = subprocess.Popen(
        ["chainlit", "run", "app.py", "--port", str(CHATBOT_PORT)],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL,  # Hide logs
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # Run in a separate process
        env={
            **os.environ,  # Inherit current environment variables
            "LOGGED_IN_EMAIL": email
        }
    )

@app.route("/", methods=["GET", "POST"])
def login():
    """Login route to authenticate users and start Chainlit."""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = get_user(email)
        if user and user[2] == password:
            print(f"🟢 LOGIN SUCCESS: {email}")

            # Start Chainlit if not running; pass the email to the new process
            threading.Thread(target=start_chainlit, args=(email,), daemon=True).start()

            # After login, display a loading page that will link to Chainlit
            chatbot_url = f"http://localhost:{CHATBOT_PORT}/"
            print(f"🔀 REDIRECTING TO: {chatbot_url}")
            return render_template("loading.html", chatbot_url=chatbot_url)
        else:
            return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Signup route to register new users."""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        role = request.form.get("role")
        if add_user(email, password, role):
            return redirect(url_for("login"))
        else:
            return render_template("signup.html", error="Email already registered.")
    return render_template("signup.html")

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)
