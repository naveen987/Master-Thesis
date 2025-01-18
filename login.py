from flask import Flask, render_template, request, redirect, url_for
from database import init_db, add_user, get_user
import subprocess
import threading
import time
import psutil
import requests

app = Flask(__name__)

# Initialize the database
init_db()

# Define the chatbot port and Flask port
FLASK_PORT = 5001  # Running Flask on port 5001
CHATBOT_PORT = 8000  # Running Chainlit on port 8000
chainlit_running = False
lock = threading.Lock()


def is_port_in_use(port):
    """Check if a port is already in use (Chainlit already running)."""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False


def wait_for_chainlit(timeout=60):
    """Wait until Chainlit is fully ready before redirecting."""
    
    # ✅ If Chainlit is already running, return immediately
    if is_port_in_use(CHATBOT_PORT):
        print("✅ Chainlit is already running. Skipping wait.")
        return True

    print(f"⏳ Waiting up to {timeout} seconds for Chainlit to be ready...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{CHATBOT_PORT}")
            if response.status_code == 200:
                print("✅ Chainlit is ready!")
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(2)  # Wait 2 seconds before retrying

    print("❌ Chainlit failed to start within the timeout! Redirecting anyway.")
    return False  # Don't block Flask indefinitely


def start_chainlit():
    """Start the Chainlit server only if it's not already running."""
    global chainlit_running
    with lock:
        if is_port_in_use(CHATBOT_PORT):  # ✅ Don't start if already running
            print("🔵 Chainlit is already running. Skipping startup.")
            return

        chainlit_running = True
        try:
            print("🔵 Starting Chainlit server...")
            subprocess.Popen(
                ["chainlit", "run", "app.py", "--port", str(CHATBOT_PORT), "--no-auto-launch"],
                stdout=subprocess.DEVNULL,  # ✅ Hide subprocess output
                stderr=subprocess.DEVNULL,
                text=True
            )

            # ✅ Wait until Chainlit is fully ready before proceeding
            if wait_for_chainlit():
                print(f"✅ Chainlit started successfully on port {CHATBOT_PORT}.")
            else:
                print("⚠️ Chainlit may still be loading. Proceeding with redirect.")

        except Exception as e:
            print(f"🔴 Error starting Chainlit: {e}")
            chainlit_running = False  # Reset flag if error occurs


@app.route("/", methods=["GET", "POST"])
def login():
    """Login route to authenticate users and redirect to Chainlit chatbot."""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Validate user credentials
        user = get_user(email)
        if user and user[2] == password:  # user[2] is the password column
            role = user[3]  # user[3] is the role column

            # ✅ Start Chainlit if it's not running
            if not is_port_in_use(CHATBOT_PORT):  # Prevent multiple instances
                threading.Thread(target=start_chainlit, daemon=True).start()
                wait_for_chainlit()  # ✅ Wait only if Chainlit was not already running

            # ✅ If Chainlit is already running, skip waiting and redirect
            chatbot_url = f"http://localhost:{CHATBOT_PORT}/?user={email}&role={role}"
            print(f"🔀 Redirecting to Chainlit: {chatbot_url}")  # Debug log
            return redirect(chatbot_url)

        else:
            return render_template("login.html", error="Invalid email or password.")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Signup route to register new users."""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        role = request.form.get("role")  # Get the selected role from the form

        # Add user to the database
        if add_user(email, password, role):
            return redirect(url_for("login"))
        else:
            return render_template("signup.html", error="Email already registered.")

    return render_template("signup.html")


if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)  # Run the Flask app on port 5001
