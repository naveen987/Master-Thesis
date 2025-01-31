from flask import Flask, render_template, request, redirect, url_for
from database import init_db, add_user, get_user
import subprocess
import threading
import psutil

app = Flask(__name__)

# Initialize the database
init_db()

# Updated ports
FLASK_PORT = 5002  # Running Flask on port 5002
CHATBOT_PORT = 8001  # Running Chainlit on port 8001

def is_port_in_use(port):
    """Check if a port is already in use (Chainlit already running)."""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False

def start_chainlit():
    """Start Chainlit server in the background if it's not already running."""
    if is_port_in_use(CHATBOT_PORT):
        print("✅ Chainlit is already running.")
        return
    
    print("🔵 Starting Chainlit server in the background...")
    subprocess.Popen(
        ["chainlit", "run", "app.py", "--port", str(CHATBOT_PORT), "--no-auto-launch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True  # Ensures it runs in the background
    )

# Ensure Chainlit starts once when Flask is launched
threading.Thread(target=start_chainlit, daemon=True).start()

@app.route("/", methods=["GET", "POST"])
def login():
    """Login route to authenticate users and redirect to Chainlit chatbot."""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = get_user(email)
        if user and user[2] == password:
            role = user[3]
            chatbot_url = f"http://localhost:{CHATBOT_PORT}/?user={email}&role={role}"
            print(f"🔀 Redirecting to Chainlit: {chatbot_url}")
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
        role = request.form.get("role")
        if add_user(email, password, role):
            return redirect(url_for("login"))
        else:
            return render_template("signup.html", error="Email already registered.")
    return render_template("signup.html")

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)
