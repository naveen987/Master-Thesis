from flask import Flask, render_template, request, redirect, url_for
from database import init_db, add_user, get_user
import subprocess
import threading
import time

app = Flask(__name__)

# Initialize the database
init_db()

# Define the chatbot port and a flag to prevent multiple launches
CHATBOT_PORT = 8000
chainlit_running = False
lock = threading.Lock()


def start_chainlit():
    """Start the Chainlit server for the chatbot."""
    global chainlit_running
    with lock:
        if not chainlit_running:
            chainlit_running = True
            # Run Chainlit without opening a browser
            subprocess.Popen(
                ["chainlit", "run", "app.py", "--port", str(CHATBOT_PORT), "--no-open"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )


@app.route("/", methods=["GET", "POST"])
def login():
    """Login route to authenticate users."""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Validate user credentials
        user = get_user(email)
        if user and user[2] == password:  # user[2] is the password column
            role = user[3]  # user[3] is the role column

            # Start the Chainlit chatbot if it's not already running
            threading.Thread(target=start_chainlit, daemon=True).start()

            # Wait for a short time to ensure Chainlit is running
            time.sleep(2)

            # Redirect to the chatbot with query parameters
            chatbot_url = f"http://localhost:{CHATBOT_PORT}?user={email}&role={role}"
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
    app.run(port=5000)  # Run the Flask app on port 5000
