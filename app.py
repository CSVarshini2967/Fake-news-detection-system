"""AI Fake News Detection System - Flask web app.

Run:  python app.py   ->  http://127.0.0.1:5000
Default admin login:  admin / admin123   (change it!)
"""
import functools
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
from flask import (Flask, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from src.predict import Predictor
from src.trainer import DATA_DIR, MODEL_DIR, run_training

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "app.db"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-in-production")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads

_predictor = None


# ---------------------------------------------------------------- model
def get_predictor():
    """Lazy-load the model; returns None if it has not been trained yet."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = Predictor()
        except FileNotFoundError:
            return None
    return _predictor


def load_metrics():
    f = MODEL_DIR / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


# ---------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            keywords TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    if not db.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():
        db.execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                   ("admin", generate_password_hash("admin123"), "admin"))
    db.commit()
    db.close()


# ---------------------------------------------------------------- auth helpers
def login_required(view):
    @functools.wraps(view)
    def wrapped(*a, **kw):
        if "user_id" not in session:
            flash("Please log in first.", "warn")
            return redirect(url_for("login"))
        return view(*a, **kw)
    return wrapped


def admin_required(view):
    @functools.wraps(view)
    def wrapped(*a, **kw):
        if session.get("role") != "admin":
            flash("Admin access only.", "error")
            return redirect(url_for("index"))
        return view(*a, **kw)
    return wrapped


# ---------------------------------------------------------------- pages
@app.route("/", methods=["GET", "POST"])
def index():
    result, text = None, ""
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if len(text) < 10:
            flash("Enter at least a full headline or sentence to analyse.", "error")
        else:
            predictor = get_predictor()
            if predictor is None:
                flash("The model has not been trained yet. Ask an admin to train it.", "error")
            else:
                result = predictor.predict(text)
                if "user_id" in session:
                    get_db().execute(
                        "INSERT INTO analyses (user_id, text, label, confidence, keywords) "
                        "VALUES (?,?,?,?,?)",
                        (session["user_id"], text, result["label"], result["confidence"],
                         ", ".join(k["word"] for k in result["keywords"])))
                    get_db().commit()
    return render_template("index.html", result=result, text=text)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if len(text) < 10:
        return jsonify(error="Field 'text' must have at least 10 characters."), 400
    predictor = get_predictor()
    if predictor is None:
        return jsonify(error="Model not trained yet."), 503
    return jsonify(predictor.predict(text))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if len(username) < 3 or len(password) < 6:
            flash("Username needs 3+ characters and password 6+ characters.", "error")
        else:
            try:
                get_db().execute(
                    "INSERT INTO users (username, password_hash) VALUES (?,?)",
                    (username, generate_password_hash(password)))
                get_db().commit()
                flash("Account created. You can log in now.", "ok")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username is taken. Choose another.", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = get_db().execute("SELECT * FROM users WHERE username=?",
                                (request.form.get("username", "").strip(),)).fetchone()
        if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
            session.clear()
            session.update(user_id=user["id"], username=user["username"], role=user["role"])
            return redirect(url_for("admin" if user["role"] == "admin" else "index"))
        flash("Wrong username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/history")
@login_required
def history():
    rows = get_db().execute(
        "SELECT * FROM analyses WHERE user_id=? ORDER BY id DESC", (session["user_id"],)).fetchall()
    return render_template("history.html", rows=rows)


@app.route("/history/delete/<int:item_id>", methods=["POST"])
@login_required
def delete_history(item_id):
    get_db().execute("DELETE FROM analyses WHERE id=? AND user_id=?", (item_id, session["user_id"]))
    get_db().commit()
    return redirect(url_for("history"))


# ---------------------------------------------------------------- admin
@app.route("/admin")
@admin_required
def admin():
    db = get_db()
    stats = {
        "users": db.execute("SELECT COUNT(*) FROM users").fetchone()[0],
        "analyses": db.execute("SELECT COUNT(*) FROM analyses").fetchone()[0],
        "fake": db.execute("SELECT COUNT(*) FROM analyses WHERE label='Likely Fake'").fetchone()[0],
        "genuine": db.execute("SELECT COUNT(*) FROM analyses WHERE label='Likely Genuine'").fetchone()[0],
    }
    recent = db.execute(
        "SELECT a.*, u.username FROM analyses a JOIN users u ON u.id=a.user_id "
        "ORDER BY a.id DESC LIMIT 10").fetchall()
    return render_template("admin.html", metrics=load_metrics(), stats=stats, recent=recent)


@app.route("/admin/upload", methods=["POST"])
@admin_required
def admin_upload():
    file = request.files.get("dataset")
    if not file or not file.filename.lower().endswith(".csv"):
        flash("Choose a .csv file.", "error")
        return redirect(url_for("admin"))
    try:
        df = pd.read_csv(file)
        df.columns = [c.strip().lower() for c in df.columns]
        if "label" not in df.columns or not ({"text", "title"} & set(df.columns)):
            raise ValueError("CSV needs 'label' (0 = fake, 1 = genuine) and 'text' columns.")
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(DATA_DIR / "news.csv", index=False)
        flash(f"Dataset saved: {len(df)} rows. Now click Retrain model.", "ok")
    except Exception as e:  # noqa: BLE001
        flash(f"Upload failed: {e}", "error")
    return redirect(url_for("admin"))


@app.route("/admin/retrain", methods=["POST"])
@admin_required
def admin_retrain():
    global _predictor
    try:
        run_training(include_rf=False)  # skip Random Forest here to keep it quick
        _predictor = None  # reload new model on next request
        flash("Model retrained and loaded.", "ok")
    except Exception as e:  # noqa: BLE001
        flash(f"Training failed: {e}", "error")
    return redirect(url_for("admin"))


init_db()

if __name__ == "__main__":
    app.run(debug=True)
