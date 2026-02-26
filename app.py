from flask import Flask, request, jsonify
from db import get_connection
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.signal import welch
from scipy.integrate import trapezoid
import model_defs
import sys
import types

# Create fake __main__ module and attach class there
main_module = sys.modules["__main__"]

from model_defs import FixedVotingClassifier
setattr(main_module, "FixedVotingClassifier", FixedVotingClassifier)

app = Flask(__name__)
def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        duration INTEGER,
        prediction INTEGER,
        confidence FLOAT
    );
    """)
    cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT,
    place TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
    cur.execute("""
ALTER TABLE sessions
ADD COLUMN IF NOT EXISTS user_id INTEGER
REFERENCES users(id) ON DELETE CASCADE;
""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS eeg_features (
        session_id INTEGER REFERENCES sessions(id),
        fp1_mean FLOAT,
        fp1_std FLOAT,
        fp2_mean FLOAT,
        fp2_std FLOAT
    );
    """)

    conn.commit()
    cur.close()
    conn.close()

init_db()
CORS(app)
from sklearn.ensemble import VotingClassifier
import numpy as np

def fix_proba(clf, X, all_classes):
    if not hasattr(clf, "predict_proba"):
        preds = clf.predict(X)
        proba = np.zeros((len(preds), len(all_classes)))
        for i, p in enumerate(preds):
            proba[i, all_classes.index(p)] = 1.0
        return proba

    proba = clf.predict_proba(X)
    n_samples = proba.shape[0]
    fixed = np.zeros((n_samples, len(all_classes)), dtype=float)

    for i, cls in enumerate(clf.classes_):
        if cls in all_classes:
            j = all_classes.index(cls)
            fixed[:, j] = proba[:, i]
    return fixed


class FixedVotingClassifier(VotingClassifier):
    def __init__(self, *args, all_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        if all_classes is None:
            raise ValueError("FixedVotingClassifier requires all_classes parameter.")
        self.all_classes = list(all_classes)

    def predict_proba(self, X):
        probas = []
        for name, est in self.estimators_:
            prob = fix_proba(est, X, self.all_classes)
            probas.append(prob)

        stacked = np.stack(probas, axis=0)

        if self.weights is not None:
            weights = np.array(self.weights)[:, None, None]
            avg = np.sum(stacked * weights, axis=0) / np.sum(self.weights)
        else:
            avg = np.mean(stacked, axis=0)

        return avg

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

clf = joblib.load(MODELS_DIR / "final_ensemble_model.pkl")
scaler = joblib.load(MODELS_DIR / "final_scaler.pkl")

# Manually define feature order (MUST match training)
FEATS = [
    f"{band}_{ch}"
    for band in ["delta","theta","alpha","beta","gamma"]
    for ch in ["Fp1","Fp2","F3","F4","O1","O2"]
]

@app.post("/predict_live")
def predict_live():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    fp1 = data.get("fp1", [])
    fp2 = data.get("fp2", [])
    fs = int(data.get("sampling_rate", 125))

    if not fp1 or not fp2:
        return jsonify({"error": "Missing fp1/fp2 data"}), 400

    sig_fp1 = np.array(fp1, dtype=np.float32)
    sig_fp2 = np.array(fp2, dtype=np.float32)

    if len(sig_fp1) < fs or len(sig_fp2) < fs:
        return jsonify({"error": "Insufficient EEG data"}), 400

    def bandpower(sig, fs, band):
        fmin, fmax = band
        freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), fs * 2))
        idx = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(idx):
            return 0.0
        return trapezoid(psd[idx], freqs[idx])

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 45),
    }

    fp1_bp = {b: bandpower(sig_fp1, fs, bands[b]) for b in bands}
    fp2_bp = {b: bandpower(sig_fp2, fs, bands[b]) for b in bands}

    def avg(a, b): return 0.5 * (a + b)

    features = {}
    for b in bands:
        features[f"{b}_Fp1"] = fp1_bp[b]
        features[f"{b}_Fp2"] = fp2_bp[b]
        features[f"{b}_F3"]  = avg(fp1_bp[b], fp2_bp[b])
        features[f"{b}_F4"]  = avg(fp1_bp[b], fp2_bp[b])
        features[f"{b}_O1"]  = fp1_bp[b]
        features[f"{b}_O2"]  = fp2_bp[b]

    X = pd.DataFrame([features])[FEATS]
    Xs = scaler.transform(X)

    pred = int(clf.predict(Xs)[0])
    proba = clf.predict_proba(Xs).tolist()[0]
    confidence = float(max(proba) * 100)
        # ================= DATABASE SAVE =================
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO sessions (user_id, duration, prediction)
    VALUES (%s, %s, %s)
    RETURNING id;
""", (
    user_id,
    len(fp1) // fs,
    pred
))

    session_id = cur.fetchone()["id"]

    fp1_mean = float(np.mean(sig_fp1))
    fp1_std  = float(np.std(sig_fp1))
    fp2_mean = float(np.mean(sig_fp2))
    fp2_std  = float(np.std(sig_fp2))

    cur.execute("""
        INSERT INTO eeg_features
        (session_id, fp1_mean, fp1_std, fp2_mean, fp2_std)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        session_id,
        fp1_mean,
        fp1_std,
        fp2_mean,
        fp2_std
    ))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "prediction": pred,
        "probabilities": proba
    })
@app.get("/sessions")
def get_sessions():
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT * FROM sessions
            ORDER BY id DESC
            LIMIT 10;
        """)

        sessions = cur.fetchall()

        cur.close()
        conn.close()

        return jsonify(sessions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.post("/users")
def create_user():
    data = request.get_json()

    name = data.get("name")
    age = data.get("age")
    gender = data.get("gender")
    place = data.get("place")

    if not name or not age:
        return jsonify({"error": "Name and age required"}), 400

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO users (name, age, gender, place)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
        """, (name, age, gender, place))

        user_id = cur.fetchone()["id"]

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"user_id": user_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.get("/users")
def get_users():
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT id, name, age, gender, place, created_at
            FROM users
            ORDER BY id DESC;
        """)

        users = cur.fetchall()

        cur.close()
        conn.close()

        return jsonify(users)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.post("/reset_db")
def reset_db():
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("DELETE FROM eeg_features;")
        cur.execute("DELETE FROM sessions;")
        cur.execute("DELETE FROM users;")

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"message": "Database cleared successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run()
