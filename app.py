"""
app.py - Flask backend for SpamGuard AI.
Run:  python app.py   then open http://127.0.0.1:5000
"""
import json
import os

import joblib
from flask import Flask, jsonify, render_template, request

from train_model import preprocess_text  # same cleaning used during training

app = Flask(__name__)

MODEL_PATH = "model/spam_model.pkl"
VEC_PATH = "model/tfidf_vectorizer.pkl"
METRICS_PATH = "model/metrics.json"

# Auto-train on first run if the model files are missing
if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
    import train_model
    train_model.main()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)
with open(METRICS_PATH) as f:
    metrics = json.load(f)


def risk_level(is_spam: bool, confidence: float) -> str:
    """Translate the prediction into a simple risk label."""
    if not is_spam:
        return "Low"
    return "High" if confidence >= 0.85 else "Medium"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True, "algorithm": metrics["algorithm"]})


@app.route("/api/stats")
def stats():
    """Real metrics + dataset numbers produced by train_model.py."""
    return jsonify(metrics)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip()
    if not email:
        return jsonify({"error": "Please enter an email message first."}), 400
    if len(email) > 10000:
        return jsonify({"error": "Email is too long (max 10,000 characters)."}), 400

    try:
        features = vectorizer.transform([preprocess_text(email)])
        proba = model.predict_proba(features)[0]          # [P(ham), P(spam)]
        is_spam = bool(proba[1] >= 0.5)
        confidence = float(proba[1] if is_spam else proba[0])
        return jsonify({
            "prediction": "Spam" if is_spam else "Not Spam",
            "confidence": round(confidence, 4),
            "risk": risk_level(is_spam, confidence),
        })
    except Exception as exc:  # never crash the server on bad input
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
