# SpamGuard AI – Spam Email Classifier

A full-stack Machine Learning web app that classifies an email as **Spam** or **Not Spam** in real time.
Frontend (HTML/CSS/JS) talks to a Flask API that runs a TF-IDF + Naive Bayes / Logistic Regression model.

## Features
- Real ML predictions with confidence % and risk level (Low / Medium / High)
- Dark / light mode, glassmorphism UI, fully responsive
- Example emails, character counter, copy button, toast notifications, loading spinner
- Recent predictions table with search and clear (saved in browser localStorage)
- Model performance and dataset numbers are **real values** read from `model/metrics.json`

## Technologies
Python, Flask, Scikit-learn, Pandas, NumPy, NLTK, HTML5, CSS3, JavaScript, Font Awesome

## Dataset
`dataset/spam.csv` – the public SMS Spam Collection (labeled spam/ham messages) plus 20 hand-written
email-style examples. Columns: `v1` (label), `v2` (message). You can replace it with any email dataset
that has the same two columns and re-run `python train_model.py`.

## ML Algorithm
1. Preprocess: lowercase, replace URLs, remove numbers/punctuation, remove stopwords, Porter stemming
2. Features: TF-IDF (unigrams + bigrams, 6000 features)
3. Train **Multinomial Naive Bayes** and **Logistic Regression**; keep the better F1 score
4. Evaluate on a 20% stratified test set; save model with Joblib

## Installation
```bash
pip install -r requirements.txt
python train_model.py     # optional: app.py trains automatically if the model is missing
```

## How to run
```bash
python app.py
```
Open http://127.0.0.1:5000

## API
**POST `/predict`**
```json
{ "email": "Congratulations! You won a prize..." }
```
Response:
```json
{ "prediction": "Spam", "confidence": 0.94, "risk": "High" }
```
Errors return `{ "error": "..." }` with HTTP 400/500.

**GET `/health`** → `{ "status": "ok", "model_loaded": true, "algorithm": "..." }`

**GET `/api/stats`** → real accuracy, precision, recall, F1 and dataset counts.

## Screenshots
Add your screenshots here (e.g. `docs/home.png`, `docs/spam-result.png`).

## Future improvements
- Train on a larger real email dataset (Enron, SpamAssassin)
- Add email header analysis and link scanning
- Try deep learning models (LSTM / BERT)
- User login with server-side history, Docker deployment
