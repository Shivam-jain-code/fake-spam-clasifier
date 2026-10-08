"""
train_model.py - trains the spam classifier and saves the model + REAL metrics.
Run once:  python train_model.py
"""
import json
import re

import joblib
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ---- Stopwords: use NLTK if its data is installed, otherwise scikit-learn's list ----
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set(ENGLISH_STOP_WORDS)

stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    """Lowercase -> keep letters -> remove stopwords -> stem. Used in training AND prediction."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)   # links are a strong spam signal
    text = re.sub(r"[^a-z\s]", " ", text)                     # remove numbers/punctuation
    words = [stemmer.stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)


def main():
    df = pd.read_csv("dataset/spam.csv").dropna()
    df.columns = ["label", "message"]
    df["target"] = (df["label"] == "spam").astype(int)      # 1 = spam, 0 = ham
    df["clean"] = df["message"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    # TF-IDF turns text into numbers (unigrams + bigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, sublinear_tf=True)
    Xtr, Xte = vectorizer.fit_transform(X_train), vectorizer.transform(X_test)

    # Train both algorithms and keep the one with the better F1 score
    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10, class_weight="balanced"),
    }
    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidates.items():
        model.fit(Xtr, y_train)
        f1 = f1_score(y_test, model.predict(Xte))
        print(f"{name}: F1 = {f1:.4f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    pred = best_model.predict(Xte)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "algorithm": best_name,
        "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "precision": round(precision_score(y_test, pred) * 100, 2),
        "recall": round(recall_score(y_test, pred) * 100, 2),
        "f1": round(f1_score(y_test, pred) * 100, 2),
        "total": int(len(df)),
        "spam": int(df["target"].sum()),
        "ham": int((df["target"] == 0).sum()),
        "train": int(len(X_train)),
        "test": int(len(X_test)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    joblib.dump(best_model, "model/spam_model.pkl")
    joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model. Metrics:\n", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
