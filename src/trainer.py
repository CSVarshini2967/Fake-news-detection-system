"""Dataset loading + model training + evaluation. Used by train.py and the admin page."""
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from .preprocess import clean_text

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
STATIC_DIR = BASE / "static"


def load_dataset(path=None) -> pd.DataFrame:
    """Return DataFrame with columns text, label (0 = fake, 1 = genuine)."""
    if path is None:
        news = DATA_DIR / "news.csv"
        fake, true = DATA_DIR / "Fake.csv", DATA_DIR / "True.csv"
        sample = DATA_DIR / "sample_news.csv"
        if news.exists():
            path = news
        elif fake.exists() and true.exists():  # ISOT dataset from Kaggle
            f, t = pd.read_csv(fake), pd.read_csv(true)
            f["label"], t["label"] = 0, 1
            df = pd.concat([f, t], ignore_index=True)
            return _normalise(df)
        elif sample.exists():
            print("WARNING: using tiny sample dataset. Download a real one (see README).")
            path = sample
        else:
            raise FileNotFoundError(
                "No dataset found. Put Fake.csv + True.csv (ISOT) or news.csv in data/.")
    return _normalise(pd.read_csv(path))


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    if "label" not in df.columns or ("text" not in df.columns and "title" not in df.columns):
        raise ValueError("CSV needs a 'label' column (0 = fake, 1 = genuine) and a 'text' column.")
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "title" in df.columns:
        df["text"] = df["title"]
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()) <= {0, 1} or df["label"].nunique() < 2:
        raise ValueError("Labels must contain both 0 (fake) and 1 (genuine).")
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def run_training(csv_path=None, include_rf=True, log=print) -> dict:
    """Train, evaluate, save the deployed model (Logistic Regression) and metrics.json."""
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_dataset(csv_path)
    log(f"Loaded {len(df)} articles. Cleaning text ...")
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    vec = TfidfVectorizer(stop_words="english", max_features=30000,
                          ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
    }
    if include_rf:
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=42)

    results = {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, y_train)
        pred = m.predict(Xte)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, pred, zero_division=0), 4),
            "train_seconds": round(time.time() - t0, 1),
        }
        log(f"{name}: {results[name]}")

    # Deployed model: Logistic Regression (gives probabilities + explainable keywords)
    deployed = models["Logistic Regression"]
    cm = confusion_matrix(y_test, deployed.predict(Xte), labels=[0, 1])
    _save_confusion_matrix(cm)

    joblib.dump(deployed, MODEL_DIR / "model.pkl")
    joblib.dump(vec, MODEL_DIR / "vectorizer.pkl")

    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "dataset": {"total": int(len(df)),
                    "fake": int((df["label"] == 0).sum()),
                    "genuine": int((df["label"] == 1).sum()),
                    "test_size": int(len(y_test))},
        "models": results,
        "deployed_model": "Logistic Regression",
        "confusion_matrix": cm.tolist(),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log("Saved models/model.pkl, models/vectorizer.pkl, models/metrics.json")
    return metrics


def _save_confusion_matrix(cm):
    STATIC_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Fake", "Genuine"])
    ax.set_yticks([0, 1], ["Fake", "Genuine"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)
