# AI Fake News Detection System

Flask web app that classifies news text as **Likely Fake** or **Likely Genuine**, with a confidence score and the keywords that influenced the result. Built with NLP (TF-IDF) and Machine Learning (Logistic Regression, Naive Bayes, Random Forest comparison).

## Setup (5 minutes)

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## 1. Get the dataset

Download **"Fake and Real News Dataset" (ISOT)** from Kaggle and put `Fake.csv` and `True.csv` inside `data/`.
Or put your own `data/news.csv` with columns `text,label` (0 = fake, 1 = genuine).
A tiny `sample_news.csv` is included only so the code runs; its scores are meaningless.

## 2. Train the model

```bash
python train.py
```

This cleans the text, trains 3 models, prints accuracy / precision / recall / F1, and saves
`models/model.pkl`, `models/vectorizer.pkl`, `models/metrics.json` and a confusion matrix image.

## 3. Run the web app

```bash
python app.py
```

Open http://127.0.0.1:5000

| Role | How |
|---|---|
| Guest | Analyse text without saving |
| User | Register, then every analysis is saved under **My history** |
| Admin | Login `admin` / `admin123` (change it in `app.py`). Upload dataset, retrain, view metrics and all recent checks |

JSON API: `POST /api/predict` with `{"text": "..."}`.

## Project structure

```
app.py              Flask routes, login, history, admin
train.py            Command line training
src/preprocess.py   Text cleaning (shared by training and prediction)
src/trainer.py      Dataset loading, training, evaluation
src/predict.py      Prediction, confidence, keywords
templates/, static/ Web pages and CSS
data/, models/      Dataset and saved model
```

## Notes for your report

- **Pipeline:** text input → cleaning (lowercase, remove URLs/HTML/punctuation) → TF-IDF (unigrams + bigrams, stop words removed) → classifier → label + probability.
- **Dataset leakage:** ISOT real articles contain "(Reuters)", which lets a model cheat. `preprocess.py` removes it.
- **Deployed model:** Logistic Regression, because it gives probabilities and its coefficients explain the keywords. Naive Bayes and Random Forest are trained for comparison.
- **Limitations:** the model learns writing style, not facts. It works best on news-style text similar to its training data, and should support, not replace, professional fact-checking.
- **Future work:** transformer models (BERT), fact database lookup, multilingual support.
- **Before deploying:** set the `SECRET_KEY` environment variable and change the admin password.
