"""Loads the trained model and returns label, confidence and influential keywords."""
import joblib
import numpy as np

from .preprocess import clean_text
from .trainer import MODEL_DIR


class Predictor:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "model.pkl")
        self.vec = joblib.load(MODEL_DIR / "vectorizer.pkl")
        self.features = np.array(self.vec.get_feature_names_out())

    def predict(self, text: str, top_k: int = 8) -> dict:
        cleaned = clean_text(text)
        X = self.vec.transform([cleaned])
        p_fake, p_real = self.model.predict_proba(X)[0]  # classes_ = [0, 1]
        is_real = p_real >= p_fake

        # Keyword contribution = tf-idf value * coefficient. Positive coef -> "genuine".
        coo = X.tocoo()
        contrib = coo.data * self.model.coef_[0][coo.col]
        sign = 1 if is_real else -1
        order = np.argsort(-(contrib * sign))[:top_k]
        keywords = [{"word": str(self.features[coo.col[i]]),
                     "weight": round(float(abs(contrib[i])), 3)}
                    for i in order if contrib[i] * sign > 0]

        warning = None
        if len(cleaned.split()) < 15:
            warning = "Very short text: the result is less reliable. Paste the full article if you can."

        return {
            "label": "Likely Genuine" if is_real else "Likely Fake",
            "is_real": bool(is_real),
            "confidence": round(float(max(p_fake, p_real)) * 100, 1),
            "prob_fake": round(float(p_fake) * 100, 1),
            "prob_real": round(float(p_real) * 100, 1),
            "keywords": keywords,
            "warning": warning,
        }
