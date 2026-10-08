"""Text cleaning shared by training (train.py / admin retrain) and prediction (app).

IMPORTANT: Training and prediction MUST use this same function.
"""
import re

_HTML = re.compile(r"<.*?>")
_URL = re.compile(r"http\S+|www\.\S+")
# The ISOT dataset leaks the label: real articles contain "(Reuters)".
# Removing it stops the model from "cheating" and gives realistic scores.
_REUTERS = re.compile(r"\(?\breuters\b\)?", re.I)
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, strip HTML/URLs/source tags/punctuation/digits, normalise spaces."""
    text = str(text)
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _REUTERS.sub(" ", text)
    text = text.lower()
    text = _NON_ALPHA.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
