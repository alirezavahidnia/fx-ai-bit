from __future__ import annotations
from typing import List, Dict, Any
import re, datetime as dt

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _HAS_TF = True
except Exception:
    _HAS_TF = False

_HAWKISH = {"tighten","hike","inflationary","restrictive","hawkish"}
_DOVISH  = {"ease","cut","accommodative","support","dovish","stimulus"}

class FinBertSentiment:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        if not _HAS_TF:
            raise ImportError("transformers/torch not installed")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def score(self, text: str) -> float:
        inputs = self.tokenizer(text[:4096], return_tensors="pt", truncation=True)
        with torch.no_grad():
            probs = self.model(**inputs).logits.softmax(-1)[0].tolist()
        neg, neu, pos = probs
        return pos - neg  # [-1,1]

def hawk_dove_score(text: str) -> float:
    t = re.findall(r"[a-zA-Z]+", text.lower())
    if not t: return 0.0
    h = sum(1 for w in t if w in _HAWKISH)
    d = sum(1 for w in t if w in _DOVISH)
    tot = h + d
    return 0.0 if tot == 0 else (h - d) / tot

def aggregate_currency_sentiment(items: List[Dict[str, Any]], half_life_minutes: int = 180) -> Dict[str, float]:
    now = dt.datetime.utcnow()
    scores, weights = {}, {}
    for it in items:
        ts = it.get("timestamp", now)
        age_min = max((now - ts).total_seconds()/60.0, 0.0)
        decay = 0.5 ** (age_min / half_life_minutes)
        rel = float(it.get("reliability", 0.5))
        w = decay * rel
        s = float(it.get("sentiment", 0.0))
        for c in it.get("currencies", []):
            scores[c] = scores.get(c, 0.0) + w * s
            weights[c] = weights.get(c, 0.0) + w
    return {c: (scores[c] / max(weights[c], 1e-9)) for c in scores}
