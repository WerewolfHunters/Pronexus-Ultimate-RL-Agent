from __future__ import annotations

import math
import re
from statistics import mean, pstdev
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data.lexicons import (
    FAILURE_INDICATORS,
    HEDGE_PHRASES,
    KNOWN_TOOLS,
    TEMPORAL_ANCHORS,
    TRIBAL_VOCAB,
    VAGUE_TEMPORAL_PHRASES,
)

try:
    import nltk
    from nltk.tokenize import sent_tokenize

    _PUNKT_READY = True
except Exception:  # pragma: no cover
    sent_tokenize = None
    _PUNKT_READY = False

try:
    import spacy

    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:  # pragma: no cover
        _NLP = spacy.blank("en")
except Exception:  # pragma: no cover
    _NLP = None

_SENTIMENT = SentimentIntensityAnalyzer()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _sentences(text: str) -> list[str]:
    if _PUNKT_READY and sent_tokenize is not None:
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except LookupError:  # pragma: no cover
            pass
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _all_text(answers: Iterable[str]) -> str:
    return " ".join(a.strip() for a in answers if a and a.strip())


def score_hedging_density(answers: list[str]) -> float:
    text = _all_text(answers).lower()
    words = max(len(text.split()), 1)
    hits = sum(text.count(phrase) for phrase in HEDGE_PHRASES)
    density = hits / words
    return _clamp(density * 35)


def score_failure_narrative_absence(answers: list[str]) -> float:
    text = _all_text(answers).lower()
    hits = sum(text.count(token) for token in FAILURE_INDICATORS)
    if not answers:
        return 0.0
    normalized_presence = min(1.0, hits / max(len(answers), 1))
    return _clamp(1.0 - normalized_presence)


def score_temporal_vagueness(answers: list[str]) -> float:
    text = _all_text(answers).lower()
    anchor_hits = sum(text.count(t) for t in TEMPORAL_ANCHORS)
    vague_hits = sum(text.count(v) for v in VAGUE_TEMPORAL_PHRASES)

    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", text))
    date_like_hits = len(re.findall(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", text))

    ner_time_hits = 0
    if _NLP is not None:
        doc = _NLP(_all_text(answers))
        ner_time_hits = sum(1 for ent in getattr(doc, "ents", []) if ent.label_ in {"DATE", "TIME"})

    concrete_time = anchor_hits + year_hits + date_like_hits + ner_time_hits
    raw = (vague_hits + 1) / (concrete_time + 1)
    return _clamp(raw / 2.5)


def score_tribal_vocabulary(answers: list[str], domain: str = "data_engineering") -> float:
    text = _all_text(answers).lower()
    words = max(len(text.split()), 1)
    vocab = TRIBAL_VOCAB.get(domain, TRIBAL_VOCAB["data_engineering"])
    hits = sum(text.count(term) for term in vocab)
    insider_density = hits / words
    return _clamp(1.0 - insider_density * 45)


def score_tool_opinion_polarity(answers: list[str]) -> float:
    text = _all_text(answers)
    sentences = _sentences(text)
    tool_sentiments: list[float] = []

    for sentence in sentences:
        lower = sentence.lower()
        if any(tool in lower for tool in KNOWN_TOOLS):
            score = _SENTIMENT.polarity_scores(sentence)["compound"]
            tool_sentiments.append(abs(score))

    if not tool_sentiments:
        return 0.5

    avg_abs = float(mean(tool_sentiments))
    return _clamp(1.0 - avg_abs)


def score_answer_length_uniformity(answers: list[str]) -> float:
    lengths = [max(len(a.split()), 1) for a in answers if a and a.strip()]
    if len(lengths) < 2:
        return 0.0
    mu = mean(lengths)
    sigma = pstdev(lengths)
    cv = sigma / mu if mu else 0.0
    return _clamp(1.0 - (cv / 0.5))


def score_structural_symmetry(answers: list[str]) -> float:
    if not answers:
        return 0.0

    signatures = []
    for a in answers:
        lines = [ln.strip() for ln in a.splitlines() if ln.strip()]
        bullet_lines = sum(bool(re.match(r"^(\d+\.|[-*])\s+", ln)) for ln in lines)
        sentence_count = len(_sentences(a))
        signatures.append((bullet_lines > 0, min(sentence_count, 6)))

    unique = len(set(signatures))
    if unique == 1:
        return 1.0
    return _clamp(1.0 - (unique - 1) / max(len(answers) - 1, 1))


def score_depth_breadth_imbalance(answers: list[str]) -> float:
    filtered = [a.strip() for a in answers if a and a.strip()]
    if len(filtered) < 2:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(filtered)

    vocab_size = max(len(vectorizer.vocabulary_), 1)
    non_zero = float(matrix.count_nonzero())
    breadth = _clamp(non_zero / (len(filtered) * vocab_size) * 3)

    avg_len = mean(len(a.split()) for a in filtered)
    depth = _clamp(avg_len / 140)

    return _clamp(0.65 * breadth + 0.35 * (1.0 - depth))


def score_all_signals(answers: list[str], domain: str = "data_engineering") -> dict:
    scores = {
        "S1_hedging": score_hedging_density(answers),
        "S2_failure_absence": score_failure_narrative_absence(answers),
        "S3_temporal_vagueness": score_temporal_vagueness(answers),
        "S4_tribal_vocab": score_tribal_vocabulary(answers, domain=domain),
        "S5_tool_polarity": score_tool_opinion_polarity(answers),
        "S6_length_uniformity": score_answer_length_uniformity(answers),
        "S7_structural_symmetry": score_structural_symmetry(answers),
        "S8_depth_breadth": score_depth_breadth_imbalance(answers),
    }
    return {k: _clamp(v) for k, v in scores.items()}
