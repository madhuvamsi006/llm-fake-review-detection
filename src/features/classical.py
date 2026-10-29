"""
Classical ML feature pipeline for XGBoost:
  - TF-IDF n-grams (unigrams + bigrams)
  - Stylometric features: avg word length, sentence count, TTR, punctuation counts
  - Readability scores: Flesch Reading Ease, Gunning Fog
  - POS tag distribution
"""
import numpy as np
import pandas as pd
import nltk
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix

for resource in ["punkt", "averaged_perceptron_tagger", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


class StylometricFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(text) for text in X])

    def _extract(self, text: str) -> list:
        words = word_tokenize(text)
        sents = sent_tokenize(text)
        n_words = max(len(words), 1)
        n_sents = max(len(sents), 1)

        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        ttr = len(set(words)) / n_words                    # type-token ratio
        avg_sent_len = n_words / n_sents
        exclamation = text.count("!")
        question = text.count("?")
        commas = text.count(",")
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        flesch = textstat.flesch_reading_ease(text)
        fog = textstat.gunning_fog(text)
        flesch_kincaid = textstat.flesch_kincaid_grade(text)

        tags = pos_tag(words)
        tag_counts = {"NN": 0, "VB": 0, "JJ": 0, "RB": 0}
        for _, tag in tags:
            for key in tag_counts:
                if tag.startswith(key):
                    tag_counts[key] += 1
        pos_ratios = [v / n_words for v in tag_counts.values()]

        return [
            avg_word_len, ttr, avg_sent_len, n_words, n_sents,
            exclamation, question, commas, uppercase_ratio,
            flesch, fog, flesch_kincaid,
        ] + pos_ratios


def build_classical_pipeline(
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
) -> dict:
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        stop_words="english",
    )
    stylo = StylometricFeatures()
    return {"tfidf": tfidf, "stylo": stylo}


def fit_transform_classical(texts_train, texts_val=None, texts_test=None, **kwargs):
    pipes = build_classical_pipeline(**kwargs)
    tfidf, stylo = pipes["tfidf"], pipes["stylo"]

    X_tfidf_train = tfidf.fit_transform(texts_train)
    X_stylo_train = csr_matrix(stylo.fit_transform(texts_train))
    X_train = hstack([X_tfidf_train, X_stylo_train])

    results = {"train": X_train, "tfidf": tfidf, "stylo": stylo}

    if texts_val is not None:
        X_val = hstack([tfidf.transform(texts_val), csr_matrix(stylo.transform(texts_val))])
        results["val"] = X_val

    if texts_test is not None:
        X_test = hstack([tfidf.transform(texts_test), csr_matrix(stylo.transform(texts_test))])
        results["test"] = X_test

    return results
