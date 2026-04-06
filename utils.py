"""
Shared NLP preprocessing utilities.
Used across all exercises in nlp-definition-analysis.
"""

import re
import sys
import nltk
from nltk.corpus import stopwords


def get_stopwords(language: str = "english") -> set:
    """Return the NLTK stopword set for the given language."""
    try:
        return set(stopwords.words(language))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words(language))


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from text."""
    return re.sub(r"[^\w\s]", "", text)


def remove_stopwords(tokens: list, lang: str = "english") -> list:
    """Remove stopwords from a list of tokens."""
    sw = get_stopwords(lang)
    return [t for t in tokens if t.lower() not in sw]


def tokenize(text: str) -> list:
    """Lowercase and split text into word tokens."""
    return text.lower().split()


def preprocess(text: str, lang: str = "english") -> list:
    """Full preprocessing pipeline: remove punctuation, tokenize, remove stopwords."""
    text = remove_punctuation(text)
    tokens = tokenize(text)
    return remove_stopwords(tokens, lang)


# Aliases for backward compatibility with individual notebooks
pre_processing = preprocess
bag_of_words = preprocess
