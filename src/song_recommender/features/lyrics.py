from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class LyricsVectorizer:
    vectorizer: TfidfVectorizer

    def fit_transform(self, lyrics: list[str]):
        return self.vectorizer.fit_transform(lyrics)

    def transform(self, lyrics: list[str]):
        return self.vectorizer.transform(lyrics)


def build_lyrics_vectorizer(max_features: int, ngram_range: tuple[int, int], min_df: int) -> LyricsVectorizer:
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words="english",
    )
    return LyricsVectorizer(vectorizer=vec)


def safe_lyrics(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    return str(text)

