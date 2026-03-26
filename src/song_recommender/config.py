from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    # config.py lives at: <repo_root>/src/song_recommender/config.py
    # so the repository root is 2 levels up from `song_recommender/` -> `src/` -> repo root.
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Paths:
    tracks_csv: Path
    processed_dir: Path
    artifacts_dir: Path
    index_path: Path
    metadata_path: Path
    tfidf_path: Path
    audio_scaler_path: Path


@dataclass(frozen=True)
class FeatureConfig:
    lyrics_max_features: int
    lyrics_ngram_range: tuple[int, int]
    lyrics_min_df: int
    audio_sample_rate: int
    audio_n_mfcc: int


@dataclass(frozen=True)
class FusionConfig:
    lyrics_weight: float
    audio_weight: float


@dataclass(frozen=True)
class AppConfig:
    paths: Paths
    features: FeatureConfig
    fusion: FusionConfig


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def load_config(path: str | Path) -> AppConfig:
    cfg = load_yaml(path)
    root = repo_root()

    tracks_csv = root / _as_path(cfg["data"]["tracks_csv"])
    processed_dir = root / _as_path(cfg["data"]["processed_dir"])
    artifacts_dir = root / _as_path(cfg["artifacts"]["dir"])

    index_path = root / _as_path(cfg["artifacts"]["index_path"])
    metadata_path = root / _as_path(cfg["artifacts"]["metadata_path"])
    tfidf_path = root / _as_path(cfg["artifacts"]["tfidf_path"])
    audio_scaler_path = root / _as_path(cfg["artifacts"]["audio_scaler_path"])

    feats = cfg["features"]
    lyrics = feats["lyrics"]
    audio = feats["audio"]
    fusion = cfg["fusion"]

    return AppConfig(
        paths=Paths(
            tracks_csv=tracks_csv,
            processed_dir=processed_dir,
            artifacts_dir=artifacts_dir,
            index_path=index_path,
            metadata_path=metadata_path,
            tfidf_path=tfidf_path,
            audio_scaler_path=audio_scaler_path,
        ),
        features=FeatureConfig(
            lyrics_max_features=int(lyrics["max_features"]),
            lyrics_ngram_range=(int(lyrics["ngram_range"][0]), int(lyrics["ngram_range"][1])),
            lyrics_min_df=int(lyrics["min_df"]),
            audio_sample_rate=int(audio["sample_rate"]),
            audio_n_mfcc=int(audio["n_mfcc"]),
        ),
        fusion=FusionConfig(
            lyrics_weight=float(fusion["lyrics_weight"]),
            audio_weight=float(fusion["audio_weight"]),
        ),
    )

