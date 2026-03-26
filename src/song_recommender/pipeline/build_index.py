from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from song_recommender.config import AppConfig
from song_recommender.features.audio import extract_mfcc_mean_std, safe_audio_path
from song_recommender.features.lyrics import build_lyrics_vectorizer, safe_lyrics
from song_recommender.models.retrieval import SimilarityIndex


@dataclass(frozen=True)
class BuildResult:
    index: SimilarityIndex
    metadata: pd.DataFrame


def _ensure_dirs(cfg: AppConfig) -> None:
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)


def build_index(cfg: AppConfig) -> BuildResult:
    _ensure_dirs(cfg)

    tracks = pd.read_csv(cfg.paths.tracks_csv)
    required = {"track_id", "title", "artist", "lyrics", "audio_path"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks.csv missing columns: {sorted(missing)}")

    tracks["track_id"] = tracks["track_id"].astype(str)
    tracks["lyrics"] = tracks["lyrics"].apply(safe_lyrics)
    tracks["audio_path"] = tracks["audio_path"].apply(safe_audio_path)

    # Lyrics features (sparse)
    lyric_vec = build_lyrics_vectorizer(
        max_features=cfg.features.lyrics_max_features,
        ngram_range=cfg.features.lyrics_ngram_range,
        min_df=cfg.features.lyrics_min_df,
    )
    X_lyrics = lyric_vec.fit_transform(tracks["lyrics"].tolist())

    # Weighted fusion:
    # - if `audio_weight == 0`, skip all audio extraction to speed up lyrics-only runs
    lyrics_w = float(cfg.fusion.lyrics_weight)
    audio_w = float(cfg.fusion.audio_weight)
    X_lyrics_w = X_lyrics.multiply(lyrics_w)

    if audio_w == 0.0:
        X = X_lyrics_w.tocsr()
        scaler = None
    else:
        # Audio features (dense)
        audio_vecs: list[np.ndarray] = []
        for p in tracks["audio_path"].tolist():
            if not p:
                audio_vecs.append(np.zeros((2 * cfg.features.audio_n_mfcc,), dtype=np.float32))
                continue
            ap = Path(p)
            if not ap.is_absolute():
                ap = cfg.paths.tracks_csv.parents[2] / ap  # repo root relative paths
            try:
                feats = extract_mfcc_mean_std(
                    ap, sample_rate=cfg.features.audio_sample_rate, n_mfcc=cfg.features.audio_n_mfcc
                )
                audio_vecs.append(feats.mfcc_mean_std)
            except Exception:
                audio_vecs.append(np.zeros((2 * cfg.features.audio_n_mfcc,), dtype=np.float32))

        X_audio = np.vstack(audio_vecs).astype(np.float32)
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_audio_scaled = scaler.fit_transform(X_audio).astype(np.float32)

        X_audio_w = X_audio_scaled * audio_w
        X = sparse.hstack([X_lyrics_w, sparse.csr_matrix(X_audio_w)], format="csr")
    index = SimilarityIndex(X=X, track_ids=tracks["track_id"].tolist())

    metadata = tracks[["track_id", "title", "artist", "audio_path"]].copy()

    joblib.dump(index, cfg.paths.index_path)
    joblib.dump(lyric_vec, cfg.paths.tfidf_path)
    if scaler is not None:
        joblib.dump(scaler, cfg.paths.audio_scaler_path)
    metadata.to_csv(cfg.paths.metadata_path, index=False)

    return BuildResult(index=index, metadata=metadata)

