from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioFeatures:
    # shape: (2*n_mfcc,)
    mfcc_mean_std: np.ndarray


def extract_mfcc_mean_std(audio_path: str | Path, sample_rate: int, n_mfcc: int) -> AudioFeatures:
    """
    Extract MFCCs and return a fixed-length vector (mean + std per coefficient).
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)
    mu = np.mean(mfcc, axis=1)
    sd = np.std(mfcc, axis=1)
    vec = np.concatenate([mu, sd], axis=0).astype(np.float32)
    return AudioFeatures(mfcc_mean_std=vec)


def safe_audio_path(p: object) -> str:
    if p is None:
        return ""
    return str(p)

