from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def _stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def repo_root() -> Path:
    # convert_dataset.py lives at: <repo_root>/src/song_recommender/scripts/convert_dataset.py
    return Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert your dataset CSV into data/raw/tracks.csv")
    ap.add_argument("--input", required=True, help="Input CSV/TSV file containing artist/song/lyrics columns")
    ap.add_argument("--output", default="data/raw/tracks.csv", help="Output CSV path (repo-root relative ok)")
    ap.add_argument("--sep", default=",", help="CSV separator (default ','). Use '\\t' for TSV.")

    ap.add_argument("--col-artist", default="artist")
    ap.add_argument("--col-title", default="song")
    ap.add_argument("--col-lyrics", default="text")

    ap.add_argument("--col-audio-path", default="", help="Optional column containing local/relative audio paths")
    ap.add_argument("--col-track-id", default="", help="Optional column for unique track_id")

    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not out.is_absolute():
        # make output relative to repo root
        out = repo_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, sep=args.sep)

    for c in [args.col_artist, args.col_title, args.col_lyrics]:
        if c and c not in df.columns:
            raise ValueError(f"Missing required column `{c}`. Available: {list(df.columns)}")

    # Normalize strings and handle missing values.
    artist = df[args.col_artist].fillna("").astype(str)
    title = df[args.col_title].fillna("").astype(str)
    lyrics = df[args.col_lyrics].fillna("").astype(str)

    if args.col_audio_path and args.col_audio_path in df.columns:
        audio_path = df[args.col_audio_path].fillna("").astype(str)
    else:
        audio_path = pd.Series([""] * len(df), index=df.index)

    if args.col_track_id and args.col_track_id in df.columns:
        track_id = df[args.col_track_id].fillna("").astype(str)
    else:
        # Prefer a stable ID from (artist, title); append a row-specific suffix if duplicates occur.
        base = artist.str.strip() + " - " + title.str.strip()
        track_id = base + "_" + pd.Series([_stable_hash(f"{b}_{i}") for i, b in enumerate(base)], index=df.index)

    out_df = pd.DataFrame(
        {
            "track_id": track_id,
            "title": title,
            "artist": artist,
            "lyrics": lyrics,
            "audio_path": audio_path,
        }
    )
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df):,} rows to: {out}")


if __name__ == "__main__":
    main()

