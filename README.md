# Song Recommendation System (Lyrics + Audio)

Hybrid song recommender that combines **lyric analysis** (TF-IDF) and **music/audio analysis** (MFCC statistics) to recommend similar songs.

## Project layout

- `data/raw/` : raw dataset here (audio + lyrics metadata)
- `data/processed/` : generated feature matrices + ids
- `models/` : vectorizers/scalers and indexes
- `src/song_recommender/` : library code
- `scripts/` : runnable entry points
- `configs/` : configuration (paths, weights)
- `tests/` : basic tests

## input data format

Columns:
- `track_id` (string, unique)
- `title` (string)
- `artist` (string)
- `lyrics` (string, can be empty)
- `audio_path` (string path to audio file; relative to repo root is ok)

Example row:
`123,Blinding Lights,The Weeknd,"I said, ooh...","data/raw/audio/123.mp3"`

Audio formats: wav/mp3/flac 

## How it works (high level)

- Lyrics: `TfidfVectorizer` produces sparse vectors.
- Audio: MFCC features are extracted using `librosa`, aggregated (mean/std).
- Fusion: weighted concatenation after scaling audio features.
- Retrieval: cosine similarity over the fused vectors.


