# Song Recommendation System (Lyrics + Audio)

Hybrid song recommender that combines **lyric analysis** (TF-IDF) and **music/audio analysis** (MFCC statistics) to recommend similar songs.

## Project layout

- `data/raw/` : put your raw dataset here (audio + lyrics metadata)
- `data/processed/` : generated feature matrices + ids
- `models/` : saved vectorizers/scalers and indexes
- `src/song_recommender/` : library code
- `scripts/` : runnable entry points (build index, run API)
- `configs/` : configuration (paths, weights)
- `tests/` : basic tests

## Expected input data format

Create a CSV at `data/raw/tracks.csv`:

Columns:
- `track_id` (string, unique)
- `title` (string)
- `artist` (string)
- `lyrics` (string, can be empty)
- `audio_path` (string path to audio file; relative to repo root is ok)

Example row:
`123,Blinding Lights,The Weeknd,"I said, ooh...","data/raw/audio/123.mp3"`

Audio formats: wav/mp3/flac (whatever `librosa` can read).

## Quickstart (Windows / PowerShell)

```powershell
cd C:\Users\Vivek\Desktop\song-recommendation-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m song_recommender.scripts.convert_dataset --input "data/raw/your_dataset.csv" --col-artist artist --col-title song --col-lyrics text --output "data/raw/tracks.csv"
python -m song_recommender.scripts.build_index --config configs/default.yaml
python -m song_recommender.scripts.run_api --config configs/default.yaml
```

Then open:
- http://127.0.0.1:8000/docs

## How it works (high level)

- Lyrics: `TfidfVectorizer` produces sparse vectors.
- Audio: MFCC features are extracted using `librosa`, aggregated (mean/std).
- Fusion: weighted concatenation after scaling audio features.
- Retrieval: cosine similarity over the fused vectors.

## Notes

- This is a solid starter template. For better quality you can later swap TF-IDF for transformer embeddings (e.g. sentence transformers) and MFCC for VGGish/OpenL3/CLAP embeddings.
