from __future__ import annotations

import argparse

from song_recommender.config import load_config
from song_recommender.pipeline.build_index import build_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    build_index(cfg)
    print(f"Saved index to: {cfg.paths.index_path}")
    print(f"Saved metadata to: {cfg.paths.metadata_path}")


if __name__ == "__main__":
    main()

