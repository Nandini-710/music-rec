from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from song_recommender.config import AppConfig
from song_recommender.models.retrieval import SimilarityIndex


class RecommendResponseItem(BaseModel):
    track_id: str
    score: float
    title: str | None = None
    artist: str | None = None


class RecommendResponse(BaseModel):
    query_track_id: str
    k: int
    results: list[RecommendResponseItem]


class HealthResponse(BaseModel):
    status: str = "ok"


def load_artifacts(cfg: AppConfig) -> tuple[SimilarityIndex, pd.DataFrame]:
    if not Path(cfg.paths.index_path).exists():
        raise FileNotFoundError(f"Missing index: {cfg.paths.index_path}")
    if not Path(cfg.paths.metadata_path).exists():
        raise FileNotFoundError(f"Missing metadata: {cfg.paths.metadata_path}")

    index: SimilarityIndex = joblib.load(cfg.paths.index_path)
    meta = pd.read_csv(cfg.paths.metadata_path)
    meta["track_id"] = meta["track_id"].astype(str)
    return index, meta


def create_app(cfg: AppConfig) -> FastAPI:
    app = FastAPI(title="Song Recommender", version="0.1.0")

    index, meta = load_artifacts(cfg)
    meta_by_id: dict[str, dict[str, Any]] = {r["track_id"]: r for r in meta.to_dict(orient="records")}
    pos: dict[str, int] = {tid: i for i, tid in enumerate(index.track_ids)}

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/recommend/{track_id}", response_model=RecommendResponse)
    def recommend(
        track_id: str,
        k: int = Query(default=10, ge=1, le=100),
    ) -> RecommendResponse:
        track_id = str(track_id)
        if track_id not in pos:
            raise HTTPException(status_code=404, detail="track_id not found in index")
        results = index.query_by_index(pos[track_id], k=k)

        items: list[RecommendResponseItem] = []
        for rid, score in results:
            m = meta_by_id.get(rid, {})
            items.append(
                RecommendResponseItem(
                    track_id=rid,
                    score=score,
                    title=m.get("title"),
                    artist=m.get("artist"),
                )
            )
        return RecommendResponse(query_track_id=track_id, k=k, results=items)

    return app

