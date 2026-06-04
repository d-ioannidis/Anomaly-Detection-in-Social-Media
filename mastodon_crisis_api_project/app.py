from __future__ import annotations

import os
import re
import json
import uuid
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from PipelineManager import PipelineManager

load_dotenv()

APP_NAME = "Crisis Mastodon Ingestion API"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "runtime_data"
RESULTS_DIR = BASE_DIR / "runtime_results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MASTODON_BASE_URL = os.getenv("MASTODON_BASE_URL", "https://mastodon.social").rstrip("/")
MASTODON_ACCESS_TOKEN = os.getenv("MASTODON_ACCESS_TOKEN", "")

app = FastAPI(title=APP_NAME, version="0.1.0")

CRISIS_KEYWORDS = [
    "earthquake", "flood", "wildfire", "fire", "hurricane", "tornado",
    "storm", "cyclone", "landslide", "tsunami", "drought", "evacuation",
    "explosion", "blackout", "outage", "war", "conflict", "shooting",
    "disaster", "crisis", "emergency", "rescue"
]


class TimelineRequest(BaseModel):
    limit: int = Field(40, ge=1, le=80)
    local: bool = False
    remote: bool = True
    only_media: bool = False
    min_keyword_matches: int = Field(1, ge=0, le=10)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    limit: int = Field(40, ge=1, le=80)
    resolve: bool = True
    offset: int = Field(0, ge=0)
    min_keyword_matches: int = Field(1, ge=0, le=10)


class NormalizeRequest(BaseModel):
    raw_json_path: str
    output_csv_name: Optional[str] = None


class IngestAndRunRequest(BaseModel):
    mode: str = Field("timeline", pattern="^(timeline|search)$")
    query: Optional[str] = None
    limit: int = Field(40, ge=1, le=80)
    local: bool = False
    remote: bool = True
    only_media: bool = False
    resolve: bool = True
    offset: int = Field(0, ge=0)
    min_keyword_matches: int = Field(1, ge=0, le=10)
    output_csv_name: Optional[str] = None


def auth_headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if MASTODON_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {MASTODON_ACCESS_TOKEN}"
    return headers


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def strip_html(text: str) -> str:
    text = unescape(str(text or ""))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def matched_keywords(text: str) -> List[str]:
    norm = normalize_text(text)
    return [kw for kw in CRISIS_KEYWORDS if kw in norm]


def infer_disaster_label(text: str) -> str:
    norm = normalize_text(text)
    priority = [
        "earthquake", "flood", "wildfire", "fire", "hurricane", "tornado",
        "storm", "cyclone", "landslide", "tsunami", "drought",
        "war", "conflict", "shooting", "explosion", "outage"
    ]
    for kw in priority:
        if kw in norm:
            return kw
    return "crisis"


def save_json(data: Any, prefix: str) -> str:
    path = DATA_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(path)


def save_framework_csv(rows: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
    if not filename:
        filename = f"mastodon_crisis_{uuid.uuid4().hex[:8]}.csv"
    path = DATA_DIR / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def mastodon_status_to_framework_row(status: Dict[str, Any]) -> Dict[str, Any]:
    account = status.get("account") or {}
    display_name = account.get("display_name") or account.get("username") or "mastodon_user"
    username = account.get("acct") or account.get("username") or "mastodon_user"
    created = status.get("created_at") or now_iso()

    text = strip_html(status.get("content", ""))
    spoiler = strip_html(status.get("spoiler_text", ""))
    merged_text = f"{spoiler}\n\n{text}".strip() if spoiler else text

    tags = status.get("tags") or []
    tag_names = [f"#{t.get('name')}" for t in tags if t.get("name")]

    keywords = matched_keywords(merged_text)
    disaster = infer_disaster_label(merged_text)

    return {
        "Name": display_name,
        "UserName": username,
        "Timestamp": created,
        "Verified": bool(account.get("bot", False)),
        "Tweets": merged_text,
        "Comments": int(status.get("replies_count", 0) or 0),
        "Retweets": int(status.get("reblogs_count", 0) or 0),
        "Likes": int(status.get("favourites_count", 0) or 0),
        "Impressions": 0,
        "Tags": " ".join(tag_names) if tag_names else (" ".join(f"#{k}" for k in keywords) if keywords else "#crisis"),
        "Tweet Link": status.get("url") or "",
        "Tweet ID": status.get("id") or "",
        "Disaster": disaster,
    }


def fetch_public_timeline(limit: int, local: bool, remote: bool, only_media: bool) -> List[Dict[str, Any]]:
    url = f"{MASTODON_BASE_URL}/api/v1/timelines/public"
    params = {
        "limit": limit,
        "local": str(local).lower(),
        "remote": str(remote).lower(),
        "only_media": str(only_media).lower(),
    }
    resp = requests.get(url, params=params, headers=auth_headers(), timeout=20)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Mastodon timeline error: {resp.status_code} {resp.text}")
    return resp.json()


def fetch_search_statuses(query: str, limit: int, resolve: bool, offset: int) -> List[Dict[str, Any]]:
    url = f"{MASTODON_BASE_URL}/api/v2/search"
    params = {
        "q": query,
        "type": "statuses",
        "limit": limit,
        "resolve": str(resolve).lower(),
        "offset": offset,
    }
    resp = requests.get(url, params=params, headers=auth_headers(), timeout=20)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Mastodon search error: {resp.status_code} {resp.text}")
    payload = resp.json()
    return payload.get("statuses", [])


def run_framework(csv_path: str) -> Dict[str, Any]:
    config = {"data_source": csv_path}
    pipeline = PipelineManager(config)
    results = pipeline.run()

    manifest = {
        "timestamp": now_iso(),
        "input_csv": csv_path,
        "results_keys": list(results.keys()) if isinstance(results, dict) else [],
        "output_files": {
            "anomaly_scores": str(BASE_DIR / "anomaly_scores.csv"),
            "fact_check_results": str(BASE_DIR / "fact_check_results.csv"),
            "anomaly_results": str(BASE_DIR / "anomaly_results.csv"),
            "sentiment_emotion_scores": str(BASE_DIR / "sentiment_emotion_scores.csv"),
        },
    }

    with open(RESULTS_DIR / "latest_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "mastodon_base_url": MASTODON_BASE_URL,
        "token_configured": str(bool(MASTODON_ACCESS_TOKEN)).lower(),
    }


@app.post("/collector/mastodon/timeline")
def collector_mastodon_timeline(req: TimelineRequest) -> Dict[str, Any]:
    raw_items = fetch_public_timeline(req.limit, req.local, req.remote, req.only_media)
    normalized_rows: List[Dict[str, Any]] = []

    for status in raw_items:
        text = strip_html(status.get("content", ""))
        if len(set(matched_keywords(text))) >= req.min_keyword_matches:
            normalized_rows.append(mastodon_status_to_framework_row(status))

    raw_json_path = save_json(raw_items, prefix="mastodon_timeline_raw")
    return {
        "mastodon_base_url": MASTODON_BASE_URL,
        "raw_count": len(raw_items),
        "crisis_filtered_count": len(normalized_rows),
        "raw_json_path": raw_json_path,
        "sample_framework_rows": normalized_rows[:3],
    }


@app.post("/collector/mastodon/search")
def collector_mastodon_search(req: SearchRequest) -> Dict[str, Any]:
    raw_items = fetch_search_statuses(req.query, req.limit, req.resolve, req.offset)
    normalized_rows: List[Dict[str, Any]] = []

    for status in raw_items:
        text = strip_html(status.get("content", ""))
        if len(set(matched_keywords(text))) >= req.min_keyword_matches:
            normalized_rows.append(mastodon_status_to_framework_row(status))

    raw_json_path = save_json(raw_items, prefix="mastodon_search_raw")
    return {
        "mastodon_base_url": MASTODON_BASE_URL,
        "query": req.query,
        "raw_count": len(raw_items),
        "crisis_filtered_count": len(normalized_rows),
        "raw_json_path": raw_json_path,
        "sample_framework_rows": normalized_rows[:3],
    }


@app.post("/preprocess/mastodon-to-framework")
def preprocess_mastodon_to_framework(req: NormalizeRequest) -> Dict[str, Any]:
    raw_path = Path(req.raw_json_path)
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail=f"Raw JSON file not found: {req.raw_json_path}")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    rows = [mastodon_status_to_framework_row(item) for item in raw_items]
    csv_path = save_framework_csv(rows, filename=req.output_csv_name)

    return {
        "rows_written": len(rows),
        "output_csv_path": csv_path,
    }


@app.post("/pipeline/ingest-and-run")
def pipeline_ingest_and_run(req: IngestAndRunRequest) -> Dict[str, Any]:
    if req.mode == "timeline":
        raw_items = fetch_public_timeline(req.limit, req.local, req.remote, req.only_media)
    else:
        if not req.query:
            raise HTTPException(status_code=400, detail="query is required when mode='search'")
        raw_items = fetch_search_statuses(req.query, req.limit, req.resolve, req.offset)

    rows: List[Dict[str, Any]] = []
    for status in raw_items:
        text = strip_html(status.get("content", ""))
        if len(set(matched_keywords(text))) >= req.min_keyword_matches:
            rows.append(mastodon_status_to_framework_row(status))

    if not rows:
        raise HTTPException(status_code=404, detail="No Mastodon posts matched the crisis filter.")

    csv_path = save_framework_csv(rows, filename=req.output_csv_name)

    try:
        manifest = run_framework(csv_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Framework run failed: {exc}") from exc

    return {
        "mastodon_base_url": MASTODON_BASE_URL,
        "mode": req.mode,
        "query": req.query,
        "rows_exported": len(rows),
        "framework_input_csv": csv_path,
        "manifest": manifest,
    }


@app.get("/results/latest")
def results_latest() -> Dict[str, Any]:
    manifest_path = RESULTS_DIR / "latest_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No pipeline run manifest found yet.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    preview = {}
    for name, path_str in manifest.get("output_files", {}).items():
        path = Path(path_str)
        if path.exists():
            try:
                preview[name] = pd.read_csv(path).head(5).to_dict(orient="records")
            except Exception:
                preview[name] = f"Could not preview {path_str}"

    return {"manifest": manifest, "preview": preview}
