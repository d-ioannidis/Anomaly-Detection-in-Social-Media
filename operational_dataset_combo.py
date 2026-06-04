import os
import time
import json
from pathlib import Path

import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"
INGEST_URL = f"{BASE_URL}/pipeline/ingest-and-run"

PROJECT_ROOT = Path(r"C:\Backup\Dimitris\Lulea\Thesis\Project").resolve()
OUTPUT_DIR = PROJECT_ROOT / "combined_api_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = ["wildfire", "forest fire", "brush fire", "evacuation order"]
OFFSETS = [0, 40, 80, 120]
LIMIT = 40

all_merged = []
run_log = []

def resolve_output_path(path_str):
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()

for query in QUERIES:
    for offset in OFFSETS:
        payload = {
            "mode": "search",
            "query": query,
            "limit": LIMIT,
            "resolve": True,
            "offset": offset,
            "min_keyword_matches": 1,
            "output_csv_name": f"{query.replace(' ', '_')}_{offset}.csv"
        }

        print(f"Running query='{query}' offset={offset} ...")

        try:
            resp = requests.post(INGEST_URL, json=payload, timeout=600)
            resp.raise_for_status()

            data = resp.json()
            print("FULL RESPONSE:")
            print(json.dumps(data, indent=2, ensure_ascii=False))

            manifest = data.get("manifest", {})
            output_files = manifest.get("output_files", {})
            merged_path_raw = output_files.get("merged_data_results")
            merged_path = resolve_output_path(merged_path_raw)

            # fallback if API does not return path
            if merged_path is None:
                merged_path = PROJECT_ROOT / "merged_data_results.csv"

            run_log.append({
                "query": query,
                "offset": offset,
                "status": "ok",
                "merged_path_raw": merged_path_raw,
                "merged_path_resolved": str(merged_path)
            })

            if merged_path.exists():
                df = pd.read_csv(merged_path)
                df["source_query"] = query
                df["source_offset"] = offset
                all_merged.append(df)
                print(f"  Loaded {len(df)} merged rows from {merged_path}")
            else:
                print("  No merged_data_results file found.")
                print(f"  Raw path from API: {merged_path_raw}")
                print(f"  Resolved path: {merged_path}")

        except Exception as e:
            run_log.append({
                "query": query,
                "offset": offset,
                "status": "error",
                "error": str(e)
            })
            print(f"  Error: {e}")

        time.sleep(2)

run_log_path = OUTPUT_DIR / "run_log.json"
with open(run_log_path, "w", encoding="utf-8") as f:
    json.dump(run_log, f, indent=2, ensure_ascii=False)

if all_merged:
    combined = pd.concat(all_merged, ignore_index=True)
    if "Tweet ID" in combined.columns:
        combined = combined.drop_duplicates(subset=["Tweet ID"], keep="first")
    else:
        combined = combined.drop_duplicates()

    combined_path = OUTPUT_DIR / "combined_merged_data_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined rows after deduplication: {len(combined)}")
    print(f"Saved combined file to: {combined_path}")
else:
    print("\nNo merged outputs were collected.")
    print(f"Check run log: {run_log_path}")