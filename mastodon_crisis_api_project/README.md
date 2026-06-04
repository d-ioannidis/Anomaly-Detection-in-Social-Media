# Mastodon Crisis API Project

This project adds a lightweight Mastodon-facing API in front of your existing crisis-management framework.

## Structure

```text
mastodon_crisis_api_project/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── runtime_data/
└── runtime_results/
```

## What it does

It mirrors the diagrammed flow:

1. **Collector**
   - Pull public posts from a Mastodon instance public timeline, or
   - search statuses by query

2. **Preprocessing Unit**
   Converts Mastodon posts into the same CSV schema your framework already expects:
   - Name
   - UserName
   - Timestamp
   - Verified
   - Tweets
   - Comments
   - Retweets
   - Likes
   - Impressions
   - Tags
   - Tweet Link
   - Tweet ID
   - Disaster

3. **Anomaly Detection / Framework Run**
   Exports the normalized Mastodon data to CSV and passes it into your existing `PipelineManager`.

4. **Results Preview**
   Exposes the latest generated CSV outputs from the framework.

## Important integration note

This API assumes your existing files are available in the same project environment:

- `PipelineManager.py`
- `Data_Collector.py`
- `Data_Preprocessor.py`
- `Anomaly_Detection.py`
- `Fact_Checker.py`
- `Feedback.py`
- `Metrics_Evaluation.py`
- `NLP_Engine.py`

The API does not replace your framework. It feeds data into it.

## Setup

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your Mastodon instance:

```bash
cp .env.example .env
```

Example:

```env
MASTODON_BASE_URL=https://mastodon.social
MASTODON_ACCESS_TOKEN=
```

`MASTODON_ACCESS_TOKEN` is optional for public timeline use on many instances, but some instances or search behaviors may work better with a token.

## Run

```bash
uvicorn app:app --reload
```

## Endpoints

### Health check
`GET /health`

### Collector: pull public timeline
`POST /collector/mastodon/timeline`

Example body:

```json
{
  "limit": 30,
  "local": false,
  "remote": true,
  "only_media": false,
  "min_keyword_matches": 1
}
```

### Collector: search statuses
`POST /collector/mastodon/search`

Example body:

```json
{
  "query": "wildfire evacuation",
  "limit": 30,
  "resolve": true,
  "offset": 0,
  "min_keyword_matches": 1
}
```

### Preprocess raw JSON to framework CSV
`POST /preprocess/mastodon-to-framework`

Example body:

```json
{
  "raw_json_path": "runtime_data/mastodon_search_raw_12345678.json",
  "output_csv_name": "mastodon_wildfire_batch.csv"
}
```

### Full pipeline run
`POST /pipeline/ingest-and-run`

Example body using search:

```json
{
  "mode": "search",
  "query": "earthquake rescue emergency",
  "limit": 50,
  "resolve": true,
  "offset": 0,
  "min_keyword_matches": 1,
  "output_csv_name": "mastodon_earthquake_run.csv"
}
```

Example body using timeline:

```json
{
  "mode": "timeline",
  "limit": 50,
  "local": false,
  "remote": true,
  "only_media": false,
  "min_keyword_matches": 1,
  "output_csv_name": "mastodon_timeline_run.csv"
}
```

### Get latest results
`GET /results/latest`

## Output directories

- `runtime_data/` holds collected JSON and exported framework CSVs
- `runtime_results/` holds the latest manifest

## Notes

- Mastodon HTML status content is cleaned into plain text before export.
- `Impressions` are filled with `0` because that field is not available in this simple ingestion flow.
- `Retweets` maps to Mastodon reblogs.
- The downstream fact-checking and anomaly modules remain unchanged.
