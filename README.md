# JIT Talent AI Score

Slack `# JD` ingestion + Exa sourcing + AI scoring pipeline.

Additional isolated workflow:
- thread reply to a scored CSV message with a numeric threshold (for example `5`)
- filters `AI Score >= threshold`
- enriches emails via SaleSQL (only emails where `type=Direct` and valid: `is_valid=true` or fallback `status=Valid`)
- verifies via Reoon then BounceBan
- creates Instantly campaign and adds filtered leads
- posts progress/final summary in the same Slack thread

## Deployment Architecture

- Vercel (`app.py`) only handles Slack Events API:
  - verifies Slack signature
  - filters channel + `# JD` format
  - enqueues job to Redis/RQ
  - returns fast `200` to Slack
- Vercel function entrypoint file: `api/index.py` (re-exports `app` from `app.py`)
- Dedicated worker (`worker.py`) runs heavy pipeline:
  - generate 15 Exa prompts
  - fetch Exa results
  - CSV + dedup + score
  - post step updates + final CSV back to Slack

## JD Message Format

Supported header formats:
- `# JD` (existing format)
- `# JD <name>` (new, optional name)
- `# JD: <name>` and `# JD | <name>` are also accepted

When a name is provided, it is used in:
- processing/progress/failure Slack messages
- final upload comment (`JD Name: ...`)
- output CSV filenames inside the worker run (slugged)

## Required Environment Variables

Set these in both:
- Vercel project env
- Worker service env
- (or locally in `.env`, starting from `.env.example`)

- `OPENAI_API_KEY`
- `EXA_API_KEY`
- `SLACK_SIGNING_SECRET`
- `SLACK_BOT_TOKEN` (preferred) or `SLACK_USER_TOKEN`
- `REDIS_URL`
- `SALESQL_API_KEY` (thread-reply enrichment workflow)
- `REOON_API_KEY` (thread-reply enrichment workflow)
- `BOUNCEBAN_API_KEY` (thread-reply enrichment workflow)
- `INSTANTLY_API_KEY` (thread-reply enrichment workflow)

Optional:
- `SLACK_CHANNEL_ID` (default `C0AF5RGPMEW`)
- `RQ_QUEUE_NAME` (default `jd-pipeline`)
- `RQ_JOB_TIMEOUT` (default `7200`)
- `RQ_WITH_SCHEDULER` (default `false`; keep disabled unless you use scheduled RQ jobs)
- `SLACK_EVENT_TTL_SECONDS` (default `86400`)
- `SLACK_EVENT_DEDUP_ENABLED` (default `true`)
- `SLACK_EVENT_DEDUP_FAIL_OPEN` (default `true`)
- `SLACK_ACK_QUEUE_UNAVAILABLE` (default `true`; returns `200` degraded responses to Slack when Redis is unavailable)
- `EXA_CONCURRENT_SEARCHES` (default `5`)
- `SCORING_CONCURRENT_CALLS` (default `24`)
- `SCORING_MAX_INFLIGHT_FACTOR` (default `2`; max in-flight scoring futures = `SCORING_CONCURRENT_CALLS * factor`)
- `SCORE_PROGRESS_NOTIFY_EVERY` (default `50`)
- `SCORE_INITIAL_SECONDS_PER_CANDIDATE` (default `2.5`)
- `INSTANTLY_CAMPAIGN_TIMEZONE` (default `Asia/Kolkata`)
- `INSTANTLY_CAMPAIGN_START_HOUR` (default `08:00`)
- `INSTANTLY_CAMPAIGN_END_HOUR` (default `18:00`)
- `INSTANTLY_CAMPAIGN_DURATION_DAYS` (default `30`)
- `REOON_MAX_WAIT_SECONDS` (default `600`)
- `REOON_POLL_INTERVAL_SECONDS` (default `15`)
- `BOUNCEBAN_MAX_WAIT_SECONDS` (default `600`)
- `BOUNCEBAN_POLL_INTERVAL_SECONDS` (default `15`)
- `INSTANTLY_FAIL_FAST` (default `false`; abort immediately on first Instantly lead add error)
- `THREAD_RESULT_STRICT` (default `true`; only allow thread-reply enrichment when thread root text matches result message prefix)
- `RESULT_MESSAGE_PREFIX` (default `AI-scored candidates CSV for this JD`)

## Slack App Setup

1. Enable **Event Subscriptions**.
2. Request URL: `https://<your-vercel-domain>/slack/events`
3. Subscribe bot events:
- `message.channels`
4. OAuth scopes:
- `chat:write`
- `files:write`
- `files:read` (required for thread-reply enrichment to download Slack CSV)
- `channels:history`
5. Reinstall app after scope/event changes.
6. Invite bot to the target channel.

If channel is private, also add:
- Event: `message.groups`
- Scope: `groups:history`

## Your Flow (Local -> GitHub -> Vercel)

1. Push repo to GitHub.
2. Import repo in Vercel.
3. Configure env vars in Vercel.
4. Deploy.

Then deploy worker separately (Railway/Render/Fly/ECS/etc.) with same env vars plus `REDIS_URL`.
For Railway, this repo includes `nixpacks.toml` so build installs dependencies and start runs `python -u worker.py`.

## Worker Run Command

```bash
python worker.py
```

Railway expected commands:
- Build: `pip install -r requirements.txt` (or leave default with `nixpacks.toml`)
- Start: `python -u worker.py`

For one-shot processing (CI/manual):

```bash
python worker.py --burst
```

## Local Development

Install:

```bash
pip install -r requirements.txt
```

Run webhook API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Run worker:

```bash
python worker.py
```

Endpoints:
- `GET /healthz`
- `POST /slack/events`
- `GET /flow` (interactive logic map editor)
- `GET /api/flow?slot=proposed|current` (load flow JSON, default `proposed`)
- `PUT /api/flow?slot=proposed|current` (save flow JSON, default `proposed`)
- `POST /api/flow/reset?slot=proposed|current` (reset slot; proposed resets from current)
- `POST /api/flow/copy-current-to-proposed` (overwrite proposed with current)
- `POST /api/flow/promote-proposed-to-current` (promote proposed to current and sync both)

## Flow Editor (Visual Pipeline)

Use `/flow` to view and edit the pipeline as a node graph (n8n-style) without touching code first.
The editor is now designed around two files:
- `current` = implemented baseline
- `proposed` = your planning/editing workspace

Main capabilities:
- drag nodes to reposition
- connect/disconnect edges visually
- add/remove nodes
- `Add Node` creates a `proposed` node
- edit node label/description in inspector
- toggle between `proposed` and `current` views
- `current` opens in read-only mode
- save/reload/reset `proposed`
- copy `current -> proposed`
- copy full flow JSON for sharing/review

Suggested workflow for this app:
1. Open `/flow`.
2. Work only in `proposed` mode (default).
3. Add proposed nodes to represent planned changes (for example, a new validation or approval stage).
4. Connect changed nodes where they should sit in the pipeline.
5. Save proposed flow.
6. During implementation, Codex maps `proposed` nodes to concrete categories (decision/data/action/etc.) and then syncs `proposed -> current`.

Persistence:
- Default current file path: `flow-current.json` in project root.
- Default proposed file path: `flow-proposed.json` in project root.
- Legacy fallback: `flow-definition.json` is still read for compatibility.
- If paths are not writable (for example in serverless runtime), it falls back to `/tmp/flow-current.json` and `/tmp/flow-proposed.json`.
- Optional overrides:
  - `FLOW_CURRENT_STORAGE_PATH=/custom/path/current.json`
  - `FLOW_PROPOSED_STORAGE_PATH=/custom/path/proposed.json`
  - Legacy `FLOW_STORAGE_PATH` continues to work as proposed-path fallback.

## Notes

- Pipeline outputs are written under `/tmp` in worker runs and cleaned up after each job.
- Step updates are posted to Slack at each stage completion.
- The 15 generated Exa prompts are uploaded to Slack as a `.txt` file per run.
