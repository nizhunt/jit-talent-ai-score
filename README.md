# JIT Talent AI Score

Slack `# JD` ingestion + Exa sourcing + AI scoring pipeline.

Additional isolated workflow:
- thread reply to a scored result message with a Google Sheet link and an exact integer threshold `1` to `10` and no other text (for example `5`)
- filters `AI Score >= threshold`
- enriches emails via SaleSQL (only emails where `type=Direct` and valid: `is_valid=true` or fallback `status=Valid`)
- verifies via Reoon then BounceBan
- creates Instantly campaign and adds filtered leads
- sets Instantly lead `personalization` from CSV `AI Email` when present (fallback: LinkedIn line)
- posts progress/final summary in the same Slack thread

## Deployment Architecture

- Vercel (`app.py`) only handles Slack Events API:
  - verifies Slack signature
  - filters channel + `# JD` format
  - enqueues source-stage job to Redis/RQ
  - returns fast `200` to Slack
- Vercel function entrypoint file: `api/index.py` (re-exports `app` from `app.py`)
- Dedicated workers (`worker.py`) run the heavy pipeline in two stages:
  - Source job (`process_jd_source_job`):
    - widen JD
    - generate Exa queries
    - fetch Exa results
    - flatten/combine CSV
    - deduplicate
    - upload handoff artifacts to Railway Storage Bucket (S3-compatible)
    - enqueue score job with lightweight payload (`run_id`, metadata)
  - Score job (`process_jd_score_job`):
    - download handoff artifacts by `run_id`
    - score candidates
    - write Google Sheet
    - post final Slack summary
    - dashboard logging

## JD Message Format

Canonical command reference:
- `SLACK_COMMANDS.md` (keep this updated whenever Slack command parsing changes)

Supported header formats:
- `# JD` (existing format)
- `# JD <name>` (new, optional name)
- `# JD: <name>` and `# JD | <name>` are also accepted
- `# JD-Quick` (low-cost test mode)
- `# JD-Quick <name>` and `# JD-Quick: <name>` are also accepted
- `# JD-Test` variants are still accepted as a legacy alias

`# JD-Quick` behavior:
- skips widening/query-generation and sends the raw JD text directly as the only Exa query
- fetches a maximum of 100 sourced profiles
- then runs the rest of the pipeline unchanged (dedup, scoring, Google Sheet, Slack summary)

When a name is provided, it is used in:
- processing/progress/failure Slack messages
- final result message (`JD Name: ...`)
- output CSV filenames inside the worker run (slugged)
- Instantly campaign name during thread-reply enrichment

## Slack Admin Commands

In the configured Slack channel, you can trigger recovery/ops actions with:

- `# JD-Runs` or `# JD-Runs 30` (list recent runs)
- `# JD-Run <run_id>` (show details for one run)
- `# JD-Retry <run_id>` (requeue score stage from bucket artifacts)
- `# JD-Cleanup` or `# JD-Cleanup 24` (dry-run cleanup preview)
- `# JD-Cleanup 24 confirm` (execute cleanup)

Responses are posted in a thread under your command message.

Full command behavior and examples:
- `SLACK_COMMANDS.md`

## Bucket Artifact Layout

Cross-job handoff artifacts are stored under `runs/{run_id}/...` (or `${PIPELINE_RUNS_PREFIX}/{run_id}/...`):

- `runs/{run_id}/jd.txt`
- `runs/{run_id}/candidates.csv`
- `runs/{run_id}/dedup.csv`
- `runs/{run_id}/jd_context_by_prompt.json`
- `runs/{run_id}/queries.json`
- `runs/{run_id}/meta.json`
- `runs/{run_id}/scored.csv` (on successful score stage)
- `runs/{run_id}/scored-partial.csv` (best-effort on score failures)
- `runs/{run_id}/scored-sheet-ready.csv` (on successful score stage)

Final-summary idempotency marker:

- `runs/{run_id}/final_summary_posted.json`

Cleanup helper (single run):

```bash
python scripts/cleanup_bucket_runs.py --run-id <run_id>
```

Cleanup helper (custom prefix):

```bash
python scripts/cleanup_bucket_runs.py --prefix runs/20260313-demo
```

Cleanup helper (retention sweep):

```bash
python scripts/cleanup_bucket_runs.py --older-than-hours 24 --dry-run
python scripts/cleanup_bucket_runs.py --older-than-hours 24
```

## Failure Recovery (Manual)

List recent runs:

```bash
python scripts/list_bucket_runs.py --limit 20
```

Download all artifacts for a run:

```bash
python scripts/download_bucket_run.py --run-id <run_id> --output-dir ./bucket-runs
```

Manually restart score stage from existing run artifacts:

```bash
python scripts/requeue_score_from_run.py --run-id <run_id>
```

Notes:
- `requeue_score_from_run.py` reads `channel_id`, `jd_name`, and `jd_test_mode` from `meta.json` by default.
- Use `--channel-id`, `--jd-name`, or `--jd-test-mode true|false` to override.
- This lets you recover without re-running Exa sourcing, as long as source-stage artifacts are still in bucket.

## Required Environment Variables

Set these in worker service env (or locally in `.env`, starting from `.env.example`).
Vercel webhook service only needs the Slack signature/token/channel + Redis vars.

- `OPENAI_API_KEY`
- `EXA_API_KEY`
- `SLACK_SIGNING_SECRET`
- `SLACK_BOT_TOKEN` (preferred) or `SLACK_USER_TOKEN`
- `REDIS_URL`
- `AWS_ENDPOINT_URL` (Railway bucket S3 endpoint)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_S3_BUCKET_NAME`
- `SALESQL_API_KEY` (thread-reply enrichment workflow)
- `REOON_API_KEY` (thread-reply enrichment workflow)
- `BOUNCEBAN_API_KEY` (thread-reply enrichment workflow)
- `INSTANTLY_API_KEY` (thread-reply enrichment workflow)
- `GOOGLE_SERVICE_ACCOUNT_JSON` (service account JSON, raw or base64-encoded JSON)
- `GOOGLE_DRIVE_FOLDER_ID` (Drive folder or Shared Drive folder where result sheets will be created)
- `GOOGLE_WORKSPACE_DOMAIN` (for domain-restricted link sharing, e.g. `calyptus.co`)

Optional:
- `SLACK_CHANNEL_ID` (default `C0AF5RGPMEW`)
- `RQ_JD_SOURCE_QUEUE_NAME` (default `jd-pipeline`)
- `RQ_JD_SCORE_QUEUE_NAME` (default `jd-pipeline-score`)
- `RQ_JD_ADMIN_QUEUE_NAME` (default `jd-admin`)
- `RQ_JD_QUEUE_NAME` (legacy fallback for source queue)
- `RQ_REPLY_QUEUE_NAME` (default `reply-enrichment`)
- `RQ_QUEUE_NAME` (legacy fallback for JD queue name only)
- `RQ_WORKER_QUEUE_TYPE` (default `source`; worker can consume `source`, `score`, `jd` (legacy source alias), `reply`, `admin`, or `all`)
- `SLACK_ADMIN_USER_IDS` (optional CSV of Slack user ids allowed to run admin commands; if unset, any user in the configured channel can run them)
- `RQ_JOB_TIMEOUT` (default `7200`)
- `RQ_WITH_SCHEDULER` (default `false`; keep disabled unless you use scheduled RQ jobs)
- `RQ_STOP_ON_JOB_FAILURE` (default `true`; when a queued job fails, stop the worker before dequeuing the next job)
- `PIPELINE_RUNS_PREFIX` (default `runs`; handoff root prefix)
- `PIPELINE_ARTIFACT_RETENTION_HOURS` (default `24`; artifacts remain recoverable in bucket for at least this many hours)
- `AWS_DEFAULT_REGION` (default `auto`)
- `S3_ADDRESSING_STYLE` (default `virtual`; set `path` only if your Railway bucket credentials page requires path style)
- `RAILWAY_BUCKET_KEY_PREFIX` (default empty; prepended to all object keys)
- `RAILWAY_BUCKET_USE_SSL` (default `true`)
- `RAILWAY_BUCKET_MAX_ATTEMPTS` (default `4`)
- `RAILWAY_BUCKET_BASE_BACKOFF_SECONDS` (default `0.5`)
- `RAILWAY_BUCKET_MAX_BACKOFF_SECONDS` (default `8`)
- Legacy compatibility aliases are still supported in code: `ENDPOINT`, `ACCESS_KEY_ID`, `SECRET_ACCESS_KEY`, `BUCKET`, `REGION`, `RAILWAY_BUCKET_ENDPOINT_URL`, `RAILWAY_BUCKET_ACCESS_KEY_ID`, `RAILWAY_BUCKET_SECRET_ACCESS_KEY`, `RAILWAY_BUCKET_REGION`, `RAILWAY_BUCKET_NAME`.
- `SLACK_EVENT_TTL_SECONDS` (default `86400`)
- `SLACK_EVENT_DEDUP_ENABLED` (default `true`)
- `SLACK_EVENT_DEDUP_FAIL_OPEN` (default `true`)
- `SLACK_ACK_QUEUE_UNAVAILABLE` (default `true`; returns `200` degraded responses to Slack when Redis is unavailable)
- `EXA_CONCURRENT_SEARCHES` (default `7`; Exa thread pool size per batch)
- `EXA_MAX_QPS` (default `2`; global Exa request cap shared across all Exa threads, including retry attempts)
- `SCORING_CONCURRENT_CALLS` (default `24`)
- `SCORING_MAX_INFLIGHT_FACTOR` (default `2`; max in-flight scoring futures = `SCORING_CONCURRENT_CALLS * factor`)
- `PIPELINE_INCLUDE_RAW_JSON` (default `false`; set `true` only for debugging, increases memory usage)
- `SCORE_PROGRESS_NOTIFY_EVERY` (default `50`)
- `SCORE_INITIAL_SECONDS_PER_CANDIDATE` (default `2.5`)
- `INSTANTLY_CAMPAIGN_TIMEZONE` (default `Asia/Kolkata`)
- `INSTANTLY_CAMPAIGN_START_HOUR` (default `08:00`)
- `INSTANTLY_CAMPAIGN_END_HOUR` (default `18:00`)
- `INSTANTLY_CAMPAIGN_DURATION_DAYS` (default `30`)
- `REOON_MAX_WAIT_SECONDS` (default `1800`)
- `REOON_POLL_INTERVAL_SECONDS` (default `15`)
- `BOUNCEBAN_MAX_WAIT_SECONDS` (default `600`)
- `BOUNCEBAN_POLL_INTERVAL_SECONDS` (default `15`)
- `INSTANTLY_FAIL_FAST` (default `false`; abort immediately on first Instantly lead add error)
- `INSTANTLY_SKIP_IF_IN_WORKSPACE` (default `true`; if true, Instantly may skip emails that already exist in the workspace)
- `THREAD_RESULT_STRICT` (default `true`; only allow thread-reply enrichment when thread root text matches result message prefix)
- `RESULT_MESSAGE_PREFIX` (default `AI-scored candidates sheet for this JD`)
- `GOOGLE_WORKSPACE_DOMAIN_ROLE` (default `writer`; set `reader` or `commenter` if needed)
- `THREAD_ENRICHMENT_VERBOSE_UPDATES` (default `false`; if true, posts intermediate stage updates in thread)
- `DASHBOARD_GOOGLE_SHEET_URL` (optional; if set, upserts JD + enrichment metrics into this dashboard sheet)
- `DASHBOARD_WORKSHEET_NAME` (optional; tab name inside `DASHBOARD_GOOGLE_SHEET_URL`; default `info-log`)

## Dashboard Logging (Optional)

Set `DASHBOARD_GOOGLE_SHEET_URL` to enable automatic dashboard rows keyed by `Scored Candidates Sheet URL`.
By default rows are written into worksheet `info-log` unless `DASHBOARD_WORKSHEET_NAME` is set.

Column names used:
- `JD Campaign Name`
- `Scored Candidates Sheet URL`
- `Total Profiles Found`
- `Profiles After Deduplication`
- `Location Mismatch (Manual Review)`
- `Score 10 Count` ... `Score 0 Count`
- `Minimum Score for Contact`
- `Candidates Entered Enrichment`
- `Emails Found via SaleSQL`
- `Emails Passed Reoon`
- `Emails Passed BounceBan`
- `pre-exist in instantly`
- `Net Leads Enrolled to Instantly`
- `Instantly Campaign Name`
- `Instantly Campaign URL`
- `Notes`

## Slack App Setup

1. Enable **Event Subscriptions**.
2. Request URL: `https://<your-vercel-domain>/slack/events`
3. Subscribe bot events:
- `message.channels`
4. OAuth scopes:
- `chat:write`
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

Dedicated workers:

```bash
python worker.py --queue-type source
python worker.py --queue-type score
python worker.py --queue-type reply
python worker.py --queue-type admin
```

Recommended production shape:
- run `2` source workers: `python worker.py --queue-type source`
- run `1` score worker: `python worker.py --queue-type score`
- run `1` reply worker: `python worker.py --queue-type reply`
- run `1` admin worker: `python worker.py --queue-type admin`

Queue topology:
- `RQ_JD_SOURCE_QUEUE_NAME` -> `worker.process_jd_source_job`
- `RQ_JD_SCORE_QUEUE_NAME` -> `worker.process_jd_score_job`
- `RQ_REPLY_QUEUE_NAME` -> `worker.process_thread_reply_enrichment_job`
- `RQ_JD_ADMIN_QUEUE_NAME` -> `worker.process_jd_admin_job`

Railway expected commands:
- Build: `pip install -r requirements.txt` (or leave default with `nixpacks.toml`)
- Start: `python -u worker.py --queue-type source`

Helper scripts committed in repo:
- `./scripts/start-jd-worker.sh`
- `./scripts/start-score-worker.sh`
- `./scripts/start-reply-worker.sh`
- `./scripts/start-admin-worker.sh`
- `./scripts/list_bucket_runs.py`
- `./scripts/download_bucket_run.py`
- `./scripts/requeue_score_from_run.py`
- `./scripts/cleanup_bucket_runs.py`

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

## Railway Setup

Railway will not create the extra worker service or replica count automatically from this repo push alone.
You need one-time service configuration in Railway after deploy.

Recommended layout:
- Service `jit-worker-source`
  - Start command: `./scripts/start-jd-worker.sh`
  - Replicas: `2`
- Service `jit-worker-score`
  - Start command: `./scripts/start-score-worker.sh`
  - Replicas: `1`
- Service `jit-worker-reply`
  - Start command: `./scripts/start-reply-worker.sh`
  - Replicas: `1`

Notes:
- `nixpacks.toml` defaults to the source worker script.
- The reply worker should be a separate Railway service that points to the same repo and same env vars, but uses the reply start command above.
- The score worker should be a separate Railway service that points to the same repo and same env vars, but uses the score start command above.
- Replica count is configured in Railway service settings, not in this repo.
- If Railway logs `Work-horse terminated unexpectedly; waitpid returned 9`, reduce memory pressure by lowering `SCORING_CONCURRENT_CALLS` (for example `4`) and keep `PIPELINE_INCLUDE_RAW_JSON=false`.

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

- Local pipeline intermediates are still written under `/tmp` inside each job and cleaned up after each run.
- Cross-job state (source -> score handoff) is persisted in the Railway bucket under `runs/{run_id}/...`.
- Step updates are posted to Slack at each stage completion.
- The 15 generated Exa prompts are uploaded to Slack as a `.txt` file per run.
