# Slack Command Reference

This is the canonical list of user-facing Slack commands accepted by this app.

Maintenance rule:
- If command parsing behavior changes in `app.py` or `thread_reply_enrichment.py`, update this file in the same PR.

General parsing rules:
- Command matching is case-insensitive.
- Header-style commands are parsed from the first line of the message.
- `# JD` and `# JD-Quick` commands use remaining lines as JD body text.
- Thread-reply enrichment command must be a bare integer message (`1` to `10`) with no extra text.

## JD Submission Commands

### 1) Full pipeline
- Command: `# JD`
- Variants:
  - `# JD <name>`
  - `# JD: <name>`
  - `# JD | <name>`
- Behavior:
  - enqueues source stage (`jd-pipeline` queue by default)
  - source stage widens JD, generates queries, fetches Exa, flattens, combines, dedups
  - uploads handoff artifacts to bucket
  - enqueues score stage

Example:
```text
# JD Senior Data Engineer
<full JD text here...>
```

### 2) Quick/test pipeline
- Command: `# JD-Quick`
- Legacy alias: `# JD-Test`
- Variants:
  - `# JD-Quick <name>`
  - `# JD-Quick: <name>`
  - `# JD-Test <name>`
- Behavior:
  - skips widening/query generation
  - uses raw JD text as the only Exa query
  - caps sourced profiles at 100
  - then runs normal dedup + score flow

Example:
```text
# JD-Quick Backend Engineer
<full JD text here...>
```

## Thread Reply Enrichment Command

### 3) Start enrichment from scored-results thread
- Command: bare integer `1`..`10` as thread reply text
- Valid: `5`
- Invalid: `score 5`, `5 please`, `5.0`
- Behavior:
  - only processed for thread replies
  - enqueues reply enrichment workflow
  - filters candidates by threshold and runs SaleSQL -> Reoon -> BounceBan -> Instantly flow

## Admin Commands (Bucket/Recovery Ops)

Admin commands run through the admin queue (`jd-admin` by default).
If `SLACK_ADMIN_USER_IDS` is set, only listed user IDs can run them.

### 4) List runs
- Command: `# JD-Runs`
- Variant: `# JD-Runs <limit>` (1..100)
- Behavior:
  - lists recent runs with status/source/score state

### 5) Show one run
- Command: `# JD-Run <run_id>`
- Behavior:
  - shows details: pipeline status, stage states, rows_after_dedup, expiry, sheet URL

### 6) Requeue score stage
- Command: `# JD-Retry <run_id>`
- Behavior:
  - validates required handoff artifacts in bucket
  - re-enqueues score stage using existing run artifacts (no re-sourcing)

### 7) Cleanup (dry-run by default)
- Command: `# JD-Cleanup`
- Variant: `# JD-Cleanup <hours>`
- Behavior:
  - dry-run preview of runs eligible for cleanup

Execute cleanup:
- Command: `# JD-Cleanup <hours> confirm`
- Behavior:
  - deletes eligible run prefixes from bucket

## Response Behavior

- Admin command responses are posted as thread replies under the command message.
- Unsupported Slack messages are ignored (`ignored=not_supported_message`).
