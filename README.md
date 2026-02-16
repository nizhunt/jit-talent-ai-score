# JIT Talent AI Score

Slack `# JD` ingestion + Exa sourcing + AI scoring pipeline.

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

Optional:
- `SLACK_CHANNEL_ID` (default `C0AF5RGPMEW`)
- `RQ_QUEUE_NAME` (default `jd-pipeline`)
- `RQ_JOB_TIMEOUT` (default `7200`)
- `SLACK_EVENT_TTL_SECONDS` (default `86400`)
- `SCORE_PROGRESS_NOTIFY_EVERY` (default `50`)
- `SCORE_INITIAL_SECONDS_PER_CANDIDATE` (default `2.5`)

## Slack App Setup

1. Enable **Event Subscriptions**.
2. Request URL: `https://<your-vercel-domain>/slack/events`
3. Subscribe bot events:
- `message.channels`
4. OAuth scopes:
- `chat:write`
- `files:write`
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

## Worker Run Command

```bash
python worker.py
```

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

## Notes

- Pipeline outputs are written under `/tmp` in worker runs.
- Step updates are posted to Slack at each stage completion.
- The 15 generated Exa prompts are uploaded to Slack as a `.txt` file per run.
