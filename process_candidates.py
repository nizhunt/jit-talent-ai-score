import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=api_key)


# ── Structured output via OpenAI's json_schema response_format ──────────
# This guarantees the model returns exactly the JSON shape we want.
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "candidate_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "Suitability score from 0 to 10",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reasoning (3‑5 sentences) for the score",
                },
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
    },
}


def get_ai_score(candidate_text: str, jd_text: str, prompt_template: str):
    """Send one candidate's data to ChatGPT and return (score, reason)."""
    if not candidate_text or pd.isna(candidate_text):
        return 0, "No candidate data available"

    # Build the prompt by replacing placeholders in prompt.md
    prompt = prompt_template.replace("[PASTE LINKEDIN DATA]", str(candidate_text))
    prompt = prompt.replace("[PASTE JD TEXT]", jd_text)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert recruiter. Evaluate the candidate "
                        "against the job description. Return your answer as JSON "
                        'with keys "score" (integer 0-10) and "reason" (string).'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format=RESPONSE_SCHEMA,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return int(data["score"]), data["reason"]
    except Exception as e:
        print(f"  ⚠  Error: {e}")
        return "Error", str(e)


def main():
    # ── File paths ──────────────────────────────────────────────────────
    prompt_path = "prompt.md"
    jd_path = "jd.md"
    csv_path = "candidates.csv"
    output_path = "candidates-scored.csv"

    print("Loading files...")
    try:
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        with open(jd_path, "r") as f:
            jd_text = f.read()
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")
        return

    total = len(df)
    print(f"Total candidates: {total}")

    # ── Resume support: if output file exists, load already‑scored rows ─
    if os.path.exists(output_path):
        scored_df = pd.read_csv(output_path)
        already_done = scored_df["ai-score"].notna().sum()
        print(f"Resuming – {already_done}/{total} already scored.")
        df["ai-score"] = scored_df["ai-score"]
        df["ai-reason"] = scored_df["ai-reason"]
    else:
        df["ai-score"] = None
        df["ai-reason"] = None

    # ── Process each row ────────────────────────────────────────────────
    pending = df["ai-score"].isna()
    pending_count = pending.sum()
    print(f"Processing {pending_count} candidates...")

    progress = tqdm(total=pending_count, desc="Scoring", unit="candidate")

    for idx in df.index:
        if not pending[idx]:
            continue

        name = df.at[idx, "name"]
        candidate_text = df.at[idx, "text"]

        score, reason = get_ai_score(candidate_text, jd_text, prompt_template)

        df.at[idx, "ai-score"] = score
        df.at[idx, "ai-reason"] = reason

        progress.set_postfix_str(f"{name} → {score}")
        progress.update(1)

        # Save after every 10 candidates so progress is not lost
        if (progress.n % 10) == 0:
            df.to_csv(output_path, index=False)

    progress.close()

    # ── Final save ──────────────────────────────────────────────────────
    df.to_csv(output_path, index=False)
    print(f"\n✅ Done! Results saved to {output_path}")
    print(df[["name", "ai-score", "ai-reason"]].head(10))


if __name__ == "__main__":
    main()
