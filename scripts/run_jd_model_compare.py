#!/usr/bin/env python3
"""Run a JD experiment comparing two models on prod extraction+scoring path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from process_candidates import (
    JD_TEST_MAX_RESULTS,
    JD_TEST_PROMPT_FILENAME,
    deduplicate_csv,
    flatten_exa_results_to_rows,
    generate_jd_email_snippet,
    normalize_url,
    read_text_file,
    run_exa_fanout,
    score_candidates_csv,
)

DEFAULT_JD_TEXT = (
    "VP Engineering | Head of Engineering | Engineering Director | Engineering Lead | Tech Lead | "
    "title contains engineering leadership | 10-20 years software engineering experience | "
    "hands-on individual contributor (IC) plus people management experience | led teams >= 6 engineers | "
    "stack: TypeScript (must), Node.js, React, Python, cloud (AWS or GCP) | "
    "experience integrating AI/ML into production systems | "
    "based in Greater Toronto Area / GTA (Toronto, Mississauga, Brampton, Markham, Vaughan, "
    "Richmond Hill, Oakville, Burlington, Oshawa, Whitby, Ajax, Pickering, Newmarket, Aurora) | "
    "SaaS, PropTech, real estate tech, or B2B marketplace background preferred | "
    "Canada work authorization required"
)


def _to_int_score(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        iv = int(float(value))
        return iv
    except Exception:
        return None


def _sentence_count(text: str) -> int:
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if c.strip()]
    return len(chunks)


def _email_has_exact_blank_line_break(email: str) -> bool:
    normalized = (email or "").strip()
    if not normalized:
        return False
    parts = re.split(r"\n\s*\n", normalized)
    return len(parts) == 2 and all(p.strip() for p in parts)


def _email_looks_plain_text(email: str) -> bool:
    lines = [ln.strip() for ln in (email or "").splitlines() if ln.strip()]
    if not lines:
        return True
    bad_prefixes = ("-", "*", "#")
    for ln in lines:
        if ln.startswith(bad_prefixes):
            return False
        if re.match(r"^\d+\.", ln):
            return False
        if re.match(r"^(score|reasoning|email)\s*:", ln.lower()):
            return False
    return True


def _build_match_key(row: pd.Series) -> str:
    linkedin = normalize_url(row.get("linkedin", ""))
    if linkedin:
        return f"li:{linkedin}"

    url = normalize_url(row.get("url", ""))
    if url:
        return f"url:{url}"

    text = str(row.get("text", "") or "").strip()
    if text:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"txt:{digest}"

    name = str(row.get("name", "") or "").strip().lower()
    location = str(row.get("location", "") or "").strip().lower()
    if name:
        return f"name:{name}|loc:{location}"

    rid = str(row.get("id", "") or "").strip()
    if rid:
        return f"id:{rid}"

    return ""


def _format_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    scores = df["ai-score"] if "ai-score" in df.columns else pd.Series([], dtype="object")
    reasons = df["ai-reason"] if "ai-reason" in df.columns else pd.Series([], dtype="object")
    emails = df["ai-email"] if "ai-email" in df.columns else pd.Series([], dtype="object")

    score_ints = scores.apply(_to_int_score)
    is_error = scores.astype(str).str.lower().eq("error")
    is_numeric = score_ints.notna()
    in_range = score_ints.apply(lambda x: x is not None and 0 <= x <= 10)

    email_non_blank = emails.fillna("").astype(str).str.strip().ne("")
    email_unique_non_blank = int(emails.fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())

    reason_sentence_counts = reasons.fillna("").astype(str).apply(_sentence_count)
    reason_4_sentences = reason_sentence_counts.eq(4)

    email_two_sentence_format = emails.fillna("").astype(str).apply(_email_has_exact_blank_line_break)
    email_plain_text = emails.fillna("").astype(str).apply(_email_looks_plain_text)

    def pct(mask: pd.Series) -> float:
        if total == 0:
            return 0.0
        return round(float(mask.sum()) * 100.0 / float(total), 2)

    return {
        "rows_scored": total,
        "score_numeric_rows": int(is_numeric.sum()),
        "score_non_numeric_rows": int((~is_numeric).sum()),
        "score_error_rows": int(is_error.sum()),
        "score_in_range_rows": int(in_range.sum()),
        "score_in_range_pct": pct(in_range),
        "reason_exact_4_sentences_rows": int(reason_4_sentences.sum()),
        "reason_exact_4_sentences_pct": pct(reason_4_sentences),
        "email_non_blank_rows": int(email_non_blank.sum()),
        "email_non_blank_pct": pct(email_non_blank),
        "email_unique_non_blank_values": email_unique_non_blank,
        "email_two_paragraph_format_rows": int(email_two_sentence_format.sum()),
        "email_two_paragraph_format_pct": pct(email_two_sentence_format),
        "email_plain_text_rows": int(email_plain_text.sum()),
        "email_plain_text_pct": pct(email_plain_text),
    }


def _run_single_model(
    *,
    client: OpenAI,
    exa_api_key: str,
    model: str,
    jd_text: str,
    scorer_prompt_template: str,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model_dir = output_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    query = jd_text.strip()
    exa_result = run_exa_fanout(
        exa_api_key=exa_api_key,
        queries=[query],
        num_results_per_query=JD_TEST_MAX_RESULTS,
    )
    items = exa_result.get("results", [])
    if len(items) > JD_TEST_MAX_RESULTS:
        items = items[:JD_TEST_MAX_RESULTS]

    rows = flatten_exa_results_to_rows(
        items,
        expansion_prompt_file=JD_TEST_PROMPT_FILENAME,
        widened_jd_context=jd_text,
        include_raw_json=False,
    )

    candidates_csv = model_dir / "candidates.csv"
    dedup_csv = model_dir / "candidates-dedup.csv"
    scored_csv = model_dir / "candidates-scored.csv"

    pd.DataFrame(rows).to_csv(candidates_csv, index=False)
    dedup_df = deduplicate_csv(str(candidates_csv), str(dedup_csv))
    generated_email, email_usage = generate_jd_email_snippet(
        client=client,
        jd_text=jd_text,
        model=model,
    )

    scored_df, usage = score_candidates_csv(
        client=client,
        input_csv=str(dedup_csv),
        output_csv=str(scored_csv),
        original_jd_text=jd_text,
        scorer_prompt_template=scorer_prompt_template,
        model=model,
        generated_email=generated_email,
        progress_callback=None,
    )

    metrics = {
        "model": model,
        "exa_results": int(len(items)),
        "rows_flattened": int(len(rows)),
        "rows_after_dedup": int(len(dedup_df)),
        "rows_scored": int(len(scored_df)),
        "usage_input_tokens": int(email_usage.get("input_tokens", 0)) + int(usage.get("input_tokens", 0)),
        "usage_output_tokens": int(email_usage.get("output_tokens", 0)) + int(usage.get("output_tokens", 0)),
        "usage_total_tokens": int(email_usage.get("total_tokens", 0)) + int(usage.get("total_tokens", 0)),
    }
    metrics.update(_format_metrics(scored_df))
    return scored_df, metrics


def _score_closeness(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, Any]:
    a = df_a.copy()
    b = df_b.copy()

    a["match_key"] = a.apply(_build_match_key, axis=1)
    b["match_key"] = b.apply(_build_match_key, axis=1)

    a = a[a["match_key"].astype(str).str.len() > 0].drop_duplicates(subset=["match_key"], keep="first")
    b = b[b["match_key"].astype(str).str.len() > 0].drop_duplicates(subset=["match_key"], keep="first")

    merged = a[["match_key", "ai-score"]].merge(
        b[["match_key", "ai-score"]],
        on="match_key",
        how="inner",
        suffixes=("_a", "_b"),
    )

    merged["score_a"] = merged["ai-score_a"].apply(_to_int_score)
    merged["score_b"] = merged["ai-score_b"].apply(_to_int_score)
    merged = merged.dropna(subset=["score_a", "score_b"]).copy()

    if merged.empty:
        return {
            "overlap_entities": 0,
            "exact_score_match_rows": 0,
            "exact_score_match_pct": 0.0,
            "within_1_point_rows": 0,
            "within_1_point_pct": 0.0,
            "mean_abs_diff": None,
            "rmse": None,
        }

    merged["abs_diff"] = (merged["score_a"] - merged["score_b"]).abs()
    exact = merged["abs_diff"].eq(0)
    within_1 = merged["abs_diff"].le(1)

    return {
        "overlap_entities": int(len(merged)),
        "exact_score_match_rows": int(exact.sum()),
        "exact_score_match_pct": round(float(exact.sum()) * 100.0 / float(len(merged)), 2),
        "within_1_point_rows": int(within_1.sum()),
        "within_1_point_pct": round(float(within_1.sum()) * 100.0 / float(len(merged)), 2),
        "mean_abs_diff": round(float(merged["abs_diff"].mean()), 4),
        "rmse": round(float(((merged["score_a"] - merged["score_b"]) ** 2).mean() ** 0.5), 4),
    }


def _build_side_by_side(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    a = df_a.copy()
    b = df_b.copy()
    a["match_key"] = a.apply(_build_match_key, axis=1)
    b["match_key"] = b.apply(_build_match_key, axis=1)

    a = a.drop_duplicates(subset=["match_key"], keep="first")
    b = b.drop_duplicates(subset=["match_key"], keep="first")

    all_cols = list(dict.fromkeys(list(a.columns) + list(b.columns)))
    all_cols = [c for c in all_cols if c != "match_key"]

    merged = a.merge(b, on="match_key", how="outer", suffixes=(f"_{model_a}", f"_{model_b}"))

    ordered_cols: List[str] = ["match_key"]
    for col in all_cols:
        ordered_cols.append(f"{col}_{model_a}")
        ordered_cols.append(f"{col}_{model_b}")

    existing = [c for c in ordered_cols if c in merged.columns]
    return merged[existing]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two models on JD test mode pipeline")
    parser.add_argument("--jd-file", default="", help="Path to file containing JD text. If omitted, uses built-in JD.")
    parser.add_argument(
        "--output-dir",
        default="api-dry-run/jd_t012t_model_compare",
        help="Directory for all outputs.",
    )
    parser.add_argument("--model-a", default="gpt-5-nano")
    parser.add_argument("--model-b", default="gpt-5-nano")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(".env")

    exa_api_key = os.getenv("EXA_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not exa_api_key:
        raise RuntimeError("Missing EXA_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    jd_text = DEFAULT_JD_TEXT
    if args.jd_file:
        jd_text = read_text_file(args.jd_file).strip()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "jd.txt").write_text(jd_text + "\n", encoding="utf-8")

    scorer_prompt_template = read_text_file("prompts/scorer_prompt.md")
    client = OpenAI(api_key=openai_api_key)

    df_a, metrics_a = _run_single_model(
        client=client,
        exa_api_key=exa_api_key,
        model=args.model_a,
        jd_text=jd_text,
        scorer_prompt_template=scorer_prompt_template,
        output_dir=output_dir,
    )

    df_b, metrics_b = _run_single_model(
        client=client,
        exa_api_key=exa_api_key,
        model=args.model_b,
        jd_text=jd_text,
        scorer_prompt_template=scorer_prompt_template,
        output_dir=output_dir,
    )

    closeness = _score_closeness(df_a, df_b)
    side_by_side_df = _build_side_by_side(df_a, df_b, args.model_a, args.model_b)

    side_by_side_csv = output_dir / "consolidated_side_by_side.csv"
    summary_json = output_dir / "comparison_summary.json"
    summary_csv = output_dir / "comparison_summary.csv"

    side_by_side_df.to_csv(side_by_side_csv, index=False)

    summary = {
        "jd_test_max_results": JD_TEST_MAX_RESULTS,
        "model_a": metrics_a,
        "model_b": metrics_b,
        "score_closeness": closeness,
        "outputs": {
            "model_a_scored_csv": str((output_dir / args.model_a / "candidates-scored.csv").resolve()),
            "model_b_scored_csv": str((output_dir / args.model_b / "candidates-scored.csv").resolve()),
            "consolidated_side_by_side_csv": str(side_by_side_csv.resolve()),
            "comparison_summary_csv": str(summary_csv.resolve()),
            "comparison_summary_json": str(summary_json.resolve()),
        },
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_rows = []
    for model_label, metrics in [(args.model_a, metrics_a), (args.model_b, metrics_b)]:
        for k, v in metrics.items():
            if k == "model":
                continue
            summary_rows.append({"section": "model_metrics", "model": model_label, "metric": k, "value": v})
    for k, v in closeness.items():
        summary_rows.append({"section": "score_closeness", "model": f"{args.model_a}_vs_{args.model_b}", "metric": k, "value": v})

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
