#!/usr/bin/env python3
"""Generate natural language profile descriptions with the OpenAI API."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any
from tqdm import tqdm


DESC_COLUMNS = [f"desc{i}" for i in range(10)]
DEFAULT_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
UNKNOWN_VALUE = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GPT descriptions for data/profile_combinations.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/profile_combinations.csv"),
        help="Input CSV of profile combinations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/profile_descriptions.csv"),
        help="Output CSV with desc0..desc9 columns.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing descriptions.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per row after API/parse errors.")
    parser.add_argument("--key", type=str, default=None, help="OpenAI API key (overrides env var)")
    parser.add_argument("--org", type=str, default=None, help="OpenAI organization (overrides env var)")
    return parser.parse_args()


def load_openai_client(api_key: str | None, org: str | None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install the OpenAI Python SDK in this environment "
            "with `python -m pip install openai`."
        ) from exc
    return OpenAI(api_key=api_key, organization=org)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def known_profile_fields(row: dict[str, str], source_columns: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    source_column_set = set(source_columns)
    for column in DEFAULT_FIELDS:
        if column not in source_column_set:
            continue
        value = (row.get(column) or "").strip()
        if value and value.lower() != UNKNOWN_VALUE:
            fields[column] = value
    return fields


def row_has_descriptions(row: dict[str, str]) -> bool:
    return all((row.get(column) or "").strip() for column in DESC_COLUMNS)


def profile_key(row: dict[str, str], source_columns: list[str]) -> tuple[tuple[str, str], ...]:
    return tuple(known_profile_fields(row, source_columns).items())


def copy_descriptions(target: dict[str, str], source: dict[str, str]) -> None:
    for column in DESC_COLUMNS:
        value = (source.get(column) or "").strip()
        if value:
            target[column] = value


def merge_existing_descriptions(
    rows: list[dict[str, str]],
    source_columns: list[str],
    existing_rows: list[dict[str, str]],
    existing_columns: list[str],
) -> int:
    existing_by_profile: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for row in existing_rows:
        if not row_has_descriptions(row):
            continue
        key = profile_key(row, existing_columns)
        if key and key not in existing_by_profile:
            existing_by_profile[key] = row

    reused = 0
    for row in rows:
        if row_has_descriptions(row):
            continue
        key = profile_key(row, source_columns)
        existing = existing_by_profile.get(key)
        if existing is None:
            continue
        copy_descriptions(row, existing)
        reused += 1
    return reused


def build_prompt(profile: dict[str, str]) -> str:
    profile_lines = "\n".join(f"- {key}: {value}" for key, value in profile.items())
    return f"""Generate 10 short, natural text descriptions of a speaker voice profile.

Use only the known fields below. Do not infer or mention any missing/unknown fields.
Each description should be one sentence and should describe the same profile in varied wording.

Known profile fields:
{profile_lines}

Return only valid JSON in this exact shape:
{{"descriptions": ["description 0", "description 1", "..."]}}
"""


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text

    if hasattr(response, "choices"):
        return response.choices[0].message.content

    raise ValueError("Could not read text from OpenAI response")


def call_openai(client: Any, model: str, prompt: str) -> str:
    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You write concise voice-profile descriptions and return strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response_text(response)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You write concise voice-profile descriptions and return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response_text(response)


def parse_descriptions(text: str) -> list[str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(text[start : end + 1])

    descriptions = payload.get("descriptions")
    if not isinstance(descriptions, list):
        raise ValueError("response JSON must contain a descriptions list")

    cleaned = [str(item).strip() for item in descriptions if str(item).strip()]
    if len(cleaned) != 10:
        raise ValueError(f"expected 10 descriptions, got {len(cleaned)}")
    return cleaned


def generate_descriptions(
    client: Any,
    model: str,
    profile: dict[str, str],
    max_retries: int,
) -> list[str]:
    prompt = build_prompt(profile)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return parse_descriptions(call_openai(client, model, prompt))
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 30))

    raise RuntimeError(f"failed after {max_retries} attempts: {last_error}") from last_error


def main() -> int:
    args = parse_args()
    source_columns, rows = read_rows(args.input)
    output_columns = source_columns + [column for column in DESC_COLUMNS if column not in source_columns]

    reused = 0
    if args.output.exists() and not args.overwrite:
        existing_columns, existing_rows = read_rows(args.output)
        if all(column in existing_columns for column in DESC_COLUMNS):
            reused = merge_existing_descriptions(
                rows=rows,
                source_columns=source_columns,
                existing_rows=existing_rows,
                existing_columns=existing_columns,
            )

    if args.limit is not None:
        rows_to_process = rows[: args.limit]
    else:
        rows_to_process = rows

    client = None
    total = len(rows_to_process)
    generated = 0
    skipped = 0

    for index, row in tqdm(enumerate(rows_to_process, start=1), total=len(rows_to_process), desc=f'prompting {args.model}'):
        if not args.overwrite and row_has_descriptions(row):
            skipped += 1
            continue

        profile = known_profile_fields(row, source_columns)
        if not profile:
            raise ValueError(f"row {index} has no known fields to send to GPT")

        # print(f"[{index}/{total}] generating descriptions for {profile}", flush=True)
        if client is None:
            client = load_openai_client(args.key, args.org)
        descriptions = generate_descriptions(
            client=client,
            model=args.model,
            profile=profile,
            max_retries=args.max_retries,
        )
        for desc_column, description in zip(DESC_COLUMNS, descriptions):
            row[desc_column] = description

        generated += 1
        write_rows(args.output, output_columns, rows)

        if args.sleep > 0:
            time.sleep(args.sleep)

    write_rows(args.output, output_columns, rows)
    print(f"Saved {args.output} generated={generated} reused={reused} skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
