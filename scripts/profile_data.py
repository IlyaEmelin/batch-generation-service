"""Профиль входных наборов: длина в токенах, дубликаты отзывов, превышение лимита.

Скрипт считает только товарное содержимое (title, category, attributes, reviews),
без system-сообщения и chat-template, поэтому реальная длина запроса не меньше
полученных значений. Токенайзер — тот же, что у модели на сервере.

Запуск:
    python scripts/profile_data.py
    python scripts/profile_data.py dev.jsonl benchmark.jsonl --limit 1500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DEFAULT_FILES = ("dev.jsonl", "benchmark.jsonl")
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: np.ndarray, q: float) -> int:
    return int(round(float(np.percentile(values, q))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Профиль входных наборов в токенах")
    parser.add_argument("files", nargs="*", default=list(DEFAULT_FILES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=1500)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    print(f"tokenizer: {args.model}")
    print()
    print(
        "| Набор | Строк | Уникальных product_id | Отзывов на товар "
        f"| Токенов в отзывах p50 / p90 / max | Товаров > {args.limit} токенов |"
    )
    print("|---|---|---|---|---|---|")

    for name in args.files:
        rows = load_rows(Path(name))
        reviews_tokens = np.array([count_tokens(" ".join(row["reviews"])) for row in rows])
        content_tokens = np.array(
            [
                count_tokens(row["title"])
                + count_tokens(row["category"])
                + count_tokens(json.dumps(row["attributes"], ensure_ascii=False))
                + count_tokens(" ".join(row["reviews"]))
                for row in rows
            ]
        )
        over_limit = int((content_tokens > args.limit).sum())
        share = over_limit / len(rows) if rows else 0.0
        reviews_per_product = [len(row["reviews"]) for row in rows]
        print(
            f"| `{Path(name).name}` | {len(rows)} "
            f"| {len({row['product_id'] for row in rows})} "
            f"| {min(reviews_per_product)}–{max(reviews_per_product)} "
            f"| {percentile(reviews_tokens, 50)} / {percentile(reviews_tokens, 90)} "
            f"/ {int(reviews_tokens.max())} "
            f"| {over_limit} ({share:.0%}) |"
        )

    print()
    print("Длинные товары (отзывы длиннее лимита) и эффект дедупликации:")
    for name in args.files:
        rows = load_rows(Path(name))
        long_rows = [row for row in rows if count_tokens(" ".join(row["reviews"])) > args.limit]
        if not long_rows:
            print(f"- `{Path(name).name}`: длинных товаров нет")
            continue
        reviews_tokens = np.array([count_tokens(" ".join(row["reviews"])) for row in long_rows])
        unique_tokens = np.array(
            [count_tokens(" ".join(dict.fromkeys(row["reviews"]))) for row in long_rows]
        )
        duplicate_share = np.array(
            [1 - len(set(row["reviews"])) / len(row["reviews"]) for row in long_rows]
        )
        fits_after_dedup = int((unique_tokens <= args.limit).sum())
        print(
            f"- `{Path(name).name}`: товаров {len(long_rows)}, "
            f"токенов до дедупликации p50 {percentile(reviews_tokens, 50)}, "
            f"после {percentile(unique_tokens, 50)}, "
            f"доля дубликатов p50 {float(np.median(duplicate_share)):.0%}, "
            f"укладываются после дедупликации {fits_after_dedup} из {len(long_rows)}"
        )


if __name__ == "__main__":
    main()
