"""Подбор параметров генерации: один фактор за раз на фиксированной подвыборке.

Скрипт не меняет продакшн-настройки сам: он печатает CSV с результатами
экспериментов, а выбранный вариант переносится в `run_batch.py` вручную и
фиксируется в README. Все конфигурации проверяются на одном и том же наборе
товаров с `max_attempts=1`: измеряется готовность модели соблюдать формат с
первого ответа, а не умение исправляющей попытки вытянуть брак.

Запуск:
    python tune_params.py --input dev.jsonl --rows 20 --concurrency 16
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer

from cardgen import DEFAULT_LIMIT, DEFAULT_MODEL, load_rows
from run_batch import CardResult, generate_card, make_client

# Базовая точка — детерминированная генерация: для строгого JSON это дешёвый
# способ получить воспроизводимый результат. Остальные конфигурации меняют
# ровно один параметр, чтобы эффект был объяснимым.
EXPERIMENTS: list[dict] = [
    {
        "name": "base",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
    },
    {
        "name": "temp_0.3_top_p_0.9",
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 0,
        "repetition_penalty": 1.0,
    },
    {
        "name": "rep_penalty_1.05",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.05,
    },
    {
        "name": "rep_penalty_1.1",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.1,
    },
    {
        "name": "max_tokens_250",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_tokens": 250,
    },
]


def __run_experiment(
    client: OpenAI,
    tokenizer,
    rows: list[dict],
    args: argparse.Namespace,
    experiment: dict,
) -> dict:
    """
    Прогоняет один вариант параметров и возвращает строку будущего CSV.

    Args:
        client: клиент
        tokenizer:
        rows: строчки
        args: аргументы скрипта
        experiment: эксперимент

    Returns:
        dict: результаты эксперимента
    """
    config = argparse.Namespace(**vars(args))
    for key, value in experiment.items():
        if key != "name":
            setattr(config, key, value)
    config.max_attempts = 1

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        results: list[CardResult] = list(
            pool.map(lambda row: generate_card(client, tokenizer, row, config), rows)
        )
    elapsed = time.perf_counter() - started

    valid = [result for result in results if result.valid]
    descriptions = [
        result.card["description"]
        for result in valid
        if isinstance(result.card, dict) and result.card.get("description")
    ]
    errors: dict[str, int] = {}
    for result in results:
        for error in result.validation_errors:
            errors[error] = errors.get(error, 0) + 1

    return {
        "config": experiment["name"],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": config.repetition_penalty,
        "max_tokens": config.max_tokens,
        "rows": len(results),
        "valid_rate": round(len(valid) / len(results), 4) if results else 0.0,
        "mean_description_chars": (
            round(statistics.mean(len(text) for text in descriptions), 1)
            if descriptions
            else 0.0
        ),
        "mean_output_tokens": round(
            statistics.mean(r.output_tokens for r in results), 1
        ),
        "hits_max_tokens": sum(
            1 for result in results if result.output_tokens >= config.max_tokens
        ),
        "elapsed_sec": round(elapsed, 2),
        "cards_per_sec": round(len(results) / elapsed, 3) if elapsed else 0.0,
        "errors_by_reason": "; ".join(
            f"{name}={count}" for name, count in sorted(errors.items())
        ),
    }


def __parse_args() -> argparse.Namespace:
    """
    Разобрать аргументы

    Returns:
        argparse.Namespace: список аргументов
    """
    parser = argparse.ArgumentParser(
        description="Подбор параметров генерации на dev-выборке"
    )
    parser.add_argument("--input", default="dev.jsonl")
    parser.add_argument(
        "--rows", type=int, default=20, help="сколько товаров брать из начала файла"
    )
    parser.add_argument("--output", default="outputs/parameter_experiments.csv")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = __parse_args()
    rows = load_rows(Path(args.input))[: max(1, args.rows)]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    client = make_client(args.base_url, args.api_key)

    records = []
    for experiment in EXPERIMENTS:
        record = __run_experiment(client, tokenizer, rows, args, experiment)
        records.append(record)
        print(
            f"{record['config']}: valid_rate={record['valid_rate']:.3f}, "
            f"описание {record['mean_description_chars']} симв., "
            f"{record['mean_output_tokens']} токенов, {record['cards_per_sec']} карточек/с",
            flush=True,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"эксперименты: {output_path}")


if __name__ == "__main__":
    main()
