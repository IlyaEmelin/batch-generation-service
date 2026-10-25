"""Сравнение локального сервиса с внешним API на одной задаче.

Используется тот же сборщик входа и тот же валидатор, что и в `run_batch.py`,
поэтому различия в результате относятся к модели, а не к промпту. Сетевые
повторы (429, 5xx, обрыв) и повторы из-за формата считаются отдельно: иначе
цена нестабильной сети выглядела бы как цена модели.

Запуск:
    python compare_api.py --rows 30
    python compare_api.py --model openai/gpt-oss-20b --concurrency 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import openai
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer

from cardgen import (
    DEFAULT_LIMIT,
    DEFAULT_MODEL,
    corrective_message,
    count_chat_tokens,
    load_rows,
    prepare_prompt,
    validate_card,
)

RETRY_RESERVE = 150
API_BASE_URL = "https://openrouter.ai/api/v1"
API_MODEL = "openai/gpt-oss-20b"
MAX_NETWORK_ATTEMPTS = 5


def __subset(rows: list[dict], count: int) -> list[dict]:
    """
    Детерминированная подвыборка: равномерно по всему файлу, без случайности.

    Args:
        rows: список строчек
        count: итоговое количество строчек

    Returns:
        list[dict]: итоговые объекты которые вошли в выборку
    """
    if count <= 0 or count >= len(rows):
        return rows
    step = len(rows) / count
    return [rows[int(index * step)] for index in range(count)]


def __request_with_backoff(
    client: OpenAI,
    args: argparse.Namespace,
    messages: list[dict],
    temperature: float,
) -> tuple[object | None, str | None, int]:
    """
    Запрос с экспоненциальной задержкой на временных ошибках.

    `400`, `401`, `403` не повторяем: это ошибка запроса или ключа, ожидание её
    не исправит. Возвращает (ответ, текст ошибки, число сетевых повторов).

    Args:
        client: ИИ - клиент
        args: аргументы запуска скрипта
        messages: сообщение к агенту ИИ
        temperature: температура агента ИИ

    Returns:
        object | None: ответ модели
        str | None: последняя ошибка
        int: количество попыток
    """
    delay = 2.0
    last_error: str | None = None
    for attempt in range(1, MAX_NETWORK_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=temperature,
                max_tokens=args.max_tokens,
            )
            return response, None, attempt - 1
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except openai.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:
                last_error = f"status {exc.status_code}"
            else:
                return None, f"status {exc.status_code}: {exc}", attempt - 1
        if attempt == MAX_NETWORK_ATTEMPTS:
            break
        time.sleep(min(delay, 30.0))
        delay *= 2
    return None, last_error, MAX_NETWORK_ATTEMPTS - 1


def __compare(args: argparse.Namespace) -> dict:
    """
    Прогоняет подвыборку через API и возвращает сводку для `report.json`.

    Args:
        args: аргументы скрипта

    Returns:
        dict: результат сравнения
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("не найден OPENROUTER_API_KEY в окружении или .env")

    rows = __subset(load_rows(Path(args.input)), args.rows)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=httpx.Timeout(180.0, connect=15.0),
        max_retries=0,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: list[dict] = []
    errors_by_reason: dict[str, int] = {}
    request_errors: dict[str, int] = {}
    started_at = datetime.now(timezone.utc)
    batch_started = time.perf_counter()

    for number, row in enumerate(rows, start=1):
        reserve = RETRY_RESERVE if args.max_attempts > 1 else 0
        prepared = prepare_prompt(row, tokenizer, limit=args.limit - reserve)
        messages = prepared.messages
        record = {
            "product_id": row["product_id"],
            "valid": False,
            "attempts": 0,
            "network_retries": 0,
            "validation_errors": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "raw_response": "",
            "error": None,
        }

        for attempt in range(1, args.max_attempts + 1):
            planned = count_chat_tokens(tokenizer, messages)
            if planned > args.limit:
                raise ValueError(
                    f"{row['product_id']}: запрос {planned} токенов > {args.limit}"
                )
            response, error, retries = __request_with_backoff(
                client, args, messages, args.temperature if attempt == 1 else 0.0
            )
            record["network_retries"] += retries
            record["attempts"] = attempt
            if response is None:
                record["error"] = error
                record["validation_errors"] = ["request_error"]
                break

            usage = response.usage
            record["input_tokens"] = usage.prompt_tokens if usage else planned
            record["output_tokens"] = usage.completion_tokens if usage else 0
            record["total_input_tokens"] += record["input_tokens"]
            record["total_output_tokens"] += record["output_tokens"]
            raw = response.choices[0].message.content or ""
            record["raw_response"] = raw

            checked = validate_card(raw, row["product_id"])
            record["valid"] = checked.valid
            record["validation_errors"] = checked.errors
            if checked.valid:
                break
            messages = prepared.messages + [
                corrective_message(checked.errors, row["product_id"])
            ]

        for reason in record["validation_errors"]:
            errors_by_reason[reason] = errors_by_reason.get(reason, 0) + 1
        if record["error"]:
            key = record["error"].split(",")[0]
            request_errors[key] = request_errors.get(key, 0) + 1
        predictions.append(record)
        print(
            f"  {number}/{len(rows)} {row['product_id']} valid={record['valid']}",
            flush=True,
        )

    elapsed = time.perf_counter() - batch_started
    with output_path.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    valid = sum(1 for record in predictions if record["valid"])
    return {
        "model": args.model,
        "api_run_date": started_at.date().isoformat(),
        "base_url": args.base_url,
        "rows": len(predictions),
        "subset": "каждый k-й товар входного файла",
        "elapsed_sec": round(elapsed, 2),
        "valid_rate": round(valid / len(predictions), 4) if predictions else 0.0,
        "retry_rate": (
            round(
                sum(1 for record in predictions if record["attempts"] > 1)
                / len(predictions),
                4,
            )
            if predictions
            else 0.0
        ),
        "network_retries": sum(record["network_retries"] for record in predictions),
        "errors_by_reason": dict(
            sorted(errors_by_reason.items(), key=lambda item: -item[1])
        ),
        "total_input_tokens": sum(
            record["total_input_tokens"] for record in predictions
        ),
        "total_output_tokens": sum(
            record["total_output_tokens"] for record in predictions
        ),
        "request_error_rows": sum(1 for record in predictions if record["error"]),
        "request_errors": dict(
            sorted(request_errors.items(), key=lambda item: -item[1])
        ),
    }


def __parse_args() -> argparse.Namespace:
    """
    Разобрать аргументы скрипта

    Returns:
        argparse.Namespace: разобранные аргументы скрипта
    """
    parser = argparse.ArgumentParser(
        description="Сравнение локального сервиса с внешним API"
    )
    parser.add_argument("--input", default="benchmark.jsonl")
    parser.add_argument("--rows", type=int, default=30, help="размер подвыборки")
    parser.add_argument("--output", default="outputs/api_predictions.jsonl")
    parser.add_argument("--summary", default="outputs/api_summary.json")
    parser.add_argument("--base-url", default=API_BASE_URL)
    parser.add_argument("--model", default=API_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=350)
    return parser.parse_args()


def main() -> None:
    args = __parse_args()
    summary = __compare(args)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"готово: {summary['rows']} карточек, valid_rate={summary['valid_rate']:.3f}, "
        f"сетевых повторов {summary['network_retries']}"
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
