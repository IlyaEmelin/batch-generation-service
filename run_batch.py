"""Пакетная генерация карточек товаров через vLLM (OpenAI-совместимый API).

Скрипт читает JSONL с товарами, собирает вход в бюджет 1500 токенов, генерирует
карточку, проверяет её контракт и при нарушении формата делает исправляющую
попытку с перечнем найденных нарушений. Результат пишется в порядке входа:
одна строка `predictions.jsonl` на каждый `product_id`.

Запуск:
    python run_batch.py --input dev.jsonl --output outputs/dev_predictions.jsonl
    python run_batch.py --input benchmark.jsonl --concurrency 8
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import OpenAI
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from cardgen import (
    DEFAULT_LIMIT,
    DEFAULT_MODEL,
    corrective_message,
    count_chat_tokens,
    load_rows,
    prepare_prompt,
    validate_card,
)

# Столько токенов держим в запасе, чтобы исправляющий запрос с перечнем
# нарушений тоже уложился в лимит 1500.
RETRY_RESERVE = 150


@dataclass
class CardResult:
    """Результат обработки одного товара, включая служебные замеры."""

    product_id: str
    valid: bool
    attempts: int
    validation_errors: list[str]
    attempt_errors: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    planned_input_tokens: int = 0
    latency_sec: float = 0.0
    request_sec: float = 0.0
    reviews_total: int = 0
    reviews_used: int = 0
    truncated: bool = False
    card: dict | None = None
    raw_response: str = ""
    error: str | None = None

    def to_row(self) -> dict:
        """Строка `predictions.jsonl`: карточка и метрики в одном объекте."""
        card = self.card or {}
        return {
            "product_id": self.product_id,
            "valid": self.valid,
            "attempts": self.attempts,
            "validation_errors": self.validation_errors,
            "attempt_errors": self.attempt_errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "planned_input_tokens": self.planned_input_tokens,
            "latency_sec": round(self.latency_sec, 3),
            "request_sec": round(self.request_sec, 3),
            "reviews_total": self.reviews_total,
            "reviews_used": self.reviews_used,
            "truncated": self.truncated,
            "description": card.get("description"),
            "pros": card.get("pros"),
            "cons": card.get("cons"),
            "tags": card.get("tags"),
            "raw_response": self.raw_response,
            "error": self.error,
        }


def make_client(base_url: str, api_key: str) -> OpenAI:
    """
    Клиент OpenAI API с таймаутом, достаточным для длинной генерации.

    Args:
        base_url: базовый путь
        api_key: апи кей

    Returns:
        OpenAI: Клиент OpenAI API
    """
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=httpx.Timeout(120.0, connect=10.0),
        max_retries=0,
    )


def generate_card(
    client: OpenAI,
    tokenizer,
    row: dict,
    args: argparse.Namespace,
) -> CardResult:
    """
    Генерирует карточку одного товара с повторами только по нарушению формата.
    TODO: похоже эту функцию надо перенести в файл 'cardgen.py'

    Args:
        client: клиент ИИ агента
        tokenizer: токенизатор
        row: прочитанная строка
        args: аргументы запуска генерации карточек

    Returns:
        CardResult: сгенерированная карточка товара
    """
    reserve = RETRY_RESERVE if args.max_attempts > 1 else 0
    prepared = prepare_prompt(row, tokenizer, limit=args.limit - reserve)
    messages = prepared.messages

    result = CardResult(
        product_id=row["product_id"],
        valid=False,
        attempts=0,
        validation_errors=[],
        reviews_total=prepared.stats.reviews_total,
        reviews_used=prepared.stats.reviews_used,
        truncated=prepared.stats.truncated,
    )

    for attempt in range(1, args.max_attempts + 1):
        planned = count_chat_tokens(tokenizer, messages)
        result.planned_input_tokens = max(result.planned_input_tokens, planned)
        if planned > args.limit:
            raise ValueError(
                f"{row['product_id']}: запрос {planned} токенов > {args.limit}"
            )

        extra_body: dict = {}
        if args.top_k > 0:
            extra_body["top_k"] = args.top_k
        if args.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = args.repetition_penalty

        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature if attempt == 1 else 0.0,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body=extra_body or None,
            )
        except Exception as exc:  # сетевые ошибки и 5xx: не роняем весь пакет
            result.error = f"{type(exc).__name__}: {exc}"
            result.validation_errors = ["request_error"]
            result.attempts = attempt
            return result

        result.request_sec = time.perf_counter() - started
        raw = response.choices[0].message.content or ""
        usage = response.usage
        result.input_tokens = usage.prompt_tokens if usage else planned
        result.output_tokens = usage.completion_tokens if usage else 0
        result.total_input_tokens += result.input_tokens
        result.total_output_tokens += result.output_tokens
        result.attempts = attempt
        result.raw_response = raw

        checked = validate_card(raw, row["product_id"])
        result.valid = checked.valid
        result.validation_errors = checked.errors
        result.card = checked.card
        if checked.valid:
            return result
        # Причины повторов копим отдельно: они нужны для разбора брака, даже
        # если исправляющая попытка в итоге удалась.
        result.attempt_errors.extend(checked.errors)

        # Повтор только по формату: смысловое качество автоматически не проверяем.
        messages = prepared.messages + [
            corrective_message(checked.errors, row["product_id"])
        ]

    return result


def __percentile(values: list[float], q: float) -> float:
    """
    Возвращает q-перцентиль методом ближайшего ранга.

    Метод не использует интерполяцию: результатом всегда будет одно из
    фактически наблюдавшихся значений. Если список пустой, возвращает 0.0.

    Args:
        values: значения
        q: Доля от 0.0 до 1.0, например 0.5 для p50 или 0.95 для p95.

    Returns:
        float: Значение q-перцентиля.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def __environment_info() -> dict:
    """
    Окружение замера: без него числа отчёта нечем воспроизвести.

    Returns:
        dict: данные об окружении замера
    """

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    gpu = None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        gpu = completed.stdout.strip() or None
    except Exception:  # nvidia-smi может отсутствовать в PATH: это не повод падать
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "openai": package_version("openai"),
    }


def __server_version(base_url: str) -> str | None:
    """
    Версия vLLM из `/version`: часть окружения в отчёте

    Args:
        base_url: базовый url к в vLLM

    Returns:
        str | None: версия vLLM
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        response = httpx.get(f"{root}/version", timeout=5.0)
        response.raise_for_status()
        return response.json().get("version")
    except Exception:
        return None


def __parse_args() -> argparse.Namespace:
    """
    Разбор аргументов скрипта

    Returns:
        argparse.Namespace: аргументы скрипта
    """
    parser = argparse.ArgumentParser(
        description="Пакетная генерация карточек товаров через vLLM"
    )
    parser.add_argument("--input", default="benchmark.jsonl")
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument(
        "--summary", default=None, help="сводка запуска (по умолчанию рядом с output)"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_MODEL)
    # 16 — измеренная точка насыщения (см. таблицу concurrency в README): на 32
    # очередь запросов растёт, а throughput падает.
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--top-k", type=int, default=0, help="0 — не передавать параметр"
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument(
        "--rows", type=int, default=0, help="взять первые N товаров (0 — все)"
    )
    return parser.parse_args()


def __build_summary(
    results: list[CardResult],
    elapsed: float,
    args: argparse.Namespace,
    started_at: datetime,
    version: str | None,
) -> dict:
    """
    Сводка запуска: то, из чего потом собирается `outputs/report.json`.

    Args:
        results: карточки результатов
        elapsed: время обработки результатов ИИ агентом
        args: аргументы запуска скрипта
        started_at: время запуска обработки карточек
        version: версия vLLM сервера.

    Returns:
        dict: сводка результат
    """
    errors: dict[str, int] = {}
    for result in results:
        for error in result.validation_errors:
            errors[error] = errors.get(error, 0) + 1

    retry_errors: dict[str, int] = {}
    for result in results:
        for error in result.attempt_errors:
            retry_errors[error] = retry_errors.get(error, 0) + 1

    latencies = [result.latency_sec for result in results]
    requests = [result.request_sec for result in results]
    valid = sum(1 for result in results if result.valid)
    retried = sum(1 for result in results if result.attempts > 1)
    return {
        "input_file": str(args.input),
        "rows": len(results),
        "started_at": started_at.isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed, 2),
        "throughput_cards_per_sec": (
            round(len(results) / elapsed, 3) if elapsed else 0.0
        ),
        "concurrency": args.concurrency,
        "params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_tokens,
            "max_attempts": args.max_attempts,
            "limit": args.limit,
            "model": args.model,
        },
        "server": {"base_url": args.base_url, "vllm_version": version},
        "environment": __environment_info(),
        "valid_rate": round(valid / len(results), 4) if results else 0.0,
        "retry_rate": round(retried / len(results), 4) if results else 0.0,
        "errors_by_reason": dict(sorted(errors.items(), key=lambda item: -item[1])),
        "retry_errors_by_reason": dict(
            sorted(retry_errors.items(), key=lambda item: -item[1])
        ),
        "latency_mean_sec": (
            round(sum(latencies) / len(latencies), 3) if latencies else 0.0
        ),
        "latency_p95_sec": round(__percentile(latencies, 0.95), 3),
        "request_mean_sec": (
            round(sum(requests) / len(requests), 3) if requests else 0.0
        ),
        "request_p95_sec": round(__percentile(requests, 0.95), 3),
        "total_input_tokens": sum(result.total_input_tokens for result in results),
        "total_output_tokens": sum(result.total_output_tokens for result in results),
        "max_planned_input_tokens": max(
            (result.planned_input_tokens for result in results), default=0
        ),
        "max_input_tokens": max((result.input_tokens for result in results), default=0),
        "truncated_rows": sum(1 for result in results if result.truncated),
    }


def __generate_cards_concurrently(
    args: Namespace,
    client: OpenAI,
    rows: list[dict],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[CardResult], float]:
    """
    Параллельная генерация карточек товаров

    Args:
        args: аргументы скрипта
        client: клиент OpenAI, для обращения к внешнему сервису
        rows: список строк товаров с отзывами
        tokenizer: токенизатор

    Returns:
        list[CardResult]: Сгенерированные карточки товаров
        float: времея генерации карточек товаров.
    """
    batch_started = time.perf_counter()
    results: list[CardResult | None] = [None] * len(rows)
    done = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {}
        for index, row in enumerate(rows):
            submitted_at = time.perf_counter()
            pending[pool.submit(generate_card, client, tokenizer, row, args)] = (
                index,
                submitted_at,
            )

        for future in as_completed(pending):
            index, submitted_at = pending[future]
            result = future.result()
            result.latency_sec = time.perf_counter() - submitted_at
            results[index] = result
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"  готово {done}/{len(rows)}", flush=True)

    elapsed = time.perf_counter() - batch_started
    finished = [result for result in results if result is not None]
    return finished, elapsed


def __write_card_results_jsonl(args: Namespace, finished: list[CardResult]) -> Path:
    """
    Запись карточек результатов в отдельный файл

    Args:
        args: аргументы вызова скрипта
        finished: карточки результатов

    Returns:
        Path: путь к выходному файлу
    """
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in finished:
            handle.write(json.dumps(result.to_row(), ensure_ascii=False) + "\n")
    return output_path


def main() -> None:
    args = __parse_args()
    rows = load_rows(Path(args.input))

    if args.rows:
        rows = rows[: args.rows]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    client = make_client(args.base_url, args.api_key)

    print(
        f"товаров: {len(rows)}, concurrency={args.concurrency}, "
        f"max_attempts={args.max_attempts}, limit={args.limit}",
        flush=True,
    )

    started_at = datetime.now(timezone.utc)
    finished, elapsed = __generate_cards_concurrently(args, client, rows, tokenizer)
    output_path = __write_card_results_jsonl(args, finished)

    # Создание итоговой сводки
    summary = __build_summary(
        finished,
        elapsed,
        args,
        started_at,
        __server_version(args.base_url),
    )
    summary_path = (
        Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"готово: {len(finished)} карточек за {elapsed:.1f} с "
        f"({summary['throughput_cards_per_sec']} карточек/с), "
        f"valid_rate={summary['valid_rate']:.3f}, retry_rate={summary['retry_rate']:.3f}"
    )
    print(f"predictions: {output_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
