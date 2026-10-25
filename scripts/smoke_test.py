"""Проверка запущенного vLLM-сервера: список моделей, холодный и тёплый запрос.

Скрипт нужен, чтобы отделить «сервер отвечает» от «сервер отвечает быстро»:
первый запрос после старта включает прогрев, второй показывает установившееся
время. Токены берутся из `usage` ответа, а не считаются локально.

Пока движок грузит модель, vLLM отвечает `503` на `/v1/models`, а порт уже
слушается: это не поломка. Поэтому сначала ждём `/health` = 200, и только
потом идём в API — иначе первый же запуск сразу после `start_vllm.sh` падает
с `openai.InternalServerError: Error code: 503`.

Запуск (из Windows-окружения проекта, сервер — в WSL2):
    python scripts/smoke_test.py
    python scripts/smoke_test.py --base-url http://127.0.0.1:8000/v1
    python scripts/smoke_test.py --ready-timeout 0     # не ждать, проверить сразу
"""

from __future__ import annotations

import argparse
import time

import httpx
from openai import OpenAI

PROMPT = 'Ответь одним JSON-объектом без пояснений: {"ok": true}'

# Период опроса /health во время загрузки модели: старт на 6 ГБ занимает
# минуты, поэтому частые запросы ничего не ускоряют.
READY_POLL_SEC = 5.0


def wait_until_ready(base_url: str, timeout_sec: float) -> None:
    """
    Ждёт, пока vLLM отдаст `GET /health` = 200.

    Args:
        base_url: базовый путь вида `http://127.0.0.1:8000/v1`.
        timeout_sec: предел ожидания в секундах; `0` — проверить один раз.

    Returns:
        None: функция либо дожидается готовности, либо завершает процесс.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    health_url = f"{root}/health"
    deadline = time.monotonic() + max(0.0, timeout_sec)

    while True:
        try:
            response = httpx.get(health_url, timeout=5.0)
            if response.status_code == 200:
                return
            state = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:  # порт ещё не слушается — тоже ждём
            state = f"{type(exc).__name__}: {exc}"

        if time.monotonic() >= deadline:
            raise SystemExit(
                f"сервер не готов за {timeout_sec:g} с ({state}). "
                f"Проверьте лог vLLM: `pgrep -af 'vllm serve'` в WSL"
            )
        print(f"ждём сервер: {state}; повтор через {READY_POLL_SEC:g} с", flush=True)
        time.sleep(READY_POLL_SEC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test vLLM-сервера")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=600.0,
        help="сколько секунд ждать /health = 200 (0 — не ждать)",
    )
    args = parser.parse_args()

    print(f"base_url: {args.base_url}")
    if args.ready_timeout > 0:
        print(f"ждём готовности сервера (до {args.ready_timeout:g} с)")
    wait_until_ready(args.base_url, args.ready_timeout)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    models = [item.id for item in client.models.list().data]
    print(f"models: {models}")

    for attempt in ("cold", "warm"):
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0,
            max_tokens=args.max_tokens,
        )
        elapsed = time.perf_counter() - started
        content = response.choices[0].message.content or ""
        print(
            f"{attempt}: {elapsed:.2f} s, "
            f"prompt_tokens={response.usage.prompt_tokens}, "
            f"completion_tokens={response.usage.completion_tokens}"
        )
        print(f"  content: {content.strip()[:200]}")


if __name__ == "__main__":
    main()
