"""Проверка запущенного vLLM-сервера: список моделей, холодный и тёплый запрос.

Скрипт нужен, чтобы отделить «сервер отвечает» от «сервер отвечает быстро»:
первый запрос после старта включает прогрев, второй показывает установившееся
время. Токены берутся из `usage` ответа, а не считаются локально.

Запуск (из Windows-окружения проекта, сервер — в WSL2):
    python scripts/smoke_test.py
    python scripts/smoke_test.py --base-url http://127.0.0.1:8000/v1
"""
from __future__ import annotations

import argparse
import time

from openai import OpenAI

PROMPT = 'Ответь одним JSON-объектом без пояснений: {"ok": true}'


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test vLLM-сервера")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    models = [item.id for item in client.models.list().data]
    print(f"base_url: {args.base_url}")
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
