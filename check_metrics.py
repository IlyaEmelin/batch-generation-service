"""Пересчёт метрик из сохранённых результатов и сборка `outputs/report.json`.

Скрипт ничего не знает про модель: он читает `predictions.jsonl`, сводки
прогонов и (если есть) результаты API-сравнения. Поэтому все числа отчёта
проверяемы без запуска генерации.

Запуск:
    python check_metrics.py
    python check_metrics.py --predictions outputs/predictions.jsonl --startup-sec 108
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from cardgen import load_rows

TARIFF_RUB_PER_HOUR = 65.0
PRICE_TARGET_RUB_PER_1000 = 25.0
API_INPUT_RUB_PER_MILLION = 20.0
API_OUTPUT_RUB_PER_MILLION = 60.0
SLA_SEC = 480.0
MONTHLY_CARDS = 50_000


def __read_json(path: Path) -> dict | None:
    """
    Читает JSON-файл и возвращает его содержимое как словарь.

    Если файл не существует, возвращает `None`.

    Args:
        path: Путь к JSON-файлу.

    Returns:
        Словарь с данными из JSON-файла или `None`, если файл не найден.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def __read_experiments(path: Path) -> list[dict] | None:
    """
    Читает CSV-файл с результатами подбора параметров.

    Значения из CSV автоматически приводятся к числовому виду, если это возможно:
    сначала выполняется попытка преобразования в `int`, затем в `float`.
    Если значение нельзя преобразовать в число, оно остаётся строкой.

    Если файл не существует, возвращает `None`.

    Args:
        path: Путь к CSV-файлу с результатами экспериментов.

    Returns:
        list[dict] | None: Список словарей, где каждый словарь соответствует одной строке CSV-файла,
        или `None`, если файл не найден.
    """
    if not path.exists():
        return None

    def cast(value: str) -> object:
        for caster in (int, float):
            try:
                return caster(value)
            except ValueError:
                continue
        return value

    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: cast(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def __cost_vllm_per_1000(elapsed_sec: float, cards: int) -> float:
    """
    Рассчитывает стоимость генерации 1000 карточек на локальной ВМ.

    Расчёт выполняется по тарифу `TARIFF_RUB_PER_HOUR` на основе фактического
    времени обработки батча и количества сгенерированных карточек.

    Args:
        elapsed_sec: Время обработки батча в секундах.
        cards: Количество обработанных карточек.

    Returns:
        float: Стоимость генерации 1000 карточек в рублях. Если `cards` равно 0,
        возвращает `0.0`.
    """
    if not cards:
        return 0.0
    return elapsed_sec / 3600 * TARIFF_RUB_PER_HOUR / cards * 1000


def __cost_api_per_1000(input_tokens: int, output_tokens: int, cards: int) -> float:
    """
    Рассчитывает стоимость генерации 1000 карточек через внешний API.

    Стоимость считается по фактическому usage: количеству входных и выходных
    токенов. Для расчёта используются тарифы `API_INPUT_RUB_PER_MILLION` и
    `API_OUTPUT_RUB_PER_MILLION`.

    Args:
        input_tokens: Общее количество входных токенов.
        output_tokens: Общее количество выходных токенов.
        cards: Количество обработанных карточек.

    Returns:
        float: Стоимость генерации 1000 карточек в рублях. Если `cards` равно 0,
        возвращает `0.0`.
    """
    if not cards:
        return 0.0
    return (
        (
            input_tokens * API_INPUT_RUB_PER_MILLION
            + output_tokens * API_OUTPUT_RUB_PER_MILLION
        )
        / 1_000_000
        / cards
        * 1000
    )


def __compute_metrics(rows: list[dict], input_rows: list[dict] | None) -> dict:
    """
    Считает итоговые метрики по строкам файла `predictions.jsonl`.

    Функция собирает статистику валидности, повторных попыток, ошибок,
    порядка строк, дублей `product_id`, токенов и длины описаний.

    Args:
        rows: Список сгенерированных строк из `predictions.jsonl`.
            Ожидаемые поля в строках:
            - `product_id`: идентификатор товара.
            - `valid`: признак валидности результата.
            - `validation_errors`: список ошибок валидации.
            - `attempt_errors`: список ошибок по попыткам генерации.
            - `attempts`: количество попыток генерации.
            - `input_tokens`: фактическое количество входных токенов.
            - `planned_input_tokens`: запланированное количество входных токенов.
            - `total_input_tokens`: суммарное количество входных токенов.
            - `total_output_tokens`: суммарное количество выходных токенов.
            - `truncated`: признак обрезки отзывов/контекста.
            - `description`: сгенерированное описание товара.
        input_rows: Исходные входные строки или `None`, если исходный файл
            недоступен. Используется для проверки сохранения порядка
            `product_id` относительно входных данных.

    Returns:
        Словарь с рассчитанными метриками. Формат результата:

        ```python
        {
            "rows": int,
            "valid_count": int,
            "valid_rate": float,
            "retry_rate": float,
            "errors_by_reason": dict[str, int],
            "retry_errors_by_reason": dict[str, int],
            "attempts_histogram": dict[str, int],
            "duplicate_product_ids": int,
            "order_matches_input": bool,
            "max_input_tokens": int,
            "max_planned_input_tokens": int,
            "rows_over_input_limit": int,
            "total_input_tokens": int,
            "total_output_tokens": int,
            "truncated_rows": int,
            "mean_description_chars": float,
        }
        ```
    """
    errors: dict[str, int] = {}
    for row in rows:
        for error in row.get("validation_errors") or []:
            errors[error] = errors.get(error, 0) + 1

    retry_errors: dict[str, int] = {}
    for row in rows:
        for error in row.get("attempt_errors") or []:
            retry_errors[error] = retry_errors.get(error, 0) + 1

    attempts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("attempts", 0))
        attempts[key] = attempts.get(key, 0) + 1

    ids = [row["product_id"] for row in rows]
    expected = [row["product_id"] for row in input_rows] if input_rows else ids
    over_limit = [row for row in rows if (row.get("planned_input_tokens") or 0) > 1500]
    valid = sum(1 for row in rows if row.get("valid"))

    return {
        "rows": len(rows),
        "valid_count": valid,
        "valid_rate": round(valid / len(rows), 4) if rows else 0.0,
        "retry_rate": (
            round(
                sum(1 for row in rows if (row.get("attempts") or 0) > 1) / len(rows), 4
            )
            if rows
            else 0.0
        ),
        "errors_by_reason": dict(sorted(errors.items(), key=lambda item: -item[1])),
        "retry_errors_by_reason": dict(
            sorted(retry_errors.items(), key=lambda item: -item[1])
        ),
        "attempts_histogram": dict(
            sorted(attempts.items(), key=lambda item: int(item[0]))
        ),
        "duplicate_product_ids": len(ids) - len(set(ids)),
        "order_matches_input": ids == expected,
        "max_input_tokens": (
            max((row.get("input_tokens") or 0) for row in rows) if rows else 0
        ),
        "max_planned_input_tokens": (
            max((row.get("planned_input_tokens") or 0) for row in rows) if rows else 0
        ),
        "rows_over_input_limit": len(over_limit),
        "total_input_tokens": sum(row.get("total_input_tokens") or 0 for row in rows),
        "total_output_tokens": sum(row.get("total_output_tokens") or 0 for row in rows),
        "truncated_rows": sum(1 for row in rows if row.get("truncated")),
        "mean_description_chars": (
            round(
                sum(len(row["description"]) for row in rows if row.get("description"))
                / max(valid, 1),
                1,
            )
            if rows
            else 0.0
        ),
    }


def __build_concurrency_table(summaries: list[dict]) -> list[dict]:
    """
    Строит таблицу сравнения запусков с разными значениями concurrency.

    На вход принимает список сводок отдельных прогонов и возвращает таблицу
    `concurrency -> throughput / latency / valid_rate`, отсортированную по
    возрастанию значения `concurrency`.

    Args:
        summaries: Список сводок прогонов. Каждый словарь может содержать поля:
            - `concurrency`: уровень параллельности запуска.
            - `rows`: количество обработанных строк.
            - `elapsed_sec`: общее время выполнения в секундах.
            - `throughput_cards_per_sec`: скорость генерации карточек в секунду.
            - `latency_mean_sec`: средняя latency одной карточки.
            - `latency_p95_sec`: p95 latency одной карточки.
            - `request_mean_sec`: среднее время API/vLLM-запроса.
            - `valid_rate`: доля валидных карточек.
            - `retry_rate`: доля карточек, потребовавших повторной генерации.
            - `errors_by_reason`: словарь ошибок валидации по причинам.

    Returns:
        Список словарей с агрегированной таблицей по concurrency.
        Формат одной строки:

        ```python
        {
            "concurrency": int | None,
            "rows": int | None,
            "elapsed_sec": float | None,
            "throughput_cards_per_sec": float | None,
            "latency_mean_sec": float | None,
            "latency_p95_sec": float | None,
            "request_mean_sec": float | None,
            "valid_rate": float | None,
            "retry_rate": float | None,
            "errors_total": int,
        }
        ```
    """
    table = []
    for summary in sorted(summaries, key=lambda item: item.get("concurrency", 0)):
        table.append(
            {
                "concurrency": summary.get("concurrency"),
                "rows": summary.get("rows"),
                "elapsed_sec": summary.get("elapsed_sec"),
                "throughput_cards_per_sec": summary.get("throughput_cards_per_sec"),
                "latency_mean_sec": summary.get("latency_mean_sec"),
                "latency_p95_sec": summary.get("latency_p95_sec"),
                "request_mean_sec": summary.get("request_mean_sec"),
                "valid_rate": summary.get("valid_rate"),
                "retry_rate": summary.get("retry_rate"),
                "errors_total": sum((summary.get("errors_by_reason") or {}).values()),
            }
        )
    return table


def __markdown_fenced_count(rows: list[dict]) -> int:
    """
    Считает ответы, которые начинаются с Markdown-ограждения ```.

    Валидатор такие ответы принимает: `cardgen.__strip_code_fences` снимает
    обёртку до разбора JSON. Но частота обёрток показывает, насколько модель
    соблюдает требование «без Markdown», поэтому она попадает в отчёт отдельно.

    Args:
        rows: строки `predictions.jsonl` или `api_predictions.jsonl`

    Returns:
        int: количество ответов с ограждением
    """
    return sum(
        1 for row in rows if (row.get("raw_response") or "").lstrip().startswith("```")
    )


def __block(items: list[dict] | None) -> dict[str, int | float] | None:
    if items is None:
        return None
    responses = sum(1 for row in items if row.get("raw_response"))
    fenced = __markdown_fenced_count(items)
    return {
        "responses": responses,
        "markdown_fenced": fenced,
        "markdown_fenced_rate": (round(fenced / responses, 4) if responses else 0.0),
    }


def __format_postprocessing(rows: list[dict], api_rows: list[dict] | None) -> dict:
    """
    Сводка по постобработке ответов: сколько раз снимались Markdown-ограждения.

    Args:
        rows: локальные строки `predictions.jsonl`
        api_rows: строки `api_predictions.jsonl` или `None`, если сравнения не было

    Returns:
        dict: блок `format_postprocessing` для `report.json`
    """
    return {"local": __block(rows), "api": __block(api_rows)}


def __tokens(row: dict, key: str) -> int:
    return int(row.get(f"total_{key}") or row.get(key) or 0)


def __local_subset(rows: list[dict], api_rows: list[dict] | None) -> dict | None:
    """
    Метрики локального прогона на тех же товарах, что попали в API-подвыборку.

    Сравнивать полный локальный батч с 30 карточками API некорректно: наборы
    товаров разные. Здесь локальные строки отбираются по `product_id` из
    `api_predictions.jsonl`, чтобы valid_rate и токены считались на одном наборе.

    Args:
        rows: локальные строки `predictions.jsonl`
        api_rows: строки `api_predictions.jsonl` или `None`

    Returns:
        dict | None: метрики подвыборки или `None`, если пересечения нет
    """
    if not api_rows:
        return None
    by_id = {row["product_id"]: row for row in rows}
    matched = [
        by_id[row["product_id"]] for row in api_rows if row.get("product_id") in by_id
    ]
    if not matched:
        return None

    valid = sum(1 for row in matched if row.get("valid"))
    return {
        "rows": len(matched),
        "valid_count": valid,
        "valid_rate": round(valid / len(matched), 4),
        "input_tokens_per_card": round(
            mean(__tokens(row, "input_tokens") for row in matched), 1
        ),
        "output_tokens_per_card": round(
            mean(__tokens(row, "output_tokens") for row in matched), 1
        ),
        "request_mean_sec": round(
            mean(float(row.get("request_sec") or 0.0) for row in matched), 2
        ),
        "latency_mean_sec": round(
            mean(float(row.get("latency_sec") or 0.0) for row in matched), 2
        ),
        "markdown_fenced": __markdown_fenced_count(matched),
        "note": (
            "Локальные карточки на тех же product_id, что и в API-подвыборке. "
            "Токены — сумма по всем попыткам, как total_* в api_comparison; "
            "latency_mean_sec включает ожидание в очереди пула, request_mean_sec — "
            "только HTTP-запрос; стоимость берётся из полного прогона, потому что "
            "тариф почасовой"
        ),
    }


def __build_costs(main: dict, api: dict | None) -> dict:
    """
    Рассчитывает стоимость генерации карточек для локального vLLM и внешнего API.

    Считает стоимость на 1000 карточек, месячную стоимость для сценария
    `MONTHLY_CARDS`, а также проверяет, укладывается ли локальная генерация
    в целевой тариф `PRICE_TARGET_RUB_PER_1000`.

    Args:
        main: Сводка основного локального запуска vLLM. Ожидаемые поля:
            - `rows`: количество обработанных карточек.
            - `elapsed_sec`: общее время обработки батча в секундах.
        api: Сводка запуска через внешний API или `None`, если сравнение с API
            не выполнялось. Если передан словарь, ожидаемые поля:
            - `rows`: количество обработанных карточек через API.
            - `model`: название API-модели.
            - `total_input_tokens`: суммарное количество входных токенов.
            - `total_output_tokens`: суммарное количество выходных токенов.
            Если суммарный фактический usage равен нулю, считаем стоимость API
            недоступной и оставляем `costs.api = None`: ноль токенов означает,
            что запросы не дошли до модели или провайдер не вернул usage, а не
            что API бесплатный.

    Returns:
        Словарь с расчётом стоимости. Формат результата:

        ```python
        {
            "tariff_rub_per_hour": float,
            "price_target_rub_per_1000": float,
            "vllm": {
                "cards": int,
                "elapsed_sec": float,
                "rub_per_1000": float,
                "rub_per_month_50k": float,
                "within_target": bool,
                "monthly_cards": int,
            },
            "vllm_24x7": {
                "rub_per_month": float,
                "note": str,
            },
            "api": None | {
                "model": str | None,
                "cards": int,
                "input_tokens": int,
                "output_tokens": int,
                "rub_per_1000": float,
                "rub_per_month_50k": float,
            },
        }
        ```
    """
    cards = main.get("rows") or 0
    elapsed = main.get("elapsed_sec") or 0.0
    per_1000 = __cost_vllm_per_1000(elapsed, cards)
    costs = {
        "tariff_rub_per_hour": TARIFF_RUB_PER_HOUR,
        "price_target_rub_per_1000": PRICE_TARGET_RUB_PER_1000,
        "vllm": {
            "cards": cards,
            "elapsed_sec": elapsed,
            "rub_per_1000": round(per_1000, 3),
            "rub_per_month_50k": round(per_1000 / 1000 * MONTHLY_CARDS, 2),
            "within_target": per_1000 <= PRICE_TARGET_RUB_PER_1000,
            "monthly_cards": MONTHLY_CARDS,
        },
        "vllm_24x7": {
            "rub_per_month": round(TARIFF_RUB_PER_HOUR * 24 * 30, 2),
            "note": "ВМ работает круглосуточно; сравнение с оплатой только времени партии",
        },
        "api": None,
    }
    if api and api.get("rows"):
        input_tokens = api.get("total_input_tokens") or 0
        output_tokens = api.get("total_output_tokens") or 0
        if input_tokens + output_tokens > 0:
            api_per_1000 = __cost_api_per_1000(
                input_tokens,
                output_tokens,
                api["rows"],
            )
            costs["api"] = {
                "model": api.get("model"),
                "cards": api["rows"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "rub_per_1000": round(api_per_1000, 3),
                "rub_per_month_50k": round(api_per_1000 / 1000 * MONTHLY_CARDS, 2),
            }
    return costs


def __build_report(
    args: argparse.Namespace,
    metrics: dict,
    main: dict,
    table: list[dict],
    costs: dict,
    api: dict | None,
    experiments: list[dict] | None = None,
    postprocessing: dict | None = None,
) -> dict:
    """
    Args:
        args: Аргументы запуска скрипта. Используются для добавления параметров
            запуска в отчёт, например `startup_sec`.
        metrics: Итоговые метрики качества генерации карточек: доли валидных
            ответов, ошибки, статистика по длинам, структуре и другим проверкам.
        main: Основная сводка текущего запуска: длительность, throughput,
            параметры модели, сведения о сервере и общие показатели батча.
        table: Таблица результатов по разным значениям concurrency. Обычно
            содержит строки со скоростью, latency и throughput для каждого
            проверенного уровня параллельности.
        costs: Расчёт стоимости запуска: оценка локальной стоимости, стоимости
            API или сравнение затрат на обработку батча.
        api: Результаты сравнения с внешним API, если сравнение запускалось.
            Если сравнение не выполнялось, передаётся `None`.
        experiments: Результаты tuning/экспериментов с параметрами генерации,
            если они запускались. Если экспериментов не было, передаётся `None`.

    Returns:
        Словарь итогового отчёта, готовый для сериализации в
        `outputs/report.json`.
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": main,
        "server": {
            "startup_sec": args.startup_sec,
            "vllm_version": (main.get("server") or {}).get("vllm_version"),
            "note": "startup_sec — время от команды запуска до первого успешного /health",
        },
        "metrics": metrics,
        "concurrency": table,
        "costs": costs,
        "tuning": experiments,
        "format_postprocessing": postprocessing,
        "api_comparison": api,
        "sla": {
            "target_sec_per_300": SLA_SEC,
            "measured_sec_per_300_estimate": round(
                300 / (main.get("throughput_cards_per_sec") or 1), 1
            ),
            "note": "Измерено на локальной GPU; SLA задания задан для T4 и сравнивается оценкой",
        },
    }


def __parse_args() -> argparse.Namespace:
    """
    Парсер аргументов

    Returns:
        argparse.Namespace: парсер аргументов
    """
    parser = argparse.ArgumentParser(
        description="Метрики и report.json из сохранённых прогонов"
    )
    parser.add_argument("--predictions", default="outputs/predictions.jsonl")
    parser.add_argument("--summary", default="outputs/predictions.summary.json")
    parser.add_argument("--sweep", default="outputs/sweep_*.summary.json")
    parser.add_argument("--api-summary", default="outputs/api_summary.json")
    parser.add_argument("--api-predictions", default="outputs/api_predictions.jsonl")
    parser.add_argument("--experiments", default="outputs/parameter_experiments.csv")
    parser.add_argument("--input", default="benchmark.jsonl")
    parser.add_argument("--report", default="outputs/report.json")
    parser.add_argument("--startup-sec", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = __parse_args()
    rows = load_rows(Path(args.predictions))

    input_path = Path(args.input)
    input_rows = load_rows(input_path) if input_path.exists() else None

    main_summary = __read_json(Path(args.summary))
    previous = __read_json(Path(args.report))
    # Без сводки прогона отчёт всё равно можно пересобрать из предыдущего
    # report.json: predictions.jsonl содержит все метрики, а run — параметры
    # запуска, которых в нём нет.
    if main_summary is None and previous and isinstance(previous.get("run"), dict):
        main_summary = previous["run"]
        print(f"внимание: нет сводки {args.summary}, беру run из {args.report}")
    if main_summary is None:
        raise SystemExit(f"нет сводки прогона: {args.summary}")

    sweep = [
        summary
        for summary in (__read_json(Path(p)) for p in sorted(glob.glob(args.sweep)))
        if summary
    ]
    table = __build_concurrency_table(sweep)
    if not table and previous and isinstance(previous.get("concurrency"), list):
        table = previous["concurrency"]
        print(f"внимание: нет sweep-сводок, беру concurrency из {args.report}")

    api_summary = __read_json(Path(args.api_summary))
    if (
        api_summary is None
        and previous
        and isinstance(previous.get("api_comparison"), dict)
    ):
        api_summary = previous["api_comparison"]
        print(f"внимание: нет {args.api_summary}, беру api_comparison из {args.report}")

    api_predictions_path = Path(args.api_predictions)
    api_rows = (
        load_rows(api_predictions_path) if api_predictions_path.exists() else None
    )

    experiments = __read_experiments(Path(args.experiments))
    if experiments is None and previous and isinstance(previous.get("tuning"), list):
        experiments = previous["tuning"]
        print(f"внимание: нет {args.experiments}, беру tuning из {args.report}")

    metrics = __compute_metrics(rows, input_rows)
    postprocessing = __format_postprocessing(rows, api_rows)
    subset = __local_subset(rows, api_rows)
    if subset:
        api_summary = dict(api_summary or {})
        api_summary["local_same_subset"] = subset

    costs = __build_costs(main_summary, api_summary)
    report = __build_report(
        args,
        metrics,
        main_summary,
        table,
        costs,
        api_summary,
        experiments,
        postprocessing=postprocessing,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"строк: {metrics['rows']}, валидных: {metrics['valid_count']} ({metrics['valid_rate']:.3f})"
    )
    print(
        f"повторов: {metrics['retry_rate']:.3f}, попытки: {metrics['attempts_histogram']}"
    )
    print(
        f"макс. вход: {metrics['max_input_tokens']} токенов (план: {metrics['max_planned_input_tokens']})"
    )
    print(
        f"порядок совпадает со входом: {metrics['order_matches_input']}, "
        f"дублей: {metrics['duplicate_product_ids']}"
    )
    if metrics["errors_by_reason"]:
        print("причины брака:")
        for reason, count in metrics["errors_by_reason"].items():
            print(f"  {reason}: {count}")
    local_format = postprocessing["local"]
    print(
        f"Markdown-обёртки: {local_format['markdown_fenced']}/"
        f"{local_format['responses']} ({local_format['markdown_fenced_rate']:.3f})"
    )
    print(
        f"стоимость ВМ: {costs['vllm']['rub_per_1000']} ₽/1000 "
        f"(цель ≤ {PRICE_TARGET_RUB_PER_1000:g}), "
        f"на 50k карточек: {costs['vllm']['rub_per_month_50k']} ₽"
    )
    if costs["api"]:
        print(f"стоимость API: {costs['api']['rub_per_1000']} ₽/1000")
    elif api_summary:
        print("стоимость API: н/д — нет фактического usage от провайдера")
    print(f"отчёт: {report_path}")


if __name__ == "__main__":
    main()
