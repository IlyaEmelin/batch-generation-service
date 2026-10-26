"""Общее ядро сервиса: загрузка данных, сборка входа в бюджет 1500 токенов и валидатор карточки.

Модуль сознательно не знает ничего про HTTP: `run_batch.py` и `compare_api.py`
используют одни и те же функции сборки промпта и проверки ответа, поэтому
сравнение локального сервера с API идёт на одинаковом входе и одинаковых
правилах приёмки.

Бюджет считается по полному собранному запросу (system-сообщение, пример
карточки, товарные данные и chat-template), а не только по отзывам: это
строгая трактовка ограничения из задания.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Generator

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LIMIT = 1500

CARD_KEYS = ("product_id", "description", "pros", "cons", "tags")
# Минимальная и максимальная длина описания
__DESCRIPTION_MIN, __DESCRIPTION_MAX = 120, 700
# регулярное выражение для разбивки на предложения.
__SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
# Описание продукта должно содержать от 2 до 5 предложений.
__DESCRIPTION_SENTENCES = (2, 5)
__PROS_RANGE = (2, 5)
__CONS_RANGE = (1, 3)
__TAGS_RANGE = (3, 8)
# CJK-диапазоны: иероглифы, кана, хангыль. Модель иногда вставляет их в русский
# текст, особенно если в отзыве был иностранный фрагмент.
__CJK_PATTERN = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)

# pros - плюсы продуктов
# cons - минусы продуктов
# tags - теги продуктов

SYSTEM_PROMPT = (
    "Ты пишешь карточку товара для маркетплейса на русском языке.\n"
    "Верни один JSON-объект и ничего больше: без Markdown, без ``` и без пояснений.\n"
    "В объекте ровно пять ключей в таком порядке: product_id, description, pros, cons, tags.\n"
    "Требования, которые проверяет валидатор:\n"
    "- product_id — строка, точно как в последней строке запроса; не придумывай новый id;\n"
    "- description — строка от 120 до 700 символов, 2–5 предложений, только факты из данных;\n"
    "- pros — массив из 2–5 непустых строк;\n"
    "- cons — массив из 1–3 непустых строк; каждый пункт — реальный минус из отзывов "
    "или ограничение из атрибутов (размер, вес, совместимость, условия эксплуатации); "
    "плюсы, выдуманные характеристики и копирование примера недопустимы; "
    "пустой массив тоже недопустим;\n"
    "- tags — массив из 3–8 непустых строк без символа #.\n"
    "Пиши description тремя предложениями: что это за товар, ключевые характеристики, "
    "для кого подходит. Ничего не выдумывай.\n"
    "Пиши только на русском языке: не используй китайские, японские и корейские символы. "
    "Если в отзыве есть иностранный фрагмент, передай его смысл по-русски.\n"
    "Пример ниже показывает только структуру и объём. Не копируй из него фразы, "
    "pros, cons или tags.\n"
    "Пример формата:\n"
    '{"product_id": "P-0000", "description": "Настольный органайзер хранит канцелярию '
    "и мелкие вещи под рукой. Несколько отделений разделяют ручки, заметки и аксессуары, "
    "поэтому рабочее место остаётся аккуратным. Модель подойдёт для письменного стола "
    'дома или в офисе.", "pros": ["Несколько отделений", "Компактный размер", '
    '"Устойчивое основание"], "cons": ["Не подходит для крупных предметов"], '
    '"tags": ["органайзер", "для стола", "канцелярия", "хранение"]}'
)

__ERROR_HINTS = {
    "not_json": "ответ не разобрался как JSON: пришли только JSON-объект, без текста вокруг",
    "not_object": "ответ должен быть JSON-объектом, а не массивом или строкой",
    "missing_keys": "в объекте нет всех пяти ключей: product_id, description, pros, cons, tags",
    "extra_keys": "в объекте лишние ключи: нужны только product_id, description, pros, cons, tags",
    "product_id_mismatch": "product_id должен быть ровно {expected_id}",
    "description_not_string": "description должен быть строкой",
    "description_length": "description должен быть длиной от 120 до 700 символов",
    "description_sentences": "description должен содержать от 2 до 5 предложений",
    "pros_not_list": "pros должен быть массивом строк",
    "pros_count": "в pros должно быть от 2 до 5 пунктов",
    "pros_item": "каждый пункт pros должен быть непустой строкой",
    "cons_not_list": "cons должен быть массивом строк",
    "cons_count": "в cons должно быть от 1 до 3 пунктов: возьми реальный минус из отзывов или ограничение из атрибутов товара",
    "cons_item": "каждый пункт cons должен быть непустой строкой",
    "tags_not_list": "tags должен быть массивом строк",
    "tags_count": "в tags должно быть от 3 до 8 пунктов",
    "tags_item": "каждый тег должен быть непустой строкой",
    "contains_cjk_chars": "текст должен быть только на русском: убери китайские, японские и корейские символы",
    "request_error": "запрос завершился ошибкой, повтори генерацию",
}


@dataclass
class _PromptStats:
    """Сколько токенов занял запрос и сколько отзывов в него попало."""

    input_tokens: int
    reviews_total: int
    reviews_used: int
    truncated: bool


@dataclass
class _PreparedPrompt:
    """Готовый запрос: сообщения для chat-completions и статистика сборки."""

    messages: list[dict]
    stats: _PromptStats


@dataclass
class _ValidationResult:
    """Результат проверки одной карточки: итог, причины брака и разобранные поля."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    # Карточка товара
    card: dict | None = None


def load_rows(path: Path) -> list[dict]:
    """
    Читает JSONL: по одному товару на строку, пустые строки пропускаются.

    Args:
        path: пусть к файлу

    Returns:
        list[dict]: список записей из предоставленного dataset описания товара
    """
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def __dedupe(items: Iterable[str]) -> list[str]:
    """
    Убирает точные дубликаты, сохраняя порядок первого появления.

    Args:
        items: список элементов

    Returns:
        list[str]: Убирает точные дубликаты, сохраняя порядок первого появления.
    """
    return list(dict.fromkeys(items))


def __spread_order(count: int) -> Generator[int, None, None]:
    """
    Порядок отбора отзывов: начало, конец, середина — чтобы покрыть весь список.

    Индексы чередуются с краёв (0, последний, 1, предпоследний, ...), поэтому
    при урезании бюджета первыми теряются повторы по смыслу, а не хвост списка
    целиком.

    Args:
        count: количество отзывов

    Returns:
        list[int]: порядок индексов
    """
    # order: list[int] = []
    left, right = 0, count - 1
    while left <= right:
        yield left
        if left != right:
            yield right
        left += 1
        right -= 1


def __build_messages(row: dict, reviews: Sequence[str]) -> list[dict[str, str]]:
    """
    Собирает chat-сообщения из товара и выбранных отзывов.

    Args:
        row: одна запись из предоставленного dataset описания товара
        reviews: список ревью которые войдут в запрос.

    Returns:
        list[dict[str, str]]: сообщение для ИИ агента.
    """
    attributes = ", ".join(
        f"{name}: {value}" for name, value in row["attributes"].items()
    )
    lines = [f"product_id: {row['product_id']}", f"Название: {row['title']}"]
    if row.get("category"):
        lines.append(f"Категория: {row['category']}")
    if attributes:
        lines.append(f"Атрибуты: {attributes}")
    lines.append("Отзывы покупателей:")
    lines.extend(f"- {text}" for text in reviews)
    if not reviews:
        lines.append("- (отзывов нет)")
    # Напоминание в конце: на длинном входе модель теряет id из первой строки и
    # придумывает новый по шаблону, поэтому id повторяется последним.
    lines.append(f"В поле product_id верни ровно: {row['product_id']}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def count_chat_tokens(tokenizer, messages: list[dict]) -> int:
    """
    Считает токены полного запроса вместе с chat-template и промптом генерации.

    Контракт зафиксирован через `return_dict=True`: токенизатор должен вернуть
    dict-like объект с `input_ids`. Это fail-fast защита от изменений
    transformers: если формат ответа изменится, скрипт упадёт с понятной ошибкой,
    а не начнёт считать `len()` от контейнера вместо реальной длины токенов.

    Args:
        tokenizer: токенизатор
        messages: сообщения

    Returns:
        int: количество токенов
    """
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
    except TypeError as exc:
        raise TypeError(
            "tokenizer.apply_chat_template должен поддерживать return_dict=True. "
            "Зафиксируйте совместимую версию transformers или поправьте "
            "count_chat_tokens под новый формат ответа."
        ) from exc

    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError(
            "tokenizer.apply_chat_template(..., return_dict=True) должен вернуть "
            "dict-like объект с ключом input_ids. Проверьте версию transformers "
            "и формат chat template."
        )

    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise TypeError(
                "tokenizer.apply_chat_template вернул batch из нескольких "
                "последовательностей; count_chat_tokens ожидает один запрос."
            )
        ids = ids[0]
    if not isinstance(ids, (list, tuple)):
        raise TypeError(
            "tokenizer.apply_chat_template(..., return_dict=True)['input_ids'] "
            "должен быть списком токенов. Проверьте версию transformers и "
            "поправьте count_chat_tokens под новый формат ответа."
        )
    return len(ids)


def __count_text_tokens(tokenizer, text: str) -> int:
    """
    Считает токены строки без специальных токенов.

    Args:
        tokenizer: токенизатор
        text:

    Returns:
        int: количество токенов
    """
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def prepare_prompt(row: dict, tokenizer, limit: int = DEFAULT_LIMIT) -> _PreparedPrompt:
    """
    Собирает запрос так, чтобы полный вход не превышал `limit` токенов.

    Алгоритм: дедупликация → оценка стоимости отзывов → отбор в порядке
    «начало, конец, середина» → точная проверка собранного запроса. Оценка по
    частям может расходиться с фактической длиной из-за границ BPE-токенов,
    поэтому после сборки длина пересчитывается по всему запросу и лишние отзывы
    отбрасываются, а последний оставшийся при необходимости обрезается по
    символам.

    Args:
        row: одна запись из предоставленного dataset описания товара
        tokenizer: токенизатор
        limit: лимит токенов

    Returns:
        _PreparedPrompt: промт запрос к ИИ модели.
    """
    base_tokens = count_chat_tokens(tokenizer, __build_messages(row, []))
    if base_tokens > limit:
        raise ValueError(
            f"{row['product_id']}: служебная часть запроса занимает {base_tokens} "
            f"токенов при лимите {limit}"
        )

    separator_tokens = __count_text_tokens(tokenizer, "\n- ")
    reviews = __dedupe(row["reviews"])
    costs = [
        __count_text_tokens(tokenizer, text) + separator_tokens for text in reviews
    ]

    selected: list[int] = []
    used = base_tokens
    for index in __spread_order(len(reviews)):
        cost = costs[index]
        if used + cost <= limit:
            selected.append(index)
            used += cost
    selected.sort()

    messages = __build_messages(row, [reviews[index] for index in selected])
    messages, stats = __fit_selected_reviews_to_token_limit(
        row=row,
        reviews=reviews,
        selected=selected,
        tokenizer=tokenizer,
        limit=limit,
        messages=messages,
    )
    return _PreparedPrompt(messages=messages, stats=stats)


def __fit_selected_reviews_to_token_limit(
    row: dict,
    reviews: list[str],
    selected: list[int],
    tokenizer,
    limit: int,
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], _PromptStats]:
    """
    Подгоняет выбранные отзывы под лимит токенов полного chat-запроса.

    Сначала пересчитывает точную длину запроса целиком. Если запрос превышает
    лимит, удаляет выбранные отзывы с конца списка. Если даже один оставшийся
    отзыв не помещается, обрезает его бинарным поиском по символам.

    Args:
        row: одна запись товара
        reviews: дедуплицированный список отзывов
        selected: индексы выбранных отзывов в списке reviews
        tokenizer: токенизатор модели
        limit: максимальное количество токенов во входном запросе
        messages: сообщение отправляемые ИИ-агенту

    Returns:
        tuple[list[dict], int, bool]: сообщения, точное число токенов и флаг частичной обрезки
    """
    exact = count_chat_tokens(tokenizer, messages)
    trimmed = False
    # Последний отзыв оставляем как кандидат на частичную обрезку, остальные
    # лишние убираем целиком.
    while exact > limit and len(selected) > 1:
        selected.pop()
        messages = __build_messages(row, [reviews[index] for index in selected])
        exact = count_chat_tokens(tokenizer, messages)

    if exact > limit and selected:
        trimmed = True
        text = reviews[selected[0]]
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            trial = __build_messages(row, [text[:middle]])
            if count_chat_tokens(tokenizer, trial) <= limit:
                low = middle
            else:
                high = middle - 1
        messages = __build_messages(row, [text[:low]] if low else [])
        exact = count_chat_tokens(tokenizer, messages)

    stats = _PromptStats(
        input_tokens=exact,
        reviews_total=len(row["reviews"]),
        reviews_used=len(selected),
        truncated=trimmed or len(selected) < len(reviews),
    )
    return messages, stats


def __strip_code_fences(text: str) -> str:
    """
    Снимает Markdown-ограждения ```json ... ```, которые модель добавляет сама.

    По контракту их быть не должно, но `Qwen2.5-1.5B-Instruct` на коротком
    промпте оборачивает JSON в ограждения. Снимаем их до разбора и отдельной
    ошибкой это не считаем: смысл проверки — содержимое карточки.

    Args:
        text: текст

    Returns:
        str: очищенный текст
    """
    stripped = text.strip()
    lines = stripped.splitlines()
    if not lines or not lines[0].strip().startswith("```"):
        return stripped
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def __count_sentences(text: str) -> int:
    """
    Считает предложения по знакам конца предложения перед пробелом.

    Разделитель требует пробела после знака, поэтому «Bluetooth 5.3» и «0.5 л»
    остаются одним предложением, а не распадаются на два.

    Args:
        text: текст

    Returns:
        int: количество предложений в тексте
    """
    return len([part for part in __SENTENCE_SPLIT.split(text.strip()) if part.strip()])


def __check_string_list(
    value: object, name: str, bounds: tuple[int, int], errors: list[str]
) -> None:
    """
    Проверка полей строковых типов

    Args:
        value: значение
        name: имя поля
        bounds: ограничение количества в массиве
        errors: общий список ошибок куда будут записаны проблемы если они есть.
    """
    if not isinstance(value, list):
        errors.append(f"{name}_not_list")
        return
    low, high = bounds
    if not low <= len(value) <= high:
        errors.append(f"{name}_count")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{name}_item")


def __contains_cjk(value: object) -> bool:
    """
    Ищет китайские, японские и корейские символы в строке или списке строк.

    Args:
        value: проверяемое значение (строка, список или что угодно ещё)

    Returns:
        bool: True, если найден хотя бы один CJK-символ
    """
    if isinstance(value, str):
        texts = [value]
    elif isinstance(value, list):
        texts = [item for item in value if isinstance(item, str)]
    else:
        return False
    return any(__CJK_PATTERN.search(text) for text in texts)


def validate_card(raw_text: str, expected_id: str) -> _ValidationResult:
    """
    Проверяет ответ модели по контракту карточки и возвращает все нарушения.

    Одна карточка может нарушать несколько правил одновременно, поэтому
    возвращается список причин, а не первая найденная. Повторяющиеся строки в
    `pros`/`cons`/`tags` форматом не запрещены и ошибкой не считаются: это
    вопрос качества текста, а не структуры.

    Args:
        raw_text: сырой ответ модели
        expected_id: ожидаемый id продукта

    Returns:
        _ValidationResult: результат валидации
    """
    errors: list[str] = []
    try:
        data = json.loads(__strip_code_fences(raw_text or ""))
    except json.JSONDecodeError:
        return _ValidationResult(valid=False, errors=["not_json"])

    if not isinstance(data, dict):
        return _ValidationResult(valid=False, errors=["not_object"])

    missing = [key for key in CARD_KEYS if key not in data]
    if missing:
        errors.append("missing_keys")
    if any(key not in CARD_KEYS for key in data):
        errors.append("extra_keys")

    if data.get("product_id") != expected_id:
        errors.append("product_id_mismatch")

    description = data.get("description")
    if not isinstance(description, str):
        errors.append("description_not_string")
    else:
        if not __DESCRIPTION_MIN <= len(description) <= __DESCRIPTION_MAX:
            errors.append("description_length")
        low, high = __DESCRIPTION_SENTENCES
        if not low <= __count_sentences(description) <= high:
            errors.append("description_sentences")

    __check_string_list(data.get("pros"), "pros", __PROS_RANGE, errors)
    __check_string_list(data.get("cons"), "cons", __CONS_RANGE, errors)
    __check_string_list(data.get("tags"), "tags", __TAGS_RANGE, errors)

    # Одна ошибка на карточку, даже если CJK нашлись в нескольких полях:
    # иначе счётчик причин брака раздувается и искажает отчёт.
    if any(
        __contains_cjk(value)
        for value in (description, data.get("pros"), data.get("cons"), data.get("tags"))
    ):
        errors.append("contains_cjk_chars")

    return _ValidationResult(valid=not errors, errors=errors, card=data)


def corrective_message(errors: Sequence[str], expected_id: str) -> dict:
    """
    Формирует сообщение с понятным перечнем нарушений для исправляющей попытки.

    Модели 1.5B бесполезно показывать коды валидатора: она не знает, что такое
    `cons_count`. Поэтому каждый код переводится в требование на русском, а
    ожидаемый `product_id` подставляется явно.

    Args:
        errors: список ошибок.
        expected_id: идентификатор продукта

    Returns:

    """
    hints = [
        __ERROR_HINTS[error].format(expected_id=expected_id)
        for error in errors
        if error in __ERROR_HINTS
    ]
    listing = "\n".join(f"- {hint}" for hint in hints)
    return {
        "role": "user",
        "content": (
            "Предыдущий ответ нарушил контракт:\n"
            f"{listing}\n"
            "Пришли исправленный JSON-объект целиком, с теми же пятью ключами "
            "и без Markdown."
        ),
    }
