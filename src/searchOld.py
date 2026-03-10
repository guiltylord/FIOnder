"""
Поиск слов в PDF.

ЕДИНАЯ ТОЧКА ВХОДА: search_in_text()

Чтобы добавить свой алгоритм поиска:
1. Найди функцию search_in_text()
2. Замени логику внутри на свой алгоритм
3. Верни список найденных слов в формате:
   [
       {
           "search_term": "что искали",
           "found_text": "что нашли",
           "page": 1,
           "x0": 100, "y0": 200, "x1": 150, "y1": 220
       },
       ...
   ]
"""

import re
import time
from collections import Counter

# =============================================================================
# ЕДИНАЯ ТОЧКА ВХОДА
# =============================================================================


def search_in_text(words_with_coords, search_terms):
    """
    Поиск указанных слов/фраз в списке слов с координатами.

    ╔═══════════════════════════════════════════════════════════╗
    ║  ВСТАВЛЯТЬ СВОЙ АЛГОРИТМ ПОИСКА СЮДА! (ниже)              ║
    ╚═══════════════════════════════════════════════════════════╝

    Args:
        words_with_coords: список слов с координатами из OCR
            [
                {"text": "слово", "page": 1, "x0": 100, "y0": 200, ...},
                ...
            ]

        search_terms: строка с поисковыми запросами (через запятую)
            "Гнетецкий ф. э." или "Иванов, Петров"

    Returns:
        Список найденных слов с координатами:
        [
            {
                "search_term": "Гнетецкий ф. э.",
                "found_text": "ГНЕТЕЦКИЙ",
                "page": 2,
                "x0": 100, "y0": 200, "x1": 150, "y1": 220
            },
            ...
        ]
    """
    start_time = time.time()

    # Парсим поисковые термины
    if isinstance(search_terms, str):
        terms = [t.strip() for t in search_terms.split(",")]
    else:
        terms = search_terms

    found = []

    # Для каждого термина пытаемся найти
    for term in terms:
        # Если термин содержит пробелы — ищем как фразу (ФИО)
        if " " in term:
            fio_matches = search_fio_universal(words_with_coords, term)
            found.extend(fio_matches)
        else:
            # Иначе ищем как отдельное слово
            word_matches = search_single_word(words_with_coords, term)
            found.extend(word_matches)

    search_time = time.time() - start_time
    print(f"[TIME] Search: {search_time:.2f}s")

    return found


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (можно менять/удалять)
# =============================================================================


def search_single_word(words_with_coords, search_term):
    """
    Поиск одного слова (точное совпадение).
    """
    term_lower = search_term.lower()
    found = []

    for word_data in words_with_coords:
        word_lower = word_data["text"].lower()
        word_clean = re.sub(r"^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$", "", word_lower)

        if term_lower == word_clean:
            found.append(
                {
                    "search_term": search_term,
                    "found_text": word_data["text"],
                    "page": word_data["page"],
                    "x0": word_data["x0"],
                    "y0": word_data["y0"],
                    "x1": word_data["x1"],
                    "y1": word_data["y1"],
                }
            )

    return found


def search_fio_universal(words_with_coords, search_phrase, max_gap=10):
    """
    Универсальный поиск ФИО в любом порядке, но слова должны быть рядом.

    Алгоритм:
    1. Находим самое длинное слово в запросе — это якорь
    2. Остальные токены → первая буква (инициалы)
    3. Ищем якорь в тексте
    4. Вокруг якоря (в пределах max_gap слов) ищем все инициалы
    5. Возвращаем координаты ВСЕХ найденных слов

    Примеры:
    - "Гнетецкий ф. э." → якорь: ГНЕТЕЦКИЙ, инициалы: Ф, Э
    - "ф. э. Гнетецкий" → якорь: ГНЕТЕЦКИЙ, инициалы: Ф, Э
    - "Гнетецкий Федор Эдуардович" → якорь: ЭДУАРДОВИЧ, инициалы: Г, Ф
    """
    # Разбиваем на слова, убираем пустые
    query_words = [w for w in search_phrase.split() if w.strip()]

    if not query_words:
        return []

    # Находим самое длинное слово в ЗАПРОСЕ — якорь
    longest = max(query_words, key=len)
    anchor = longest.strip().upper().rstrip(".")

    # Остальные токены запроса → инициалы (первая буква)
    initials = []
    for word in query_words:
        word = word.strip().upper().rstrip(".")
        if word != anchor and word:
            initials.append(word[0])

    initials_count = Counter(initials)

    # Нормализуем текст
    text_words = []
    text_initials = []

    for w in words_with_coords:
        word = w["text"].strip().upper().rstrip(".")
        if word:
            text_words.append(word)
            text_initials.append(word[0])

    # 1. Ищем якорь (полное совпадение)
    anchor_indices = [i for i, w in enumerate(text_words) if w == anchor]

    if not anchor_indices:
        return []

    # 2. Для каждой позиции якоря ищем инициалы рядом
    for anchor_index in anchor_indices:
        # Окно вокруг якоря
        start = max(0, anchor_index - max_gap)
        end = min(len(text_words), anchor_index + max_gap + 1)

        window_words = text_words[start:end]
        window_indices = list(range(start, end))

        # Считаем инициалы в окне (только короткие слова!)
        window_initials = []
        window_initial_indices = []

        for i, word in enumerate(window_words):
            word_clean = word.rstrip(".")
            # Только короткие слова (инициалы)
            if len(word_clean) <= 2 and word[0] in initials:
                window_initials.append(word[0])
                window_initial_indices.append(window_indices[i])

        window_initials_count = Counter(window_initials)

        # Проверяем, все ли инициалы найдены в окне
        all_found = True
        for initial, required in initials_count.items():
            if window_initials_count.get(initial, 0) < required:
                all_found = False
                break

        if all_found:
            # Найдено! Собираем координаты
            found_words = [words_with_coords[anchor_index]]
            used_indices = {anchor_index}

            for initial, required in initials_count.items():
                count = 0
                for i, init in enumerate(window_initials):
                    if (
                        init == initial
                        and window_initial_indices[i] not in used_indices
                        and count < required
                    ):
                        found_words.append(words_with_coords[window_initial_indices[i]])
                        used_indices.add(window_initial_indices[i])
                        count += 1

            return [
                {
                    "search_term": search_phrase,
                    "found_text": w["text"],
                    "page": w["page"],
                    "x0": w["x0"],
                    "y0": w["y0"],
                    "x1": w["x1"],
                    "y1": w["y1"],
                }
                for w in found_words
            ]

    return []
