"""
Поиск слов в PDF с координатами и подсветка.
"""

import re

import fitz


def normalize_term(term, is_first=False):
    """
    Нормализация слова:
    - "Ф." → "Ф"
    - "Ф" → "Ф"
    - "Федор" → "Ф" (если это не первое слово)
    - "ГНЕТЕЦКИЙ" → "ГНЕТЕЦКИЙ"
    """
    term = term.strip().upper().rstrip(".")  # Удаляем точку если есть
    if not term:
        return ""
    # Первое слово (фамилия) оставляем полностью
    if is_first:
        return term
    # Короткие слова (1-2 буквы) — это инициалы
    if len(term) <= 2:
        return term[0]
    # Длинные слова после первого — тоже сокращаем до первой буквы
    return term[0]


def normalize_phrase(phrase):
    """
    Нормализация всей фразы:
    "Гнетецкий Федор Эдуардович" → ["ГНЕТЕЦКИЙ", "Ф", "Э"]
    "гнетецкий ф. э." → ["ГНЕТЕЦКИЙ", "Ф", "Э"]
    """
    words = phrase.split()
    normalized = []
    for i, word in enumerate(words):
        norm = normalize_term(word, is_first=(i == 0))
        if norm:
            normalized.append(norm)
    return normalized


def search_words(words_with_coords, search_terms):
    """
    Поиск указанных слов в списке слов с координатами.
    O(n) сложность — один проход по списку.
    """
    # Парсим поисковые термины
    if isinstance(search_terms, str):
        terms = [t.strip().lower() for t in search_terms.split(",")]
    else:
        terms = [t.lower() for t in search_terms]

    found = []
    for word_data in words_with_coords:
        word_lower = word_data["text"].lower()
        # Очищаем слово от спецсимволов для сравнения
        word_clean = re.sub(
            r"^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$", "", word_lower
        )

        for term in terms:
            if term == word_clean:  # Точное совпадение
                found.append(
                    {
                        "search_term": term,
                        "found_text": word_data["text"],
                        "page": word_data["page"],
                        "x0": word_data["x0"],
                        "y0": word_data["y0"],
                        "x1": word_data["x1"],
                        "y1": word_data["y1"],
                    }
                )
                break  # Нашли совпадение — переходим к следующему слову

    return found


def search_phrase_with_gaps(words_with_coords, search_phrase, max_gap=15):
    """
    Поиск фразы с пропусками между словами.
    Нормализует запрос и текст, ищет последовательность.

    max_gap — максимальное количество пропускаемых слов между элементами фразы
    """
    # Нормализуем запрос
    query_terms = normalize_phrase(search_phrase)
    if not query_terms:
        return []

    found = []
    n = len(words_with_coords)
    first_term = query_terms[0]

    for i in range(n):
        word_text = words_with_coords[i]["text"].strip().upper().rstrip(".")
        word_first = normalize_term(word_text, is_first=True)

        if word_first != first_term:
            continue

        # Пытаемся найти всю последовательность
        matched_words = [words_with_coords[i]]
        query_idx = 1
        text_idx = i + 1
        gap_count = 0

        while query_idx < len(query_terms) and text_idx < n:
            next_word_text = words_with_coords[text_idx]["text"].strip().upper().rstrip(
                "."
            )
            next_word_normalized = normalize_term(next_word_text, is_first=False)

            if next_word_normalized == query_terms[query_idx]:
                matched_words.append(words_with_coords[text_idx])
                query_idx += 1
                gap_count = 0
            else:
                gap_count += 1
                # Если слишком большой разрыв — прерываем
                if gap_count > max_gap:
                    break

            text_idx += 1

        # Если все слова найдены
        if query_idx == len(query_terms):
            found_phrase = {
                "search_term": search_phrase,
                "found_text": " ".join(w["text"] for w in matched_words),
                "page": matched_words[0]["page"],
                "x0": matched_words[0]["x0"],
                "y0": matched_words[0]["y0"],
                "x1": matched_words[-1]["x1"],
                "y1": matched_words[-1]["y1"],
            }
            found.append(found_phrase)

    return found


def highlight_words_in_pdf(pdf_path, output_path, found_words):
    """
    Подсветка найденных слов в PDF файле.
    """
    doc = fitz.open(pdf_path)

    for item in found_words:
        page_num = item["page"] - 1  # fitz использует 0-based индекс
        page = doc[page_num]

        # Создаём прямоугольник вокруг слова
        rect = fitz.Rect(item["x0"], item["y0"], item["x1"], item["y1"])

        # Рисуем красную рамку (толщина 2px)
        page.draw_rect(rect, color=(1, 0, 0), width=2)

    doc.save(output_path)
    doc.close()


def find_and_highlight(pdf_path, output_path, words_with_coords, search_terms):
    """
    Полный пайплайн: поиск → подсветка.
    """
    found = []

    # Парсим поисковые термины
    if isinstance(search_terms, str):
        terms = [t.strip() for t in search_terms.split(",")]
    else:
        terms = search_terms

    # Для каждого термина пытаемся найти как фразу
    for term in terms:
        # Если термин содержит пробелы — ищем как фразу с пропусками
        if " " in term:
            phrase_matches = search_phrase_with_gaps(words_with_coords, term)
            found.extend(phrase_matches)
        else:
            # Иначе ищем как отдельное слово (старый метод)
            word_matches = search_words(words_with_coords, [term])
            found.extend(word_matches)

    # Подсвечиваем в PDF
    if found:
        highlight_words_in_pdf(pdf_path, output_path, found)

    return found
