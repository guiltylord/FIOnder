"""
Поиск слов в PDF с координатами и подсветка.
"""

import re

import fitz


def search_words(words_with_coords, search_terms):
    """
    Поиск указанных слов в списке слов с координатами.
    O(n) сложность — один проход по списку.

    words_with_coords: список слов с координатами из ocr_search.py
    [
        {"text": "Денис", "page": 1, "x0": 100, "y0": 200, "x1": 150, "y1": 220},
        ...
    ]

    search_terms: строка с словами через запятую или список

    Возвращает список найденных слов с координатами.
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
        word_clean = re.sub(r"^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$", "", word_lower)

        for term in terms:
            if term in word_clean or word_clean in term:
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


def highlight_words_in_pdf(pdf_path, output_path, found_words):
    """
    Подсветка найденных слов в PDF файле.

    Рисует красные прямоугольники вокруг найденных слов.
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

    Возвращает список найденных слов и сохраняет подсвеченный PDF.
    """
    # Ищем нужные слова
    found = search_words(words_with_coords, search_terms)

    # Подсвечиваем в PDF
    if found:
        highlight_words_in_pdf(pdf_path, output_path, found)

    return found
