"""
Подсветка найденных слов в PDF.
"""

import fitz


def highlight_in_pdf(pdf_path, output_path, found_words):
    """
    Подсветка найденных слов в PDF файле.

    Рисует красные прямоугольники вокруг найденных слов.

    Args:
        pdf_path: путь к исходному PDF
        output_path: путь для сохранения подсвеченного PDF
        found_words: список найденных слов с координатами
            [
                {
                    "text": "слово",
                    "page": 1,
                    "x0": 100, "y0": 200, "x1": 150, "y1": 220
                },
                ...
            ]
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


def apply_highlight(pdf_path, output_path, found_words):
    """
    Применяет подсветку к PDF.

    Args:
        output_path: путь для сохранения подсвеченного PDF
        found_words: список найденных слов с координатами

    Returns:
        bool: True если подсветка применена, False если нет
    """
    if not found_words:
        return False

    # Путь к исходному PDF берём из настроек main.py
    # (передаётся через параметры)
    highlight_in_pdf(pdf_path, output_path, found_words)
    return True
