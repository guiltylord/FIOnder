"""
Подсветка найденных слов в PDF.

Рисует цветные прямоугольники вокруг найденных позиций.
"""

import fitz

# Цвет рамки (R, G, B) в диапазоне 0–1
HIGHLIGHT_COLOR = (1, 0, 0)   # красный
HIGHLIGHT_WIDTH = 2            # толщина рамки в pt


def highlight_in_pdf(pdf_path: str, output_path: str, found_words: list) -> None:
    """
    Рисует прямоугольники вокруг найденных слов и сохраняет новый PDF.

    Args:
        pdf_path:    путь к исходному PDF.
        output_path: путь для сохранения результата.
        found_words: список найденных позиций (выход search_in_text):
            [{"page": int, "x0": float, "y0": float, "x1": float, "y1": float, ...}, ...]
    """
    if not found_words:
        print("[WARN] highlight_in_pdf: список пуст, PDF не изменён.")
        return

    doc = fitz.open(pdf_path)

    for item in found_words:
        page_idx = item["page"] - 1          # fitz: 0-based
        if page_idx < 0 or page_idx >= len(doc):
            continue
        page = doc[page_idx]
        rect = fitz.Rect(item["x0"], item["y0"], item["x1"], item["y1"])
        page.draw_rect(rect, color=HIGHLIGHT_COLOR, width=HIGHLIGHT_WIDTH)

    doc.save(output_path)
    doc.close()

    print(f"[INFO] Подсвеченный PDF → {output_path}")


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
>>>>>>> 703724c (модуль main определяет, нужно ли использовать окр)
