"""
Точка входа для PDF Analyzer.

Архитектура:
1. extractor.py — извлечение слов из PDF (OCR)
2. search.py — поиск слов (единая точка входа: search_in_text())
3. highlight.py — подсветка найденного в PDF
"""

import os
import time

from extractor import extract_words_with_coords
from search import search_in_text
from highlight import highlight_in_pdf

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

FILE_INPUT = "doc"  # Без расширения .pdf
SEARCH_TERMS = "Гнетецкий ф. э."  # Искомые слова через запятую
SAVE_TEXT_FILE = False  # Сохранять ли текст в TXT (True/False)

# Пути
INPUT_DIR = "input"
OUTPUT_DIR = "output"
PDF_INPUT = f"{INPUT_DIR}\\{FILE_INPUT}.pdf"

# =============================================================================


def main():
    start = time.time()
    ts = int(start)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[TIME] Init: {time.time() - start:.2f}s")

    # 1. Извлечение слов с координатами из PDF
    coords_start = time.time()
    words_with_coords = extract_words_with_coords(PDF_INPUT)
    print(f"[TIME] extract_words_with_coords: {time.time() - coords_start:.2f}s")
    print(f"Найдено слов (coords): {len(words_with_coords)}")

    # 2. Сохранение текста в TXT (опционально)
    if SAVE_TEXT_FILE:
        output_txt = f"{OUTPUT_DIR}/{FILE_INPUT}Output{ts}.txt"
        text = " ".join(w["text"] for w in words_with_coords)
        save_start = time.time()
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[TIME] save_txt: {time.time() - save_start:.2f}s")
        print(f"Текст сохранён в: {output_txt}")

    # 3. Поиск слов (ЕДИНАЯ ТОЧКА ВХОДА — search_in_text())
    output_pdf = f"{OUTPUT_DIR}/{FILE_INPUT}Highlighted{ts}.pdf"
    search_start = time.time()
    found = search_in_text(words_with_coords, SEARCH_TERMS)
    print(f"[TIME] search_in_text: {time.time() - search_start:.2f}s")
    print(f"Найдено совпадений: {len(found)}")

    # 4. Подсветка найденного в PDF
    if found:
        highlight_start = time.time()
        highlight_in_pdf(PDF_INPUT, output_pdf, found)
        print(f"[TIME] highlight_in_pdf: {time.time() - highlight_start:.2f}s")
        print_results(found, output_pdf)
    else:
        print("\nСовпадения не найдены")

    print(f"\nОбщее время: {time.time() - start:.2f} сек.")


def print_results(found, output_pdf):
    """Вывод результатов поиска."""
    print("\nРезультаты:")
    for item in found:
        print(f"  Стр. {item['page']}: {item['found_text']}")
    print(f"\nПодсвеченный PDF: {output_pdf}")


if __name__ == "__main__":
    main()
