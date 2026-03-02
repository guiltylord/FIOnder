"""
Точка входа для PDF Analyzer.
"""

import os
import time

from extractor import extract_words_with_coords, process_pdf
from search import find_and_highlight

# === НАСТРОЙКИ ===
FILE_INPUT = "doc"  # Без расширения .pdf
SEARCH_TERMS = "Гнетецкий ф."  # Искомые слова через запятую
# =================


def main():
    start = time.time()
    ts = int(start)

    os.makedirs("output", exist_ok=True)

    # 1. Извлечение текста (OCR + фильтрация) → TXT
    output_txt = f"output/{FILE_INPUT}Output{ts}.txt"
    extract_text(FILE_INPUT, output_txt)

    # 2. Извлечение слов с координатами
    words_with_coords = extract_words_with_coords(f"{FILE_INPUT}.pdf")
    print(f"Найдено слов: {len(words_with_coords)}")

    # 3. Поиск и подсветка
    output_pdf = f"output/{FILE_INPUT}Highlighted{ts}.pdf"
    found = highlight_words(FILE_INPUT, output_pdf, words_with_coords)

    # 4. Вывод результатов
    print_results(found, output_pdf)

    print(f"\nОбщее время: {time.time() - start:.2f} сек.")


def extract_text(file_input, output_txt):
    """Извлечение текста из PDF и сохранение в TXT."""
    print(f"Обработка {file_input}.pdf...")
    result = process_pdf(f"{file_input}.pdf")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Текст сохранён в: {output_txt}")


def highlight_words(file_input, output_pdf, words_with_coords):
    """Поиск и подсветка слов в PDF."""
    print(f'\nПоиск: "{SEARCH_TERMS}"...')
    found = find_and_highlight(
        f"{file_input}.pdf", output_pdf, words_with_coords, SEARCH_TERMS
    )
    return found


def print_results(found, output_pdf):
    """Вывод результатов поиска."""
    print(f"Найдено совпадений: {len(found)}")
    if found:
        print("\nРезультаты:")
        for item in found:
            print(f"  Стр. {item['page']}: {item['found_text']}")
        print(f"\nПодсвеченный PDF: {output_pdf}")


if __name__ == "__main__":
    main()
