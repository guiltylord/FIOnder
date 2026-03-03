"""
Точка входа для PDF Analyzer.
"""

import os
import time

from extractor import extract_words_with_coords
from search import find_and_highlight

# === НАСТРОЙКИ ===
FILE_INPUT = "doc"  # Без расширения .pdf
SEARCH_TERMS = "Гнетецкий ф. э."  # Искомые слова через запятую
SAVE_TEXT_FILE = False  # Сохранять ли текст в TXT (True/False)
# =================


def main():
    start = time.time()
    ts = int(start)

    os.makedirs("output", exist_ok=True)
    print(f"[TIME] Init: {time.time() - start:.2f}s")
    

    # 1. Извлечение слов с координатами
    coords_start = time.time()
    words_with_coords = extract_words_with_coords(f"{FILE_INPUT}.pdf")
    print(f"[TIME] extract_words_with_coords: {time.time() - coords_start:.2f}s")
    print(f"Найдено слов (coords): {len(words_with_coords)}")

    # 1.2. Сохранение текста в TXT (опционально)
    if SAVE_TEXT_FILE:
        output_txt = f"output/{FILE_INPUT}Output{ts}.txt"
        # Извлекаем текст из слов
        text = " ".join(w["text"] for w in words_with_coords)
        save_start = time.time()
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[TIME] save_txt: {time.time() - save_start:.2f}s")
        print(f"Текст сохранён в: {output_txt}")

    # 3. Поиск и подсветка
    output_pdf = f"output/{FILE_INPUT}Highlighted{ts}.pdf"
    search_start = time.time()
    found = highlight_words(FILE_INPUT, output_pdf, words_with_coords)
    print(f"[TIME] highlight_words: {time.time() - search_start:.2f}s")

    # 4. Вывод результатов
    print_results(found, output_pdf)

    print(f"\nОбщее время: {time.time() - start:.2f} сек.")


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
