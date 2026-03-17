# main.py

import os
import time
import pprint  # Импортируем для красивого вывода в консоль

from extractor import extract_words_with_coords
from highlight import highlight_in_pdf
from search import search_in_text

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

FILE_INPUT = "pravo"  # Имя файла без расширения .pdf
SEARCH_TERMS = "Татьяна Эдуардовна Шпилевская"  # Искомые слова
SAVE_TEXT_FILE = True  # Сохранять ли распознанный текст в TXT (True/False)

# Пути к папкам
INPUT_DIR = "input"
OUTPUT_DIR = "output"
PDF_INPUT = os.path.join(INPUT_DIR, f"{FILE_INPUT}.pdf")

# =============================================================================


def main():
    start_time = time.time()
    timestamp = int(start_time)

    # Создаем папку для результатов, если ее нет
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[TIME] Init: {time.time() - start_time:.2f}s")

    # 1. Извлечение слов с координатами из PDF с улучшенной обработкой
    coords_start = time.time()
    words_with_coords = extract_words_with_coords(PDF_INPUT)
    print(f"[TIME] extract_words_with_coords: {time.time() - coords_start:.2f}s")
    print(f"Найдено слов (с координатами): {len(words_with_coords)}")

    # =========================================================================
    # ВЫВОД РЕЗУЛЬТАТОВ TESSERACT В КОНСОЛЬ
    # =========================================================================
    print("\n[INFO] Результаты распознавания Tesseract:")
    if words_with_coords:
        print("Пример первых 20 распознанных слов:")
        for word_data in words_with_coords[:20]:
            print(f"  Стр. {word_data['page']}, Текст: '{word_data['text']}', Координаты: (x0={word_data['x0']}, y0={word_data['y0']})")
        # Для полного вывода всех слов раскомментируйте следующую строку:
        # pprint.pprint(words_with_coords)
    else:
        print("  Слова не были распознаны. Проверьте PDF-файл и настройки.")
    print("-" * 40)
    # =========================================================================

    # 2. Сохранение всего распознанного текста в TXT (опционально)
    if SAVE_TEXT_FILE:
        output_txt = os.path.join(OUTPUT_DIR, f"{FILE_INPUT}_Output_{timestamp}.txt")
        full_text = " ".join(w["text"] for w in words_with_coords)
        save_start = time.time()
        try:
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"[TIME] save_txt: {time.time() - save_start:.2f}s")
            print(f"Распознанный текст сохранён в: {output_txt}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить текстовый файл: {e}")

    # 3. Поиск заданных слов
    output_pdf = os.path.join(OUTPUT_DIR, f"{FILE_INPUT}_Highlighted_{timestamp}.pdf")
    search_start = time.time()
    found_items = search_in_text(words_with_coords, SEARCH_TERMS)
    print(f"[TIME] search_in_text: {time.time() - search_start:.2f}s")
    print(f"Найдено совпадений: {len(found_items)}")

    # 4. Подсветка найденного в новом PDF-файле
    if found_items:
        highlight_start = time.time()
        highlight_in_pdf(PDF_INPUT, output_pdf, found_items)
        print(f"[TIME] highlight_in_pdf: {time.time() - highlight_start:.2f}s")
        print_results(found_items, output_pdf)
    else:
        print("\nСовпадения для подсветки не найдены.")

    print(f"\nОбщее время выполнения: {time.time() - start_time:.2f} сек.")


def print_results(found, output_pdf_path):
    """Аккуратный вывод результатов поиска в консоль."""
    print("\n--- РЕЗУЛЬТАТЫ ПОИСКА ---")
    for item in found:
        print(f"  > Стр. {item['page']}: найдено '{item['found_text']}'")
    print(f"\nПодсвеченный PDF сохранён как: {output_pdf_path}")
    print("--------------------------")


if __name__ == "__main__":
    main()