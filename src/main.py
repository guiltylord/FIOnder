# main.py
 
import os
import sys
import time
#import pprint  # Импортируем для красивого вывода в консоль
 
import fitz  # PyMuPDF
 
from extractor import extract_words_with_coords
from highlight import highlight_in_pdf
from search import search_in_text
 
# =============================================================================
# НАСТРОЙКИ
# =============================================================================
 
FILE_INPUT = "vseros_removed" # Имя файла без расширения .pdf
SEARCH_TERMS = "Ангабаева О С" # Искомые слова
SAVE_TEXT_FILE = True  # Сохранять ли распознанный текст в TXT (True/False)
 
# Пути к папкам
INPUT_DIR = "input"
OUTPUT_DIR = "output"
PDF_INPUT = os.path.join(INPUT_DIR, f"{FILE_INPUT}.pdf")
 
# Минимальное среднее количество символов на страницу, чтобы считать PDF читаемым.
# Если меньше порога — документ считается сканом и запускается OCR.
READABLE_CHARS_THRESHOLD = 50
 
# =============================================================================
 
 
def is_pdf_readable(pdf_path: str, threshold: int = READABLE_CHARS_THRESHOLD) -> bool:
    """
    Проверяет, содержит ли PDF читаемый текстовый слой.
 
    Открывает документ через PyMuPDF и подсчитывает среднее количество
    символов на страницу. Если среднее ниже порога — PDF считается сканом.
 
    Args:
        pdf_path: путь к PDF-файлу.
        threshold: минимальное среднее число символов на страницу.
 
    Returns:
        True — если документ читаемый (есть текстовый слой),
        False — если документ нечитаемый (скан, нужен OCR).
    """
    doc = fitz.open(pdf_path)
    total_chars = 0
    num_pages = len(doc)
 
    for page in doc:
        text = page.get_text("text")
        total_chars += len(text.strip())
 
    doc.close()
 
    if num_pages == 0:
        return False
 
    avg_chars = total_chars / num_pages
    is_readable = avg_chars >= threshold
 
    print(f"[PDF] Страниц: {num_pages}, "
          f"символов всего: {total_chars}, "
          f"среднее на стр.: {avg_chars:.1f} — "
          f"{'читаемый ✓' if is_readable else 'скан/OCR нужен ✗'}")
 
    return is_readable
 
 
def extract_words_native(pdf_path: str) -> list[dict]:
    """
    Извлекает слова с координатами из читаемого PDF через PyMuPDF.
 
    Возвращает список словарей в том же формате, что и extractor.py:
      { "text": str, "page": int, "x0": float, "y0": float,
        "x1": float, "y1": float }
 
    Args:
        pdf_path: путь к PDF-файлу.
 
    Returns:
        Список слов с координатами.
    """
    doc = fitz.open(pdf_path)
    words_with_coords = []
 
    for page_index, page in enumerate(doc, start=1):
        # get_text("words") → список кортежей:
        # (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        raw_words = page.get_text("words")
        for w in raw_words:
            x0, y0, x1, y1, text, *_ = w
            text = text.strip()
            if text:
                words_with_coords.append({
                    "text": text,
                    "page": page_index,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                })
 
    doc.close()
    return words_with_coords
 
 
def main():
    start_time = time.time()
    timestamp = int(start_time)
 
    # Создаем папку для результатов, если ее нет
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[TIME] Init: {time.time() - start_time:.2f}s")
 
    # -------------------------------------------------------------------------
    # 1. Определяем тип PDF и выбираем метод извлечения
    # -------------------------------------------------------------------------
    check_start = time.time()
    # readable = is_pdf_readable(PDF_INPUT)
    readable = False
    print(f"[TIME] is_pdf_readable: {time.time() - check_start:.2f}s")
 
    coords_start = time.time()
 
    if readable:
        print("[EXTRACT] Используем нативное извлечение (PyMuPDF, без OCR)...")
        words_with_coords = extract_words_native(PDF_INPUT)
    else:
        print("[EXTRACT] PDF нечитаемый — запускаем OCR (extractor.py)...")
        words_with_coords = extract_words_with_coords(PDF_INPUT)
 
    print(f"[TIME] Извлечение текста: {time.time() - coords_start:.2f}s")
    print(f"Найдено слов (с координатами): {len(words_with_coords)}")
 
    # =========================================================================
    # ВЫВОД РЕЗУЛЬТАТОВ ИЗВЛЕЧЕНИЯ В КОНСОЛЬ
    # =========================================================================
    print("\n[INFO] Результаты извлечения текста:")
    if words_with_coords:
        print("Пример первых 20 распознанных слов:")
        for word_data in words_with_coords[:20]:
            print(f"  Стр. {word_data['page']}, Текст: '{word_data['text']}', "
                  f"Координаты: (x0={word_data['x0']}, y0={word_data['y0']})")
        # Для полного вывода всех слов раскомментируйте следующую строку:
        # pprint.pprint(words_with_coords)
    else:
        print("  Слова не были извлечены. Проверьте PDF-файл и настройки.")
    print("-" * 40)
    # =========================================================================
 
    # -------------------------------------------------------------------------
    # 2. Сохранение всего распознанного текста в TXT (опционально)
    # -------------------------------------------------------------------------
    if SAVE_TEXT_FILE:
        output_txt = os.path.join(OUTPUT_DIR, f"{FILE_INPUT}_Output_{timestamp}.txt")
        full_text = " ".join(w["text"] for w in words_with_coords)
        save_start = time.time()
        try:
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"[TIME] save_txt: {time.time() - save_start:.2f}s")
            print(f"Текст сохранён в: {output_txt}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить текстовый файл: {e}")
 
    # -------------------------------------------------------------------------
    # 3. Поиск заданных слов
    # -------------------------------------------------------------------------
    output_pdf = os.path.join(OUTPUT_DIR, f"{FILE_INPUT}_Highlighted_{timestamp}.pdf")
    search_start = time.time()
    found_items = search_in_text(words_with_coords, SEARCH_TERMS)
    print(f"[TIME] search_in_text: {time.time() - search_start:.2f}s")
    print(f"Найдено совпадений: {len(found_items)}")
 
    # -------------------------------------------------------------------------
    # 4. Подсветка найденного в новом PDF-файле
    # -------------------------------------------------------------------------
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