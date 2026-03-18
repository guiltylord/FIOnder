import os
import sys
import time

# Добавляем путь к src, если запускаем из корня
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from extractor import extract_words_with_coords
from highlight import highlight_in_pdf
from search import search_in_text

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

FILE_INPUT = "vseros"
SEARCH_TERMS = "Ангабаева Ольга Сергеевна, Андреева Наталья Александровна, Андреева Наталья Владимировна, Андриевская М Б, Н. А. Антонова"
SAVE_TEXT_FILE = True

INPUT_DIR  = "input"
OUTPUT_DIR = "output"
PDF_INPUT  = os.path.join(INPUT_DIR, f"{FILE_INPUT}.pdf")

# =============================================================================

def main():
    start = time.time()
    ts = int(start)

    if not os.path.exists(PDF_INPUT):
        print(f"[ОШИБКА] Не найден входной файл: {PDF_INPUT}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Начало обработки файла: {PDF_INPUT}")

    # 1. Извлечение слов
    try:
        words_with_coords = extract_words_with_coords(PDF_INPUT)
    except Exception as e:
        print(f"[КРИТИЧЕСКАЯ ОШИБКА OCR]: {e}")
        return

    if not words_with_coords:
        print("[ВНИМАНИЕ] Слова не извлечены. Проверьте PDF или настройки OCR.")
        return

    print(f"Найдено слов: {len(words_with_coords)}")

    # 2. Сохранение текста (TXT)
    if SAVE_TEXT_FILE:
        output_txt = os.path.join(OUTPUT_DIR, f"{FILE_INPUT}Output{ts}.txt")
        text = " ".join(w["text"] for w in words_with_coords)
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Текст сохранён в: {output_txt}")

    # 3. Поиск
    found = search_in_text(words_with_coords, SEARCH_TERMS)
    print(f"Найдено совпадений: {len(found)}")

    # 4. Подсветка
    if found:
        output_pdf = os.path.join(OUTPUT_DIR, f"{FILE_INPUT}Highlighted{ts}.pdf")
        highlight_in_pdf(PDF_INPUT, output_pdf, found)
        print_results(found, output_pdf)
    else:
        print("\nСовпадения не найдены.")

    print(f"\nОбщее время: {time.time() - start:.2f} сек.")


def print_results(found, output_pdf):
    print("\nРезультаты:")
    pages = {}
    for item in found:
        p = item['page']
        if p not in pages:
            pages[p] = []
        pages[p].append(item['found_text'])

    for p in sorted(pages.keys()):
        print(f"  Стр. {p}: {', '.join(pages[p])}")
    print(f"\nПодсвеченный PDF: {output_pdf}")


if __name__ == "__main__":
    main()