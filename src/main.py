"""
Точка входа. Только оркестрация: читаем → ищем → подсвечиваем.
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from extractor import extract_words_with_coords
from highlight import highlight_in_pdf
from search import search_in_text

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

FILE_INPUT   = "pravo"
SEARCH_TERMS = ("Т Э Шпилевская")
SAVE_TEXT  = True   # сохранять ли извлечённый текст в .txt

INPUT_DIR  = "input"
OUTPUT_DIR = "output"
PDF_INPUT  = os.path.join(INPUT_DIR, f"{FILE_INPUT}.pdf")

# =============================================================================


def main():
    start = time.time()
    ts    = int(start)

    if not os.path.exists(PDF_INPUT):
        print(f"[ОШИБКА] Файл не найден: {PDF_INPUT}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Обработка: {PDF_INPUT}\n")

    # ── 1. Извлечение ────────────────────────────────────────────────────────
    words = extract_words_with_coords(PDF_INPUT)
    if not words:
        print("[ВНИМАНИЕ] Слова не извлечены. Проверьте PDF или настройки OCR.")
        return
    print(f"\n[INFO] Слов извлечено: {len(words)}")

    # ── 2. Сохранение текста ─────────────────────────────────────────────────
    if SAVE_TEXT:
        txt_path = os.path.join(OUTPUT_DIR, f"{FILE_INPUT}Output{ts}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(" ".join(w["text"] for w in words))
        print(f"[INFO] Текст → {txt_path}")

    # ── 3. Поиск ─────────────────────────────────────────────────────────────
    found = search_in_text(words, SEARCH_TERMS)
    print(f"[INFO] Найдено совпадений: {len(found)}")

    # ── 4. Подсветка ─────────────────────────────────────────────────────────
    if found:
        pdf_out = os.path.join(OUTPUT_DIR, f"{FILE_INPUT}Highlighted{ts}.pdf")
        highlight_in_pdf(PDF_INPUT, pdf_out, found)
        _print_results(found, pdf_out)
    else:
        print("\nСовпадения не найдены.")

    print(f"\n[TIME] Итого: {time.time() - start:.2f}с")


def _print_results(found: list, pdf_path: str) -> None:
    pages: dict = {}
    for item in found:
        pages.setdefault(item["page"], []).append(item["found_text"])

    print("\nРезультаты:")
    for p in sorted(pages):
        texts = ", ".join(t for t in pages[p] if t)
        print(f"  Стр. {p}: {texts}")

    print(f"\nПодсвеченный PDF: {pdf_path}")


if __name__ == "__main__":
    main()