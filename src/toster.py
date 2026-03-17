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

FILE_INPUT=r"C:\Users\User\Desktop\Файлы для теста\doc"
  # Без расширения .pdf
SEARCH_TERMS = [
        'К.Н.', 'Лебедев', 'Е.А.', 'Кулаков', 'Ю.А.', 'И.В.', 'Дзюба', 'Д.С.',
        'Морозов', 'Коровин', 'М.С.', 'Е.В.', 'Гасанов', 'Фёдоров', 'М.Л.', 'Ю.П.',
        'М.Р.', 'Л.О.', 'Иванов', 'Нармин', 'Мария', 'Орлов', 'П.А.', 'Котова',
        'Смирнов', 'Викторович', 'Игнатов', 'Б.С.', 'Э.Д.', 'Джонсон', 'Эльдар','Т.А. Постоваловой',
        'Казаков', 'Фернандес', 'С.Н.', 'Попова', 'Иванова', 'Ли', 'Луиджи',
        'Николай', 'А.В.', 'Китов', 'С.П.', 'Кузнецов', 'Шмидт', 'А.С.', 'Акимов', 'Виноградова Е.В.',
        'оглы', 'Марио', 'Пётр', 'Николаевна', 'Амангельдинов', 'Соколова', 
        'Такаяма Анна Тацумиевна', 'Анай-оол Чодураа Сергеевна', 'Александрова Марина Александровна',
          'Лушников', 'И.И.', 'Валько Олеся Юрьевна', 'Иванов И.И.', 'Иванов Иван Иванович',  'В.Н. Шангин',
        'Т.В.', 'Соколова Татьяна Валерьевна', 'ДЕНЬЕР ВАН ДЕР', 'Тронин Артем Николаевич', 'Шангин Виктор Николаевич',
        'Останина Надежда Геннадьевна', 'Зиязетдинова Олеся Хаузировна', 'Гнетецкий Ф.Э.','О.С. Курылёвой','АРМСТРОНГ-БРАУН', 'бэ', 'Финагина Л.Н.',
    ]# Искомые слова через запятую
SAVE_TEXT_FILE = False  # Сохранять ли текст в TXT (True/False)

# =============================================================================


def main():
    start = time.time()
    ts = int(start)

    os.makedirs("output", exist_ok=True)
    print(f"[TIME] Init: {time.time() - start:.2f}s")

    # 1. Извлечение слов с координатами из PDF
    coords_start = time.time()
    words_with_coords = extract_words_with_coords(f"{FILE_INPUT}.pdf")
    anay = [w for w in words_with_coords if "анай" in w["text"].lower() or "оол" in w["text"].lower()]
    # После получения words_with_coords от OCR, перед search_in_text:
    page1 = [w for w in words_with_coords if w["page"] == 1]
    ivan = [w for w in page1 if 'иван' in w['text'].lower()]
    print("Токены с 'иван':", ivan)
    page1 = [w for w in words_with_coords if w["page"] == 1 and 160 < w["y0"] < 210]
    print("Токены стр.1 y0∈(160,210):", page1)
    # Печатаем все токены со страницы 1 около нужной строки
    page1_tokens = [w for w in words_with_coords if w["page"] == 1 and 260 < w["y0"] < 290]
    page1_tokens = [w for w in words_with_coords if w["page"] == 1 and 100 < w["y0"] < 260]
    for t in page1_tokens:
     print(f"  y0={t['y0']:.1f} {repr(t['text'])}")
    print(f"  y0={t['y0']:.1f} {repr(t['text'])}")
    print("Токены стр.1 y0∈(260,290):", page1_tokens)
    print("OCR токены Анай-оол:", anay)
    print(f"[TIME] extract_words_with_coords: {time.time() - coords_start:.2f}s")
    print(f"Найдено слов (coords): {len(words_with_coords)}")

    # 2. Сохранение текста в TXT (опционально)
    if SAVE_TEXT_FILE:
        output_txt = f"output/{FILE_INPUT}Output{ts}.txt"
        text = " ".join(w["text"] for w in words_with_coords)
        save_start = time.time()
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[TIME] save_txt: {time.time() - save_start:.2f}s")
        print(f"Текст сохранён в: {output_txt}")

    # 3. Поиск слов (ЕДИНАЯ ТОЧКА ВХОДА — search_in_text())
    filename = os.path.basename(FILE_INPUT)
    output_pdf = f"output/{filename}Highlighted{ts}.pdf"
    search_start = time.time()
    found = search_in_text(words_with_coords, SEARCH_TERMS)
    for f in found:

     print(f)  # покажет все поля включая found_text
    print(f"DEBUG: terms={SEARCH_TERMS!r}, found={len(found)}")
    print(f"[TIME] search_in_text: {time.time() - search_start:.2f}s")
    print(f"Найдено совпадений: {len(found)}")

    # 4. Подсветка найденного в PDF
    if found:
        highlight_start = time.time()
        highlight_in_pdf(f"{FILE_INPUT}.pdf", output_pdf, found)
        full_path = os.path.abspath(output_pdf)
        print(f"\nPDF сохранён: {full_path}")
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