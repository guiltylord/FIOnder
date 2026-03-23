# main.py
 
import os
import time
#import pprint  # Импортируем для красивого вывода в консоль
 
import fitz  # PyMuPDF
 
from extractor import extract_words_with_coords
from highlight import highlight_in_pdf
from search import search_in_text
 
# =============================================================================
# НАСТРОЙКИ
# =============================================================================
 
FILE_INPUT = "nechdokprot" # Имя файла без расширения .pdf
SEARCH_TERMS = [
    'К.Н.', 'Лебедев', 'Е.А.', 'Кулаков', 'Ю.А.', 'И.В.', 'Дзюба', 'Д.С.',
    'Морозов', 'Коровин', 'М.С.', 'Е.В.', 'Гасанов', 'Фёдоров', 'М.Л.', 'Ю.П.',
    'М.Р.', 'Л.О.', 'Иванов', 'Нармин', 'Мария', 'Орлов', 'П.А.', 'Котова','Ториков В.Е.', 'Черепнин С А',
    'Смирнов', 'Викторович', 'Игнатов', 'Б.С.', 'Э.Д.', 'Джонсон', 'Эльдар', 'Т.А. Постоваловой',
    'Казаков', 'Фернандес', 'С.Н.', 'Попова', 'Иванова', 'Ли', 'Луиджи','Белоус М.Ф.','Евдокименко С.Н.','Высоцкий И.Г.',
    'Николай', 'А.В.', 'Китов', 'С.П.', 'Кузнецов', 'Шмидт', 'А.С.', 'Акимов', 'Виноградова Е.В.', 'Кузьмина Светлана Индисовна ','Тихонов Андрей Павлович', 'Загускин Н.Н.', 'Любимов М.В.',
    'оглы', 'Марио', 'Пётр', 'Николаевна', 'Амангельдинов', 'Соколова','Берлин Михаил Игоревич', 'Пустовойтов С.А.',  'Сперанский О.В.', 'Чайников А.В.', 
    'Такаяма Анна Тацумиевна', 'Анай-оол Чодураа Сергеевна', 'Александрова Марина Александровна','А.В. Власов','В.С. Тимонин','А.А. Климов',
    'Лушников', 'И.И.', 'Валько Олеся Юрьевна', 'Иванов И.И.', 'В.Н. Шангин','Наседкину Марию Алексеевну','Гриднев Владимир Михайлович',
    'Т.В. Соколова', 'Соколова Татьяна Валерьевна', 'ДЕНЬЕР ВАН ДЕР', 'Тронин Артем Николаевич', 'Шангин Виктор Николаевич', 'С.И. Сивоха', 'Мажаров',
    'Останина Надежда Геннадьевна', 'Зиязетдинова Олеся Хаузировна', 'Гнетецкий Ф.Э.', 'О.С. Курылёвой', 'И.И.', 'Ганина Нелли Ивановна', 'А.В.Захарченко',
    'АРМСТРОНГ-БРАУН', 'бэ', 'Финагина Л.Н.','Бондаренко Василию Фёдоровичу','В.Н. Шангин', 'Иванова Татьяна Борисовна', 'Т.Б. Иванова','Аскаров','Галченко Лариса Викторовна','В.И. Яхонтов','Дудина С.А.','Стефанская Анастасия','Стефанская А.А.'
] # Искомые слова
SAVE_TEXT_FILE = True  # Сохранять ли распознанный текст в TXT (True/False)
 
# Пути к папкам
INPUT_DIR = "Файлы для теста"
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
    readable = is_pdf_readable(PDF_INPUT)
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