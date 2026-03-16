"""
Тестовый скрипт для пакетной обработки PDF файлов.

Каждый файл тестируется столько раз, сколько указано в TEST_CASES (1, 2, 3 или больше).

Структура выходных файлов:
    test/{global_timestamp}/
        ├── report.txt                 — отчёт по всем тестам
        ├── vseros_1710612350.pdf      — результат теста 1 (vseros)
        ├── vseros_1710612355.pdf      — результат теста 2 (vseros)
        ├── CROC_1710612352.pdf        — результат теста 1 (CROC)
        └── ...

Где:
    global_timestamp — время начала ВСЕХ тестов (одинаково для всех файлов)
    file_timestamp — время начала теста конкретного файла (может отличаться)

Использование:
    python test_batch.py
"""

import os
import time
from datetime import datetime

from extractor import extract_words_with_coords
from highlight import highlight_in_pdf
from search import search_in_text

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

# Словарь: { имя_файла: [список_тестов] }
# Каждый файл может иметь 1, 2, 3 или больше тестов — любое количество!
TEST_CASES = {
    "vseros": [
        # Тест 1: полные ФИО
        [
            "Ангабаева Ольга Сергеевна",
            "Андреева Наталья Александровна",
            "Андреева Наталья Владимировна",
            "Андриевская М Б",
            "Н. А. Антонова",
        ],
    #     # Тест 2: инициалы + короткие формы
    #     [
    #         "Александрова М А",
    #         "Андриевская М Б",
    #         "Анненкова В В",
    #     ],
    ],

    "CROC": [
        # Тест 1: полные имена
        [
            "Панин Иван",
            "Сабина Керимова",
        ],
        # Тест 2: короткие
        [
            "Денис",
            "Панин",
            "Сабина",
        ],
    ],

    "chemistry": [
        # Тест 1: учёные
        [
            "Менделеев Дмитрий",
            "Бутлеров Александр",
            "Зелинский Николай",
        ],
        # Тест 2: другие
        [
            "Стародумов В И",
            "Ураков К Ю",
        ],
    ],

    "doc": [
        # Тест 1
        [
            "Гнетецкий Ф Э",
            "Борисов Б",
        ],
    ],

    "new": [
        # Тест 1
        [
            "Дезмал",
            "Стефания",
        ],
    ],

    "participants": [
        # Тест 1
        [
            "Романо Даниэла",
            "Абель Кей",
            "Пол Ашфорд",
        ],
    ],

    "pravo": [
        # Тест 1
        [
            "Е В Мотина",
            "Татьяна Эдуардовна Шпилевская",
            "Войтик А А",
        ],
    ],

    "test": [
        # Тест 1
        [
            "Иванов И И",
        ],
        # Тест 2
        [
            "Иванов Иван Иваныч",
        ],
    ],
}

# Папки
INPUT_DIR = "input"
BASE_OUTPUT_DIR = "test"  # Базовая папка для тестов
OUTPUT_DIR = "output"  # Старая папка (для совместимости)

# Сохранять ли текст в TXT
SAVE_TEXT_FILE = False

# =============================================================================


def run_test(pdf_name: str, search_terms: list, test_num: int, global_ts: int) -> dict:
    """
    Запуск теста для одного PDF файла с набором поисковых запросов.

    Args:
        pdf_name: имя файла без расширения
        search_terms: список поисковых запросов
        test_num: номер теста (1 или 2)
        global_ts: единый timestamp для всех файлов (начало всех тестов)

    Returns:
        статистика теста
    """
    pdf_path = f"{INPUT_DIR}\\{pdf_name}.pdf"
    
    # Timestamp начала теста именно этого файла
    file_ts = int(time.time())

    # Проверка существования файла
    if not os.path.exists(pdf_path):
        print(f"❌ Файл не найден: {pdf_path}")
        return {
            "error": "File not found",
            "pdf_name": pdf_name,
            "test_num": test_num,
            "total_found": 0,
            "search_times": [],
            "total_time": 0,
        }

    print(f"\n{'='*60}")
    print(f"📄 Файл: {pdf_name}.pdf | Тест #{test_num}")
    print(f"🔍 Запросов: {len(search_terms)}")
    print(f"{'='*60}")

    start_total = time.time()

    # 1. Извлечение слов с координатами (один раз на файл)
    extract_start = time.time()
    words_with_coords = extract_words_with_coords(pdf_path)
    extract_time = time.time() - extract_start
    print(f"[TIME] Extract: {extract_time:.2f}s | слов: {len(words_with_coords)}")

    # 2. Создаём папку для результатов: test/{global_ts}/
    file_output_dir = f"{BASE_OUTPUT_DIR}\\{global_ts}"
    os.makedirs(file_output_dir, exist_ok=True)

    # 3. Сохранение текста (опционально)
    if SAVE_TEXT_FILE:
        # Имя файла: {pdf_name}_{file_ts}.txt
        output_txt = f"{file_output_dir}\\{pdf_name}_{file_ts}.txt"
        text = " ".join(w["text"] for w in words_with_coords)
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"💾 Текст сохранён: {output_txt}")

    # 4. Поиск по каждому запросу
    results = []
    search_times = []

    for term in search_terms:
        search_start = time.time()
        found = search_in_text(words_with_coords, term)
        search_time = time.time() - search_start
        search_times.append(search_time)

        results.append({
            "term": term,
            "found_count": len(found),
            "found_items": found,
            "time": search_time,
        })

        status = "✅" if found else "❌"
        print(f"  {status} Запрос: '{term}' → найдено: {len(found)} ({search_time:.3f}s)")

    # 5. Подсветка (если что-то найдено)
    all_found = [item for r in results for item in r["found_items"]]
    if all_found:
        # Имя файла: {pdf_name}_{file_ts}.pdf
        output_pdf = f"{file_output_dir}\\{pdf_name}_{file_ts}.pdf"
        highlight_start = time.time()
        highlight_in_pdf(pdf_path, output_pdf, all_found)
        highlight_time = time.time() - highlight_start
        print(f"[TIME] Highlight: {highlight_time:.2f}s")
        print(f"💾 Подсвеченный PDF: {output_pdf}")

    total_time = time.time() - start_total

    return {
        "pdf_name": pdf_name,
        "test_num": test_num,
        "total_found": len(all_found),
        "search_times": search_times,
        "extract_time": extract_time,
        "total_time": total_time,
        "results": results,
        "output_dir": file_output_dir,
    }


def print_summary(all_results: list, global_ts: int):
    """Вывод сводной статистики по всем тестам."""
    print(f"\n{'='*60}")
    print("📊 СВОДНАЯ СТАТИСТИКА")
    print(f"{'='*60}")

    total_tests = len(all_results)
    total_found = sum(r.get("total_found", 0) for r in all_results)
    total_time = sum(r.get("total_time", 0) for r in all_results)
    errors = sum(1 for r in all_results if "error" in r)

    print(f"Тестов выполнено: {total_tests}")
    print(f"Ошибок: {errors}")
    print(f"Всего найдено совпадений: {total_found}")
    print(f"Общее время: {total_time:.2f}s")
    print(f"Среднее время на тест: {total_time / max(total_tests, 1):.2f}s")

    # Детали по каждому файлу
    print(f"\n{'='*60}")
    print("ДЕТАЛИ ПО ФАЙЛАМ:")
    print(f"{'='*60}")

    # Группируем по файлам
    files_seen = {}
    for r in all_results:
        pdf_name = r.get('pdf_name', '???')
        if pdf_name not in files_seen:
            files_seen[pdf_name] = []
        files_seen[pdf_name].append(r)

    for pdf_name, results in files_seen.items():
        print(f"\n📁 {pdf_name}.pdf ({len(results)} тестов):")
        for r in results:
            if "error" in r:
                print(f"   ❌ Тест #{r.get('test_num', '?')}: {r['error']}")
            else:
                print(f"   ✅ Тест #{r['test_num']}: найдено {r['total_found']} за {r['total_time']:.2f}s")


def main():
    """Запуск пакетного тестирования."""
    # Единый timestamp для ВСЕХ файлов (перед началом тестов)
    global_ts = int(time.time())

    print("="*60)
    print("🚀 ПАКЕТНОЕ ТЕСТИРОВАНИЕ PDF ANALYZER")
    print(f"Timestamp: {global_ts}")
    print(f"Дата: {datetime.fromtimestamp(global_ts).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Папка результатов: {BASE_OUTPUT_DIR}\\{global_ts}\\")
    print("="*60)

    start = time.time()

    # Создаём базовую папку для тестов
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Запуск тестов: каждый файл тестируется столько раз, сколько указано в TEST_CASES
    all_results = []

    for pdf_name, test_sets in TEST_CASES.items():
        # Перебираем ВСЕ тесты для этого файла (1, 2, 3 или больше)
        for test_num, search_terms in enumerate(test_sets, start=1):
            result = run_test(pdf_name, search_terms, test_num=test_num, global_ts=global_ts)
            all_results.append(result)

    # Сводка
    print_summary(all_results, global_ts)

    # Итоговое время
    total = time.time() - start
    print(f"\n⏱ Общее время тестирования: {total:.2f}s")

    # Сохранение отчёта в папке test/{global_ts}/report.txt
    report_path = f"{BASE_OUTPUT_DIR}\\{global_ts}\\report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PDF ANALYZER — ОТЧЁТ О ТЕСТИРОВАНИИ\n")
        f.write(f"Timestamp: {global_ts}\n")
        f.write(f"Дата: {datetime.fromtimestamp(global_ts).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Папка: {BASE_OUTPUT_DIR}\\{global_ts}\\\n")
        f.write("="*60 + "\n\n")
        print_summary_to_file(all_results, f)

    print(f"📄 Отчёт сохранён: {report_path}")


def print_summary_to_file(all_results: list, f):
    """Запись сводки в файл."""
    total_tests = len(all_results)
    total_found = sum(r.get("total_found", 0) for r in all_results)
    total_time = sum(r.get("total_time", 0) for r in all_results)
    errors = sum(1 for r in all_results if "error" in r)

    f.write(f"Тестов выполнено: {total_tests}\n")
    f.write(f"Ошибок: {errors}\n")
    f.write(f"Всего найдено совпадений: {total_found}\n")
    f.write(f"Общее время: {total_time:.2f}s\n\n")

    # Группируем по файлам
    files_seen = {}
    for r in all_results:
        pdf_name = r.get('pdf_name', '???')
        if pdf_name not in files_seen:
            files_seen[pdf_name] = []
        files_seen[pdf_name].append(r)

    for pdf_name, results in files_seen.items():
        f.write(f"\n📁 {pdf_name}.pdf ({len(results)} тестов):\n")
        for r in results:
            if "error" in r:
                f.write(f"   ❌ Тест #{r.get('test_num', '?')}: {r['error']}\n")
            else:
                f.write(f"   ✅ Тест #{r['test_num']}: найдено {r['total_found']} за {r['total_time']:.2f}s\n")
                for res in r.get("results", []):
                    f.write(f"      • '{res['term']}' → {res['found_count']} ({res['time']:.3f}s)\n")


if __name__ == "__main__":
    main()
