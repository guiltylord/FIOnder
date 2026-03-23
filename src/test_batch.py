"""
Пакетное тестирование PDF Analyzer.

Запуск:
    python src/test_batch.py

Результаты сохраняются в:
    test/{timestamp}/
        ├── report.txt
        ├── vseros_{ts}.pdf
        └── ...
"""

import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from extractor import extract_words_with_coords
from highlight import highlight_in_pdf
from search import search_in_text

# =============================================================================
# ТЕСТ-КЕЙСЫ
# Каждый файл — список тестов. Закомментированные тесты не запускаются.
# =============================================================================

TEST_CASES = {
    "vseros_removed": [
        [
            "Ангабаева Ольга Сергеевна",
            "Андреева Наталья Александровна",
            "Андреева Наталья Владимировна",
            "Андриевская М Б",
            "Н. А. Антонова",
        ],
    ],

    # "CROC": [
    #     [
    #         "Панин Иван",
    #         "Сабина Керимова",
    #     ],
    # ],

    # "sokolova": [
    #     [
    #         "Соколова Т В",
    #     ],
    # ],

    "spisokstudentov_removed": [
        [
            "А В Власов",
        ],
    ],

    "chemistry": [
        [
            "Стародумов В И",
            "Ураков К Ю",
        ],
    ],

    "doc_removed": [
        [
            "Гнетецкий Ф Э",
            "Борисов Б",
        ],
    ],

    # "participants_removed": [
    #     [
    #         "Бейкер Джеймс А",
    #         "Абель Кей",
    #         "Пол Ашфорд",
    #     ],
    # ],

    # "pravo_removed": [
    #     [
    #         "Е В Мотина",
    #         "Татьяна Эдуардовна Шпилевская",
    #         "Войтик А А",
    #     ],
    # ],

    # "test": [
    #     [
    #         "Иванов И И",
    #     ],
    #     [
    #         "Иванов Иван Иванович",
    #     ],
    # ],
    # "autoschool": [
    #     [
    #         "Шангин В Н",
    #     ]
    # ],
    # "badminton": [
    #     [
    #         "Невгень Сергей",
    #     ]
    # ],
    "nechdokkyzyl": [
        [
            "А Сарыглар",
            "Монгуш С. В."
        ]
    ],
    "nechdoktehnologia": [
        [
            "Пономарева Людмила Михайловна",
            "Куролесова Е В"
        ]
    ],
    # "nechdokysm": [
    #     [
    #         "Р А Белов",
    #     ]
    # ],
    # "nechdokobuchenie": [
    #     [
    #         " Загвоздина Любовь Генриховна",
    #     ]
    # ],
    # "nov": [
    #     [
    #         " Тормышов П.Е",
    #     ]
    # ],
    # "nechdokpersdan": [
    #     [
    #         "Архипова Ю.Г."
    #     ]
    # ],
    # "gazprom": [
    #     [
    #         "Иванова Т Б,",
    #     ]
    # ],
    # "nechdokkrivoi": [
    #     [
    #         " И А Грачева",
    #     ]
    # ],
    # "rasporyazhenie": [
    #     [
    #         "Н И Ганина",
    #     ]
    # ],
}

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

INPUT_DIR       = "input"
BASE_OUTPUT_DIR = "test"
SAVE_TEXT_FILE  = True

# =============================================================================


def run_test(pdf_name, search_terms, test_num, global_ts):
    pdf_path = os.path.join(INPUT_DIR, f"{pdf_name}.pdf")
    file_ts  = int(time.time())

    if not os.path.exists(pdf_path):
        print(f"  ❌ Файл не найден: {pdf_path}, пропускаем.")
        return {"error": "File not found", "pdf_name": pdf_name,
                "test_num": test_num, "total_found": 0, "total_time": 0}

    print(f"\n{'='*60}")
    print(f"📄 {pdf_name}.pdf  |  Тест #{test_num}  |  Запросов: {len(search_terms)}")
    print(f"{'='*60}")

    start_total = time.time()

    # 1. Извлечение
    t0 = time.time()
    words_with_coords = extract_words_with_coords(pdf_path)
    extract_time = time.time() - t0
    print(f"[Извлечение] {extract_time:.2f}s  |  слов: {len(words_with_coords)}")

    if not words_with_coords:
        print("  ⚠️  Слова не найдены, пропускаем поиск.")
        return {"pdf_name": pdf_name, "test_num": test_num,
                "total_found": 0, "total_time": time.time() - start_total,
                "extract_time": extract_time, "results": [], "search_times": []}

    # 2. Папка результатов
    out_dir = os.path.join(BASE_OUTPUT_DIR, str(global_ts))
    os.makedirs(out_dir, exist_ok=True)

    # 3. Сохранение текста (опционально)
    if SAVE_TEXT_FILE:
        txt_path = os.path.join(out_dir, f"{pdf_name}_{file_ts}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(" ".join(w["text"] for w in words_with_coords))
        print(f"  💾 Текст: {txt_path}")

    # 4. Поиск
    results     = []
    search_times = []
    all_found   = []

    for term in search_terms:
        t0    = time.time()
        found = search_in_text(words_with_coords, term)
        st    = time.time() - t0
        search_times.append(st)
        all_found.extend(found)
        results.append({"term": term, "found_count": len(found),
                         "found_items": found, "time": st})
        icon = "✅" if found else "❌"
        print(f"  {icon} '{term}' → {len(found)} ({st:.3f}s)")

    # 5. Подсветка
    if all_found:
        pdf_out = os.path.join(out_dir, f"{pdf_name}_{file_ts}.pdf")
        t0 = time.time()
        highlight_in_pdf(pdf_path, pdf_out, all_found)
        print(f"  💾 PDF: {pdf_out}  ({time.time()-t0:.2f}s)")

    total_time = time.time() - start_total
    return {"pdf_name": pdf_name, "test_num": test_num,
            "total_found": len(all_found), "search_times": search_times,
            "extract_time": extract_time, "total_time": total_time,
            "results": results, "out_dir": out_dir}


def print_summary(all_results):
    print(f"\n{'='*60}")
    print("📊 ИТОГО")
    print(f"{'='*60}")

    total_found = sum(r.get("total_found", 0) for r in all_results)
    total_time  = sum(r.get("total_time",  0) for r in all_results)
    errors      = sum(1 for r in all_results if "error" in r)

    print(f"Тестов:    {len(all_results)}")
    print(f"Ошибок:    {errors}")
    print(f"Найдено:   {total_found}")
    print(f"Время:     {total_time:.2f}s")

    # Группировка по файлам
    files: dict = {}
    for r in all_results:
        files.setdefault(r["pdf_name"], []).append(r)

    print()
    for pdf_name, results in files.items():
        print(f"📁 {pdf_name}.pdf")
        for r in results:
            if "error" in r:
                print(f"   ❌ Тест #{r['test_num']}: {r['error']}")
            else:
                print(f"   ✅ Тест #{r['test_num']}: {r['total_found']} найдено  ({r['total_time']:.2f}s)")
                for res in r.get("results", []):
                    icon = "✅" if res["found_count"] else "❌"
                    print(f"      {icon} '{res['term']}' → {res['found_count']}")

    return total_found, total_time, errors


def save_report(all_results, global_ts):
    out_dir     = os.path.join(BASE_OUTPUT_DIR, str(global_ts))
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PDF ANALYZER — ОТЧЁТ\n")
        f.write(f"Timestamp: {global_ts}\n")
        f.write(f"Дата: {datetime.fromtimestamp(global_ts).strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        files: dict = {}
        for r in all_results:
            files.setdefault(r["pdf_name"], []).append(r)

        for pdf_name, results in files.items():
            f.write(f"\n{pdf_name}.pdf\n")
            for r in results:
                if "error" in r:
                    f.write(f"  Тест #{r['test_num']}: {r['error']}\n")
                else:
                    f.write(f"  Тест #{r['test_num']}: {r['total_found']} найдено ({r['total_time']:.2f}s)\n")
                    for res in r.get("results", []):
                        f.write(f"    • '{res['term']}' → {res['found_count']} ({res['time']:.3f}s)\n")

    print(f"\n📄 Отчёт: {report_path}")


def main():
    global_ts = int(time.time())

    print("=" * 60)
    print("🚀 PDF ANALYZER — ПАКЕТНОЕ ТЕСТИРОВАНИЕ")
    print(f"   {datetime.fromtimestamp(global_ts).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Результаты: {BASE_OUTPUT_DIR}/{global_ts}/")
    print("=" * 60)

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    all_results = []
    
    # Считаем общее количество файлов для обработки
    total_files = sum(len(test_sets) for test_sets in TEST_CASES.values())
    file_counter = 0
    
    for pdf_name, test_sets in TEST_CASES.items():
        for test_num, search_terms in enumerate(test_sets, start=1):
            file_counter += 1
            print(f"\n[{file_counter}/{total_files}] Обработка: {pdf_name}.pdf (тест #{test_num})")
            result = run_test(pdf_name, search_terms, test_num, global_ts)
            all_results.append(result)

    print_summary(all_results)
    save_report(all_results, global_ts)

    total = sum(r.get("total_time", 0) for r in all_results)
    print(f"\n⏱  Общее время: {total:.2f}s")


if __name__ == "__main__":
    main()