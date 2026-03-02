"""
Пример: Сохранить текст из PDF в TXT файл
"""

import os
import time

from ocr_search import get_text_with_stats, save_to_txt_clean

# === НАСТРОЙКИ ===
FILE_INPUT = "CROC"  # Без расширения .pdf
# =================

start = time.time()
ts = int(start)

os.makedirs("output", exist_ok=True)
output = f"output/{FILE_INPUT}Output{ts}.txt"

print(f"Обработка {FILE_INPUT}.pdf...")

save_to_txt_clean(f"{FILE_INPUT}.pdf", output)

# Статистика
stats = get_text_with_stats(f"{FILE_INPUT}.pdf")

print("\nРезультаты:")
print(f"  Страниц: {stats['pages']}")
print(f"  Средняя уверенность: {stats['confidence']:.1f}%")
print(f"  Распознано слов: {stats['words']}")
print(f"  Время обработки: {stats['elapsed_time']:.2f} сек.")
print(f"\nСохранено в: {output}")
print(f"Общее время: {time.time() - start:.2f} сек.")
