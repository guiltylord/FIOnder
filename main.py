"""
Пример: Сохранить текст из PDF в TXT файл
"""

from ocr_search import save_to_txt_clean, get_text_with_stats
import time
import os

# === НАСТРОЙКИ ===
FILE_INPUT = 'CROC'  # Без расширения .pdf
# =================

start = time.time()
ts = int(start)

os.makedirs(FILE_INPUT, exist_ok=True)
output = f'output/{FILE_INPUT}Output{ts}.txt'

print(f"Обработка {FILE_INPUT}.pdf...")

save_to_txt_clean(f'{FILE_INPUT}.pdf', output)

# Статистика
stats = get_text_with_stats(f'{FILE_INPUT}.pdf')

print(f'\nРезультаты:')
print(f'  Страниц: {stats["pages"]}')
print(f'  Средняя уверенность: {stats["avg_confidence"]:.1f}%')
print(f'  Время обработки: {stats["elapsed_time"]:.2f} сек.')
print(f'\nСохранено в: {output}')
print(f'Общее время: {time.time()-start:.2f} сек.')
