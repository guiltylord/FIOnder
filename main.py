"""
Пример: Сохранить текст из PDF в TXT файл
"""

from ocr_search import save_to_txt
import time
import os

# === НАСТРОЙКИ ===
WITH_COORDS = True  # True — координаты, False — текст
FILE_INPUT = 'FILENAME' # Без расширения .pdf
# =================

start = time.time()
ts = int(start)

os.makedirs(FILE_INPUT, exist_ok=True)
output = f'output/{FILE_INPUT}Output{ts}.txt'

save_to_txt(f'{FILE_INPUT}.pdf', output, with_coords=WITH_COORDS)

print(f'Сохранено в: {output}')
print(f'Время: {time.time()-start:.2f} сек.')
