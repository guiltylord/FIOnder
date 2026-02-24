"""
Пример 2: Сохранить текст из PDF в TXT файл
"""

from ocr_search import save_to_txt
import time
import os

start_time = time.time()
timestamp = int(start_time)

# Сохраняем в файл
input = 'CROC.pdf'
pdf_name = os.path.splitext(os.path.basename(input))[0]

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

output = os.path.join(output_dir, f'{pdf_name}Output{timestamp}.txt')
save_to_txt(input, output)

elapsed = time.time() - start_time
print(f'Готово! Текст сохранён в {output}')
print(f'Время работы: {elapsed:.2f} сек.')
