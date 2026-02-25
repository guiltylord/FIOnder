"""
Сравнение предобработок для OCR
Запуск: python compare.py [pdf_file]
"""

import fitz
import pytesseract
from PIL import Image
import io
import time
import os
import re

from preprocess import prepare_image as prepare_simple


def get_text(image):
    start = time.time()
    text = pytesseract.image_to_string(image, lang='rus+eng', config='--psm 3')
    t = time.time() - start
    
    data = pytesseract.image_to_data(image, lang='rus+eng', config='--psm 3', output_type=pytesseract.Output.DICT)
    confs = [float(c) for c in data['conf'] if c and float(c) > 0]
    avg_conf = sum(confs) / len(confs) if confs else 0
    
    return t, avg_conf, text


def clean_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Пропускаем строки с большим количеством спецсимволов
        special_chars = len(re.findall(r'[^\w\sА-Яа-яA-Za-z\"\'&;,\(\)]', line))
        if special_chars / max(len(line), 1) > 0.5:
            continue
        # Пропускаем очень короткие строки из мусора
        if len(line) < 3 and not re.search(r'[А-Яа-яA-Za-z]{2,}', line):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def save_text(filepath, text, page_num, time_sec, conf):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== Страница {page_num} ===\n")
        f.write(f"Время: {time_sec:.2f} сек | Уверенность: {conf:.1f}%\n\n")
        f.write(text)


def compare(pdf_path, page_num=0):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    pix_base = page.get_pixmap()
    img_base = Image.open(io.BytesIO(pix_base.tobytes('png')))
    
    pix_2x = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_2x = Image.open(io.BytesIO(pix_2x.tobytes('png')))
    
    pix_4x = page.get_pixmap(matrix=fitz.Matrix(4, 4))
    img_4x = Image.open(io.BytesIO(pix_4x.tobytes('png')))
    
    img_prepared = prepare_simple(img_4x)
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ts = int(time.time())
    
    print("=" * 60)
    print(f"Файл: {pdf_path}, Страница {page_num + 1}")
    print("=" * 60)
    
    # 1. Без обработки (1x)
    t, conf, text = get_text(img_base)
    text = clean_text(text)
    path1 = f"output/compare_{base_name}_{ts}_1_no_process.txt"
    save_text(path1, text, page_num + 1, t, conf)
    print(f"\n[1] Без обработки (1x)")
    print(f"    Время: {t:.2f} сек | Уверенность: {conf:.1f}%")
    print(f"    Сохранено: {path1}")
    
    # 2. Масштаб 2x
    t, conf, text = get_text(img_2x)
    text = clean_text(text)
    path2 = f"output/compare_{base_name}_{ts}_2_scale_2x.txt"
    save_text(path2, text, page_num + 1, t, conf)
    print(f"\n[2] Масштаб 2x")
    print(f"    Время: {t:.2f} сек | Уверенность: {conf:.1f}%")
    print(f"    Сохранено: {path2}")
    
    # 3. Масштаб 4x + Grayscale
    t, conf, text = get_text(img_prepared)
    text = clean_text(text)
    path3 = f"output/compare_{base_name}_{ts}_3_scale_4x_grayscale.txt"
    save_text(path3, text, page_num + 1, t, conf)
    print(f"\n[3] Масштаб 4x + Grayscale (текущий)")
    print(f"    Время: {t:.2f} сек | Уверенность: {conf:.1f}%")
    print(f"    Сохранено: {path3}")
    
    doc.close()
    print("\n" + "=" * 60)


if __name__ == '__main__':
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else 'CROC.pdf'
    try:
        compare(pdf)
    except FileNotFoundError:
        print(f"Файл {pdf} не найден. Положите PDF в папку проекта.")
