"""
Простой OCR для PDF.

Две функции:
    1. get_text(pdf_path) — возвращает текст из PDF
    2. save_to_txt(pdf_path, txt_path, with_coords=False) — сохраняет текст в файл
"""

import fitz
import pytesseract
from PIL import Image
import io
import re
from preprocess import prepare_image


def get_text(pdf_path):
    """
    Получает текст из PDF.

    pdf_path: путь к PDF файлу

    Возвращает: список строк (по одной строке на страницу)
    """
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
        img_data = pix.tobytes('png')
        image = Image.open(io.BytesIO(img_data))
        img_preprocessed = prepare_image(image)

        text = pytesseract.image_to_string(img_preprocessed, lang='rus+eng', config='--psm 3')
        all_text.append(text)

    doc.close()
    return all_text


def clean_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        special_chars = len(re.findall(r'[^\w\sА-Яа-яA-Za-z\"\'&;]', line))
        if special_chars / max(len(line), 1) > 0.4:
            continue
        if len(line) < 3 and not re.search(r'[А-Яа-яA-Za-z]{2,}', line):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def save_to_txt(pdf_path, txt_path, with_coords=False):
    """
    Сохраняет текст из PDF в TXT файл.

    pdf_path: путь к PDF файлу
    txt_path: путь куда сохранить TXT
    with_coords: True — координаты, False — простой текст
    """
    doc = fitz.open(pdf_path)

    with open(txt_path, 'w', encoding='utf-8') as f:
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
            img_data = pix.tobytes('png')
            image = Image.open(io.BytesIO(img_data))
            img_preprocessed = prepare_image(image)

            data = pytesseract.image_to_data(img_preprocessed, lang='rus+eng', config='--psm 3', output_type=pytesseract.Output.DICT)

            for i in range(len(data['text'])):
                txt = data['text'][i].strip()
                conf = float(data['conf'][i])
                if not txt or conf < 40:
                    continue

                if with_coords:
                    f.write(f"{page_num}|{txt}|{data['left'][i]}|{data['top'][i]}|{data['width'][i]}|{data['height'][i]}|{conf:.1f}\n")
                else:
                    f.write(txt + ' ')

    doc.close()
