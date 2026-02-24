"""
Простой OCR для PDF.

Две функции:
    1. get_text(pdf_path) — возвращает текст из PDF
    2. save_to_txt(pdf_path, txt_path) — сохраняет текст в файл
"""

import fitz
import pytesseract
from PIL import Image
import io


def get_text(pdf_path):
    """
    Получает текст из PDF.
    
    pdf_path: путь к PDF файлу
    
    Возвращает: список строк (по одной строке на страницу)
    """
    doc = fitz.open(pdf_path)
    all_text = []
    
    for page in doc:
        # Превращаем страницу в картинку
        pix = page.get_pixmap()
        img_data = pix.tobytes('png')
        image = Image.open(io.BytesIO(img_data))
        
        # Распознаём текст
        text = pytesseract.image_to_string(image, lang='rus+eng')
        all_text.append(text)
    
    doc.close()
    return all_text


def save_to_txt(pdf_path, txt_path):
    """
    Сохраняет текст из PDF в TXT файл.
    
    pdf_path: путь к PDF файлу
    txt_path: путь куда сохранить TXT
    """
    text_list = get_text(pdf_path)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(text_list, start=1):
            f.write(f'=== Страница {i} ===\n')
            f.write(text)
            f.write('\n\n')
