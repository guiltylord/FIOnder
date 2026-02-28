"""
Универсальный OCR для PDF файлов.
Минимальная фильтрация — только явный мусор.
"""

import fitz
import pytesseract
from PIL import Image, ImageEnhance
import io
import time
import re
import os
from typing import Dict, List


# =============================================================================
# НАСТРОЙКИ
# =============================================================================

LANG = 'rus+eng'
MIN_CONFIDENCE = 30
SCALE = 2.0
CONTRAST = 1.6


# =============================================================================
# ПРЕДОБРАБОТКА
# =============================================================================

def preprocess_image(image: Image.Image) -> Image.Image:
    """Предобработка изображения."""
    img = image.convert('L')
    w, h = img.size
    img = img.resize((int(w * SCALE), int(h * SCALE)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(CONTRAST)
    return img


# =============================================================================
# ОЧИСТКА ТЕКСТА (МИНИМАЛЬНАЯ)
# =============================================================================

class TextCleaner:
    """Минимальная очистка текста — только явный мусор."""

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Очистка текста.
        
        Удаляет:
        - Пустые строки
        - Строки где >50% спецсимволы
        - Строки без букв
        
        Сохраняет:
        - Все слова с буквами
        - Инициалы, ФИО
        - Слова с &
        """
        if not text:
            return ""

        lines = text.split('\n')
        valid_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Пропуск строк где >50% спецсимволы
            special = len(re.findall(r'[^\w\sА-Яа-яA-Za-z\"\'&;,\(\)\.\-]', line))
            if special / max(len(line), 1) > 0.5:
                continue

            # Пропуск строк без букв
            if not re.search(r'[А-Яа-яA-Za-z]', line):
                continue

            valid_lines.append(line)

        return ' '.join(valid_lines)


# =============================================================================
# OCR
# =============================================================================

class UniversalOCR:
    """Универсальный OCR для PDF."""

    def __init__(self, lang: str = LANG, min_confidence: int = MIN_CONFIDENCE):
        self.lang = lang
        self.min_confidence = min_confidence

    def process(self, pdf_path: str) -> 'OCRResult':
        """Обработка PDF файла."""
        start_time = time.time()

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")

        doc = fitz.open(pdf_path)
        pages_count = len(doc)
        all_text = []
        all_words_data = []
        all_confs = []

        for page_num, page in enumerate(doc):
            # Рендер
            pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
            img_data = pix.tobytes('png')
            image = Image.open(io.BytesIO(img_data))

            # Предобработка
            img = preprocess_image(image)

            # OCR
            text = pytesseract.image_to_string(img, lang=self.lang, config='--psm 3 --oem 3')
            all_text.append(text)

            # Данные
            data = pytesseract.image_to_data(
                img, lang=self.lang, config='--psm 3 --oem 3',
                output_type=pytesseract.Output.DICT
            )

            for i in range(len(data['text'])):
                txt = data['text'][i].strip()
                conf = float(data['conf'][i]) if data['conf'][i] else 0
                if txt:
                    all_words_data.append({
                        'text': txt,
                        'confidence': conf,
                        'page': page_num + 1,
                        'bbox': {
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i],
                        }
                    })
                    if conf >= self.min_confidence:
                        all_confs.append(conf)

        doc.close()

        raw_text = '\n'.join(all_text)
        cleaned_text = TextCleaner.clean(raw_text)
        avg_confidence = sum(all_confs) / len(all_confs) if all_confs else 0
        elapsed = time.time() - start_time

        return OCRResult(
            text=raw_text,
            cleaned_text=cleaned_text,
            confidence=avg_confidence,
            pages=pages_count,
            time_elapsed=elapsed,
            words_data=all_words_data,
        )


# =============================================================================
# РЕЗУЛЬТАТ
# =============================================================================

class OCRResult:
    """Результат OCR."""

    def __init__(self, text: str, cleaned_text: str, confidence: float,
                 pages: int, time_elapsed: float, words_data: List[Dict]):
        self.text = text
        self.cleaned_text = cleaned_text
        self.confidence = confidence
        self.pages = pages
        self.time_elapsed = time_elapsed
        self.words_data = words_data
        self.preprocess_name = f'scale_{SCALE}_contrast_{CONTRAST}'
        self.tesseract_config = '--psm 3 --oem 3'


# =============================================================================
# ФУНКЦИИ
# =============================================================================

def get_text(pdf_path: str) -> str:
    """Получить текст из PDF."""
    return UniversalOCR().process(pdf_path).cleaned_text


def save_to_txt_clean(pdf_path: str, txt_path: str):
    """Сохранить текст в TXT."""
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(get_text(pdf_path))


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
        pdf_file = pdf_files[0] if pdf_files else None

    if not pdf_file:
        print("Нет PDF файлов")
        sys.exit(1)

    print(f"Обработка: {pdf_file}")
    print("=" * 60)

    ocr = UniversalOCR()
    result = ocr.process(pdf_file)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    print(f"Страниц: {result.pages}")
    print(f"Уверенность: {result.confidence:.1f}%")
    print(f"Слов: {len(result.cleaned_text.split())}")
    print(f"Время: {result.time_elapsed:.2f} сек")
    print("\nТЕКСТ:")
    print("-" * 60)
    print(result.cleaned_text[:2000] + "..." if len(result.cleaned_text) > 2000 else result.cleaned_text)
