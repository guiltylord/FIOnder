"""
Универсальный OCR для PDF файлов.
Фильтрация мусора через координаты слов.
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
# ОЧИСТКА ТЕКСТА
# =============================================================================

class TextCleaner:
    """Очистка текста через уверенность и плотность."""

    SHORT_WORDS = {
        'и', 'в', 'на', 'по', 'с', 'к', 'у', 'о', 'а', 'но', 'же', 'бы', 'ли',
        'что', 'за', 'под', 'над', 'при', 'без', 'для', 'от', 'до', 'из', 'об', 'во',
        'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они', 'её', 'его', 'мне', 'тебе',
        'is', 'a', 'the', 'to', 'of', 'and', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'or', 'an', 'be', 'are', 'was', 'were', 'has', 'have', 'had'
    }

    @classmethod
    def clean(cls, text: str, words_data: List[Dict] = None) -> str:
        """
        Очистка текста.
        
        Алгоритм:
        1. Фильтрация по уверенности (слова < 30% — мусор)
        2. Фильтрация по длине и составу
        3. Удаление изолированных коротких слов в конце
        """
        if not text:
            return ""

        # Строим мапу уверенность по словам
        conf_map = {}
        if words_data:
            for item in words_data:
                txt = item.get('text', '').strip()
                conf = item.get('confidence', 0)
                if txt:
                    # Нормализуем слово
                    clean = re.sub(r'^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$', '', txt)
                    if clean:
                        if clean not in conf_map:
                            conf_map[clean] = []
                        conf_map[clean].append(conf)

        lines = text.split('\n')
        valid_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            special_ratio = len(re.findall(r'[^\w\sА-Яа-яA-Za-z\"\'&;,\(\)\.\-]', line)) / max(len(line), 1)
            if special_ratio > 0.4:
                continue

            if not re.search(r'[А-Яа-яA-Za-z]', line):
                continue

            valid_lines.append(line)

        result = ' '.join(valid_lines)
        words = result.split()
        filtered = []

        for word in words:
            clean = re.sub(r'^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$', '', word)
            if not clean:
                continue

            # Слова с & валидны
            if '&' in clean and len(clean) > 3:
                filtered.append(clean)
                continue

            # Короткие слова из списка
            if len(clean) <= 2:
                if clean.lower() in cls.SHORT_WORDS:
                    filtered.append(clean)
                continue

            # Проверяем уверенность
            confs = conf_map.get(clean, conf_map.get(clean.lower(), [50]))
            avg_conf = sum(confs) / len(confs) if confs else 50

            # Слова с низкой уверенностью (<35%) и длиной <4 — мусор
            if len(clean) < 4 and avg_conf < 35:
                continue

            # Соотношение букв
            letters = len(re.findall(r'[А-Яа-яA-Za-z]', clean))
            if letters / len(clean) < 0.5:
                continue

            filtered.append(clean)

        # Удаление мусора в конце (изолированные короткие слова)
        if filtered:
            # Находим конец полезного текста
            end = len(filtered)
            consecutive_short = 0
            
            for i in range(len(filtered) - 1, -1, -1):
                w = filtered[i]
                w_clean = w.replace('.', '')
                
                # Короткое слово (1-2 буквы)
                if len(w_clean) <= 2:
                    consecutive_short += 1
                    # Если 3+ коротких слов подряд — это мусор в конце
                    if consecutive_short >= 3:
                        end = i
                        break
                else:
                    # Сбрасываем счётчик если нашли нормальное слово
                    if consecutive_short > 0 and len(w_clean) >= 3:
                        end = i + 1
                        break

            filtered = filtered[:end]

        return ' '.join(filtered)


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
        
        # Очистка текста
        cleaned_text = TextCleaner.clean(raw_text, all_words_data)
        
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
