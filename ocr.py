"""
Универсальный OCR для PDF.
Умная фильтрация мусора на основе лингвистической статистики.
"""

import fitz
import pytesseract
from PIL import Image, ImageEnhance
import io
import re
from typing import Dict, List

LANG = 'rus+eng'
MIN_CONF = 30
SCALE = 2.0
CONTRAST = 1.6
VOWELS = set('аеёиоуыэюяaeiouyAEIOUYАЕЁИОУЫЭЮЯ')
SHORT_WORDS = {
    'и', 'в', 'на', 'по', 'с', 'к', 'у', 'о', 'а', 'но', 'же', 'бы', 'ли',
    'что', 'за', 'под', 'над', 'при', 'без', 'для', 'от', 'до', 'из', 'об', 'во',
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они', 'её', 'его', 'мне', 'тебе',
    'is', 'a', 'the', 'to', 'of', 'and', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'or', 'an', 'be', 'are', 'was', 'were', 'has', 'have', 'had'
}


def preprocess(img: Image.Image) -> Image.Image:
    """Предобработка: grayscale + scale + contrast."""
    img = img.convert('L')
    w, h = img.size
    img = img.resize((int(w * SCALE), int(h * SCALE)), Image.Resampling.LANCZOS)
    return ImageEnhance.Contrast(img).enhance(CONTRAST)


def is_valid(word: str, conf: float) -> bool:
    """Проверка слова: повторы, гласные, уверенность."""
    if '&' in word and len(word) > 3:
        return True
    if len(word) <= 2:
        return word.lower() in SHORT_WORDS
    if re.search(r'(.)\1{2,}', word):
        return False
    letters = [c for c in word if c.isalpha()]
    if letters and len(word) >= 3:
        ratio = sum(1 for c in letters if c in VOWELS) / len(letters)
        if not (0.25 <= ratio <= 0.75):
            return False
    if len(word) <= 4 and conf < 40:
        return False
    return True


def clean(text: str, words_data: List[Dict]) -> str:
    """Очистка текста от мусора."""
    if not text:
        return ""
    
    # Мапа уверенности
    conf_map = {}
    for item in words_data:
        txt = item.get('text', '').strip()
        conf = item.get('confidence', 0)
        if txt:
            c = re.sub(r'^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$', '', txt)
            if c:
                conf_map.setdefault(c, []).append(conf)
    
    # Фильтрация строк и слов
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if len(re.findall(r'[^\w\sА-Яа-яA-Za-z\"\'&;,\(\)\.\-]', line)) / max(len(line), 1) > 0.5:
            continue
        if not re.search(r'[А-Яа-яA-Za-z]', line):
            continue
        lines.append(line)
    
    result = ' '.join(lines)
    filtered = []
    for word in result.split():
        c = re.sub(r'^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$', '', word)
        if not c:
            continue
        confs = conf_map.get(c, conf_map.get(c.lower(), [50]))
        avg_conf = sum(confs) / len(confs) if confs else 50
        if is_valid(c, avg_conf):
            filtered.append(c)
    
    # Удаление мусора в конце
    if filtered:
        end = len(filtered)
        short_count = 0
        for i in range(len(filtered) - 1, -1, -1):
            w = filtered[i].replace('.', '')
            confs = conf_map.get(filtered[i], conf_map.get(filtered[i].lower(), [50]))
            avg_conf = sum(confs) / len(confs) if confs else 50
            
            if len(w) <= 2:
                short_count += 1
                if short_count >= 2:
                    end = i
                    break
            elif len(w) < 5 and avg_conf < 40:
                end = i
                break
            elif short_count > 0:
                end = i + 1
                break
        filtered = filtered[:end]
    
    return ' '.join(filtered)


class OCR:
    """OCR процессор."""
    
    def __init__(self, lang: str = LANG, min_conf: int = MIN_CONF):
        self.lang = lang
        self.min_conf = min_conf
    
    def process(self, pdf_path: str) -> Dict:
        """Обработка PDF."""
        import time
        import os
        
        start = time.time()
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        pages_count = len(doc)
        all_text, all_words, all_confs = [], [], []
        
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
            img = Image.open(io.BytesIO(pix.tobytes('png')))
            img = preprocess(img)
            
            text = pytesseract.image_to_string(img, lang=self.lang, config='--psm 3 --oem 3')
            all_text.append(text)
            
            data = pytesseract.image_to_data(img, lang=self.lang, config='--psm 3 --oem 3',
                                            output_type=pytesseract.Output.DICT)
            for i in range(len(data['text'])):
                txt = data['text'][i].strip()
                conf = float(data['conf'][i]) if data['conf'][i] else 0
                if txt:
                    all_words.append({'text': txt, 'confidence': conf, 'page': page_num + 1,
                                     'bbox': {'left': data['left'][i], 'top': data['top'][i],
                                             'width': data['width'][i], 'height': data['height'][i]}})
                    if conf >= self.min_conf:
                        all_confs.append(conf)
        
        doc.close()
        raw = '\n'.join(all_text)
        cleaned = clean(raw, all_words)
        
        return {
            'text': cleaned,
            'raw_text': raw,
            'confidence': sum(all_confs) / len(all_confs) if all_confs else 0,
            'pages': pages_count,
            'time': time.time() - start,
            'words': len(cleaned.split()),
            'words_data': all_words
        }


def get_text(pdf_path: str) -> str:
    """Получить текст из PDF."""
    return OCR().process(pdf_path)['text']


def save_to_txt(pdf_path: str, txt_path: str):
    """Сохранить текст в TXT."""
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(get_text(pdf_path))
