"""OCR для PDF файлов с умной фильтрацией мусора."""

import fitz
import pytesseract
from PIL import Image, ImageEnhance
import io
import re
import time
import os
import sys


# =============================================================================
# НАСТРОЙКИ
# =============================================================================

SCALE = 2.0
CONTRAST = 1.6
MIN_CONFIDENCE = 30

VOWELS = set('аеёиоуыэюяaeiouyАЕЁИОУЫЭЮЯ')

SHORT_WORDS = {
    'и', 'в', 'на', 'по', 'с', 'к', 'у', 'о', 'а', 'но', 'же', 'бы', 'ли',
    'что', 'за', 'под', 'над', 'при', 'без', 'для', 'от', 'до', 'из', 'об', 'во',
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они', 'её', 'его', 'мне', 'тебе',
    'is', 'a', 'the', 'to', 'of', 'and', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'or', 'an', 'be', 'are', 'was', 'were', 'has', 'have', 'had'
}


# =============================================================================
# ФУНКЦИИ
# =============================================================================

def is_valid_word(word, confidence):
    """Проверка слова на валидность."""
    if '&' in word and len(word) > 3:
        return True
    
    if len(word) <= 2:
        return word.lower() in SHORT_WORDS
    
    if re.search(r'(.)\1{2,}', word):
        return False
    
    letters = [char for char in word if char.isalpha()]
    if letters and len(word) >= 3:
        vowel_ratio = sum(1 for char in letters if char in VOWELS) / len(letters)
        if not (0.25 <= vowel_ratio <= 0.75):
            return False
    
    return len(word) > 4 or confidence >= 40


def preprocess_image(image):
    """Предобработка изображения перед OCR."""
    image = image.convert('L')
    
    width, height = image.size
    image = image.resize(
        (int(width * SCALE), int(height * SCALE)),
        Image.Resampling.LANCZOS
    )
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(CONTRAST)
    
    return image


def extract_text_from_pdf(pdf_path):
    """Распознавание текста из PDF с помощью Tesseract OCR."""
    pages_text = []
    all_words = []
    confidences = []
    
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        image = Image.open(io.BytesIO(pixmap.tobytes('png')))
        image = preprocess_image(image)
        
        text = pytesseract.image_to_string(
            image,
            lang='rus+eng',
            config='--psm 3 --oem 3'
        )
        pages_text.append(text)
        
        data = pytesseract.image_to_data(
            image,
            lang='rus+eng',
            config='--psm 3 --oem 3',
            output_type=pytesseract.Output.DICT
        )
        
        for i in range(len(data['text'])):
            word_text = data['text'][i].strip()
            word_conf = float(data['conf'][i]) if data['conf'][i] else 0
            
            if word_text:
                all_words.append({
                    'text': word_text,
                    'confidence': word_conf,
                    'page': page_num + 1
                })
                
                if word_conf >= MIN_CONFIDENCE:
                    confidences.append(word_conf)
    
    doc.close()
    
    return pages_text, all_words, confidences


def build_confidence_map(words):
    """Построение карты уверенности: слово -> список уверенностей."""
    conf_map = {}
    
    for word_data in words:
        text = word_data['text']
        conf = word_data['confidence']
        
        clean = re.sub(r'^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$', '', text)
        
        if clean:
            if clean not in conf_map:
                conf_map[clean] = []
            conf_map[clean].append(conf)
    
    return conf_map


def filter_text(pages_text, confidence_map):
    """Фильтрация текста от мусора."""
    filtered = []
    
    for page_text in pages_text:
        for line in page_text.split('\n'):
            line = line.strip()
            
            if not line:
                continue
            
            if not re.search(r'[А-Яа-яA-Za-z]', line):
                continue
            
            special_chars = re.findall(r'[^\w\sА-Яа-яA-Za-z\"\'&;,\(\)\.\-]', line)
            if len(special_chars) / max(len(line), 1) > 0.5:
                continue
            
            for word in line.split():
                clean = re.sub(r'^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$', '', word)
                
                if not clean:
                    continue
                
                conf_list = confidence_map.get(clean, confidence_map.get(clean.lower(), [50]))
                avg_conf = sum(conf_list) / len(conf_list) if conf_list else 50
                
                if is_valid_word(clean, avg_conf):
                    filtered.append(clean)
    
    return filtered


def remove_trailing_garbage(words, confidence_map):
    """Удаление мусора в конце текста."""
    if not words:
        return words
    
    end_index = len(words)
    short_count = 0
    
    for i in range(len(words) - 1, -1, -1):
        word = words[i]
        word_clean = word.replace('.', '')
        
        conf_list = confidence_map.get(word, confidence_map.get(word.lower(), [50]))
        avg_conf = sum(conf_list) / len(conf_list) if conf_list else 50
        
        if len(word_clean) <= 2:
            short_count += 1
            if short_count >= 2:
                end_index = i
                break
        
        elif len(word_clean) < 5 and avg_conf < 40:
            end_index = i
            break
        
        elif short_count > 0:
            end_index = i + 1
            break
    
    return words[:end_index]


def process_pdf(pdf_path):
    """Обработка PDF файла: OCR + фильтрация."""
    start_time = time.time()
    
    pages_text, all_words, confidences = extract_text_from_pdf(pdf_path)
    pages_count = len(pages_text)
    
    conf_map = build_confidence_map(all_words)
    
    filtered_words = filter_text(pages_text, conf_map)
    filtered_words = remove_trailing_garbage(filtered_words, conf_map)
    
    result_text = ' '.join(filtered_words)
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    elapsed_time = time.time() - start_time
    
    return {
        'text': result_text,
        'pages': pages_count,
        'confidence': avg_confidence,
        'time': elapsed_time,
        'words': len(result_text.split())
    }


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

def main():
    """Запуск OCR обработки."""
    start_time = time.time()
    timestamp = int(start_time)
    
    filename = 'CROC'
    pdf_file = filename + '.pdf'
    
    if not os.path.exists(pdf_file):
        sys.exit(f"Файл '{pdf_file}' не найден!")
    
    os.makedirs('output', exist_ok=True)
    output_file = f'output/{filename}_output_{timestamp}.txt'
    
    print(f"Обработка: {pdf_file}...")
    print("=" * 60)
    
    result = process_pdf(pdf_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Файл: {pdf_file}\n")
        f.write(f"Страниц: {result['pages']}\n")
        f.write(f"Уверенность: {result['confidence']:.1f}%\n")
        f.write(f"Время обработки: {result['time']:.2f} сек\n")
        f.write(f"Распознано слов: {result['words']}\n")
        f.write("\n" + "=" * 60 + "\n\n")
        f.write(result['text'])
    
    print("\nРЕЗУЛЬТАТЫ:")
    print("=" * 60)
    print(f"  Страниц: {result['pages']}")
    print(f"  Средняя уверенность: {result['confidence']:.1f}%")
    print(f"  Распознано слов: {result['words']}")
    print(f"  Время обработки: {result['time']:.2f} сек.")
    print(f"  Сохранено в: {output_file}")
    print(f"\nОбщее время: {time.time() - start_time:.2f} сек.")
    
    print("\n" + "-" * 60)
    print("ТЕКСТ:")
    print("-" * 60)
    for line in result['text'].split('\n')[:5]:
        print(line[:100])
    if len(result['text']) > 500:
        print("...")


if __name__ == '__main__':
    main()
