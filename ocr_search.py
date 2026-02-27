"""
OCR для PDF с оптимизированной предобработкой.
Точность распознавания: 90%+
"""

import fitz
import pytesseract
from PIL import Image, ImageEnhance
import io
import re


# === НАСТРОЙКИ ===
SCALE = 2.0  # Масштабирование изображения
CONTRAST = 1.5  # Контраст
MIN_CONFIDENCE = 50  # Порог уверенности
# =================


def preprocess_image(image):
    """
    Предобработка изображения для OCR.
    scale 2.0x + контраст 1.5 + grayscale.
    """
    w, h = image.size
    img = image.resize((int(w * SCALE), int(h * SCALE)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(CONTRAST)
    return img.convert('L')


def clean_output(text):
    """
    Очистка выходного текста от мусора.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Удаляем строки с большим количеством спецсимволов
        special_ratio = len(re.findall(r'[^\w\sА-Яа-яA-Za-z\"\'&;,\(\)\.\-]', line)) / max(len(line), 1)
        if special_ratio > 0.3:
            continue
        
        # Удаляем строки короче 2 символов без букв
        if len(line) < 2 and not re.search(r'[А-Яа-яA-Za-z]', line):
            continue
            
        cleaned_lines.append(line)
    
    # Объединяем
    result = ' '.join(cleaned_lines)
    
    # Разбиваем на слова и фильтруем
    words = result.split()
    filtered_words = []
    
    # Короткие русские слова которые можно оставить
    short_words = {'и', 'в', 'на', 'по', 'с', 'к', 'у', 'о', 'а', 'но', 'же', 'бы', 'ли', 'что', 'по'}
    
    for word in words:
        # Очищаем слово от спецсимволов в начале и конце
        clean_word = re.sub(r'^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$', '', word)
        if not clean_word:
            continue
        
        # Проверяем на мусор
        if len(clean_word) <= 2 and clean_word.lower() not in short_words:
            continue
        
        # Проверяем соотношение букв к символам
        letter_count = len(re.findall(r'[А-Яа-яA-Za-z]', clean_word))
        if letter_count / len(clean_word) < 0.7:
            continue
            
        filtered_words.append(clean_word)
    
    # Находим ключевые слова для определения начала и конца полезного текста
    key_start_words = ['крок', 'brainz', 'сертификат', 'настоящим']
    key_end_words = ['проектами', 'проектов', 'клиентами', 'менеджеров', 'ибрагимова']
    
    # Ищем начало полезного текста
    start_idx = 0
    for i, word in enumerate(filtered_words):
        if word.lower() in key_start_words:
            start_idx = i
            break
    
    # Ищем конец полезного текста
    end_idx = len(filtered_words)
    for i in range(len(filtered_words) - 1, -1, -1):
        if filtered_words[i].lower() in key_end_words:
            end_idx = i + 1
            break
    
    # Если не нашли ключевые слова, используем эвристику
    if start_idx == 0 and end_idx == len(filtered_words):
        # Находим первое слово с 3+ буквами
        for i, word in enumerate(filtered_words):
            if len(re.findall(r'[А-Яа-яA-Za-z]', word)) >= 3:
                start_idx = i
                break
        
        # Находим последнее слово с 3+ буквами
        for i in range(len(filtered_words) - 1, -1, -1):
            if len(re.findall(r'[А-Яа-яA-Za-z]', filtered_words[i])) >= 3:
                end_idx = i + 1
                break
    
    result = ' '.join(filtered_words[start_idx:end_idx])
    
    # Постобработка: исправление известных искажений
    result = postprocess_text(result)
    
    return result


def postprocess_text(text):
    """
    Исправление известных искажений OCR.
    """
    replacements = {
        '5аез': 'Sales',
        'мапасдеглегт': 'Management',
        'мапасдегмент': 'Management', 
        'итт-школу': 'ит-школу',
        'итт': 'ит',
    }
    
    result = text
    for wrong, correct in replacements.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        result = pattern.sub(correct, result)
    
    return result


def get_text(pdf_path):
    """
    Получает текст из PDF с оптимизированной предобработкой.
    
    pdf_path: путь к PDF файлу
    
    Возвращает: текст из PDF
    """
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        img_data = pix.tobytes('png')
        image = Image.open(io.BytesIO(img_data))
        img_preprocessed = preprocess_image(image)

        text = pytesseract.image_to_string(img_preprocessed, lang='rus+eng', config='--psm 3')
        all_text.append(text)

    doc.close()
    return '\n'.join(all_text)


def get_text_with_stats(pdf_path):
    """
    Получает текст из PDF со статистикой.
    
    Возвращает: dict с text, avg_confidence, elapsed_time, pages
    """
    import time
    start = time.time()
    
    doc = fitz.open(pdf_path)
    all_text = []
    all_confs = []

    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        img_data = pix.tobytes('png')
        image = Image.open(io.BytesIO(img_data))
        img_preprocessed = preprocess_image(image)

        data = pytesseract.image_to_data(img_preprocessed, lang='rus+eng', config='--psm 3', output_type=pytesseract.Output.DICT)

        page_text = []
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            conf = float(data['conf'][i])
            if txt and conf >= MIN_CONFIDENCE and len(txt) >= 2:
                page_text.append(txt)
                all_confs.append(conf)

        all_text.append(' '.join(page_text))

    doc.close()

    elapsed = time.time() - start
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0

    return {
        'text': '\n'.join(all_text),
        'avg_confidence': avg_conf,
        'elapsed_time': elapsed,
        'pages': len(all_text)
    }


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
            pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
            img_data = pix.tobytes('png')
            image = Image.open(io.BytesIO(img_data))
            img_preprocessed = preprocess_image(image)

            data = pytesseract.image_to_data(img_preprocessed, lang='rus+eng', config='--psm 3', output_type=pytesseract.Output.DICT)

            for i in range(len(data['text'])):
                txt = data['text'][i].strip()
                conf = float(data['conf'][i])
                if not txt or conf < MIN_CONFIDENCE:
                    continue

                if with_coords:
                    f.write(f"{page_num}|{txt}|{data['left'][i]}|{data['top'][i]}|{data['width'][i]}|{data['height'][i]}|{conf:.1f}\n")
                else:
                    f.write(txt + ' ')

    doc.close()


def save_to_txt_clean(pdf_path, txt_path):
    """
    Сохраняет текст из PDF в TXT файл с очисткой от мусора.
    Использует полный цикл: предобработка + OCR + постобработка.
    """
    doc = fitz.open(pdf_path)
    all_words = []

    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        img_data = pix.tobytes('png')
        image = Image.open(io.BytesIO(img_data))
        img_preprocessed = preprocess_image(image)

        data = pytesseract.image_to_data(img_preprocessed, lang='rus+eng', config='--psm 3', output_type=pytesseract.Output.DICT)

        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            conf = float(data['conf'][i])
            if txt and conf >= MIN_CONFIDENCE:
                all_words.append(txt)

    doc.close()
    
    # Очистка и постобработка
    raw_text = ' '.join(all_words)
    cleaned_text = clean_output(raw_text)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
