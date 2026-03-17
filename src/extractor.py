import logging
import os

# Отключаем проверку обновлений моделей (ускоряет запуск)
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
# Подавляем логи дубляжа нейросети
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

# Полностью глушим логи PaddleOCR программно
logging.getLogger("ppocr").setLevel(logging.ERROR)


import io
import re
import time

import fitz
import numpy as np
from PIL import Image

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ НЕЙРОСЕТИ PADDLE OCR
# =============================================================================
try:
    import logging

    from paddleocr import PaddleOCR

    # Отключаем лишний спам от библиотеки в консоль
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    
    print("[INFO] Загрузка нейросети PaddleOCR (CPU)...")
    # use_gpu=False - работает строго на процессоре
    # lang='ru' - включает поддержку и русского, и английского текста
    # show_log=False - отключаем технические логи в консоли
    ocr_model = PaddleOCR(lang='ru')
except ImportError:
    print("[ОШИБКА] Не установлен PaddleOCR! Выполните команду в терминале:")
    print("pip install paddlepaddle paddleocr")
    exit(1)


# =============================================================================
# НАСТРОЙКИ
# =============================================================================

SCALE = 2.0  # Масштаб 2.0 идеален для нейросетей (они не любят слишком огромные картинки)
MIN_CONFIDENCE = 40  # Порог уверенности (теперь работает от 0 до 100)

VOWELS = set("аеёиоуыэюяaeiouyАЕЁИОУЫЭЮЯ")

SHORT_WORDS = {
    "и", "в", "на", "по", "с", "к", "у", "о", "а", "но", "же", "бы", "ли", 
    "что", "за", "под", "над", "при", "без", "для", "от", "до", "из", "об", 
    "во", "я", "ты", "он", "она", "оно", "мы", "вы", "они", "её", "его", 
    "мне", "тебе", "is", "a", "the", "to", "of", "and", "in", "for", "on", 
    "with", "at", "by", "from", "as", "or", "an", "be", "are", "was", "were", 
    "has", "have", "had",
}


# =============================================================================
# АЛГОРИТМ ИЗВЛЕЧЕНИЯ КООРДИНАТ
# =============================================================================

def extract_words_with_coords(pdf_path):
    """
    Извлечение всех слов с координатами из PDF с помощью PaddleOCR.
    PaddleOCR возвращает целые строки. Мы аккуратно разрезаем их на слова
    с высчитыванием координат, чтобы сохранить совместимость со старым кодом.
    """
    start_time = time.time()
    words_with_coords =[]
    doc = fitz.open(pdf_path)

    ocr_time = 0
    process_time = 0

    for page_num, page in enumerate(doc):
        process_start = time.time()
        
        # Получаем картинку страницы
        pixmap = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        
        # Нейросети едят сырые картинки. Никакой бинаризации больше нет!
        img_np = np.array(image)
        
        scale_x = page.rect.width / image.width
        scale_y = page.rect.height / image.height
        process_time += time.time() - process_start

        # =================================================================
        # МАГИЯ PADDLE OCR
        # =================================================================
        ocr_start = time.time()
        result = ocr_model.ocr(img_np, cls=False)
        ocr_time += time.time() - ocr_start

        process_start = time.time()
        lines = result[0] if result and result[0] else[]
        
        for line in lines:
            box, (text, conf) = line
            
            # Уверенность нейросети (возвращает от 0.0 до 1.0)
            if conf * 100 < 30:
                continue
                
            # box содержит 4 угла: [TopLeft, TopRight, BottomRight, BottomLeft]
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x0, x1 = min(x_coords), max(x_coords)
            y0, y1 = min(y_coords), max(y_coords)

            # Возвращаем координаты в масштаб оригинального PDF
            pdf_x0 = x0 * scale_x
            pdf_x1 = x1 * scale_x
            pdf_y0 = y0 * scale_y
            pdf_y1 = y1 * scale_y

            text = text.strip()
            if not text:
                continue

            # Разрезаем распознанную строку на отдельные слова для поиска ФИО
            words = text.split()
            if len(words) == 1:
                words_with_coords.append({
                    "text": words[0], "page": page_num + 1,
                    "x0": pdf_x0, "y0": pdf_y0, "x1": pdf_x1, "y1": pdf_y1,
                })
            else:
                # Если в строке несколько слов (например "Иванов И. И.")
                # Высчитываем пропорциональные координаты каждого слова
                char_w = (pdf_x1 - pdf_x0) / max(len(text), 1)
                curr_idx = 0
                for w in words:
                    start_char = text.find(w, curr_idx)
                    if start_char == -1: start_char = curr_idx
                    end_char = start_char + len(w)

                    w_x0 = pdf_x0 + start_char * char_w
                    w_x1 = pdf_x0 + end_char * char_w

                    words_with_coords.append({
                        "text": w, "page": page_num + 1,
                        "x0": w_x0, "y0": pdf_y0, "x1": w_x1, "y1": pdf_y1,
                    })
                    curr_idx = end_char
                    
        process_time += time.time() - process_start

    doc.close()

    total_time = time.time() - start_time
    print(f"\n[TIME] OCR (Paddle): {ocr_time:.2f}s ({ocr_time / total_time * 100:.0f}%)")
    print(f"[TIME] Coords Prep: {process_time:.2f}s ({process_time / total_time * 100:.0f}%)")
    print(f"[TIME] TOTAL EXTRACT: {total_time:.2f}s")

    return words_with_coords


# =============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ (Сохранены для совместимости)
# =============================================================================

def extract_text_from_pdf(pdf_path):
    """Распознавание сплошного текста (Аналог старой функции на базе Paddle)."""
    pages_text = []
    all_words =[]
    confidences =[]

    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        img_np = np.array(image)

        result = ocr_model.ocr(img_np, cls=False)
        lines = result[0] if result and result[0] else []
        
        page_str =[]
        if lines:
            for line in lines:
                box, (text, conf) = line
                page_str.append(text)
                
                conf_100 = conf * 100
                if conf_100 >= MIN_CONFIDENCE:
                    confidences.append(conf_100)
                
                for w in text.split():
                    all_words.append({
                        "text": w, "confidence": conf_100, "page": page_num + 1
                    })
                    
        pages_text.append("\n".join(page_str))

    doc.close()
    return pages_text, all_words, confidences


def build_confidence_map(words):
    """Построение карты уверенности: слово -> список уверенностей."""
    conf_map = {}
    for word_data in words:
        text = word_data["text"]
        conf = word_data["confidence"]
        clean = re.sub(r"^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$", "", text)
        if clean:
            if clean not in conf_map:
                conf_map[clean] =[]
            conf_map[clean].append(conf)
    return conf_map


def is_valid_word(word, confidence):
    """Проверка слова на валидность."""
    if re.match(r"^[А-Яа-яA-Za-z]\.?$", word):
        return confidence >= 20

    if len(word) <= 2:
        return word.lower() in SHORT_WORDS

    if re.search(r"(.)\1{2,}", word):
        return False

    letters =[char for char in word if char.isalpha()]
    if letters and len(word) >= 3:
        vowel_ratio = sum(1 for char in letters if char in VOWELS) / len(letters)
        if not (0.25 <= vowel_ratio <= 0.75):
            return False

    return len(word) > 4 or confidence >= 40


def filter_text(pages_text, confidence_map):
    """Фильтрация текста от мусора."""
    filtered =[]

    for page_text in pages_text:
        for line in page_text.split("\n"):
            line = line.strip()
            if not line or not re.search(r"[А-Яа-яA-Za-z]", line):
                continue

            special_chars = re.findall(r"[^\w\sА-Яа-яA-Za-z\"\'&;,\(\)\.\-]", line)
            if len(special_chars) / max(len(line), 1) > 0.5:
                continue

            for word in line.split():
                if re.match(r"^[А-Яа-яA-Za-z]\.?$", word):
                    clean = word
                else:
                    clean = re.sub(r"^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$", "", word)
                if not clean:
                    continue

                conf_list = confidence_map.get(
                    clean, confidence_map.get(clean.lower(), [50])
                )
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
        word_clean = word.replace(".", "")
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

    result_text = " ".join(filtered_words)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    elapsed_time = time.time() - start_time

    return {
        "text": result_text,
        "pages": pages_count,
        "confidence": avg_confidence,
        "elapsed_time": elapsed_time,
        "words": len(result_text.split()),
    }