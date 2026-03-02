"""
OCR для PDF файлов с умной фильтрацией мусора.
"""

import io
import re
import time

import fitz
import pytesseract
from PIL import Image, ImageEnhance

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

SCALE = 2.0  # Базовое разрешение
CONTRAST = 2.0  # Базовый контраст
MIN_CONFIDENCE = 30  # Базовый порог уверенности

VOWELS = set("аеёиоуыэюяaeiouyАЕЁИОУЫЭЮЯ")

SHORT_WORDS = {
    "и",
    "в",
    "на",
    "по",
    "с",
    "к",
    "у",
    "о",
    "а",
    "но",
    "же",
    "бы",
    "ли",
    "что",
    "за",
    "под",
    "над",
    "при",
    "без",
    "для",
    "от",
    "до",
    "из",
    "об",
    "во",
    "я",
    "ты",
    "он",
    "она",
    "оно",
    "мы",
    "вы",
    "они",
    "её",
    "его",
    "мне",
    "тебе",
    "is",
    "a",
    "the",
    "to",
    "of",
    "and",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "or",
    "an",
    "be",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
}


# =============================================================================
# АЛГОРИТМ
# =============================================================================


def preprocess_image(image):
    """Предобработка изображения перед OCR."""
    image = image.convert("L")
    width, height = image.size
    image = image.resize(
        (int(width * SCALE), int(height * SCALE)), Image.Resampling.LANCZOS
    )
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(CONTRAST)


def extract_text_from_pdf(pdf_path):
    """Распознавание текста из PDF с помощью Tesseract OCR."""
    pages_text = []
    all_words = []
    confidences = []

    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        image = preprocess_image(image)

        text = pytesseract.image_to_string(
            image, lang="rus+eng", config="--psm 3 --oem 3"
        )
        pages_text.append(text)

        data = pytesseract.image_to_data(
            image,
            lang="rus+eng",
            config="--psm 3 --oem 3",
            output_type=pytesseract.Output.DICT,
        )

        for i in range(len(data["text"])):
            word_text = data["text"][i].strip()
            word_conf = float(data["conf"][i]) if data["conf"][i] else 0

            if word_text:
                all_words.append(
                    {"text": word_text, "confidence": word_conf, "page": page_num + 1}
                )

                if word_conf >= MIN_CONFIDENCE:
                    confidences.append(word_conf)

    doc.close()
    return pages_text, all_words, confidences


def extract_words_with_coords(pdf_path):
    """
    Извлечение всех слов с координатами из PDF.

    Возвращает список слов с координатами bounding box:
    [
        {"text": "слово", "page": 1, "x0": 100, "y0": 200, "x1": 150, "y1": 220},
        ...
    ]
    """
    import time
    start_time = time.time()
    
    words_with_coords = []
    doc = fitz.open(pdf_path)
    
    ocr_time = 0
    process_time = 0

    for page_num, page in enumerate(doc):
        # Замер OCR
        ocr_start = time.time()
        pixmap = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        image = preprocess_image(image)

        data = pytesseract.image_to_data(
            image,
            lang="rus+eng",
            config="--psm 3 --oem 3",
            output_type=pytesseract.Output.DICT,
        )
        ocr_time += time.time() - ocr_start
        
        # Замер обработки
        process_start = time.time()
        scale_x = page.rect.width / image.width
        scale_y = page.rect.height / image.height

        for i in range(len(data["text"])):
            word_text = data["text"][i].strip()
            if word_text:
                # Координаты от Tesseract (в масштабе изображения)
                x0 = data["left"][i]
                y0 = data["top"][i]
                x1 = x0 + data["width"][i]
                y1 = y0 + data["height"][i]

                # Конвертируем в координаты PDF
                pdf_x0 = x0 * scale_x
                pdf_y0 = y0 * scale_y
                pdf_x1 = x1 * scale_x
                pdf_y1 = y1 * scale_y

                words_with_coords.append(
                    {
                        "text": word_text,
                        "page": page_num + 1,
                        "x0": pdf_x0,
                        "y0": pdf_y0,
                        "x1": pdf_x1,
                        "y1": pdf_y1,
                    }
                )
        process_time += time.time() - process_start

    doc.close()
    
    total_time = time.time() - start_time
    print(f"\n[TIME] OCR: {ocr_time:.2f}s ({ocr_time/total_time*100:.0f}%)")
    print(f"[TIME] Coords: {process_time:.2f}s ({process_time/total_time*100:.0f}%)")
    print(f"[TIME] TOTAL: {total_time:.2f}s")
    
    return words_with_coords


def build_confidence_map(words):
    """Построение карты уверенности: слово -> список уверенностей."""
    conf_map = {}
    for word_data in words:
        text = word_data["text"]
        conf = word_data["confidence"]
        clean = re.sub(r"^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$", "", text)
        if clean:
            if clean not in conf_map:
                conf_map[clean] = []
            conf_map[clean].append(conf)
    return conf_map


def is_valid_word(word, confidence):
    """Проверка слова на валидность."""
    # Инициалы: одна заглавная буква (с точкой или без: Ф, Ф., A, A.)
    if re.match(r"^[А-Яа-яA-Za-z]\.?$", word):
        return confidence >= 20  # Пропускаем с низкой уверенностью

    if len(word) <= 2:
        return word.lower() in SHORT_WORDS

    if re.search(r"(.)\1{2,}", word):
        return False

    letters = [char for char in word if char.isalpha()]
    if letters and len(word) >= 3:
        vowel_ratio = sum(1 for char in letters if char in VOWELS) / len(letters)
        if not (0.25 <= vowel_ratio <= 0.75):
            return False

    return len(word) > 4 or confidence >= 40


def filter_text(pages_text, confidence_map):
    """Фильтрация текста от мусора."""
    filtered = []

    for page_text in pages_text:
        for line in page_text.split("\n"):
            line = line.strip()
            if not line or not re.search(r"[А-Яа-яA-Za-z]", line):
                continue

            special_chars = re.findall(r"[^\w\sА-Яа-яA-Za-z\"\'&;,\(\)\.\-]", line)
            if len(special_chars) / max(len(line), 1) > 0.5:
                continue

            for word in line.split():
                # Для инициалов (Ф, Ф., Э, Э.) сохраняем как есть
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
    # filtered_words = remove_trailing_garbage(filtered_words, conf_map)

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
