import cv2
import fitz          # PyMuPDF
import numpy as np
import pytesseract
from deskew import determine_skew

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

# Путь к исполняемому файлу Tesseract (для Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Папка с tessdata_best моделями (оставьте None, чтобы использовать системные модели)
TESSDATA_DIR = None

# Языки для распознавания
TESSERACT_LANG = "rus+eng"

# DPI рендеринга страницы PDF.
# 300 — стандарт. 400–600 — для документов с мелким шрифтом (< 9pt).
PDF_DPI = 300

# Минимальный порог уверенности Tesseract (0–100).
MIN_CONFIDENCE = 60

# Порог угла перекоса (градусы).
DESKEW_THRESHOLD_DEG = 0.5

# Режим сегментации страницы (PSM).
TESSERACT_PSM = 6

# Движок OCR (OEM).
TESSERACT_OEM = 1

# =============================================================================
# ТАБЛИЦА ЗАМЕНЫ: латинские символы → кириллические двойники
# =============================================================================
# Только символы, которые Tesseract реально путает при смешанном rus+eng режиме.
# Ключ — латиница, значение — кириллица, идентичная визуально.

_LATIN_TO_CYR: dict[str, str] = {
    # Строчные
    "a": "а",
    "e": "е",
    "o": "о",
    "p": "р",
    "c": "с",
    "x": "х",
    "y": "у",
    # Прописные
    "A": "А",
    "B": "В",
    "C": "С",
    "E": "Е",
    "H": "Н",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "X": "Х",
    "Y": "У",
}

# Предварительно строим обратный набор: все кириллические символы Unicode-диапазона
_CYR_CHARS = set(chr(c) for c in range(ord("А"), ord("я") + 1))
_CYR_CHARS.update({"Ё", "ё"})


# =============================================================================
# ИСПРАВЛЕНИЕ СМЕШАННОГО АЛФАВИТА
# =============================================================================

def _cyrillic_ratio(text: str) -> float:
    """Доля кириллических букв в строке (0.0 – 1.0). Учитываются только буквы."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if c in _CYR_CHARS)
    return cyr / len(letters)


def fix_mixed_script(text: str, cyr_threshold: float = 0.5) -> str:
    """
    Исправляет слово, в котором OCR случайно подставил латинские символы
    вместо кириллических двойников.

    Логика:
      • Подсчитываем долю кириллицы среди букв слова.
      • Если доля ≥ cyr_threshold — слово кириллическое; заменяем все
        визуально совпадающие латинские буквы на кириллические.
      • Если слово явно латинское (английское) — не трогаем.

    Примеры ошибок Tesseract, которые исправляет функция:
      «дoкyмeнт»  → «документ»   (o,y,e — латиница)
      «cтaтья»    → «статья»     (c,a — латиница)
      «ЗAЯВЛEНИE» → «ЗАЯВЛЕНИЕ»  (A,E — латиница)
    """
    # Быстрая проверка: если нет ни одного латинского символа из таблицы — выходим
    if not any(c in _LATIN_TO_CYR for c in text):
        return text

    ratio = _cyrillic_ratio(text)

    # Слово преимущественно кириллическое — исправляем латинские вкрапления
    if ratio >= cyr_threshold:
        return "".join(_LATIN_TO_CYR.get(c, c) for c in text)

    # Слово преимущественно латинское — не трогаем (например, "PDF", "API")
    return text


# =============================================================================
# СКЛЕЙКА ПЕРЕНОСОВ
# =============================================================================

def _join_hyphenated_lines(raw_words: list[dict]) -> list[dict]:
    """
    Объединяет слова, разбитые переносом через дефис между строками.

    Tesseract возвращает поля block_num, line_num, word_num, которые мы
    используем для определения позиции слова на странице.

    Алгоритм:
      1. Группируем слова по (page, block_num, line_num).
      2. Если последнее слово строки оканчивается на «-» и следующая строка
         того же блока существует — соединяем их, убирая дефис.
      3. Координаты объединённого слова: минимальный bbox обоих.
      4. Уверенность: минимальная из двух (консервативная оценка).

    Аргумент:
        raw_words — список, обогащённый служебными полями
        ``_block``, ``_line``, ``_word_idx`` (добавляются внутри extract_words_with_coords).

    Возвращает очищенный список (без служебных полей).
    """
    if not raw_words:
        return []

    # --- Группировка по (page, block, line) ---
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for w in raw_words:
        key = (w["page"], w.get("_block", 0), w.get("_line", 0))
        groups[key].append(w)

    # Сортируем строки по странице → блоку → строке → позиции слова
    sorted_keys = sorted(groups.keys())

    merged: list[dict] = []
    skip_next_first_word: set[tuple] = set()  # ключи строк, у которых съедено первое слово

    for i, key in enumerate(sorted_keys):
        line_words = groups[key]  # слова уже упорядочены по word_num

        start_idx = 0
        if key in skip_next_first_word:
            start_idx = 1  # первое слово уже поглощено предыдущей строкой

        for j, word in enumerate(line_words[start_idx:], start=start_idx):
            is_last_in_line = (j == len(line_words) - 1)

            if is_last_in_line and word["text"].endswith("-") and i + 1 < len(sorted_keys):
                # Проверяем, что следующая строка в том же блоке
                next_key = sorted_keys[i + 1]
                same_page = next_key[0] == key[0]
                same_block = next_key[1] == key[1]

                if same_page and same_block and next_key not in skip_next_first_word:
                    next_line_words = groups[next_key]
                    if next_line_words:
                        continuation = next_line_words[0]

                        # Собираем склеенное слово
                        base_text = word["text"][:-1]  # убираем дефис
                        joined_text = base_text + continuation["text"]

                        joined_word = {
                            "text": joined_text,
                            "page": word["page"],
                            "confidence": min(word["confidence"], continuation["confidence"]),
                            "x0": min(word["x0"], continuation["x0"]),
                            "y0": min(word["y0"], continuation["y0"]),
                            "x1": max(word["x1"], continuation["x1"]),
                            "y1": max(word["y1"], continuation["y1"]),
                            "_block": word.get("_block"),
                            "_line": word.get("_line"),
                        }
                        merged.append(joined_word)
                        skip_next_first_word.add(next_key)
                        continue  # не добавляем исходный «word» с дефисом

            merged.append(word)

    # Удаляем служебные поля перед возвратом
    for w in merged:
        w.pop("_block", None)
        w.pop("_line", None)

    return merged


# =============================================================================
# PREPROCESSING
# =============================================================================

def _build_tesseract_config() -> str:
    parts = [f"--oem {TESSERACT_OEM}", f"--psm {TESSERACT_PSM}"]
    if TESSDATA_DIR:
        parts.append(f'--tessdata-dir "{TESSDATA_DIR}"')
    return " ".join(parts)


def _deskew_image(gray: np.ndarray) -> np.ndarray:
    angle = determine_skew(gray)
    if angle is None or abs(angle) < DESKEW_THRESHOLD_DEG:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    print(f"  [DESKEW] Угол исправлен: {angle:.2f}°")
    return rotated


def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Полный пайплайн предобработки изображения перед OCR.

    Этапы:
      1. Grayscale         — убираем цвет.
      2. Deskew            — исправляем наклон страницы.
      3. Denoise           — подавляем шум.
      4. CLAHE             — локальное выравнивание контраста.
      5. Adaptive threshold — бинаризация.
      6. Morphology        — закрываем разрывы в символах.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _deskew_image(gray)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    binary = cv2.adaptiveThreshold(
        src=enhanced,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=4,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return processed


def _is_valid_word(text: str) -> bool:
    if not text:
        return False
    if len(text) == 1 and not text.isalnum():
        return False
    allowed = sum(1 for c in text if c.isalnum() or c in "-.,:/")
    return (allowed / len(text)) >= 0.5


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================

def extract_words_with_coords(pdf_path: str) -> list[dict]:
    """
    Извлекает все слова и их координаты из PDF-файла.

    Улучшения по сравнению с базовой версией:
      • fix_mixed_script()      — заменяет латинские «двойники» на кириллицу
                                  в словах, которые OCR частично прочитал латиницей.
      • _join_hyphenated_lines()— склеивает слова, разбитые переносом
                                  в конце строки (убирает «-» и объединяет части).

    Возвращает список словарей:
      {
        "text": str,       — распознанное слово (после исправлений)
        "page": int,       — номер страницы (начиная с 1)
        "confidence": int, — уверенность Tesseract (0–100)
        "x0": int, "y0": int, "x1": int, "y1": int,
      }
    """
    all_words: list[dict] = []
    tess_config = _build_tesseract_config()

    print(f"[INFO] Начата обработка файла: {pdf_path}")
    print(f"[INFO] Tesseract config: {tess_config}")

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        

        for page_num, page in enumerate(doc, start=1):
            print(f"[INFO] Страница {page_num}/{total_pages}...")

            pix = page.get_pixmap(dpi=PDF_DPI)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                opencv_image = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:
                opencv_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            processed = preprocess_image_for_ocr(opencv_image)

            data = pytesseract.image_to_data(
                processed,
                lang=TESSERACT_LANG,
                config=tess_config,
                output_type=pytesseract.Output.DICT,
            )

            # --- Сбор слов с сохранением позиционных мета-полей ---
            page_words_raw: list[dict] = []
            fixed_count = 0

            for i in range(len(data["level"])):
                confidence = int(data["conf"][i])
                if confidence < MIN_CONFIDENCE:
                    continue

                text = data["text"][i].strip()
                if not _is_valid_word(text):
                    continue

                # ── Исправление смешанного алфавита ──────────────────────────
                fixed_text = fix_mixed_script(text)
                if fixed_text != text:
                    fixed_count += 1
                    print(f"  [FIX] «{text}» → «{fixed_text}»")

                x, y, w, h = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )

                page_words_raw.append({
                    "text": fixed_text,
                    "page": page_num,
                    "confidence": confidence,
                    "x0": x,
                    "y0": y,
                    "x1": x + w,
                    "y1": y + h,
                    # Служебные поля для склейки переносов
                    "_block": data["block_num"][i],
                    "_line": data["line_num"][i],
                })

            # ── Склейка переносов ─────────────────────────────────────────────
            page_words = _join_hyphenated_lines(page_words_raw)

            joined_count = len(page_words_raw) - len(page_words)
           

            all_words.extend(page_words)

        doc.close()
        print(f"[INFO] Готово. Всего слов извлечено: {len(all_words)}")

    except FileNotFoundError:
        print(f"[ERROR] Файл не найден: {pdf_path}")
    except Exception as e:
        print(f"[ERROR] Критическая ошибка: {e}")

    return all_words


# =============================================================================
# УТИЛИТЫ ДЛЯ ДИАГНОСТИКИ
# =============================================================================

def preview_preprocessing(pdf_path: str, page_num: int = 1, output_dir: str = "."):
    """
    Сохраняет промежуточные изображения пайплайна для визуальной диагностики.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=PDF_DPI)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        original = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
    else:
        original = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    doc.close()

    cv2.imwrite(os.path.join(output_dir, "0_original.png"), original)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "1_gray.png"), gray)
    deskewed = _deskew_image(gray)
    cv2.imwrite(os.path.join(output_dir, "2_deskewed.png"), deskewed)
    denoised = cv2.fastNlMeansDenoising(deskewed, h=10, templateWindowSize=7, searchWindowSize=21)
    cv2.imwrite(os.path.join(output_dir, "3_denoised.png"), denoised)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    cv2.imwrite(os.path.join(output_dir, "4_clahe.png"), enhanced)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 4)
    cv2.imwrite(os.path.join(output_dir, "5_binary.png"), binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    final = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, "6_final.png"), final)

    print(f"[PREVIEW] Промежуточные изображения сохранены в: {output_dir}/")


def benchmark_confidence(pdf_path: str, max_pages: int = 3) -> dict:
    """
    Анализирует распределение уверенности Tesseract по документу.
    """
    all_confidences = []
    tess_config = _build_tesseract_config()

    doc = fitz.open(pdf_path)
    pages_to_check = min(max_pages, len(doc))

    for page_num in range(pages_to_check):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=PDF_DPI)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

        processed = preprocess_image_for_ocr(img)
        data = pytesseract.image_to_data(processed, lang=TESSERACT_LANG,
                                          config=tess_config,
                                          output_type=pytesseract.Output.DICT)
        for conf in data["conf"]:
            c = int(conf)
            if c >= 0:
                all_confidences.append(c)

    doc.close()

    if not all_confidences:
        return {}

    arr = np.array(all_confidences)
    stats = {
        "total_tokens": len(arr),
        "mean_confidence": round(float(np.mean(arr)), 1),
        "median_confidence": round(float(np.median(arr)), 1),
        "pct_above_40": round(float(np.mean(arr >= 40)) * 100, 1),
        "pct_above_60": round(float(np.mean(arr >= 60)) * 100, 1),
        "pct_above_80": round(float(np.mean(arr >= 80)) * 100, 1),
    }

    print("\n[BENCHMARK] Статистика уверенности Tesseract:")
    print(f"  Всего токенов:        {stats['total_tokens']}")
    print(f"  Средняя уверенность:  {stats['mean_confidence']}")
    print(f"  Медиана:              {stats['median_confidence']}")
    print(f"  Токенов с conf ≥ 40:  {stats['pct_above_40']}%")
    print(f"  Токенов с conf ≥ 60:  {stats['pct_above_60']}%")
    print(f"  Токенов с conf ≥ 80:  {stats['pct_above_80']}%")
    print(f"\n  Рекомендация: если pct_above_60 > 70% — ставьте MIN_CONFIDENCE=60.")
    print(f"                если pct_above_60 < 50% — документ сложный, используйте 40–50.\n")

    return stats

