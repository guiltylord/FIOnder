"""
Извлечение слов с координатами из PDF через EasyOCR.

ВЫХОД:
    list[dict] — каждый элемент:
        {
            "text": str,      # слово
            "page": int,      # номер страницы (1-based)
            "x0": float,      # левый край  (в единицах PDF)
            "y0": float,      # верхний край
            "x1": float,      # правый край
            "y1": float,      # нижний край
        }
"""

import io
import os
import sys
import time

import fitz
import numpy as np
from PIL import Image

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

SCALE    = 2.0   # масштаб при рендере страницы (больше = точнее, но медленнее)
TARGET_DPI = 300  # Стандарт для качественного распознавания
MIN_CONF = 0.3   # минимальная уверенность OCR-блока
PAD      = 40    # отступ в пикселях вокруг страницы перед OCR

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ EASYOCR
# =============================================================================

try:
    import easyocr
    print("[INFO] Загрузка EasyOCR (ru)...")
    _reader = easyocr.Reader(["ru"], gpu=False, verbose=False)
    print("[INFO] EasyOCR готов.")
except ImportError:
    print("[ОШИБКА] pip install easyocr")
    sys.exit(1)
except Exception as e:
    print(f"[КРИТИЧЕСКАЯ ОШИБКА] {e}")
    sys.exit(1)


# =============================================================================
# ПУБЛИЧНЫЙ API
# =============================================================================

def extract_words_with_coords(pdf_path: str) -> list[dict]:
    """
    Открывает PDF, прогоняет каждую страницу через OCR,
    возвращает список слов с координатами в системе координат PDF.

    Args:
        pdf_path: путь к PDF-файлу.

    Returns:
        Список словарей с полями: text, page, x0, y0, x1, y1.
    """
    if not os.path.exists(pdf_path):
        print(f"[ОШИБКА] Файл не найден: {pdf_path}")
        return []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ОШИБКА] Не удалось открыть PDF: {e}")
        return []

    words     = []
    total     = len(doc)
    t_ocr     = 0.0
    t_prep    = 0.0
    t_start   = time.time()

    for page_num, page in enumerate(doc, start=1):
        print(f"  [стр. {page_num}/{total}] рендер...", end=" ", flush=True)        
        
        # ── Рендер с динамическим масштабом ────────────────────────────────────
        t0 = time.time()

        # Рассчитываем масштаб исходя из DPI. 
        # В PDF 1 единица = 1/72 дюйма. 300 DPI / 72 = 4.16.
        # Это гарантирует, что буквы будут одного размера в пикселях на любом формате.
        zoom = TARGET_DPI / 72
        mat = fitz.Matrix(zoom, zoom)
        
        pixmap = page.get_pixmap(matrix=mat)
        image  = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        img_np = np.array(image)

        # Коэффициенты обратного масштабирования (ВАЖНО!)
        # Используем фактические размеры полученного изображения
        scale_x = page.rect.width  / image.width
        scale_y = page.rect.height / image.height

        # Добавляем отступ (PAD)
        img_np = np.pad(
            img_np,
            ((PAD, PAD), (PAD, PAD), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        t_prep += time.time() - t0

        # ── OCR ──────────────────────────────────────────────────────────────
        t0      = time.time()
        results = _reader.readtext(img_np, detail=1, paragraph=False)
        elapsed = time.time() - t0
        t_ocr  += elapsed
        print(f"OCR {elapsed:.1f}с, блоков: {len(results)}")

        # ── Разбиение блоков на слова ─────────────────────────────────────────
        for bbox, text, conf in results:
            if conf < MIN_CONF:
                continue
            text = text.strip()
            if not text:
                continue

            # Координаты блока в пикселях (с вычетом PAD-отступа)
            xs = [p[0] - PAD for p in bbox]
            ys = [p[1] - PAD for p in bbox]
            bx0, bx1 = min(xs), max(xs)
            by0, by1 = min(ys), max(ys)

            # Перевод в единицы PDF
            pdf_x0 = bx0 * scale_x
            pdf_x1 = bx1 * scale_x
            pdf_y0 = by0 * scale_y
            pdf_y1 = by1 * scale_y

            word_list = text.split()

            if len(word_list) == 1:
                words.append(_make_word(word_list[0], page_num, pdf_x0, pdf_y0, pdf_x1, pdf_y1))
            else:
                # Аппроксимируем координаты каждого слова по ширине символов
                words.extend(
                    _split_block_into_words(
                        word_list, text, page_num, pdf_x0, pdf_y0, pdf_x1, pdf_y1
                    )
                )

    doc.close()

    total_t = time.time() - t_start
    print(f"[TIME] OCR: {t_ocr:.2f}с | Подготовка: {t_prep:.2f}с | Итого: {total_t:.2f}с")
    return words


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def _make_word(text: str, page: int, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"text": text, "page": page, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _split_block_into_words(
    word_list: list,
    full_text: str,
    page: int,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> list[dict]:
    """
    Делит блок OCR на отдельные слова, аппроксимируя X-координаты
    пропорционально позиции символов в строке.
    """
    char_w   = (bx1 - bx0) / max(len(full_text), 1)
    result   = []
    curr_idx = 0

    for w in word_list:
        sc = full_text.find(w, curr_idx)
        if sc == -1:
            sc = curr_idx
        ec = sc + len(w)

        result.append(_make_word(
            w, page,
            bx0 + sc * char_w,
            by0,
            bx0 + ec * char_w,
            by1,
        ))
        curr_idx = ec

    return result