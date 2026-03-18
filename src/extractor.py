import os
import sys
import io
import time

import fitz
import numpy as np
from PIL import Image

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

SCALE    = 2.0   # 3.0 слишком тяжело для EasyOCR на CPU
MIN_CONF = 0.3
PAD      = 40

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ EASYOCR
# =============================================================================

try:
    import easyocr
    print("[INFO] Загрузка EasyOCR (ru)...")
    reader = easyocr.Reader(['ru'], gpu=False, verbose=False)
    print("[INFO] EasyOCR готов.")
except ImportError:
    print("[ОШИБКА] pip install easyocr")
    sys.exit(1)
except Exception as e:
    print(f"[КРИТИЧЕСКАЯ ОШИБКА] {e}")
    sys.exit(1)

# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================

def extract_words_with_coords(pdf_path):
    start_time = time.time()
    words_with_coords = []

    if not os.path.exists(pdf_path):
        print(f"[ОШИБКА] Файл не найден: {pdf_path}")
        return []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ОШИБКА] {e}")
        return []

    total_pages = len(doc)
    ocr_time     = 0
    process_time = 0

    for page_num, page in enumerate(doc, start=1):
        print(f"  [стр. {page_num}/{total_pages}] обработка...", end=' ', flush=True)
        t0      = time.time()
        pixmap  = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        image   = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        img_np  = np.array(image)
        scale_x = page.rect.width  / image.width
        scale_y = page.rect.height / image.height
        img_np  = np.pad(img_np, ((PAD, PAD), (PAD, PAD), (0, 0)),
                         mode='constant', constant_values=255)
        process_time += time.time() - t0

        t0     = time.time()
        result = reader.readtext(img_np, detail=1, paragraph=False)
        elapsed = time.time() - t0
        ocr_time += elapsed
        print(f"{elapsed:.1f}s, найдено блоков: {len(result)}")

        for (bbox, text, conf) in result:
            if conf < MIN_CONF:
                continue
            text = text.strip()
            if not text:
                continue

            x_coords = [p[0] - PAD for p in bbox]
            y_coords = [p[1] - PAD for p in bbox]
            x0, x1   = min(x_coords), max(x_coords)
            y0, y1   = min(y_coords), max(y_coords)

            pdf_x0 = x0 * scale_x
            pdf_x1 = x1 * scale_x
            pdf_y0 = y0 * scale_y
            pdf_y1 = y1 * scale_y

            words = text.split()
            if len(words) == 1:
                words_with_coords.append({
                    "text": words[0], "page": page_num,
                    "x0": pdf_x0, "y0": pdf_y0,
                    "x1": pdf_x1, "y1": pdf_y1,
                })
            else:
                char_w   = (pdf_x1 - pdf_x0) / max(len(text), 1)
                curr_idx = 0
                for w in words:
                    sc = text.find(w, curr_idx)
                    if sc == -1: sc = curr_idx
                    ec = sc + len(w)
                    words_with_coords.append({
                        "text": w, "page": page_num,
                        "x0": pdf_x0 + sc * char_w,
                        "y0": pdf_y0,
                        "x1": pdf_x0 + ec * char_w,
                        "y1": pdf_y1,
                    })
                    curr_idx = ec

    doc.close()
    total = time.time() - start_time
    print(f"\n[TIME] OCR: {ocr_time:.2f}s | Prep: {process_time:.2f}s | TOTAL: {total:.2f}s")
    return words_with_coords