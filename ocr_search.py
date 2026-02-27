"""
OCR для PDF с универсальным адаптивным алгоритмом.
Использует universal_ocr для лучшего качества распознавания.
"""

# Импортируем функции из универсального модуля для обратной совместимости
from universal_ocr import (
    UniversalOCR,
    TextCleaner,
    get_text,
    get_text_with_stats,
    save_to_txt_clean,
)

# Для обратной совместимости со старым кодом
# Эти функции теперь используют универсальный OCR

def save_to_txt(pdf_path, txt_path, with_coords=False):
    """
    Сохраняет текст из PDF в TXT файл.
    """
    ocr = UniversalOCR(min_confidence=40)
    result = ocr.process(pdf_path, auto_config=True)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        if with_coords:
            for word_data in result.words_data:
                bbox = word_data['bbox']
                f.write(f"{word_data['page']}|{word_data['text']}|{bbox['left']}|{bbox['top']}|{bbox['width']}|{bbox['height']}|{word_data['confidence']:.1f}\n")
        else:
            f.write(result.cleaned_text)


# Старые константы для совместимости
SCALE = 3.5  # Теперь используется в UniversalOCR
CONTRAST = 1.6
MIN_CONFIDENCE = 40
TESSERACT_CONFIG = '--psm 3'
