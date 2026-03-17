# extractor.py

import cv2  # OpenCV для обработки изображений
import fitz  # PyMuPDF
import numpy as np
import pytesseract

# =============================================================================
# НАСТРОЙКИ TESSERACT И ОБРАБОТКИ
# =============================================================================
# Если Tesseract не в системном PATH, укажите путь к нему:
# Например, для Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Языки для распознавания. Для документов РФ 'rus+eng' - лучший выбор.
TESSERACT_LANG = 'rus+eng'

# DPI (точек на дюйм) для рендеринга страниц PDF. 300 - хороший баланс качества и скорости.
PDF_DPI = 300

# Минимальный уровень "уверенности" Tesseract в распознанном слове (от 0 до 100).
# Слова с уверенностью ниже этого порога будут отброшены.
MIN_CONFIDENCE = 40
# =============================================================================


def preprocess_image_for_ocr(image):
    """
    Выполняет предварительную обработку изображения для улучшения качества OCR.
    """
    # 1. Преобразование в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Адаптивная бинаризация (превращение в черно-белое изображение).
    # Этот метод отлично работает для документов с неравномерным освещением или тенями,
    # так как он вычисляет порог для разных участков изображения индивидуально.
    processed_image = cv2.adaptiveThreshold(
        src=gray_image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,  # Размер соседней области для вычисления порога
        C=2            # Константа, вычитаемая из среднего
    )

    return processed_image


def extract_words_with_coords(pdf_path: str) -> list:
    """
    Извлекает все слова и их координаты из PDF-файла, используя PyMuPDF и Tesseract.

    Для каждой страницы выполняется рендеринг в изображение, предобработка
    с помощью OpenCV и затем OCR с помощью Tesseract.
    """
    all_words = []
    print(f"[INFO] Начата обработка файла: {pdf_path}")

    try:
        # 1. Открываем PDF-файл
        doc = fitz.open(pdf_path)
        print(f"[INFO] PDF успешно открыт, страниц: {len(doc)}.")

        # 2. Итерируемся по каждой странице документа
        for page_num, page in enumerate(doc, 1):
            print(f"[INFO] Обработка страницы {page_num}/{len(doc)}...")

            # 3. Конвертируем страницу в изображение (объект Pixmap)
            pix = page.get_pixmap(dpi=PDF_DPI)

            # 4. Конвертируем Pixmap в формат Numpy array, понятный для OpenCV
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            # OpenCV использует порядок BGR, а PyMuPDF - RGB. Меняем каналы местами.
            if img_data.shape[2] == 4: # RGBA
                opencv_image = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else: # RGB
                opencv_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            # 5. Применяем улучшающую предобработку
            processed_image = preprocess_image_for_ocr(opencv_image)

            # 6. Распознаем текст с помощью Tesseract, получая детальную информацию
            data = pytesseract.image_to_data(processed_image, lang=TESSERACT_LANG, output_type=pytesseract.Output.DICT)

            # 7. Фильтруем и сохраняем результаты
            num_boxes = len(data['level'])
            for i in range(num_boxes):
                # Берем только элементы, являющиеся словами, с достаточной уверенностью
                confidence = int(data['conf'][i])
                if confidence > MIN_CONFIDENCE:
                    text = data['text'][i].strip()
                    if text:  # Пропускаем пустые строки
                        # Координаты слова
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        
                        word_data = {
                            "text": text,
                            "page": page_num,
                            "x0": x,
                            "y0": y,
                            "x1": x + w,
                            "y1": y + h
                        }
                        all_words.append(word_data)
        
        doc.close()
        print("[INFO] Обработка всех страниц завершена.")

    except Exception as e:
        print(f"[ERROR] Произошла критическая ошибка при обработке PDF: {e}")
        print("[ERROR] Убедитесь, что файл существует и не поврежден.")

    return all_words
