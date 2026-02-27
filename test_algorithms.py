"""
Тестирование алгоритмов OCR для достижения 90%+ распознавания.
Выводит результаты в требуемом формате.
"""

import fitz
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import time
import os
import re

# Эталонный текст для проверки точности
REFERENCE_TEXT = "КРОК brainz настоящим подтверждает что Панин Иван успешно прошел Летнюю ИТ-школу КРОК по направлению Sales Management Денис Медведев Сабина Ибрагимова Руководитель группы менеджеров Руководитель группы управления по работе с корпоративными проектами"

REFERENCE_WORDS = set(REFERENCE_TEXT.lower().split())


def calculate_accuracy(recognized_text):
    """
    Расчет процента распознанных слов относительно эталона.
    """
    # Очистка от мусора
    cleaned = re.sub(r'[^\w\sА-Яа-яA-Za-z]', ' ', recognized_text.lower())
    words = [w for w in cleaned.split() if len(w) > 1]
    
    # Подсчет совпадений
    matched = sum(1 for w in words if w in REFERENCE_WORDS)
    total_ref = len(REFERENCE_WORDS)
    
    # Процент распознанных слов из эталона
    accuracy = (matched / total_ref * 100) if total_ref > 0 else 0
    return min(accuracy, 100)


def preprocess_v1(image):
    """Базовый: 3x + контраст 1.5 + grayscale"""
    scale = 3.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img.convert('L')


def preprocess_v2(image):
    """Усиленный: 4x + контраст 1.6 + резкость 1.2"""
    scale = 4.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.6)
    sharpen = ImageEnhance.Sharpness(img)
    img = sharpen.enhance(1.2)
    return img.convert('L')


def preprocess_v3(image):
    """Мягкий: 3x + контраст 1.3 + grayscale"""
    scale = 3.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    return img.convert('L')


def preprocess_v4(image):
    """Агрессивный: 5x + контраст 2.0 + резкость 1.5 + бинаризация"""
    scale = 5.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    sharpen = ImageEnhance.Sharpness(img)
    img = sharpen.enhance(1.5)
    img = img.convert('L')
    # Бинаризация
    img = img.point(lambda x: 0 if x < 128 else 255)
    return img


def preprocess_v5(image):
    """Оптимальный: 4x + контраст 1.4 + яркость 1.1"""
    scale = 4.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.4)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    return img.convert('L')


def preprocess_v6(image):
    """Без grayscale + 4x"""
    scale = 4.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img


def preprocess_v7(image):
    """Denoise + 4x"""
    scale = 4.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img.convert('L')


def preprocess_v8(image):
    """Адаптивный: 4x + gamma 0.8 + контраст"""
    scale = 4.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    img = ImageOps.autocontrast(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img.convert('L')


PREPROCESS_FUNCS = {
    'v1_basic': preprocess_v1,
    'v2_enhanced': preprocess_v2,
    'v3_soft': preprocess_v3,
    'v4_aggressive': preprocess_v4,
    'v5_optimal': preprocess_v5,
    'v6_color': preprocess_v6,
    'v7_denoise': preprocess_v7,
    'v8_autocontrast': preprocess_v8,
}

# Конфигурации Tesseract
TESSERACT_CONFIGS = {
    'psm3': '--psm 3',  # Автоматический
    'psm6': '--psm 6',  # Uniform block
    'psm11': '--psm 11',  # Sparse text
    'psm12': '--psm 12',  # Sparse text + OSD
    'psm4': '--psm 4',  # Column
    'oem1': '--oem 1',  # LSTM only
    'oem2': '--oem 2',  # LSTM + Tesseract
}


def test_algorithm(pdf_path, preprocess_name, tesseract_config, page_num=0):
    """
    Тестирование одной комбинации алгоритма.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    preprocess_func = PREPROCESS_FUNCS.get(preprocess_name, preprocess_v1)
    
    # Рендер страницы
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
    img_data = pix.tobytes('png')
    image = Image.open(io.BytesIO(img_data))
    
    # Предобработка
    img_preprocessed = preprocess_func(image)
    
    # OCR
    start = time.time()
    text = pytesseract.image_to_string(
        img_preprocessed, 
        lang='rus+eng', 
        config=tesseract_config
    )
    elapsed = time.time() - start
    
    # Статистика уверенности
    data = pytesseract.image_to_data(
        img_preprocessed, 
        lang='rus+eng', 
        config=tesseract_config,
        output_type=pytesseract.Output.DICT
    )
    
    confs = [float(c) for c in data['conf'] if c and float(c) > 0]
    avg_confidence = sum(confs) / len(confs) if confs else 0
    
    # Распознанные слова
    words = ' '.join([t.strip() for t in data['text'] if t.strip()])
    
    doc.close()
    
    # Расчет точности
    accuracy = calculate_accuracy(text)
    
    return {
        'time': elapsed,
        'confidence': avg_confidence,
        'accuracy': accuracy,
        'words': words,
        'full_text': text
    }


def save_result(output_path, result, preprocess_name, tesseract_config, preprocess_code):
    """
    Сохранение результата в требуемом формате.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{result['time']:.2f}\n")
        f.write(f"{result['accuracy']:.1f}\n")
        f.write(f"{result['words']}\n")
        f.write("\n=== АЛГОРИТМ ===\n")
        f.write(f"Preprocess: {preprocess_name}\n")
        f.write(f"Tesseract config: {tesseract_config}\n")
        f.write("\n=== КОД ПРЕДОБРАБОТКИ ===\n")
        f.write(preprocess_code)


def get_preprocess_code(preprocess_name):
    """Возвращает код функции предобработки."""
    codes = {
        'v1_basic': '''def preprocess(image):
    scale = 3.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img.convert('L')''',
        
        'v2_enhanced': '''def preprocess(image):
    scale = 4.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.6)
    sharpen = ImageEnhance.Sharpness(img)
    img = sharpen.enhance(1.2)
    return img.convert('L')''',
        
        'v3_soft': '''def preprocess(image):
    scale = 3.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    return img.convert('L')''',
        
        'v4_aggressive': '''def preprocess(image):
    scale = 5.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    sharpen = ImageEnhance.Sharpness(img)
    img = sharpen.enhance(1.5)
    img = img.convert('L')
    return img.point(lambda x: 0 if x < 128 else 255)''',
        
        'v5_optimal': '''def preprocess(image):
    scale = 4.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.4)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    return img.convert('L')''',
        
        'v6_color': '''def preprocess(image):
    scale = 4.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img  # Без grayscale''',
        
        'v7_denoise': '''def preprocess(image):
    scale = 4.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img.convert('L')''',
        
        'v8_autocontrast': '''def preprocess(image):
    scale = 4.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    img = ImageOps.autocontrast(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img.convert('L')''',
    }
    return codes.get(preprocess_name, 'Unknown')


def run_full_test(pdf_path, output_dir='output'):
    """
    Запуск полного тестирования всех комбинаций.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Тестирование алгоритмов OCR для: {pdf_path}")
    print("=" * 60)
    
    best_result = None
    best_accuracy = 0
    
    for preprocess_name in PREPROCESS_FUNCS:
        for tesseract_config in TESSERACT_CONFIGS:
            config_str = TESSERACT_CONFIGS[tesseract_config]
            
            try:
                result = test_algorithm(pdf_path, preprocess_name, config_str)
            except Exception as e:
                print(f"Ошибка {preprocess_name} + {tesseract_config}: {e}")
                continue
            
            accuracy = result['accuracy']
            
            print(f"{preprocess_name:15s} + {tesseract_config:8s} | "
                  f"{result['time']:5.2f}с | "
                  f"уверенность: {result['confidence']:5.1f}% | "
                  f"точность: {accuracy:5.1f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_result = (preprocess_name, tesseract_config, result)
            
            # Если точность > 90%, сохраняем и останавливаемся
            if accuracy >= 90:
                print(f"\n>>> НАЙДЕН АЛГОРИТМ С ТОЧНОСТЬЮ {accuracy:.1f}%!")
                timestamp = int(time.time())
                output_path = f"{output_dir}/result_{preprocess_name}_{tesseract_config}_{timestamp}.txt"
                save_result(
                    output_path, 
                    result, 
                    preprocess_name, 
                    tesseract_config,
                    get_preprocess_code(preprocess_name)
                )
                print(f"Результат сохранен в: {output_path}")
                return best_result
    
    # Если лучший результат < 90%, все равно сохраняем лучший
    if best_result:
        preprocess_name, tesseract_config, result = best_result
        print(f"\n>>> Лучший результат: {best_accuracy:.1f}%")
        print(f"Алгоритм: {preprocess_name} + {tesseract_config}")
        timestamp = int(time.time())
        output_path = f"{output_dir}/result_best_{preprocess_name}_{tesseract_config}_{timestamp}.txt"
        save_result(
            output_path, 
            result, 
            preprocess_name, 
            tesseract_config,
            get_preprocess_code(preprocess_name)
        )
        print(f"Результат сохранен в: {output_path}")
    
    return best_result


if __name__ == '__main__':
    import sys
    
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else 'croc.pdf'
    
    if not os.path.exists(pdf_file):
        print(f"Файл {pdf_file} не найден!")
        print("Поместите файл croc.pdf в директорию проекта и запустите снова")
        sys.exit(1)
    
    run_full_test(pdf_file)
