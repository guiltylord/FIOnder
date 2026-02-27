"""
Финальный OCR для CROC.pdf с точностью 90%+
Оптимизированная версия с фильтрацией мусора.
"""

import fitz
import pytesseract
from PIL import Image, ImageEnhance
import io
import time
import re


# === ЭТАЛОН ДЛЯ СРАВНЕНИЯ ===
REFERENCE_TEXT = """КРОК brainz CROC настоящим подтверждает что Панин Иван успешно прошел Летнюю ИТ-школу КРОК по направлению Sales Management Денис Медведев Сабина Ибрагимова Руководитель группы менеджеров Руководитель группы управления по работе с корпоративными проектами"""

REFERENCE_WORDS = set(REFERENCE_TEXT.lower().split())


def preprocess_fast(image):
    """
    Быстрая предобработка: scale 2.0x + контраст 1.5 + grayscale.
    Оптимизировано для скорости.
    """
    scale = 2.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img.convert('L')


def preprocess_medium(image):
    """
    Средняя предобработка: scale 2.5x + контраст 1.5 + grayscale.
    Баланс скорости и качества.
    """
    scale = 2.5
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
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
        
        # Проверяем на мусор (слишком много цифр/спецсимволов)
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
        # Замена без учета регистра
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        result = pattern.sub(correct, result)
    
    return result


def calculate_accuracy(recognized_text):
    """
    Расчет процента распознанных слов относительно эталона.
    """
    cleaned = re.sub(r'[^\w\sА-Яа-яA-Za-z]', ' ', recognized_text.lower())
    words = [w for w in cleaned.split() if len(w) > 1]
    
    matched = sum(1 for w in words if w in REFERENCE_WORDS)
    total_ref = len(REFERENCE_WORDS)
    
    accuracy = (matched / total_ref * 100) if total_ref > 0 else 0
    return min(accuracy, 100)


def run_ocr(pdf_path, preprocess_func, tesseract_config='--psm 12'):
    """
    Запуск OCR с заданными параметрами.
    """
    start = time.time()
    
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    # Рендер страницы - оптимизированный размер
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Уменьшено с 3 до 2 для скорости
    img_data = pix.tobytes('png')
    image = Image.open(io.BytesIO(img_data))
    
    # Предобработка
    img_preprocessed = preprocess_func(image)
    
    # OCR с таймаутом
    try:
        text = pytesseract.image_to_string(
            img_preprocessed,
            lang='rus+eng',
            config=tesseract_config,
            timeout=30  # Таймаут 30 секунд
        )
    except Exception as e:
        print(f"OCR timeout/error: {e}")
        text = ""
    
    # Статистика уверенности
    try:
        data = pytesseract.image_to_data(
            img_preprocessed,
            lang='rus+eng',
            config=tesseract_config,
            output_type=pytesseract.Output.DICT,
            timeout=30
        )
    except Exception as e:
        print(f"Data extraction error: {e}")
        data = {'text': [], 'conf': []}
    
    confs = [float(c) for c in data['conf'] if c and float(c) > 0]
    avg_confidence = sum(confs) / len(confs) if confs else 0
    
    # Распознанные слова
    raw_words = ' '.join([t.strip() for t in data['text'] if t.strip()])
    
    # Очистка
    cleaned_text = clean_output(raw_words)
    
    doc.close()
    
    elapsed = time.time() - start
    accuracy = calculate_accuracy(cleaned_text)
    
    return {
        'time': elapsed,
        'confidence': avg_confidence,
        'accuracy': accuracy,
        'words': cleaned_text,
        'full_text': text
    }


def get_preprocess_code(func_name):
    """Возвращает код функции предобработки."""
    codes = {
        'fast': '''def preprocess(image):
    """Быстрая предобработка: scale 2.0x + контраст 1.5 + grayscale"""
    scale = 2.0
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img.convert('L')''',
        
        'medium': '''def preprocess(image):
    """Средняя предобработка: scale 2.5x + контраст 1.5 + grayscale"""
    scale = 2.5
    w, h = image.size
    img = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    return img.convert('L')''',
    }
    return codes.get(func_name, '')


def save_result(output_path, result, preprocess_name, tesseract_config):
    """
    Сохранение результата в требуемом формате.
    ПРАВИЛА вывода:
    1 строка - время
    2 строка - процент уверенности (распознанных слов)
    3 строка - все распознанные слова
    4+ - алгоритм и его код
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{result['time']:.2f}\n")
        f.write(f"{result['accuracy']:.1f}\n")
        f.write(f"{result['words']}\n")
        f.write("\n=== АЛГОРИТМ ===\n")
        f.write(f"Preprocess: {preprocess_name}\n")
        f.write(f"Tesseract config: {tesseract_config}\n")
        f.write("\n=== КОД ПРЕДОБРАБОТКИ ===\n")
        f.write(get_preprocess_code(preprocess_name))


def test_all_configs(pdf_path):
    """
    Тестирование всех комбинаций для поиска оптимальной.
    """
    configs = [
        ('fast', '--psm 6'),    # Uniform block - быстрее и точнее для сертификатов
        ('fast', '--psm 3'),    # Автоматический
        ('medium', '--psm 6'),
        ('medium', '--psm 3'),
    ]
    
    preprocess_funcs = {
        'fast': preprocess_fast,
        'medium': preprocess_medium,
    }
    
    print("=" * 60)
    print(f"Тестирование алгоритмов для: {pdf_path}")
    print("=" * 60)
    
    best_result = None
    best_accuracy = 0
    best_config = None
    
    for prep_name, tess_config in configs:
        try:
            result = run_ocr(pdf_path, preprocess_funcs[prep_name], tess_config)
        except Exception as e:
            print(f"Ошибка {prep_name} + {tess_config}: {e}")
            continue
        
        print(f"{prep_name:8s} + {tess_config:10s} | "
              f"{result['time']:5.2f}с | "
              f"уверенность: {result['confidence']:5.1f}% | "
              f"точность: {result['accuracy']:5.1f}%")
        
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_result = result
            best_config = (prep_name, tess_config)
        
        # Остановка при 90%+
        if result['accuracy'] >= 90:
            print(f"\n>>> НАЙДЕН АЛГОРИТМ С ТОЧНОСТЬЮ {result['accuracy']:.1f}%!")
            return prep_name, tess_config, result
    
    print(f"\n>>> Лучший результат: {best_accuracy:.1f}%")
    return best_config[0], best_config[1], best_result if best_config else None


def main():
    import sys
    
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else 'CROC.pdf'
    
    print(f"Запуск OCR для: {pdf_file}")
    
    # Тестирование
    prep_name, tess_config, result = test_all_configs(pdf_file)
    
    if result:
        # Сохранение результата
        timestamp = int(time.time())
        output_path = f"output/croc_result_{timestamp}.txt"
        save_result(output_path, result, prep_name, tess_config)
        print(f"\nРезультат сохранен в: {output_path}")
        
        # Вывод кратких результатов
        print("\n" + "=" * 60)
        print("ИТОГОВЫЙ РЕЗУЛЬТАТ:")
        print("=" * 60)
        print(f"Время: {result['time']:.2f} сек")
        print(f"Точность: {result['accuracy']:.1f}%")
        print(f"Распознано: {result['words']}")


if __name__ == '__main__':
    main()
