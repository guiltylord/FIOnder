# PDFinder — OCR для PDF

Простой скрипт для распознавания текста в PDF с помощью Tesseract OCR.

---

## Установка

### 1. Tesseract OCR

Скачайте установщик с официальной страницы:  
**https://github.com/UB-Mannheim/tesseract/wiki**

Рекомендуется версия **5.x** для Windows.

### 2. Языковые данные (traineddata)

По умолчанию устанавливается только английский язык. Для русского выполните в PowerShell (от администратора):

```powershell
# Скачивание русского языка
Invoke-WebRequest -Uri "https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata" -OutFile "$env:TEMP\rus.traineddata"

# Копирование в папку Tesseract (требуются права администратора)
Copy-Item "$env:TEMP\rus.traineddata" "C:\Program Files\Tesseract-OCR\tessdata\rus.traineddata"

# Проверка
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
```

> **Примечание:** Если нужен обычный язык, а не улучшенная модель — уберите `_best` в ссылке:  
> `https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata`

### 3. Python-зависимости

```bash
pip install -r requirements.txt
```

---

## Использование

### 1. Настройка в main.py

Откройте `main.py` и измените две строки:

```python
# === НАСТРОЙКИ ===
WITH_COORDS = True  # True — координаты, False — текст
FILE_INPUT = 'FILENAME' # Без расширения .pdf
# =================
```

### 2. Запуск

```bash
python main.py
```

Результат сохранится в `output/<Filename>Output<timestamp>.txt`,
где <Filename> - название вашего файла, а <timestamp> - Unix-время начала распознавания файла

---

## Формат вывода

**С координатами (`WITH_COORDS = True`):**
```
1|ООО|120|45|35|18|95.5
1|РОМАШКА|160|45|80|18|87.3
```
Поля: `страница | текст | left | top | width | height | confidence`

**Без координат (`WITH_COORDS = False`):**
```
ООО РОМАШКА ВЕКТОР ...
```

---

## Структура проекта

```
PDFAnalyzer/
├── main.py           # Точка входа (настройки)
├── ocr_search.py     # Функции OCR
├── requirements.txt  # Зависимости
└── output/           # Результаты
```
