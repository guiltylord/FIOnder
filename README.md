# PDFinder — OCR для PDF

Простой скрипт для распознавания текста в PDF с помощью Tesseract OCR.

---

## Установка

### 1. Tesseract OCR

Скачайте установщик с официальной страницы:
**[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)**

Рекомендуется версия **5.x** для Windows.

> **Важно:** При установке отметьте галочками **Cyrillic** и **Russian** в разделе *Additional language data* — это добавит поддержку русского языка.

### 2. Добавление Tesseract в PATH

Чтобы Python мог найти Tesseract, добавьте его в системную переменную `PATH`.

**Способ A: Через интерфейс Windows (рекомендуется)**

1. Нажмите `Win + R`, введите `sysdm.cpl` → Enter
2. Вкладка **Дополнительно** → **Переменные среды**
3. В разделе **Системные переменные** найдите `Path` → **Изменить**
4. **Создать** → вставьте `C:\Program Files\Tesseract-OCR` → OK
5. Перезапустите терминал/VS Code

**Способ B: Через PowerShell (для текущего пользователя)**

```powershell
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$userPath;C:\Program Files\Tesseract-OCR", "User")
```

**Способ C: Быстрая проверка без PATH**

Если не хотите менять переменные, добавьте в начало скрипта:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### 3. Языковые данные (traineddata)

По умолчанию устанавливается только английский язык. Если русский не был добавлен при установке, выполните в PowerShell (от администратора):

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

### 4. Python-зависимости

```bash
pip install -r requirements.txt
```

---

## Использование

### 1. Настройка в main.py

Откройте `main.py` и измените две строки:

```python
WITH_COORDS = True   # True — координаты, False — текст
save_to_txt('CROC.pdf', output, with_coords=WITH_COORDS)  # укажите ваш PDF
```

### 2. Запуск

```bash
python main.py
```

Результат сохранится в `output/CROCOutput<timestamp>.txt`

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

## Как это работает

### Что такое OCR?

**OCR (Optical Character Recognition)** — технология распознавания текста на изображениях.  
Если ваш PDF содержит сканы документов (картинки, а не текст), обычный копипаст не сработает.  
OCR «смотрит» на картинку и превращает буквы в настоящий текст.

### Какие библиотеки используются?


| Библиотека         | Зачем нужна                                           |
| ------------------ | ----------------------------------------------------- |
| **fitz (PyMuPDF)** | Открывает PDF и превращает каждую страницу в картинку |
| **PIL (Pillow)**   | Работает с изображениями (подготовка для Tesseract)   |
| **pytesseract**    | Python-обёртка для Tesseract OCR (распознаёт текст)   |


### Как работает код?

**Шаг 1: Открытие PDF**

```python
doc = fitz.open(pdf_path)  # Открываем PDF файл
```

**Шаг 2: Превращаем страницу в картинку**

```python
pix = page.get_pixmap()           # Рендерим страницу
img_data = pix.tobytes('png')     # Конвертируем в PNG
image = Image.open(...)           # Открываем как изображение
```

**Шаг 3: Распознаём текст**

```python
# Простой текст (без координат)
text = pytesseract.image_to_string(image, lang='rus+eng')

# Текст с координатами (каждое слово + позиция)
data = pytesseract.image_to_data(image, lang='rus+eng', output_type=pytesseract.Output.DICT)
```

**Шаг 4: Фильтрация мусора**

```python
if not txt or conf < 40:
    continue  # Пропускаем пустые и ненадёжные результаты
```

Tesseract иногда ошибается. Параметр `conf` (confidence) показывает уверенность от 0 до 100.  
Значения ниже 40 — скорее всего мусор.

**Шаг 5: Сохранение**

```python
# С координатами
f.write(f"{page_num}|{txt}|{left}|{top}|{width}|{height}|{conf}\n")

# Без координат
f.write(txt + ' ')
```

### Структура проекта

```
PDFAnalyzer/
├── main.py           # Точка входа (настройки)
├── ocr_search.py     # Функции OCR
├── requirements.txt  # Зависимости
└── output/           # Результаты
```

### Чем отличаются режимы?

`**WITH_COORDS = False` (простой текст):**

- Возвращает весь текст подряд
- Подходит для чтения, поиска, копирования
- Невозможно понять, где какое слово на странице

`**WITH_COORDS = True` (координаты):**

- Каждое слово + его позиция (левый верхний угол, размеры)
- Можно подсветить слово в PDF
- Можно сделать кликабельный текст поверх скана
- Формат: `страница | слово | x | y | ширина | высота | уверенность`

---

## Частые проблемы


| Проблема                                           | Решение                                                                 |
| -------------------------------------------------- | ----------------------------------------------------------------------- |
| `Tesseract is not installed`                       | Установите Tesseract и добавьте в PATH (пункт 2)                        |
| `TesseractNotFoundError`                           | Укажите путь в коде: `pytesseract.pytesseract.tesseract_cmd = ...`      |
| `Data file for language 'rus' not found`           | Скачайте `rus.traineddata` (пункт 3)                                    |
| `ModuleNotFoundError: No module named 'fitz'`      | `pip install -r requirements.txt`                                       |
| Плохое распознавание                               | Убедитесь, что PDF не размыт, попробуйте `_best` модель                 |


