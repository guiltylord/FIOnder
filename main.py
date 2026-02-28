"""OCR для PDF. Простая фильтрация мусора."""

import fitz, pytesseract
from PIL import Image, ImageEnhance
import io, re, time, os

SCALE = 2.0
CONTRAST = 1.6
VOWELS = set('аеёиоуыэюяaeiouyАЕЁИОУЫЭЮЯ')
SHORT = {'и','в','на','по','с','к','у','о','а','но','же','бы','ли','что','за','под','над','при','без','для','от','до','из','об','во','я','ты','он','она','оно','мы','вы','они','её','его','мне','тебе','is','a','the','to','of','and','in','for','on','with','at','by','from','as','or','an','be','are','was','were','has','have','had'}

# def is_valid(w, c):
#     """Проверка слова: повторы символов, гласные, уверенность."""
#     if '&' in w and len(w) > 3: return True
#     if len(w) <= 2: return w.lower() in SHORT
#     if re.search(r'(.)\1{2,}', w): return False  # 3+ одинаковых подряд — мусор
#     letters = [x for x in w if x.isalpha()]
#     if letters and len(w) >= 3:
#         r = sum(1 for x in letters if x in VOWELS) / len(letters)
#         if not (0.25 <= r <= 0.75): return False  # Гласных 25-75%
#     return len(w) > 4 or c >= 40  # Короткие слова — только с высокой уверенностью

def ocr(pdf):
    """Распознавание PDF + фильтрация."""
    t0 = time.time()
    doc = fitz.open(pdf)
    pages = len(doc)
    text, words, confs = [], [], []
    
    for pn, page in enumerate(doc):
        # Рендер + предобработка
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        img = Image.open(io.BytesIO(pix.tobytes('png')))
        img = img.convert('L')
        w, h = img.size
        img = img.resize((int(w*SCALE), int(h*SCALE)), Image.Resampling.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(CONTRAST)
        
        # OCR
        text.append(pytesseract.image_to_string(img, lang='rus+eng', config='--psm 3 --oem 3'))
        data = pytesseract.image_to_data(img, lang='rus+eng', config='--psm 3 --oem 3', output_type=pytesseract.Output.DICT)
        
        for i in range(len(data['text'])):
            t = data['text'][i].strip()
            c = float(data['conf'][i]) if data['conf'][i] else 0
            if t:
                words.append({'text': t, 'conf': c, 'page': pn+1})
                if c >= 30: confs.append(c)
    
    doc.close()
    
    # Мапа: слово -> уверенность
    cm = {}
    for w in words:
        t = re.sub(r'^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$', '', w['text'])
        if t: cm.setdefault(t, []).append(w['conf'])
    
    # Фильтрация строк и слов
    res = []
    for line in '\n'.join(text).split('\n'):
        line = line.strip()
        if not line or not re.search(r'[А-Яа-яA-Za-z]', line): 
            continue
        if len(re.findall(r'[^\w\sА-Яа-яA-Za-z\"\'&;,\(\)\.\-]', line)) / max(len(line),1) > 0.5: 
            continue
        for w in line.split():
            c = re.sub(r'^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$', '', w)
            if not c: continue
            cf = sum(cm.get(c, cm.get(c.lower(), [50]))) / len(cm.get(c, cm.get(c.lower(), [50])))
            if is_valid(c, cf): 
                res.append(c)
    
    # Удаление мусора в конце (2+ коротких слова подряд)
    end, sc = len(res), 0
    for i in range(len(res)-1, -1, -1):
        w = res[i].replace('.', '')
        cf = sum(cm.get(res[i], cm.get(res[i].lower(), [50]))) / len(cm.get(res[i], cm.get(res[i].lower(), [50])))
        if len(w) <= 2:
            sc += 1
            if sc >= 2: 
                end = i
                break
        elif len(w) < 5 and cf < 40: 
            end = i
            break
        elif sc > 0: 
            end = i + 1
            break
    
    txt = ' '.join(res[:end])
    return {'text': txt, 'pages': pages, 'conf': sum(confs)/len(confs) if confs else 0, 'time': time.time()-t0, 'words': len(txt.split())}

def main():
    t0 = time.time()
    ts = int(t0)
    filename = 'CROC'
    pdf = filename + '.pdf'

    if not os.path.exists(pdf):
        sys.exit(f"Файл '{pdf}' не найден!")

    os.makedirs('output', exist_ok=True)
    out = f'output/{filename}_output_{ts}.txt'
    
    print(f"Обработка: {pdf}...\n{'='*60}")
    
    r = ocr(pdf)
    
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f"Файл: {pdf}\nСтраниц: {r['pages']}\nУверенность: {r['conf']:.1f}%\nВремя: {r['time']:.2f} сек\nСлов: {r['words']}\n\n{'='*60}\n\n{r['text']}")
    
    print(f"\nРЕЗУЛЬТАТЫ:\n  Страниц: {r['pages']}\n  Уверенность: {r['conf']:.1f}%\n  Слов: {r['words']}\n  Время: {r['time']:.2f} сек.\n  Файл: {out}")
    print(f"\nОбщее время: {time.time()-t0:.2f} сек.\n\nТЕКСТ:\n{'-'*60}")
    for l in r['text'].split('\n')[:5]: 
        print(l[:100])
    if len(r['text']) > 500: 
        print("...")

if __name__ == '__main__':
    main()
