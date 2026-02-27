import fitz
import pytesseract
from PIL import Image, ImageEnhance
import io

SCALE = 2.0
CONTRAST = 1.5

doc = fitz.open('CROC.pdf')
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
img_data = pix.tobytes('png')
image = Image.open(io.BytesIO(img_data))

w, h = image.size
img = image.resize((int(w * SCALE), int(h * SCALE)), Image.Resampling.LANCZOS)
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(CONTRAST)
img = img.convert('L')

# Сырой текст
text = pytesseract.image_to_string(img, lang='rus+eng', config='--psm 3')
print('=== СЫРОЙ ТЕКСТ ===')
print(text)
print()

# Детали по словам
data = pytesseract.image_to_data(img, lang='rus+eng', config='--psm 3', output_type=pytesseract.Output.DICT)
print('=== СЛОВА С УВЕРЕННОСТЬЮ > 30 ===')
for i in range(len(data['text'])):
    txt = data['text'][i].strip()
    conf = float(data['conf'][i])
    if txt and conf > 30:
        print(f'{conf:5.1f} | {txt}')

doc.close()
