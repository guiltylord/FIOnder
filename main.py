"""
OCR для PDF файлов.
"""

from ocr import OCR
import time
import os
import sys

FILE_INPUT = 'CROC'
OUTPUT_DIR = 'output'


def find_pdf(name: str) -> str:
    """Поиск PDF файла."""
    path = name if name.endswith('.pdf') else f'{name}.pdf'
    if os.path.exists(path):
        return path
    for root, _, files in os.walk('.'):
        for f in files:
            if f.lower() == path.lower():
                return os.path.join(root, f)
    pdfs = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    if pdfs:
        print(f"Используем: {pdfs[0]}")
        return pdfs[0]
    raise SystemExit(f"PDF '{path}' не найден!")


def main():
    start = time.time()
    ts = int(start)
    
    file_input = sys.argv[1] if len(sys.argv) > 1 else FILE_INPUT
    pdf_path = find_pdf(file_input)
    file_input = os.path.splitext(os.path.basename(pdf_path))[0]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = f'{OUTPUT_DIR}/{file_input}_output_{ts}.txt'
    
    print(f"Обработка: {pdf_path}...")
    print("=" * 60)
    
    result = OCR().process(pdf_path)
    
    with open(output, 'w', encoding='utf-8') as f:
        f.write(f"Файл: {pdf_path}\n")
        f.write(f"Страниц: {result['pages']}\n")
        f.write(f"Уверенность: {result['confidence']:.1f}%\n")
        f.write(f"Время: {result['time']:.2f} сек\n")
        f.write(f"Слов: {result['words']}\n\n")
        f.write("=" * 60 + "\n\n")
        f.write(result['text'])
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    print(f"  Страниц: {result['pages']}")
    print(f"  Уверенность: {result['confidence']:.1f}%")
    print(f"  Слов: {result['words']}")
    print(f"  Время: {result['time']:.2f} сек.")
    print(f"  Файл: {output}")
    print(f"\nОбщее время: {time.time() - start:.2f} сек.")
    
    print("\n" + "-" * 60)
    print("ТЕКСТ:")
    print("-" * 60)
    for line in result['text'].split('\n')[:5]:
        print(line[:100])
    if len(result['text']) > 500:
        print("...")


if __name__ == '__main__':
    main()
