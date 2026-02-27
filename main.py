"""
Универсальный OCR для PDF файлов.
"""

from universal_ocr import UniversalOCR
import time
import os
import sys

FILE_INPUT = 'CROC'
OUTPUT_DIR = 'output'


def find_pdf_file(file_input: str) -> str:
    """Поиск PDF файла."""
    pdf_path = file_input if file_input.endswith('.pdf') else f'{file_input}.pdf'

    if os.path.exists(pdf_path):
        return pdf_path

    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.lower() == pdf_path.lower():
                return os.path.join(root, f)

    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    if pdf_files:
        print(f"Используем: {pdf_files[0]}")
        return pdf_files[0]

    raise SystemExit(f"PDF файл '{pdf_path}' не найден!")


def main():
    start = time.time()
    ts = int(start)

    file_input = sys.argv[1] if len(sys.argv) > 1 else FILE_INPUT
    pdf_path = find_pdf_file(file_input)
    file_input = os.path.splitext(os.path.basename(pdf_path))[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = f'{OUTPUT_DIR}/{file_input}_output_{ts}.txt'

    print(f"Обработка: {pdf_path}...")
    print("=" * 60)

    ocr = UniversalOCR()
    result = ocr.process(pdf_path)

    with open(output, 'w', encoding='utf-8') as f:
        f.write(f"Файл: {pdf_path}\n")
        f.write(f"Страниц: {result.pages}\n")
        f.write(f"Уверенность: {result.confidence:.1f}%\n")
        f.write(f"Время: {result.time_elapsed:.2f} сек\n")
        f.write(f"Слов: {len(result.cleaned_text.split())}\n")
        f.write("\n" + "=" * 60 + "\n\n")
        f.write(result.cleaned_text)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    print(f'  Страниц: {result.pages}')
    print(f'  Уверенность: {result.confidence:.1f}%')
    print(f'  Слов: {len(result.cleaned_text.split())}')
    print(f'  Время: {result.time_elapsed:.2f} сек.')
    print(f'  Файл: {output}')
    print(f'\nОбщее время: {time.time()-start:.2f} сек.')

    print("\n" + "-" * 60)
    print("ТЕКСТ:")
    print("-" * 60)
    lines = result.cleaned_text.split('\n')
    for line in lines[:5]:
        print(line[:100])
    if len(result.cleaned_text) > 500:
        print("...")


if __name__ == '__main__':
    main()
