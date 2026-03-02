import fitz
import os
import re
from rapidfuzz import fuzz


def normalize(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fuzzy_match(a, b):
    score1 = fuzz.partial_ratio(a, b)
    score2 = fuzz.token_set_ratio(a, b)
    return max(score1, score2)


def highlight_fio_in_pdf(input_pdf, output_pdf, fio, threshold=85):

    if not os.path.exists(input_pdf):
        print("❌ PDF файл не найден.")
        return

    doc = fitz.open(input_pdf)
    target = normalize(fio)
    found = False

    print("\n🔎 Поиск и выделение...\n")

    for page_number, page in enumerate(doc, start=1):

        text_instances = page.get_text("blocks")

        for block in text_instances:
            block_text = block[4]
            normalized_block = normalize(block_text)

            score = fuzzy_match(normalized_block, target)

            if score >= threshold:
                found = True

                # Ищем точные совпадения для выделения
                areas = page.search_for(block_text)

                for area in areas:
                    highlight = page.add_highlight_annot(area)
                    highlight.set_colors(stroke=(1, 1, 0))  # жёлтый
                    highlight.update()

                print(f"✅ Выделено на странице {page_number} (совпадение {score}%)")

    if found:
        doc.save(output_pdf)
        print(f"\n💾 Сохранён файл: {output_pdf}")
    else:
        print("❌ ФИО не найдено.")

    doc.close()


if __name__ == "__main__":

    input_pdf = r"C:\Users\User\Desktop\Python\arbitr\poiskpdf\Предварительные-результаты-МЭ-ВсОШ-по-химии-9кл-2020.pdf"
    output_pdf = r"C:\Users\User\Desktop\Python\arbitr\poiskpdf\RESULT_HIGHLIGHTED.pdf"

    fio_input = input("Введите ФИО для поиска: ").strip()

    if fio_input:
        highlight_fio_in_pdf(input_pdf, output_pdf, fio_input)
    else:
        print("Вы не ввели ФИО.")