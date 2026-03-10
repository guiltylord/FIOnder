"""
Поиск ФИО в OCR-тексте PDF.

ЕДИНАЯ ТОЧКА ВХОДА:
    search_in_text(words_with_coords, search_terms)

Аргументы:
    words_with_coords — список словарей:
        {"text": str, "page": int, "x0": float, "y0": float, "x1": float, "y1": float}
    search_terms — строка через запятую или список строк

Возвращает список словарей:
    search_term, found_text, page, x0, y0, x1, y1
"""

import re
import time


# -----------------------------------------------------------------------------
# Настройки
# -----------------------------------------------------------------------------
MAX_GAP              = 6
MAX_VERTICAL_DIST    = 80
MAX_HORIZONTAL_DIST  = 400
# Для вертикального поиска инициалов (таблицы)
MAX_VERTICAL_DIST_V  = 50    # расстояние по Y для поиска инициалов в таблицах
COL_TOLERANCE        = 35    # допуск по x для «той же колонки»


# =============================================================================
# НОРМАЛИЗАЦИЯ
# =============================================================================

try:
    from pymorphy3 import MorphAnalyzer as _MorphAnalyzer
    import threading as _threading
    _morph       = _MorphAnalyzer()
    _morph_lock  = _threading.Lock()
    _norm_cache  = {}
    _forms_cache = {}
    _USE_PYMORPHY = True
    print("[search_fio] Используется pymorphy3")
except ImportError:
    _USE_PYMORPHY = False
    print("[search_fio] Используются встроенные правила")

_RULES = [
    ("СКОГО","СКИЙ"),("СКОМУ","СКИЙ"),("СКИМ","СКИЙ"),
    ("ЦКОГО","ЦКИЙ"),("ЦКОМУ","ЦКИЙ"),("ЦКИМ","ЦКИЙ"),
    ("СКОЙ","СКИЙ"),("ЦКОЙ","ЦКИЙ"),
    ("ОВЫМ","ОВ"),("ЕВЫМ","ЕВ"),("ОВОМ","ОВ"),("ЕВОМ","ЕВ"),
    ("ОВОГО","ОВ"),("ЕВОГО","ЕВ"),("ОВОМУ","ОВ"),("ЕВОМУ","ЕВ"),
    ("ОВЫХ","ОВ"),("ЕВЫХ","ЕВ"),("ОВОЙ","ОВ"),("ЕВОЙ","ЕВ"),
    ("ОВУ","ОВ"),("ЕВУ","ЕВ"),("ОВЕ","ОВ"),("ЕВЕ","ЕВ"),
    ("ОВЫ","ОВ"),("ЕВЫ","ЕВ"),("ОВА","ОВ"),("ЕВА","ЕВ"),
    ("ИНЫМ","ИН"),("ИНОГО","ИН"),("ИНОМУ","ИН"),
    ("ИНЫХ","ИН"),("ИНОЙ","ИН"),("ИНУ","ИН"),("ИНЕ","ИН"),
    ("ИНЫ","ИН"),("ИНА","ИН"),
]


def normalize_surname(word: str) -> str:
    word = word.strip()
    if not word:
        return ""
    key = word.upper()

    if _USE_PYMORPHY:
        if key not in _norm_cache:
            with _morph_lock:
                parsed = _morph.parse(word.lower())
            for p in parsed:
                tag = str(p.tag)
                if 'Surn' in tag and 'masc' in tag:
                    _norm_cache[key] = p.normal_form.upper()
                    break
            else:
                for p in parsed:
                    if 'Surn' in str(p.tag):
                        _norm_cache[key] = p.normal_form.upper()
                        break
                else:
                    _norm_cache[key] = parsed[0].normal_form.upper()
        return _norm_cache[key]

    w = key
    for suffix, replacement in _RULES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)] + replacement
    return w


def _all_surname_forms(word: str) -> frozenset:
    word = word.strip()
    if not word:
        return frozenset()
    key = word.upper()
    if not _USE_PYMORPHY:
        return frozenset([normalize_surname(word)])
    if key not in _forms_cache:
        with _morph_lock:
            parsed = _morph.parse(word.lower())
        _forms_cache[key] = frozenset(p.normal_form.upper() for p in parsed)
    return _forms_cache[key]


def _get_all_word_forms(word: str) -> frozenset:
    """
    Возвращает все возможные формы слова (для поиска).
    Например: "Дезмал" → {"ДЕЗМАЛ", "ДЕЗМАЛА", "ДЕЗМАЛУ", ...}
    """
    word = word.strip()
    if not word:
        return frozenset()
    key = word.upper()
    
    if not _USE_PYMORPHY:
        # Без pymorphy: генерируем все возможные падежные формы
        # Для мужского рода (Дезмал): Дезмал, Дезмала, Дезмалу, Дезмалом, Дезмале
        # Для женского рода (Иванова): Иванова, Ивановой, Иванову, Ивановой, Иванову
        forms = {key}
        
        # Если слово оканчивается на согласную (мужской род)
        if key[-1] not in "АЕИОУЫЭЮЯ":
            for ending in ["А", "У", "ОМ", "Е"]:
                forms.add(key + ending)
            # Родительный/Винительный: -а
            if not key.endswith("А"):
                forms.add(key + "А")
        
        # Если слово оканчивается на -а/-я (женский род или мужской типа "судья")
        elif key.endswith(("А", "Я")):
            base = key[:-1]
            for ending in ["Ы", "Е", "У", "ОЙ", "ОЮ", "Е"]:
                forms.add(base + ending)
            # Родительный: -ы/-и
            forms.add(base + "Ы")
        
        return frozenset(forms)
    
    # С pymorphy: получаем все формы
    cache_key = "FORMS:" + key
    if cache_key not in _forms_cache:
        with _morph_lock:
            parsed = _morph.parse(word.lower())
        # Собираем все формы для всех разборов
        all_forms = set()
        for p in parsed:
            # Добавляем все формы этого разбора
            for form in p.lexeme:
                all_forms.add(form.word.upper())
        _forms_cache[cache_key] = frozenset(all_forms)
    
    return _forms_cache[cache_key]


# =============================================================================
# КЛАССИФИКАЦИЯ ТОКЕНОВ
# =============================================================================

def _is_initial(text: str) -> bool:
    return bool(re.fullmatch(r"[А-ЯЁ]\.?", text.upper()))

def _is_double_initial(text: str) -> bool:
    return bool(re.fullmatch(r"[А-ЯЁ]\.?[А-ЯЁ]\.?", text.upper()))

def _split_double_initial(text: str) -> list:
    return re.findall(r"[А-ЯЁ]", text.upper())

def _is_word(text: str) -> bool:
    return bool(re.fullmatch(r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?", text))


# =============================================================================
# ПАРСИНГ ЗАПРОСА
# =============================================================================

def parse_query(query: str) -> dict:
    query = re.sub(r"([А-ЯЁа-яё]\.)([А-ЯЁа-яё]\.?)", r"\1 \2", query)
    parts = re.findall(r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?|[А-ЯЁ]\.", query)

    words    = []
    initials = []

    for p in parts:
        if _is_initial(p):
            initials.append(p[0].upper())
        else:
            words.append(p.upper())

    surname   = words[0] if words else None
    name_full = words[1] if len(words) >= 2 else None
    patr_full = words[2] if len(words) >= 3 else None

    name_initial = (
        name_full[0] if name_full
        else (initials[0] if initials else None)
    )
    patr_initial = (
        patr_full[0] if patr_full
        else (initials[1] if len(initials) >= 2 else None)
    )

    return {
        "surname":            surname,
        "surname_norm":       normalize_surname(surname) if surname else None,
        "name_initial":       name_initial,
        "name_full":          name_full,
        "patronymic_initial": patr_initial,
        "patronymic_full":    patr_full,
    }


# =============================================================================
# ПОДГОТОВКА ТОКЕНОВ
# =============================================================================

def _strip_numbering(text: str) -> tuple:
    m = re.match(r"^\d+[.)\s]+", text)
    if m:
        return text[m.end():], m.end()
    return text, 0


def _strip_punctuation(text: str) -> str:
    """
    Удаляет пунктуацию в начале и конце слова.
    "Дезмала)" → "Дезмала"
    "(слово" → "слово"
    """
    return re.sub(r"^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$", "", text)


def prepare_tokens(words_with_coords: list) -> list:
    raw_tokens = []

    for i, w in enumerate(words_with_coords):
        original = w["text"].strip()
        if not original:
            continue

        cleaned, prefix_len = _strip_numbering(original)
        if not cleaned:
            continue

        # Сохраняем информацию о дефисе в конце ДО удаления пунктуации
        # Это нужно для склейки переносов (Ива- + нов → Иванов)
        ends_with_dash = cleaned.endswith("-")

        # Удаляем пунктуацию в начале и конце
        cleaned = _strip_punctuation(cleaned)
        if not cleaned:
            continue

        # Восстанавливаем дефис если он был (для склейки переносов)
        # Но только если после удаления пунктуации осталась буква
        if ends_with_dash and cleaned and not cleaned.endswith("-"):
            cleaned = cleaned + "-"

        if prefix_len and len(original) > 0:
            char_width = (w["x1"] - w["x0"]) / len(original)
            new_x0 = w["x0"] + char_width * prefix_len
        else:
            new_x0 = w["x0"]

        raw_tokens.append({
            "text": cleaned,
            "page": w["page"],
            "x0":   new_x0,
            "y0":   w["y0"],
            "x1":   w["x1"],
            "y1":   w["y1"],
            "idx":  i,
        })

    # Склейка переносов: «Ива-» + «нов» → «Иванов»
    # Работает как для горизонтальных, так и для вертикальных переносов
    merged    = []
    skip_next = False

    for k, rw in enumerate(raw_tokens):
        if skip_next:
            skip_next = False
            continue
        text = rw["text"]
        if not text.endswith("-"):
            merged.append(rw)
            continue

        # Ищем продолжение переноса среди следующих токенов
        found_continuation = False
        for j in range(k + 1, min(k + 5, len(raw_tokens))):
            nxt = raw_tokens[j]
            if not _is_word(nxt["text"]):
                continue

            # Проверка: первая часть без дефиса должна быть словом
            if not _is_word(text[:-1]):
                break

            # Горизонтальный перенос (соседний токен)
            is_horizontal = (j == k + 1)

            # Вертикальный перенос (следующая строка)
            y_diff = nxt["y0"] - rw["y0"]
            line_height = rw["y1"] - rw["y0"]
            is_vertical = (
                y_diff > 0 and
                y_diff < line_height * 3 and  # Не слишком далеко по вертикали
                abs(nxt["x0"] - rw["x0"]) < 100  # Примерно та же колонка
            )

            if is_horizontal or is_vertical:
                # Для склейки создаём правильный bounding box
                # который охватывает обе части слова
                merged.append({
                    **rw,
                    "text": text[:-1] + nxt["text"],
                    "x1": max(rw["x1"], nxt["x1"]),  # Правая граница
                    "y1": max(rw["y1"], nxt["y1"]),  # Нижняя граница (для вертикальных переносов)
                })
                skip_next = (j == k + 1)
                # Пропускаем все токены между k и j
                for _ in range(j - k - 1):
                    merged.append({"text": "", "page": 0, "x0": 0, "y0": 0, "x1": 0, "y1": 0, "idx": 0})
                found_continuation = True
                break

            # Если не нашли продолжение в разумных пределах — прерываем
            if y_diff > line_height * 3:
                break

        if not found_continuation:
            merged.append(rw)

    # Классификация
    tokens = []

    for rw in merged:
        text = rw["text"]
        # Пропускаем пустые токены (заполнители для вертикальных переносов)
        if not text:
            continue
        base = {k: rw[k] for k in ("page", "x0", "y0", "x1", "y1", "idx")}

        if _is_double_initial(text):
            for letter in _split_double_initial(text):
                tokens.append({**base, "type": "initial", "text": letter, "raw": letter})
            continue

        if _is_initial(text):
            ttype = "initial"
        elif _is_word(text):
            ttype = "word"
        else:
            ttype = "junk"

        tokens.append({**base, "type": ttype, "text": text.upper(), "raw": text})

    return tokens


# =============================================================================
# СОПОСТАВЛЕНИЕ
# =============================================================================

def _surname_matches(token: dict, query: dict) -> bool:
    if not query["surname_norm"]:
        return False
    if normalize_surname(token["text"]) == query["surname_norm"]:
        return True
    token_forms = _all_surname_forms(token["text"])
    query_forms = _all_surname_forms(query["surname"])
    return bool(token_forms & query_forms)


def _name_matches(token: dict, initial, full) -> bool:
    if initial is None:
        return False
    if token["text"][0].upper() != initial.upper():
        return False
    if token["type"] == "initial":
        return True
    if full is not None:
        return normalize_surname(token["text"]) == normalize_surname(full)
    return token["type"] == "word"


# =============================================================================
# ПРОСТРАНСТВЕННАЯ БЛИЗОСТЬ
# =============================================================================

def _tokens_are_close(anchor: dict, candidate: dict) -> bool:
    """Горизонтальная близость — токены на одной строке."""
    if abs(anchor["y0"] - candidate["y0"]) > MAX_VERTICAL_DIST:
        return False
    gap = max(anchor["x0"], candidate["x0"]) - min(anchor["x1"], candidate["x1"])
    return gap <= MAX_HORIZONTAL_DIST


def _same_column(anchor: dict, candidate: dict) -> bool:
    """
    Вертикальная близость — токен в той же колонке таблицы.
    Центры по X совпадают с допуском COL_TOLERANCE.
    """
    anchor_cx    = (anchor["x0"]    + anchor["x1"])    / 2
    candidate_cx = (candidate["x0"] + candidate["x1"]) / 2
    return abs(anchor_cx - candidate_cx) <= COL_TOLERANCE


# =============================================================================
# ПОИСК ИНИЦИАЛОВ В ОКНЕ (горизонтальный)
# =============================================================================

def _find_initials_in_window(tokens, start, direction, anchor, query):
    """
    Ищет инициалы имени и отчества начиная с позиции start,
    двигаясь в сторону direction (+1 вперёд, -1 назад).
    
    Важно: ищем только в пределах одной строки (по Y).
    Токены в другой строке (даже в соседней) не считаются.
    """
    n        = len(tokens)
    name_tok = None
    patr_tok = None
    matched  = []
    gap      = 0
    j        = start

    # Определяем "границы строки" якоря
    # Считаем, что токены на одной строке, если их Y пересекаются или
    # разница меньше половины высоты строки якоря
    anchor_line_height = anchor["y1"] - anchor["y0"]
    y_tolerance = anchor_line_height * 0.8  # Допуск для "той же строки"

    while 0 <= j < n and gap <= MAX_GAP:
        t = tokens[j]

        page_diff = t["page"] - anchor["page"]
        if abs(page_diff) > 1:
            break

        # Для горизонтального поиска на той же странице требуем,
        # чтобы токены были примерно на одной строке
        if page_diff == 0:
            # Проверяем, что Y токена находится в пределах строки якоря
            y_diff = abs(t["y0"] - anchor["y0"])
            if y_diff > y_tolerance:
                # Токен на другой строке — прекращаем горизонтальный поиск
                break
            
            # Также проверяем горизонтальную близость (чтобы не искать
            # инициалы в другой колонке, если они далеко по X)
            if not _tokens_are_close(anchor, t):
                break

        if t["type"] in ("word", "initial"):
            if name_tok is None and _name_matches(
                t, query["name_initial"], query["name_full"]
            ):
                name_tok = t
                matched.append(t)
            elif patr_tok is None and _name_matches(
                t, query["patronymic_initial"], query["patronymic_full"]
            ):
                patr_tok = t
                matched.append(t)
            else:
                break
        else:
            gap += 1

        j += direction

    return name_tok, patr_tok, matched


# =============================================================================
# ПОИСК ИНИЦИАЛОВ ПО ВЕРТИКАЛИ (для таблиц)
# =============================================================================

def _find_initials_vertically(tokens: list, surname_tok: dict, query: dict):
    """
    Ищет инициалы имени и отчества строго под (или над) фамилией —
    в той же колонке таблицы.

    Возвращает (name_tok, patr_tok, matched_list).
    """
    page        = surname_tok["page"]
    sy0         = surname_tok["y0"]
    sy1         = surname_tok["y1"]

    needs_name  = query["name_initial"] is not None
    needs_patr  = query["patronymic_initial"] is not None

    # Собираем кандидатов: та же страница, та же колонка, инициалы ИЛИ слова
    # Для полных имён (не инициалов) тоже ищем по вертикали
    below = [
        t for t in tokens
        if t["page"] == page
        and t["type"] in ("initial", "word")           # ← ищем и инициалы, и слова
        and t["y0"] > sy1                          # ниже фамилии
        and t["y0"] < sy0 + MAX_VERTICAL_DIST_V    # не слишком далеко
        and _same_column(surname_tok, t)
    ]
    above = [
        t for t in tokens
        if t["page"] == page
        and t["type"] in ("initial", "word")           # ← ищем и инициалы, и слова
        and t["y1"] < sy0                          # выше фамилии
        and t["y1"] > sy0 - MAX_VERTICAL_DIST_V
        and _same_column(surname_tok, t)
    ]

    below.sort(key=lambda t: t["y0"])
    above.sort(key=lambda t: t["y0"], reverse=True)

    name_tok = None
    patr_tok = None
    matched  = []

    for group in (below, above):
        for t in group:
            if name_tok is None and needs_name and _name_matches(
                t, query["name_initial"], query["name_full"]
            ):
                name_tok = t
                matched.append(t)
            elif patr_tok is None and needs_patr and _name_matches(
                t, query["patronymic_initial"], query["patronymic_full"]
            ):
                patr_tok = t
                matched.append(t)

        # Если нашли хотя бы одно совпадение в этой группе — не ищем в другой
        if matched:
            break

    return name_tok, patr_tok, matched


# =============================================================================
# ОСНОВНОЙ ПОИСК
# =============================================================================

def _search_by_surname_only(tokens: list, query: dict) -> list:
    """
    Поиск только по фамилии/слову.
    Работает через нормализацию + точное совпадение + все формы слова.
    """
    results = []
    seen    = set()

    # Получаем все формы поискового запроса
    search_forms = _get_all_word_forms(query["surname"])

    for tok in tokens:
        if tok["type"] != "word":
            continue
        
        # Проверка 1: токен входит в набор форм поиска
        token_in_forms = tok["text"].upper() in search_forms
        
        # Проверка 2: обратная проверка (форма токена входит в формы запроса)
        token_forms = _get_all_word_forms(tok["text"])
        query_in_token_forms = query["surname"].upper() in token_forms
        
        if not (token_in_forms or query_in_token_forms):
            continue

        key = (tok["page"], round(tok["x0"], 1), round(tok["y0"], 1))
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "search_term": query.get("_raw", ""),
            "found_text":  tok["raw"],
            "page":        tok["page"],
            "x0":          tok["x0"],
            "y0":          tok["y0"],
            "x1":          tok["x1"],
            "y1":          tok["y1"],
        })

    return results


def _search_by_initials_only(tokens: list, query: dict) -> list:
    results    = []
    seen       = set()
    needs_name = query["name_initial"] is not None
    needs_patr = query["patronymic_initial"] is not None
    n          = len(tokens)

    for i, tok in enumerate(tokens):
        if tok["type"] != "initial":
            continue

        name_tok = None
        patr_tok = None
        matched  = []

        if needs_name and tok["text"][0] == query["name_initial"]:
            name_tok = tok
            matched.append(tok)

            if needs_patr:
                for j in range(i + 1, min(i + 1 + MAX_GAP, n)):
                    t = tokens[j]
                    if t["type"] == "junk":
                        continue
                    if t["type"] != "initial":
                        break
                    if not _tokens_are_close(tok, t):
                        break
                    if t["text"][0] == query["patronymic_initial"]:
                        patr_tok = t
                        matched.append(t)
                        break

        elif needs_patr and not needs_name and tok["text"][0] == query["patronymic_initial"]:
            patr_tok = tok
            matched.append(tok)

        name_ok = (not needs_name) or (name_tok is not None)
        patr_ok = (not needs_patr) or (patr_tok is not None)

        if not (name_ok and patr_ok):
            continue

        for m in matched:
            key = (m["page"], round(m["x0"], 1), round(m["y0"], 1))
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "search_term": query.get("_raw", ""),
                "found_text":  m["raw"],
                "page":        m["page"],
                "x0":          m["x0"],
                "y0":          m["y0"],
                "x1":          m["x1"],
                "y1":          m["y1"],
            })

    return results


def _search_fio(tokens: list, query: dict) -> list:
    """
    Режим 3: фамилия + инициалы.

    Стратегия:
      1. Горизонтальный поиск инициалов (вперёд и назад по токенам).
      2. Если не нашли — вертикальный поиск в той же колонке таблицы.
    """
    results    = []
    seen       = set()
    needs_name = query["name_initial"] is not None
    needs_patr = query["patronymic_initial"] is not None

    for i, tok in enumerate(tokens):
        if tok["type"] != "word":
            continue
        if not _surname_matches(tok, query):
            continue

        # --- Шаг 1: горизонтальный поиск ---
        name_f, patr_f, match_f = _find_initials_in_window(
            tokens, i + 1, +1, tok, query
        )
        name_b, patr_b, match_b = _find_initials_in_window(
            tokens, i - 1, -1, tok, query
        )

        name_tok = name_f or name_b
        patr_tok = patr_f or patr_b
        matched  = [tok] + match_f + match_b

        # --- Шаг 2: вертикальный поиск (fallback для таблиц) ---
        still_needs_name = needs_name and name_tok is None
        still_needs_patr = needs_patr and patr_tok is None

        if still_needs_name or still_needs_patr:
            vname, vpatr, vmatch = _find_initials_vertically(tokens, tok, query)

            if still_needs_name and vname is not None:
                name_tok = vname
                matched.extend(vmatch)
            if still_needs_patr and vpatr is not None:
                patr_tok = vpatr
                # избегаем дублей, если vmatch уже добавлен
                for m in vmatch:
                    if m not in matched:
                        matched.append(m)

        # --- Проверяем, нашли ли всё необходимое ---
        if needs_name and name_tok is None:
            continue
        if needs_patr and patr_tok is None:
            continue

        for m in matched:
            key = (m["page"], round(m["x0"], 1), round(m["y0"], 1))
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "search_term": query.get("_raw", ""),
                "found_text":  m["raw"],
                "page":        m["page"],
                "x0":          m["x0"],
                "y0":          m["y0"],
                "x1":          m["x1"],
                "y1":          m["y1"],
            })

    return results


# =============================================================================
# ЕДИНАЯ ТОЧКА ВХОДА
# =============================================================================

def search_in_text(words_with_coords: list, search_terms) -> list:
    """
    Ищет ФИО в OCR-тексте PDF.

    Аргументы:
        words_with_coords — список словарей:
            {"text": str, "page": int,
             "x0": float, "y0": float, "x1": float, "y1": float}
        search_terms — строка через запятую или список строк

    Возвращает список словарей:
        search_term, found_text, page, x0, y0, x1, y1
    """
    start = time.time()

    if isinstance(search_terms, str):
        terms = [t.strip() for t in search_terms.split(",") if t.strip()]
    else:
        terms = [t.strip() for t in search_terms if t.strip()]

    tokens = prepare_tokens(words_with_coords)
    found  = []

    for term in terms:
        query         = parse_query(term)
        query["_raw"] = term

        has_surname  = query["surname"] is not None
        has_initials = (
            query["name_initial"] is not None or
            query["patronymic_initial"] is not None
        )

        if has_surname and has_initials:
            found.extend(_search_fio(tokens, query))
        elif has_surname:
            found.extend(_search_by_surname_only(tokens, query))
        elif has_initials:
            found.extend(_search_by_initials_only(tokens, query))

    print(
        f"[TIME] Search: {time.time() - start:.2f}s | "
        f"запросов: {len(terms)} | найдено: {len(found)}"
    )

    return found


