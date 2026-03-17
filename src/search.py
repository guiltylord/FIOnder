"""
Поиск ФИО в OCR-тексте PDF.

ЕДИНАЯ ТОЧКА ВХОДА:
    search_in_text(words_with_coords, search_terms)

ИСПРАВЛЕНИЯ:
    1. _is_double_initial — требует хотя бы одну точку (иначе 'до','по','из'
       разбивались на фиктивные инициалы).
    2. Дедупликация в _search_fio/_search_by_surname_only/_search_by_initials_only —
       добавлен text в ключ, чтобы два инициала с одинаковыми координатами
       (слитый токен 'Ф.Э.') не выбрасывали друг друга.
    3. _find_initials_vertically — anchor_tok исключён из кандидатов,
       чтобы найденный инициал имени не блокировал поиск инициала отчества.
    4. _fix_ocr — исправление типичных OCR-ошибок до классификации токена:
       '%.' → 'Э.', латинские буквы → кириллические аналоги и т.п.
"""

import re
import time

# -----------------------------------------------------------------------------
# Настройки
# -----------------------------------------------------------------------------
MAX_GAP = 6
MAX_VERTICAL_DIST = 150
MAX_HORIZONTAL_DIST = 400
COL_TOLERANCE = 35

# -----------------------------------------------------------------------------
# Таблица исправления OCR-ошибок
#
# OCR часто путает визуально похожие символы. Применяется в prepare_tokens
# к каждому токену до его классификации.
# -----------------------------------------------------------------------------

# Замена всего токена целиком (наивысший приоритет)
_OCR_TOKEN_FIXES = {
    '%.' : 'Э.',   # % ↔ Э  — самая частая ошибка при низком DPI
    '%-' : 'Э-',
    '3.' : 'З.',   # цифра 3 вместо буквы З
    '0.' : 'О.',   # цифра 0 вместо буквы О
    'l.' : 'І.',   # латинская l вместо И/І
}

# Посимвольная замена (применяется если токен содержит символы из таблицы)
_OCR_CHAR_FIXES = {
    # Латиница → Кириллица (визуально идентичны)
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
    'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
    'X': 'Х', 'Y': 'У', 'a': 'а', 'c': 'с', 'e': 'е',
    'o': 'о', 'p': 'р', 'x': 'х', 'y': 'у',
    # Спецсимволы → Кириллица
    '%': 'Э',
    '§': 'Б',
    '&': 'Ъ',
}


def _fix_ocr(text: str) -> str:
    """
    Исправляет типичные OCR-ошибки в тексте токена.

    1. Точное совпадение всего токена (_OCR_TOKEN_FIXES):
       '%.' → 'Э.' — OCR полностью заменил инициал спецсимволом.
    2. Посимвольная замена (_OCR_CHAR_FIXES):
       только если токен содержит символы из таблицы (не трогаем
       нормально распознанные кириллические слова).
    """
    fixed = _OCR_TOKEN_FIXES.get(text)
    if fixed is not None:
        return fixed

    if not any(ch in _OCR_CHAR_FIXES for ch in text):
        return text

    return "".join(_OCR_CHAR_FIXES.get(ch, ch) for ch in text)


# =============================================================================
# НОРМАЛИЗАЦИЯ
# =============================================================================

try:
    import threading as _threading
    from pymorphy3 import MorphAnalyzer as _MorphAnalyzer

    _morph = _MorphAnalyzer()
    _morph_lock = _threading.Lock()
    _norm_cache = {}
    _forms_cache = {}
    _USE_PYMORPHY = True
    print("[search_fio] Используется pymorphy3")
except ImportError:
    _USE_PYMORPHY = False
    print("[search_fio] Используются встроенные правила")

_RULES = [
    ("СКОГО", "СКИЙ"), ("СКОМУ", "СКИЙ"), ("СКИМ", "СКИЙ"), ("ЦКОГО", "ЦКИЙ"),
    ("ЦКОМУ", "ЦКИЙ"), ("ЦКИМ", "ЦКИЙ"), ("СКОЙ", "СКИЙ"), ("ЦКОЙ", "ЦКИЙ"),
    ("ОВЫМ", "ОВ"), ("ЕВЫМ", "ЕВ"), ("ОВОМ", "ОВ"), ("ЕВОМ", "ЕВ"),
    ("ОВОГО", "ОВ"), ("ЕВОГО", "ЕВ"), ("ОВОМУ", "ОВ"), ("ЕВОМУ", "ЕВ"),
    ("ОВЫХ", "ОВ"), ("ЕВЫХ", "ЕВ"), ("ОВОЙ", "ОВ"), ("ЕВОЙ", "ЕВ"),
    ("ОВУ", "ОВ"), ("ЕВУ", "ЕВ"), ("ОВЕ", "ОВ"), ("ЕВЕ", "ЕВ"),
    ("ОВЫ", "ОВ"), ("ЕВЫ", "ЕВ"), ("ОВА", "ОВ"), ("ЕВА", "ЕВ"),
    ("ИНЫМ", "ИН"), ("ИНОГО", "ИН"), ("ИНОМУ", "ИН"), ("ИНЫХ", "ИН"),
    ("ИНОЙ", "ИН"), ("ИНУ", "ИН"), ("ИНЕ", "ИН"), ("ИНЫ", "ИН"), ("ИНА", "ИН"),
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
                if "Surn" in tag and "masc" in tag:
                    _norm_cache[key] = p.normal_form.upper()
                    break
            else:
                for p in parsed:
                    if "Surn" in str(p.tag):
                        _norm_cache[key] = p.normal_form.upper()
                        break
                else:
                    _norm_cache[key] = parsed[0].normal_form.upper()
        return _norm_cache[key]

    w = key
    for suffix, replacement in _RULES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)] + replacement
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
    word = word.strip()
    if not word:
        return frozenset()
    key = word.upper()

    if not _USE_PYMORPHY:
        forms = {key}
        if key[-1] not in "АЕИОУЫЭЮЯ":
            for ending in ["А", "У", "ОМ", "Е"]:
                forms.add(key + ending)
            if not key.endswith("А"):
                forms.add(key + "А")
        elif key.endswith(("А", "Я")):
            base = key[:-1]
            for ending in ["Ы", "Е", "У", "ОЙ", "ОЮ", "Е"]:
                forms.add(base + ending)
            forms.add(base + "Ы")
        return frozenset(forms)

    cache_key = "FORMS:" + key
    if cache_key not in _forms_cache:
        with _morph_lock:
            parsed = _morph.parse(word.lower())
        all_forms = set()
        for p in parsed:
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
    # ИСПРАВЛЕНИЕ 1: точка обязательна хотя бы при одном инициале.
    # Иначе двухбуквенные слова 'до', 'по', 'из', 'на' и т.п.
    # ошибочно распознаются как двойные инициалы.
    return bool(re.fullmatch(r"[А-ЯЁ]\.([А-ЯЁ]\.?)|[А-ЯЁ]\.?[А-ЯЁ]\.", text.upper()))


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

    words = []
    initials = []
    for p in parts:
        if _is_initial(p):
            initials.append(p[0].upper())
        else:
            words.append(p.upper())

    if not words:
        return {"surname": None, "surname_norm": None, "name_initial": None,
                "name_full": None, "patronymic_initial": None, "patronymic_full": None}

    best_surn_idx = 0
    max_score = -1

    patr_suffixes = ("ОВИЧ", "ЕВИЧ", "ИЧ", "ОВНА", "ЕВНА", "ИЧНА", "ИНИЧНА")
    surn_suffixes = ("ОВ", "ОВА", "ЕВ", "ЕВА", "ИН", "ИНА", "СКИЙ", "СКАЯ", "ЦКИЙ", "ЦКАЯ")

    for idx, w in enumerate(words):
        score = 0
        if w.endswith(surn_suffixes):
            score += 10
        if _USE_PYMORPHY:
            with _morph_lock:
                parsed = _morph.parse(w.lower())
                if any("Surn" in p.tag for p in parsed):
                    score += 5
                if any("Name" in p.tag for p in parsed):
                    score -= 5

        if idx == 2 and len(words) == 3:
            if any(words[1].endswith(s) for s in patr_suffixes):
                score += 20

        if score > max_score:
            max_score = score
            best_surn_idx = idx

    surname = words[best_surn_idx]
    remaining_words = [w for i, w in enumerate(words) if i != best_surn_idx]

    name_full = remaining_words[0] if len(remaining_words) >= 1 else None
    patr_full = remaining_words[1] if len(remaining_words) >= 2 else None

    name_initial = name_full[0] if name_full else (initials[0] if len(initials) >= 1 else None)
    patr_initial = patr_full[0] if patr_full else (initials[1] if len(initials) >= 2 else None)

    return {
        "surname": surname,
        "surname_norm": normalize_surname(surname),
        "name_initial": name_initial,
        "name_full": name_full,
        "patronymic_initial": patr_initial,
        "patronymic_full": patr_full,
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
    text = re.sub(r"^[^\wА-Яа-яA-Za-z]+", "", text)
    text = re.sub(r"[^\wА-Яа-яA-Za-z\.]+$", "", text)
    if text.endswith("."):
        if not re.fullmatch(r"([А-Яа-яA-Za-z]\.)+", text):
            text = text[:-1]
            text = re.sub(r"[^\wА-Яа-яA-Za-z]+$", "", text)
    return text


def prepare_tokens(words_with_coords: list) -> list:
    raw_tokens = []

    for i, w in enumerate(words_with_coords):
        original = w["text"].strip()
        if not original:
            continue

        cleaned, prefix_len = _strip_numbering(original)
        if not cleaned:
            continue

        ends_with_dash = cleaned.endswith("-")

        # Исправление OCR-ошибок ДО strip_punctuation:
        # '%.' должно стать 'Э.' прежде чем punctuation-стриппер
        # выбросит '%' как не-буквенный символ и обнулит токен.
        cleaned = _fix_ocr(cleaned)
        if not cleaned:
            continue

        cleaned = _strip_punctuation(cleaned)
        if not cleaned:
            continue

        if ends_with_dash and cleaned and not cleaned.endswith("-"):
            cleaned = cleaned + "-"

        if prefix_len and len(original) > 0:
            char_width = (w["x1"] - w["x0"]) / len(original)
            new_x0 = w["x0"] + char_width * prefix_len
        else:
            new_x0 = w["x0"]

        raw_tokens.append({
            "text": cleaned, "page": w["page"],
            "x0": new_x0, "y0": w["y0"], "x1": w["x1"], "y1": w["y1"],
            "idx": i,
        })

    merged = []
    skip_next = False

    for k, rw in enumerate(raw_tokens):
        if skip_next:
            skip_next = False
            continue
        text = rw["text"]
        if not text.endswith("-"):
            merged.append(rw)
            continue

        found_continuation = False
        for j in range(k + 1, min(k + 5, len(raw_tokens))):
            nxt = raw_tokens[j]
            if not _is_word(nxt["text"]):
                continue
            if not _is_word(text[:-1]):
                break

            is_horizontal = j == k + 1
            y_diff = nxt["y0"] - rw["y0"]
            line_height = rw["y1"] - rw["y0"]
            is_vertical = (
                y_diff > 0 and y_diff < line_height * 3
                and abs(nxt["x0"] - rw["x0"]) < 100
            )

            if is_horizontal or is_vertical:
                merged.append({
                    **rw,
                    "text": text[:-1] + nxt["text"],
                    "x1": max(rw["x1"], nxt["x1"]),
                    "y1": max(rw["y1"], nxt["y1"]),
                    "parts": [
                        {"x0": rw["x0"], "y0": rw["y0"], "x1": rw["x1"], "y1": rw["y1"]},
                        {"x0": nxt["x0"], "y0": nxt["y0"], "x1": nxt["x1"], "y1": nxt["y1"]},
                    ],
                })
                skip_next = j == k + 1
                for _ in range(j - k - 1):
                    merged.append({"text": "", "page": 0, "x0": 0, "y0": 0, "x1": 0, "y1": 0, "idx": 0})
                found_continuation = True
                break

            if y_diff > line_height * 3:
                break

        if not found_continuation:
            merged.append(rw)

    tokens = []
    for rw in merged:
        text = rw["text"]
        if not text:
            continue
        base = {k: rw[k] for k in ("page", "x0", "y0", "x1", "y1", "idx")}
        if "parts" in rw:
            base["parts"] = rw["parts"]

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
# СОПОСТАВЛЕНИЕ И ПРОСТРАНСТВЕННАЯ БЛИЗОСТЬ
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
    STOP_WORDS = {
        "ФАМИЛИЯ", "ИМЯ", "ОТЧЕСТВО",
        "ПОДПИСЬ", "ДАТА", "ДОЛЖНОСТЬ",
        "М.П.", "МП", "РУКОВОДИТЕЛЬ", "ДИРЕКТОР",
    }
    return token["type"] == "word" and token["text"].upper() not in STOP_WORDS


def _same_column(anchor: dict, candidate: dict) -> bool:
    anchor_cx = (anchor["x0"] + anchor["x1"]) / 2
    candidate_cx = (candidate["x0"] + candidate["x1"]) / 2
    if abs(anchor_cx - candidate_cx) <= COL_TOLERANCE:
        return True
    overlap = min(anchor["x1"], candidate["x1"]) - max(anchor["x0"], candidate["x0"])
    return overlap > -10


def _tok_key(t: dict) -> tuple:
    """Уникальный ключ токена для дедупликации."""
    return (t["page"], round(t["x0"], 1), round(t["y0"], 1), t["text"])


# =============================================================================
# ПОИСК ИНИЦИАЛОВ — ПРОСТРАНСТВЕННЫЙ ПОДХОД
# =============================================================================

def _collect_candidates(tokens: list, anchor: dict, direction: int) -> list:
    """
    Собирает все токены, пространственно близкие к anchor.

    Два класса кандидатов:
      1. Та же строка   — любой токен в пределах MAX_HORIZONTAL_DIST по x,
                          на нужной стороне (direction).
      2. Другая строка  — только токены из той же колонки (_same_column),
                          в пределах MAX_VERTICAL_DIST по y,
                          в нужном вертикальном направлении.

    Итерируем по ВСЕМ токенам страницы, а не линейно — поэтому «чужие»
    токены из другой колонки не вызывают преждевременный break.
    """
    page = anchor["page"]
    anchor_cy = (anchor["y0"] + anchor["y1"]) / 2
    anchor_cx = (anchor["x0"] + anchor["x1"]) / 2
    line_h = max(anchor["y1"] - anchor["y0"], 8)
    same_line_threshold = line_h * 0.7

    candidates = []
    for t in tokens:
        if t["page"] != page or t["type"] not in ("word", "initial"):
            continue

        t_cy = (t["y0"] + t["y1"]) / 2
        t_cx = (t["x0"] + t["x1"]) / 2
        y_delta = t_cy - anchor_cy
        abs_y = abs(y_delta)
        on_same_line = abs_y < same_line_threshold

        if on_same_line:
            gap = max(anchor["x0"], t["x0"]) - min(anchor["x1"], t["x1"])
            if gap > MAX_HORIZONTAL_DIST:
                continue
            if direction > 0 and t_cx < anchor_cx - 5:
                continue
            if direction < 0 and t_cx > anchor_cx + 5:
                continue
        else:
            if abs_y > MAX_VERTICAL_DIST:
                continue
            if not _same_column(anchor, t):
                continue
            if direction > 0 and y_delta < 0:
                continue
            if direction < 0 and y_delta > 0:
                continue

        candidates.append(t)

    candidates.sort(key=lambda t: (
        0 if abs((t["y0"] + t["y1"]) / 2 - anchor_cy) < same_line_threshold else 1,
        abs((t["y0"] + t["y1"]) / 2 - anchor_cy),
        abs((t["x0"] + t["x1"]) / 2 - anchor_cx),
    ))
    return candidates


def _find_initials_in_window(tokens: list, start: int, direction: int,
                              anchor: dict, query: dict):
    """Ищет инициалы имени/отчества пространственно рядом с anchor."""
    candidates = _collect_candidates(tokens, anchor, direction)

    name_tok, patr_tok = None, None
    matched = []
    for t in candidates:
        if name_tok is None and _name_matches(t, query["name_initial"], query["name_full"]):
            name_tok = t
            matched.append(t)
        elif patr_tok is None and _name_matches(t, query["patronymic_initial"], query["patronymic_full"]):
            patr_tok = t
            matched.append(t)

        if (not query["name_initial"] or name_tok) and (not query["patronymic_initial"] or patr_tok):
            break

    return name_tok, patr_tok, matched


def _find_initials_vertically(tokens: list, anchor_tok: dict, query: dict,
                               needs_name: bool, needs_patr: bool):
    """
    Ищет инициалы строго в той же колонке выше/ниже anchor_tok.
    anchor_tok — фамилия ИЛИ уже найденный инициал имени (шаг 3 в _search_fio).

    ИСПРАВЛЕНИЕ 3: anchor_tok исключён из списка кандидатов.
    Без этого найденный инициал имени (например 'Ф.') мог совпасть
    с patronymic_initial при поиске отчества, блокируя нахождение 'Э.'.
    """
    page = anchor_tok["page"]
    sy0 = anchor_tok["y0"]
    sy1 = anchor_tok["y1"]
    sy_mid = (sy0 + sy1) / 2

    line_h = max(sy1 - sy0, 10)
    # MAX_VERTICAL_DIST * 2 — при поиске от инициала имени (шаг 3)
    # отчество может стоять дальше порога от фамилии
    SEARCH_DIST = max(MAX_VERTICAL_DIST * 2, line_h * 8)

    # Ключ для исключения самого anchor из кандидатов
    anchor_key = (anchor_tok["page"], round(anchor_tok["x0"], 1), round(anchor_tok["y0"], 1))

    below = sorted([
        t for t in tokens
        if t["page"] == page
        and t["type"] in ("initial", "word")
        and t["y0"] >= sy_mid - 10
        and t["y0"] < sy0 + SEARCH_DIST
        and _same_column(anchor_tok, t)
        and (t["page"], round(t["x0"], 1), round(t["y0"], 1)) != anchor_key
    ], key=lambda t: t["y0"])

    above = sorted([
        t for t in tokens
        if t["page"] == page
        and t["type"] in ("initial", "word")
        and t["y1"] <= sy_mid + 10
        and t["y1"] > sy0 - SEARCH_DIST
        and _same_column(anchor_tok, t)
        and (t["page"], round(t["x0"], 1), round(t["y0"], 1)) != anchor_key
    ], key=lambda t: t["y0"], reverse=True)

    name_tok, patr_tok = None, None
    matched = []

    for group in (below, above):
        for t in group:
            if name_tok is None and needs_name and _name_matches(t, query["name_initial"], query["name_full"]):
                name_tok = t
                matched.append(t)
            elif patr_tok is None and needs_patr and _name_matches(t, query["patronymic_initial"], query["patronymic_full"]):
                patr_tok = t
                matched.append(t)
        if matched:
            break

    return name_tok, patr_tok, matched


# =============================================================================
# ОСНОВНОЙ ПОИСК
# =============================================================================

def _search_by_surname_only(tokens: list, query: dict) -> list:
    results, seen = [], set()
    search_forms = _get_all_word_forms(query["surname"])

    for tok in tokens:
        if tok["type"] != "word":
            continue

        token_in_forms = tok["text"].upper() in search_forms
        token_forms = _get_all_word_forms(tok["text"])
        query_in_token_forms = query["surname"].upper() in token_forms

        if not (token_in_forms or query_in_token_forms):
            continue

        # ИСПРАВЛЕНИЕ 2: text включён в ключ дедупликации
        key = _tok_key(tok)
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "search_term": query.get("_raw", ""), "found_text": tok["raw"],
            "page": tok["page"], "x0": tok["x0"], "y0": tok["y0"],
            "x1": tok["x1"], "y1": tok["y1"],
        })

    return results


def _search_by_initials_only(tokens: list, query: dict) -> list:
    results, seen = [], set()
    needs_name = query["name_initial"] is not None
    needs_patr = query["patronymic_initial"] is not None
    n = len(tokens)

    for i, tok in enumerate(tokens):
        if tok["type"] != "initial":
            continue

        name_tok, patr_tok, matched = None, None, []

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
                    if t["text"][0] == query["patronymic_initial"]:
                        patr_tok = t
                        matched.append(t)
                        break
        elif needs_patr and not needs_name and tok["text"][0] == query["patronymic_initial"]:
            patr_tok = tok
            matched.append(tok)

        if not ((not needs_name or name_tok) and (not needs_patr or patr_tok)):
            continue

        for m in matched:
            # ИСПРАВЛЕНИЕ 2: text включён в ключ дедупликации
            key = _tok_key(m)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "search_term": query.get("_raw", ""), "found_text": m["raw"],
                "page": m["page"], "x0": m["x0"], "y0": m["y0"], "x1": m["x1"], "y1": m["y1"],
            })

    return results


def _search_fio(tokens: list, query: dict) -> list:
    results, seen = [], set()
    needs_name = query["name_initial"] is not None
    needs_patr = query["patronymic_initial"] is not None

    for i, tok in enumerate(tokens):
        if tok["type"] != "word" or not _surname_matches(tok, query):
            continue

        # Шаг 1: пространственный поиск вперёд и назад от фамилии
        name_f, patr_f, match_f = _find_initials_in_window(tokens, i + 1, +1, tok, query)
        name_b, patr_b, match_b = _find_initials_in_window(tokens, i - 1, -1, tok, query)

        score_f = (1 if name_f else 0) + (1 if patr_f else 0)
        score_b = (1 if name_b else 0) + (1 if patr_b else 0)

        if score_f >= score_b and score_f > 0:
            name_tok, patr_tok, matched = name_f, patr_f, [tok] + match_f
        elif score_b > score_f:
            name_tok, patr_tok, matched = name_b, patr_b, [tok] + match_b
        else:
            name_tok, patr_tok, matched = None, None, [tok]

        # Шаг 2: вертикальный поиск от фамилии для недостающих инициалов
        still_needs_name = needs_name and name_tok is None
        still_needs_patr = needs_patr and patr_tok is None

        if still_needs_name or still_needs_patr:
            vname, vpatr, vmatch = _find_initials_vertically(
                tokens, tok, query, still_needs_name, still_needs_patr
            )
            if still_needs_name and vname:
                name_tok = vname
                matched.extend(vmatch)
            if still_needs_patr and vpatr:
                patr_tok = vpatr
                for m in vmatch:
                    if m not in matched:
                        matched.append(m)

        # Шаг 3: имя найдено, отчество нет — ищем вертикально от имени
        if needs_patr and patr_tok is None and name_tok is not None:
            _, vpatr, vmatch = _find_initials_vertically(
                tokens, name_tok, query, needs_name=False, needs_patr=True
            )
            if vpatr:
                patr_tok = vpatr
                for m in vmatch:
                    if m not in matched:
                        matched.append(m)

        if (needs_name and not name_tok) or (needs_patr and not patr_tok):
            continue

        for m in matched:
            if "parts" in m:
                for part_idx, part in enumerate(m["parts"]):
                    # ИСПРАВЛЕНИЕ 2: text включён в ключ дедупликации
                    key = (m["page"], round(part["x0"], 1), round(part["y0"], 1), m["text"])
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "search_term": query.get("_raw", ""),
                        "found_text": m["raw"] if part_idx == 0 else "",
                        "page": m["page"], "x0": part["x0"], "y0": part["y0"],
                        "x1": part["x1"], "y1": part["y1"],
                    })
            else:
                # ИСПРАВЛЕНИЕ 2: text включён в ключ дедупликации
                key = _tok_key(m)
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "search_term": query.get("_raw", ""), "found_text": m["raw"],
                    "page": m["page"], "x0": m["x0"], "y0": m["y0"],
                    "x1": m["x1"], "y1": m["y1"],
                })

    return results


# =============================================================================
# ЕДИНАЯ ТОЧКА ВХОДА
# =============================================================================

def search_in_text(words_with_coords: list, search_terms) -> list:
    start = time.time()
    terms = [
        t.strip()
        for t in (search_terms.split(",") if isinstance(search_terms, str) else search_terms)
        if t.strip()
    ]
    tokens = prepare_tokens(words_with_coords)
    found = []

    for term in terms:
        query = parse_query(term)
        query["_raw"] = term
        has_surname = query["surname"] is not None
        has_initials = (query["name_initial"] is not None
                        or query["patronymic_initial"] is not None)

        if has_surname and has_initials:
            found.extend(_search_fio(tokens, query))
        elif has_surname:
            found.extend(_search_by_surname_only(tokens, query))
        elif has_initials:
            found.extend(_search_by_initials_only(tokens, query))

    print(f"[TIME] Search: {time.time() - start:.2f}s | запросов: {len(terms)} | найдено: {len(found)}")
    return found


# =============================================================================
# ТЕСТ: запуск через  python search.py
# =============================================================================

def _make_word(text, page, x0, y0, x1, y1):
    return {"text": text, "page": page, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


_NOISE_PAGE2 = [
    _make_word("21",           2,  68.0, 428.4,  80.0, 440.4),
    _make_word("Фамилия",      2,  93.4, 428.4, 140.1, 440.4),
    _make_word("Имя",          2,  93.4, 440.4, 115.2, 452.4),
    _make_word("Отчество",     2,  93.4, 452.4, 140.9, 464.4),
    _make_word("22",           2,  68.0, 468.6,  80.0, 480.6),
    _make_word("ГРН",          2,  93.4, 468.6, 110.0, 480.6),
    _make_word("и",            2, 112.0, 468.6, 118.0, 480.6),
    _make_word("дата",         2, 120.0, 468.6, 145.0, 480.6),
    _make_word("внесения",     2, 147.0, 468.6, 195.0, 480.6),
    _make_word("в",            2, 197.0, 468.6, 203.0, 480.6),
    _make_word("ЕГРЮЛ",        2, 205.0, 468.6, 248.0, 480.6),
    _make_word("1107746894683",2, 323.9, 468.6, 430.0, 480.6),
]


def _run_case(words_with_coords, search_term, label, expect_found):
    results = search_in_text(words_with_coords, search_term)
    found   = len(results) > 0
    ok      = found == expect_found
    sym     = "PASS" if ok else "FAIL"
    exp_str = "найти" if expect_found else "не найти"
    print(f"  [{sym}] '{search_term}'  ({label})")
    print(f"         ожидалось: {exp_str} | результатов: {len(results)}")
    for r in results:
        print(f"         -> {r['found_text']!r:12s} стр={r['page']}"
              f"  x0={r['x0']:.1f} y0={r['y0']:.1f}")


def test_gnetecky():
    SEP = "-" * 64

    print(SEP)
    print("Сценарий 1: нормальный OCR — 'Ф.'  и  'Э.'  раздельно")
    print(SEP)
    words_s1 = _NOISE_PAGE2 + [
        _make_word("ГНЕТЕЦКИЙ", 2, 323.9, 428.4, 395.5, 440.4),
        _make_word("Ф.",         2, 323.9, 440.4, 336.4, 452.4),
        _make_word("Э.",         2, 323.9, 452.4, 334.9, 464.4),
    ]
    _run_case(words_s1, "ГНЕТЕЦКИЙ Ф.",   label="только Ф.", expect_found=True)
    _run_case(words_s1, "ГНЕТЕЦКИЙ Ф.Э.", label="Ф. + Э.",   expect_found=True)

    print()
    print(SEP)
    print("Сценарий 2: OCR без точек — 'Ф'  и  'Э'  отдельно")
    print(SEP)
    words_s2 = _NOISE_PAGE2 + [
        _make_word("ГНЕТЕЦКИЙ", 2, 323.9, 428.4, 395.5, 440.4),
        _make_word("Ф",          2, 323.9, 440.4, 332.0, 452.4),
        _make_word("Э",          2, 323.9, 452.4, 330.0, 464.4),
    ]
    _run_case(words_s2, "ГНЕТЕЦКИЙ Ф.",   label="только Ф.", expect_found=True)
    _run_case(words_s2, "ГНЕТЕЦКИЙ Ф.Э.", label="Ф + Э",     expect_found=True)

    print()
    print(SEP)
    print("Сценарий 3: OCR склеил 'Ф.Э.' в один токен")
    print(SEP)
    words_s3 = _NOISE_PAGE2 + [
        _make_word("ГНЕТЕЦКИЙ", 2, 323.9, 428.4, 395.5, 440.4),
        _make_word("Ф.Э.",       2, 323.9, 440.4, 345.0, 452.4),
    ]
    _run_case(words_s3, "ГНЕТЕЦКИЙ Ф.",   label="только Ф.",   expect_found=True)
    _run_case(words_s3, "ГНЕТЕЦКИЙ Ф.Э.", label="слитый Ф.Э.", expect_found=True)
    res3 = search_in_text(words_s3, "ГНЕТЕЦКИЙ Ф.Э.")
    has_E = any(r["found_text"] in ("Э", "Э.") for r in res3)
    print(f"  {'OK ' if has_E else 'БАГ'} координаты Э. в results: {has_E}"
          f"  (tokens=[{', '.join(r['found_text'] for r in res3)}])")

    print()
    print(SEP)
    far_y = 428.4 + MAX_VERTICAL_DIST + 30
    print(f"Сценарий 4: Э. дальше MAX_VERTICAL_DIST={MAX_VERTICAL_DIST}px от фамилии")
    print(f"            y0(Э.)={far_y:.0f} — шаг 3 ищет от Ф.")
    print(SEP)
    words_s4 = _NOISE_PAGE2 + [
        _make_word("ГНЕТЕЦКИЙ", 2, 323.9, 428.4,  395.5, 440.4),
        _make_word("Ф.",         2, 323.9, 440.4,  336.4, 452.4),
        _make_word("Э.",         2, 323.9, far_y,  334.9, far_y + 12),
    ]
    _run_case(words_s4, "ГНЕТЕЦКИЙ Ф.Э.",
              label=f"Э. на y={far_y:.0f}", expect_found=True)

    print()
    print(SEP)
    print("Сценарий 5: реальный OCR — '%.' вместо 'Э.'  (низкий DPI)")
    print(SEP)
    words_s5 = _NOISE_PAGE2 + [
        _make_word("ГНЕТЕЦКИЙ", 2, 325.0, 427.5, 396.0, 439.5),
        _make_word("Ф.",         2, 324.5, 442.0, 336.5, 454.0),
        _make_word("%.",         2, 324.5, 454.0, 334.5, 466.0),  # OCR-ошибка
    ]
    _run_case(words_s5, "ГНЕТЕЦКИЙ Ф.",   label="только Ф.",          expect_found=True)
    _run_case(words_s5, "ГНЕТЕЦКИЙ Ф.Э.", label="'%.' исправлен в 'Э.'", expect_found=True)


if __name__ == "__main__":
    test_gnetecky()

