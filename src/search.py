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
import threading

# =============================================================================
# НАСТРОЙКИ
# =============================================================================
MAX_GAP = 6
MAX_VERTICAL_DIST = 80
MAX_HORIZONTAL_DIST = 400
MAX_VERTICAL_DIST_V = 50
COL_TOLERANCE = 35

# =============================================================================
# МОРФОЛОГИЯ (pymorphy3 + fallback)
# =============================================================================

_RULES = [
    ("СКОГО", "СКИЙ"), ("СКОМУ", "СКИЙ"), ("СКИМ", "СКИЙ"),
    ("ЦКОГО", "ЦКИЙ"), ("ЦКОМУ", "ЦКИЙ"), ("ЦКИМ", "ЦКИЙ"),
    ("СКОЙ", "СКИЙ"), ("ЦКОЙ", "ЦКИЙ"),
    ("ОВЫМ", "ОВ"), ("ЕВЫМ", "ЕВ"), ("ОВОМ", "ОВ"), ("ЕВОМ", "ЕВ"),
    ("ОВОГО", "ОВ"), ("ЕВОГО", "ЕВ"), ("ОВОМУ", "ОВ"), ("ЕВОМУ", "ЕВ"),
    ("ОВЫХ", "ОВ"), ("ЕВЫХ", "ЕВ"), ("ОВОЙ", "ОВ"), ("ЕВОЙ", "ЕВ"),
    ("ОВУ", "ОВ"), ("ЕВУ", "ЕВ"), ("ОВЕ", "ОВ"), ("ЕВЕ", "ЕВ"),
    ("ОВЫ", "ОВ"), ("ЕВЫ", "ЕВ"), ("ОВА", "ОВ"), ("ЕВА", "ЕВ"),
    ("ИНЫМ", "ИН"), ("ИНОГО", "ИН"), ("ИНОМУ", "ИН"), ("ИНЫХ", "ИН"),
    ("ИНОЙ", "ИН"), ("ИНУ", "ИН"), ("ИНЕ", "ИН"), ("ИНЫ", "ИН"), ("ИНА", "ИН"),
]

try:
    from pymorphy3 import MorphAnalyzer
    _morph = MorphAnalyzer()
    _morph_lock = threading.Lock()
    _USE_PYMORPHY = True
    print("[search] Используется pymorphy3")
except ImportError:
    _USE_PYMORPHY = False
    print("[search] Используются встроенные правила")

_norm_cache = {}
_forms_cache = {}


def normalize(word: str) -> str:
    """Приводит слово к начальной форме (именительный падеж)."""
    word = word.strip()
    if not word:
        return ""
    key = word.upper()

    if key not in _norm_cache:
        if _USE_PYMORPHY:
            with _morph_lock:
                parsed = _morph.parse(word.lower())
            for p in parsed:
                if "Surn" in str(p.tag) and "masc" in str(p.tag):
                    _norm_cache[key] = p.normal_form.upper()
                    break
            else:
                for p in parsed:
                    if "Surn" in str(p.tag):
                        _norm_cache[key] = p.normal_form.upper()
                        break
                else:
                    _norm_cache[key] = parsed[0].normal_form.upper()
        else:
            w = key
            for suffix, replacement in _RULES:
                if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                    _norm_cache[key] = w[: -len(suffix)] + replacement
                    break
            else:
                _norm_cache[key] = key

    return _norm_cache[key]


def get_all_forms(word: str) -> frozenset:
    """Возвращает все возможные формы слова для поиска."""
    word = word.strip()
    if not word:
        return frozenset()
    key = word.upper()
    cache_key = "FORMS:" + key

    if cache_key not in _forms_cache:
        if not _USE_PYMORPHY:
            forms = {key}
            if key[-1] not in "АЕИОУЫЭЮЯ":
                for ending in ["А", "У", "ОМ", "Е"]:
                    forms.add(key + ending)
            elif key.endswith(("А", "Я")):
                base = key[:-1]
                for ending in ["Ы", "Е", "У", "ОЙ", "ОЮ"]:
                    forms.add(base + ending)
            _forms_cache[cache_key] = frozenset(forms)
        else:
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
    return bool(re.fullmatch(r"[А-ЯЁ]\.?[А-ЯЁ]\.?", text.upper()))


def _split_double_initial(text: str) -> list:
    return re.findall(r"[А-ЯЁ]", text.upper())


def _is_word(text: str) -> bool:
    return bool(re.fullmatch(r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?", text))

# =============================================================================
# ПОДГОТОВКА ТОКЕНОВ
# =============================================================================


def _strip_numbering(text: str) -> tuple:
    m = re.match(r"^\d+[.)\s]+", text)
    if m:
        return text[m.end():], m.end()
    return text, 0


def _strip_punctuation(text: str) -> str:
    return re.sub(r"^[^\wА-Яа-яA-Za-z]+|[^\wА-Яа-яA-Za-z]+$", "", text)


def prepare_tokens(words_with_coords: list) -> list:
    """Подготовка токенов: удаление нумерации, склейка переносов, классификация."""
    raw_tokens = []

    for i, w in enumerate(words_with_coords):
        original = w["text"].strip()
        if not original:
            continue

        cleaned, prefix_len = _strip_numbering(original)
        if not cleaned:
            continue

        ends_with_dash = cleaned.endswith("-")
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
            "text": cleaned,
            "page": w["page"],
            "x0": new_x0, "y0": w["y0"],
            "x1": w["x1"], "y1": w["y1"],
            "idx": i,
        })

    # Склейка переносов
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

        found = False
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
                found = True
                break

            if y_diff > line_height * 3:
                break

        if not found:
            merged.append(rw)

    # Классификация
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

        ttype = "initial" if _is_initial(text) else ("word" if _is_word(text) else "junk")
        tokens.append({**base, "type": ttype, "text": text.upper(), "raw": text})

    return tokens

# =============================================================================
# ПРОСТРАНСТВЕННАЯ БЛИЗОСТЬ
# =============================================================================


def _same_line(anchor: dict, candidate: dict) -> bool:
    """Токены на одной строке (горизонтально)."""
    if abs(anchor["y0"] - candidate["y0"]) > MAX_VERTICAL_DIST:
        return False
    gap = max(anchor["x0"], candidate["x0"]) - min(anchor["x1"], candidate["x1"])
    return gap <= MAX_HORIZONTAL_DIST


def _same_column(anchor: dict, candidate: dict) -> bool:
    """Токены в одной колонке (вертикально, для таблиц)."""
    anchor_cx = (anchor["x0"] + anchor["x1"]) / 2
    candidate_cx = (candidate["x0"] + candidate["x1"]) / 2
    return abs(anchor_cx - candidate_cx) <= COL_TOLERANCE

# =============================================================================
# ПОИСК ИНИЦИАЛОВ
# =============================================================================


def _find_initials(tokens: list, anchor: dict, query: dict, horizontal: bool = True) -> list:
    """
    Ищет инициалы имени и отчества.
    horizontal=True — вдоль строки, False — по вертикали (таблицы).
    """
    page = anchor["page"]
    needs_name = query["name_initial"] is not None
    needs_patr = query["patronymic_initial"] is not None

    if horizontal:
        candidates = []
        for direction in [+1, -1]:
            gap = 0
            j = tokens.index(anchor) + direction
            while 0 <= j < len(tokens) and gap <= MAX_GAP:
                t = tokens[j]
                if t["page"] != page:
                    break
                if not _same_line(anchor, t):
                    break
                if t["type"] in ("word", "initial"):
                    candidates.append(t)
                else:
                    gap += 1
                j += direction
    else:
        sy0, sy1 = anchor["y0"], anchor["y1"]
        candidates = [
            t for t in tokens
            if t["page"] == page
            and t["type"] in ("initial", "word")
            and _same_column(anchor, t)
            and (sy1 < t["y0"] < sy0 + MAX_VERTICAL_DIST_V or sy0 - MAX_VERTICAL_DIST_V < t["y1"] < sy0)
        ]

    matched = []
    found_name = False
    found_patr = False

    for t in candidates:
        if needs_name and not found_name:
            if t["text"][0].upper() == query["name_initial"]:
                matched.append(t)
                found_name = True
                continue
        if needs_patr and not found_patr:
            if t["text"][0].upper() == query["patronymic_initial"]:
                matched.append(t)
                found_patr = True
                continue
        if horizontal and t["type"] in ("word", "initial"):
            break

    return matched if (found_name or not needs_name) and (found_patr or not needs_patr) else []

# =============================================================================
# ЕДИНЫЙ ПОИСК
# =============================================================================


def _search(tokens: list, query: dict) -> list:
    """
    Универсальный поиск ФИО. Автоматически определяет режим по наличию фамилии и инициалов.
    """
    results = []
    seen = set()

    has_surname = query["surname"] is not None
    has_initials = query["name_initial"] is not None or query["patronymic_initial"] is not None

    for i, tok in enumerate(tokens):
        # Режим 1: только инициалы
        if not has_surname and has_initials:
            if tok["type"] != "initial":
                continue
            matched = _find_initials(tokens, tok, query, horizontal=True)
            if matched:
                for m in [tok] + matched:
                    key = (m["page"], round(m["x0"], 1), round(m["y0"], 1))
                    if key not in seen:
                        seen.add(key)
                        results.append(_make_result(query, m))

        # Режим 2: только фамилия/слово
        elif has_surname and not has_initials:
            if tok["type"] != "word":
                continue
            token_forms = get_all_forms(tok["text"])
            query_forms = get_all_forms(query["surname"])
            if not (tok["text"].upper() in query_forms or query["surname"].upper() in token_forms):
                continue
            key = (tok["page"], round(tok["x0"], 1), round(tok["y0"], 1))
            if key not in seen:
                seen.add(key)
                results.append(_make_result(query, tok))

        # Режим 3: ФИО полностью
        elif has_surname and has_initials:
            if tok["type"] != "word":
                continue
            token_forms = get_all_forms(tok["text"])
            query_forms = get_all_forms(query["surname"])
            if not (normalize(tok["text"]) == query["surname_norm"] or
                    tok["text"].upper() in query_forms or query["surname"].upper() in token_forms):
                continue

            matched = _find_initials(tokens, tok, query, horizontal=True)
            if not matched or len(matched) < (1 if query["name_initial"] and not query["patronymic_initial"] else
                                               2 if query["name_initial"] and query["patronymic_initial"] else 1):
                matched = _find_initials(tokens, tok, query, horizontal=False)

            if not matched:
                continue

            if query["name_initial"] and not any(t["text"][0] == query["name_initial"] for t in matched):
                continue
            if query["patronymic_initial"] and not any(t["text"][0] == query["patronymic_initial"] for t in matched):
                continue

            for m in [tok] + matched:
                if "parts" in m:
                    for part_idx, part in enumerate(m["parts"]):
                        key = (m["page"], round(part["x0"], 1), round(part["y0"], 1))
                        if key not in seen:
                            seen.add(key)
                            results.append({
                                "search_term": query.get("_raw", ""),
                                "found_text": m["raw"] if part_idx == 0 else "",
                                "page": m["page"],
                                "x0": part["x0"], "y0": part["y0"],
                                "x1": part["x1"], "y1": part["y1"],
                            })
                else:
                    key = (m["page"], round(m["x0"], 1), round(m["y0"], 1))
                    if key not in seen:
                        seen.add(key)
                        results.append(_make_result(query, m))

    return results


def _make_result(query: dict, token: dict) -> dict:
    """Создаёт результат поиска."""
    return {
        "search_term": query.get("_raw", ""),
        "found_text": token["raw"],
        "page": token["page"],
        "x0": token["x0"], "y0": token["y0"],
        "x1": token["x1"], "y1": token["y1"],
    }

# =============================================================================
# ПАРСИНГ ЗАПРОСА
# =============================================================================


def parse_query(query: str) -> dict:
    """Парсит запрос вида «Иванов А.И.» или «Иванов»."""
    query = re.sub(r"([А-ЯЁа-яё]\.)([А-ЯЁа-яё]\.?)", r"\1 \2", query)
    parts = re.findall(r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?|[А-ЯЁ]\.", query)

    words = []
    initials = []

    for p in parts:
        if _is_initial(p):
            initials.append(p[0].upper())
        else:
            words.append(p.upper())

    surname = words[0] if words else None
    name_initial = initials[0] if initials else None
    patr_initial = initials[1] if len(initials) >= 2 else None

    return {
        "surname": surname,
        "surname_norm": normalize(surname) if surname else None,
        "name_initial": name_initial,
        "patronymic_initial": patr_initial,
        "_raw": query,
    }

# =============================================================================
# ЕДИНАЯ ТОЧКА ВХОДА
# =============================================================================


def search_in_text(words_with_coords: list, search_terms) -> list:
    """
    Ищет ФИО в OCR-тексте PDF.

    Аргументы:
        words_with_coords — список словарей:
            {"text": str, "page": int, "x0": float, "y0": float, "x1": float, "y1": float}
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
    found = []

    for term in terms:
        query = parse_query(term)
        found.extend(_search(tokens, query))

    print(
        f"[TIME] Search: {time.time() - start:.2f}s | "
        f"запросов: {len(terms)} | найдено: {len(found)}"
    )

    return found
