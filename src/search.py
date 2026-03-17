"""
Поиск ФИО в OCR-тексте PDF.

ЕДИНАЯ ТОЧКА ВХОДА:
    search_in_text(words_with_coords, search_terms)
"""

import re
import time

# -----------------------------------------------------------------------------
# Настройки
# -----------------------------------------------------------------------------
MAX_GAP = 6
MAX_VERTICAL_DIST = 80
MAX_HORIZONTAL_DIST = 400
MAX_VERTICAL_DIST_V = 50
COL_TOLERANCE = 35


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

_RULES =[
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
    if not word: return ""
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
    if not word: return frozenset()
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
    if not word: return frozenset()
    key = word.upper()

    if not _USE_PYMORPHY:
        forms = {key}
        if key[-1] not in "АЕИОУЫЭЮЯ":
            for ending in["А", "У", "ОМ", "Е"]: forms.add(key + ending)
            if not key.endswith("А"): forms.add(key + "А")
        elif key.endswith(("А", "Я")):
            base = key[:-1]
            for ending in["Ы", "Е", "У", "ОЙ", "ОЮ", "Е"]: forms.add(base + ending)
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
    return bool(re.fullmatch(r"[А-ЯЁA-Z]\.?", text.upper()))

def _is_word(text: str) -> bool:
    return bool(re.fullmatch(r"[А-ЯЁа-яёA-Za-z]+(?:-[А-ЯЁа-яёA-Za-z]+)?", text))


# =============================================================================
# ПАРСИНГ ЗАПРОСА
# =============================================================================

def parse_query(query: str) -> dict:
    query = re.sub(r"([А-ЯЁа-яё]\.)([А-ЯЁа-яё]\.?)", r"\1 \2", query)
    parts = re.findall(r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?|[А-ЯЁ]\.", query)

    words = []
    initials =[]
    for p in parts:
        if _is_initial(p): initials.append(p[0].upper())
        else: words.append(p.upper())

    if not words:
        return {"surname": None, "surname_norm": None, "name_initial": None, 
                "name_full": None, "patronymic_initial": None, "patronymic_full": None}

    best_surn_idx = 0
    max_score = -1

    patr_suffixes = ("ОВИЧ", "ЕВИЧ", "ИЧ", "ОВНА", "ЕВНА", "ИЧНА", "ИНИЧНА")
    surn_suffixes = ("ОВ", "ОВА", "ЕВ", "ЕВА", "ИН", "ИНА", "СКИЙ", "СКАЯ", "ЦКИЙ", "ЦКАЯ")

    for idx, w in enumerate(words):
        score = 0
        if w.endswith(surn_suffixes): score += 10
        if _USE_PYMORPHY:
            with _morph_lock:
                parsed = _morph.parse(w.lower())
                if any("Surn" in p.tag for p in parsed): score += 5
                if any("Name" in p.tag for p in parsed): score -= 5
        
        if idx == 2 and len(words) == 3:
            if any(words[1].endswith(s) for s in patr_suffixes):
                score += 20
        
        if score > max_score:
            max_score = score
            best_surn_idx = idx

    surname = words[best_surn_idx]
    remaining_words =[w for i, w in enumerate(words) if i != best_surn_idx]
    
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
    if m: return text[m.end() :], m.end()
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
    expanded_words =[]
    
    # 1. ЭЛЕГАНТНАЯ РАЗВЕРТКА СЛИПШИХСЯ СЛОВ (Т.В.Соколова) Без математики!
    for w in words_with_coords:
        text = w["text"].strip()
        if not text: continue
        
        # Искусственно вставляем пробелы в склейки с помощью lookbehind
        spaced = re.sub(r'([А-ЯЁA-Z]\.)(?=[А-ЯЁA-Z]\.)', r'\1 ', text)       # А.Б. -> А. Б.
        spaced = re.sub(r'(?<=\.)(?=[А-ЯЁа-яёA-Za-z])', ' ', spaced)         # Т.В -> Т. В
        spaced = re.sub(r'(?<=[А-ЯЁа-яёA-Za-z]{2})(?=[А-ЯЁA-Z]\.)', ' ', spaced) # СоколоваТ. -> Соколова Т.
        
        parts = spaced.split()
        if len(parts) > 1:
            for p in parts:
                expanded_words.append({
                    "text": p,
                    "page": w["page"],
                    "x0": w["x0"], # МЫ НЕ ВЫСЧИТЫВАЕМ КООРДИНАТЫ, ПРОСТО ДУБЛИРУЕМ РОДИТЕЛЯ!
                    "y0": w["y0"], # При подсветке обведется весь блок целиком
                    "x1": w["x1"],
                    "y1": w["y1"]
                })
        else:
            expanded_words.append(w)

    raw_tokens =[]
    for i, w in enumerate(expanded_words):
        original = w["text"].strip()
        if not original: continue

        cleaned, prefix_len = _strip_numbering(original)
        if not cleaned: continue

        ends_with_dash = cleaned.endswith("-")
        cleaned = _strip_punctuation(cleaned)
        if not cleaned: continue

        if ends_with_dash and cleaned and not cleaned.endswith("-"):
            cleaned = cleaned + "-"

        # Префикс (нумерация) сдвигает только если это оригинальное слово, а не кусок
        if prefix_len and len(original) > 0 and len(original) == len(w["text"].strip()):
            char_width = (w["x1"] - w["x0"]) / len(original)
            new_x0 = w["x0"] + char_width * prefix_len
        else:
            new_x0 = w["x0"]

        raw_tokens.append({
            "text": cleaned, "page": w["page"],
            "x0": new_x0, "y0": w["y0"], "x1": w["x1"], "y1": w["y1"], "idx": i,
        })

    merged =[]
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
            if not _is_word(nxt["text"]): continue
            if not _is_word(text[:-1]): break

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
                    "parts":[
                        {"x0": rw["x0"], "y0": rw["y0"], "x1": rw["x1"], "y1": rw["y1"]},
                        {"x0": nxt["x0"], "y0": nxt["y0"], "x1": nxt["x1"], "y1": nxt["y1"]},
                    ],
                })
                skip_next = j == k + 1
                for _ in range(j - k - 1):
                    merged.append({"text": "", "page": 0, "x0": 0, "y0": 0, "x1": 0, "y1": 0, "idx": 0})
                found_continuation = True
                break

            if y_diff > line_height * 3: break

        if not found_continuation:
            merged.append(rw)

    tokens =[]
    for rw in merged:
        text = rw["text"]
        if not text: continue
        base = {k: rw[k] for k in ("page", "x0", "y0", "x1", "y1", "idx")}
        if "parts" in rw: base["parts"] = rw["parts"]

        if _is_initial(text): ttype = "initial"
        elif _is_word(text): ttype = "word"
        else: ttype = "junk"

        tokens.append({**base, "type": ttype, "text": text.upper(), "raw": text})

    return tokens


# =============================================================================
# СОПОСТАВЛЕНИЕ И ПРОСТРАНСТВЕННАЯ БЛИЗОСТЬ
# =============================================================================

def _surname_matches(token: dict, query: dict) -> bool:
    if not query["surname_norm"]:
        return False
        
    # БЛОКИРОВКА ЗАХВАТА ДВОЙНЫХ ФАМИЛИЙ: 
    # Если в тексте дефис есть, а мы искали без дефиса — игнорируем
    if "-" in token["text"] and "-" not in query["surname"]:
        return False

    if normalize_surname(token["text"]) == query["surname_norm"]:
        return True
        
    token_forms = _all_surname_forms(token["text"])
    query_forms = _all_surname_forms(query["surname"])
    return bool(token_forms & query_forms)


def _name_matches(token: dict, initial, full) -> bool:
    if initial is None: return False
    if token["text"][0].upper() != initial.upper(): return False
    if token["type"] == "initial": return True
    if full is not None:
        return normalize_surname(token["text"]) == normalize_surname(full)
        
    STOP_WORDS = {"ФАМИЛИЯ", "ИМЯ", "ОТЧЕСТВО", "ПОДПИСЬ", "ДАТА", "ДОЛЖНОСТЬ", "М.П.", "МП", "РУКОВОДИТЕЛЬ", "ДИРЕКТОР"}
    return token["type"] == "word" and token["text"].upper() not in STOP_WORDS


def _tokens_are_close(anchor: dict, candidate: dict) -> bool:
    if abs(anchor["y0"] - candidate["y0"]) > MAX_VERTICAL_DIST: return False
    gap = max(anchor["x0"], candidate["x0"]) - min(anchor["x1"], candidate["x1"])
    return gap <= MAX_HORIZONTAL_DIST

def _same_column(anchor: dict, candidate: dict) -> bool:
    anchor_cx = (anchor["x0"] + anchor["x1"]) / 2
    candidate_cx = (candidate["x0"] + candidate["x1"]) / 2
    if abs(anchor_cx - candidate_cx) <= COL_TOLERANCE: return True
    overlap = min(anchor["x1"], candidate["x1"]) - max(anchor["x0"], candidate["x0"])
    if overlap > -10: return True
    return False


# =============================================================================
# ПОИСК ИНИЦИАЛОВ
# =============================================================================

def _find_initials_in_window(tokens, start, direction, anchor, query):
    n = len(tokens)
    line_h = max(anchor["y1"] - anchor["y0"], 8)
    anchor_cy = (anchor["y0"] + anchor["y1"]) / 2
    anchor_cx = (anchor["x0"] + anchor["x1"]) / 2

    SAME_LINE_Y  = line_h * 0.7
    MAX_WRAP_Y   = line_h * 2.0

    name_tok, patr_tok = None, None
    matched =[]
    wrap_line_cy = None
    unrelated_words = 0

    j = start
    while 0 <= j < n:
        t = tokens[j]
        j += direction

        if t["page"] != anchor["page"]: break
        if t["type"] not in ("word", "initial"): continue

        t_cy = (t["y0"] + t["y1"]) / 2
        t_cx = (t["x0"] + t["x1"]) / 2
        y_delta = t_cy - anchor_cy
        abs_y = abs(y_delta)

        if abs_y > MAX_WRAP_Y: break

        on_anchor_line = abs_y < SAME_LINE_Y

        if on_anchor_line:
            if not _tokens_are_close(anchor, t): continue
            if direction > 0 and t_cx < anchor_cx - 5: continue
            if direction < 0 and t_cx > anchor_cx + 5: continue
        else:
            # РАЗРЕШЕН ПОИСК НА ПРЕДЫДУЩУЮ И СЛЕДУЮЩУЮ СТРОКУ (ОБРАБОТКА ПЕРЕНОСА)
            if direction > 0 and y_delta < 0: continue # Ищем вперед, но токен выше
            if direction < 0 and y_delta > 0: continue # Ищем назад, но токен ниже

            if wrap_line_cy is None:
                wrap_line_cy = t_cy
            elif abs(t_cy - wrap_line_cy) > SAME_LINE_Y:
                break

            if t["type"] == "word":
                idx_dist = abs(t["idx"] - anchor["idx"])
                is_sequential = 0 < idx_dist <= 5
                if not _same_column(anchor, t) and not is_sequential:
                    continue

        # ЗАЩИТА ОТ КРАЖИ ИНИЦИАЛОВ ЧУЖОГО ЧЕЛОВЕКА
        if t["type"] == "word":
            is_name = _name_matches(t, query["name_initial"], query["name_full"])
            is_patr = _name_matches(t, query["patronymic_initial"], query["patronymic_full"])
            # Если мы наткнулись на чужое слово (например, другую фамилию) - сворачиваемся
            if not is_name and not is_patr:
                unrelated_words += 1
                if unrelated_words > 1: # Допускаем максимум 1 мусорное слово (например, предлог)
                    break

        if name_tok is None and _name_matches(t, query["name_initial"], query["name_full"]):
            name_tok = t
            matched.append(t)
        elif patr_tok is None and _name_matches(t, query["patronymic_initial"], query["patronymic_full"]):
            patr_tok = t
            matched.append(t)

        found_all = True
        if query["name_initial"] and not name_tok: found_all = False
        if query["patronymic_initial"] and not patr_tok: found_all = False
        
        if found_all: break

    return name_tok, patr_tok, matched


def _find_initials_vertically(tokens: list, surname_tok: dict, query: dict, needs_name: bool, needs_patr: bool):
    page = surname_tok["page"]
    sy0 = surname_tok["y0"]
    sy1 = surname_tok["y1"]
    sy_mid = (sy0 + sy1) / 2
    
    SEARCH_DIST = 80

    below =[
        t for t in tokens if t["page"] == page and t["type"] in ("initial", "word")
        and t["y0"] >= sy_mid - 10 and t["y0"] < sy0 + SEARCH_DIST
        and _same_column(surname_tok, t)
    ]
    above =[
        t for t in tokens if t["page"] == page and t["type"] in ("initial", "word")
        and t["y1"] <= sy_mid + 10 and t["y1"] > sy0 - SEARCH_DIST
        and _same_column(surname_tok, t)
    ]

    below.sort(key=lambda t: t["y0"])
    above.sort(key=lambda t: t["y0"], reverse=True)

    name_tok, patr_tok = None, None
    matched =[]

    for group in (below, above):
        for t in group:
            if name_tok is None and needs_name and _name_matches(t, query["name_initial"], query["name_full"]):
                name_tok = t
                matched.append(t)
            elif patr_tok is None and needs_patr and _name_matches(t, query["patronymic_initial"], query["patronymic_full"]):
                patr_tok = t
                matched.append(t)
        if matched: break

    return name_tok, patr_tok, matched


# =============================================================================
# ОСНОВНОЙ ПОИСК
# =============================================================================

def _search_by_surname_only(tokens: list, query: dict) -> list:
    results, seen =[], set()
    search_forms = _get_all_word_forms(query["surname"])

    for tok in tokens:
        if tok["type"] != "word": continue

        token_in_forms = tok["text"].upper() in search_forms
        token_forms = _get_all_word_forms(tok["text"])
        query_in_token_forms = query["surname"].upper() in token_forms

        if not (token_in_forms or query_in_token_forms): continue

        key = (tok["page"], round(tok["x0"], 1), round(tok["y0"], 1))
        if key in seen: continue
        seen.add(key)
        
        results.append({
            "search_term": query.get("_raw", ""), "found_text": tok["raw"],
            "page": tok["page"], "x0": tok["x0"], "y0": tok["y0"],
            "x1": tok["x1"], "y1": tok["y1"],
        })

    return results


def _search_by_initials_only(tokens: list, query: dict) -> list:
    results, seen =[], set()
    needs_name = query["name_initial"] is not None
    needs_patr = query["patronymic_initial"] is not None
    n = len(tokens)

    for i, tok in enumerate(tokens):
        if tok["type"] != "initial": continue

        name_tok, patr_tok, matched = None, None,[]

        if needs_name and tok["text"][0] == query["name_initial"]:
            name_tok = tok
            matched.append(tok)
            if needs_patr:
                for j in range(i + 1, min(i + 1 + MAX_GAP, n)):
                    t = tokens[j]
                    if t["type"] == "junk": continue
                    if t["type"] != "initial" or not _tokens_are_close(tok, t): break
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
            key = (m["page"], round(m["x0"], 1), round(m["y0"], 1))
            if key in seen: continue
            seen.add(key)
            results.append({
                "search_term": query.get("_raw", ""), "found_text": m["raw"],
                "page": m["page"], "x0": m["x0"], "y0": m["y0"], "x1": m["x1"], "y1": m["y1"],
            })

    return results


def _search_fio(tokens: list, query: dict) -> list:
    results, seen =[], set()
    needs_name = query["name_initial"] is not None
    needs_patr = query["patronymic_initial"] is not None

    for i, tok in enumerate(tokens):
        if tok["type"] != "word" or not _surname_matches(tok, query):
            continue

        # Шаг 1: горизонтальный поиск
        name_f, patr_f, match_f = _find_initials_in_window(tokens, i + 1, +1, tok, query)
        name_b, patr_b, match_b = _find_initials_in_window(tokens, i - 1, -1, tok, query)

        score_f = (1 if name_f else 0) + (1 if patr_f else 0)
        score_b = (1 if name_b else 0) + (1 if patr_b else 0)

        if score_f > score_b:
            name_tok, patr_tok, matched = name_f, patr_f, [tok] + match_f
        elif score_b > score_f:
            name_tok, patr_tok, matched = name_b, patr_b, [tok] + match_b
        elif score_f == score_b and score_f > 0:
            name_tok, patr_tok, matched = name_f, patr_f, [tok] + match_f
        else:
            name_tok, patr_tok, matched = None, None, [tok]

        # Шаг 2: вертикальный поиск 
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

        if (needs_name and not name_tok) or (needs_patr and not patr_tok):
            continue

        for m in matched:
            if "parts" in m:
                for part_idx, part in enumerate(m["parts"]):
                    key = (m["page"], round(part["x0"], 1), round(part["y0"], 1))
                    if key in seen: continue
                    seen.add(key)
                    results.append({
                        "search_term": query.get("_raw", ""),
                        "found_text": m["raw"] if part_idx == 0 else "",
                        "page": m["page"], "x0": part["x0"], "y0": part["y0"],
                        "x1": part["x1"], "y1": part["y1"],
                    })
            else:
                key = (m["page"], round(m["x0"], 1), round(m["y0"], 1))
                if key in seen: continue
                seen.add(key)
                results.append({
                    "search_term": query.get("_raw", ""), "found_text": m["raw"],
                    "page": m["page"], "x0": m["x0"], "y0": m["y0"], "x1": m["x1"], "y1": m["y1"],
                })

    return results

# =============================================================================
# ЕДИНАЯ ТОЧКА ВХОДА
# =============================================================================

def search_in_text(words_with_coords: list, search_terms) -> list:
    start = time.time()
    terms =[t.strip() for t in (search_terms.split(",") if isinstance(search_terms, str) else search_terms) if t.strip()]
    # print(words_with_coords)
    tokens = prepare_tokens(words_with_coords)
    found =[]

    for term in terms:
        query = parse_query(term)
        query["_raw"] = term
        has_surname = query["surname"] is not None
        has_initials = query["name_initial"] is not None or query["patronymic_initial"] is not None

        if has_surname and has_initials: found.extend(_search_fio(tokens, query))
        elif has_surname: found.extend(_search_by_surname_only(tokens, query))
        elif has_initials: found.extend(_search_by_initials_only(tokens, query))

    print(f"[TIME] Search: {time.time() - start:.2f}s | запросов: {len(terms)} | найдено: {len(found)}")
    return found