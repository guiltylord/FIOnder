"""
Поиск ФИО в OCR-тексте PDF.

ЕДИНАЯ ТОЧКА ВХОДА:
    search_in_text(words_with_coords, search_terms)
"""

import math
import re
import time

# -----------------------------------------------------------------------------
# Настройки
# -----------------------------------------------------------------------------
# Хардкор-пиксельные лимиты убраны из основного FIO поиска, 
# они оставлены только для базовых/старых эвристик инициалов.
MAX_GAP = 6
MAX_VERTICAL_DIST = 80
MAX_HORIZONTAL_DIST = 400


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
    return bool(re.fullmatch(r"[А-ЯЁа-яёA-Za-z]+(?:[-–—−][А-ЯЁа-яёA-Za-z]+)?", text))


# =============================================================================
# ПАРСИНГ ЗАПРОСА
# =============================================================================

def parse_query(query: str) -> dict:
    query = re.sub(r"[–—−]", "-", query)
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

    # СПЕЦ-ПРОВЕРКА: если пользователь ввел "Тормышов ПЕ" или "Архипова ЮГ"
    # Если второе слово это 2 заглавные буквы (и нет других инициалов), разбиваем на Имя и Отчество
    if name_full and len(name_full) == 2 and name_full.isupper() and not patr_full and len(initials) == 0:
        name_initial = name_full[0]
        patr_initial = name_full[1]
        name_full = None

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
    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"^[^\wА-ЯЁа-яёA-Za-z\-]+", "", text)
    text = re.sub(r"[^\wА-ЯЁа-яёA-Za-z\.\-]+$", "", text)
    if text.endswith("."):
        if not re.fullmatch(r"([А-ЯЁа-яёA-Za-z]\.)+", text):
            text = text[:-1]
            text = re.sub(r"[^\wА-ЯЁа-яёA-Za-z\-]+$", "", text)
    return text

def prepare_tokens(words_with_coords: list) -> list:
    expanded_words =[]
    
    # 1. Расклейка (Т.В.Соколова -> Т. В. Соколова)
    for w in words_with_coords:
        text = re.sub(r"[–—−]", "-", w["text"].strip())
        if not text: continue
        
        spaced = re.sub(r'([А-ЯЁA-Z]\.)(?=[А-ЯЁA-Z]\.)', r'\1 ', text)
        spaced = re.sub(r'(?<=\.)(?=[А-ЯЁа-яёA-Za-z])', ' ', spaced)
        spaced = re.sub(r'(?<=[А-ЯЁа-яёA-Za-z]{2})(?=[А-ЯЁA-Z]\.)', ' ', spaced)
        
        parts = spaced.split()
        if len(parts) > 1:
            for p in parts:
                expanded_words.append({
                    "text": p, "raw": p, 
                    "page": w["page"], "x0": w["x0"], "y0": w["y0"], "x1": w["x1"], "y1": w["y1"]
                })
        else:
            expanded_words.append({**w, "text": text, "raw": text})

    raw_tokens =[]
    for i, w in enumerate(expanded_words):
        original = w["text"].strip()
        if not original: continue

        cleaned, prefix_len = _strip_numbering(original)
        if not cleaned: continue

        ends_with_dash = bool(re.search(r"-$", cleaned))
        cleaned = _strip_punctuation(cleaned)
        if not cleaned: continue

        if ends_with_dash and cleaned and not cleaned.endswith("-"):
            cleaned = cleaned + "-"

        if prefix_len and len(original) > 0 and len(original) == len(w["raw"]):
            char_width = (w["x1"] - w["x0"]) / len(original)
            new_x0 = w["x0"] + char_width * prefix_len
        else:
            new_x0 = w["x0"]

        raw_tokens.append({
            "text": cleaned, "raw": w["raw"], "page": w["page"],
            "x0": new_x0, "y0": w["y0"], "x1": w["x1"], "y1": w["y1"], "idx": i,
        })

    # 2. ИДЕАЛЬНАЯ СКЛЕЙКА ПЕРЕНОСОВ
    merged =[]
    i = 0
    n = len(raw_tokens)
    
    while i < n:
        rw = raw_tokens[i]
        text = rw["text"]
        
        if not text.endswith("-"):
            merged.append(rw)
            i += 1
            continue
            
        found_continuation = False
        for j in range(i + 1, min(i + 5, n)):
            nxt = raw_tokens[j]
            if not _is_word(nxt["text"]): continue
            if not _is_word(text[:-1]): break

            is_horizontal = j == i + 1
            y_diff = nxt["y0"] - rw["y0"]
            line_height = max(rw["y1"] - rw["y0"], 8)
            is_vertical = (
                y_diff > 0 and y_diff < line_height * 4 
                and abs(nxt["x0"] - rw["x0"]) < 150
            )

            if is_horizontal or is_vertical:
                merged_text = text + nxt["text"] if is_horizontal else text[:-1] + nxt["text"]
                merged.append({
                    **rw,
                    "text": merged_text,
                    "raw": rw["raw"] + nxt["raw"],
                    "x1": max(rw["x1"], nxt["x1"]),
                    "y1": max(rw["y1"], nxt["y1"]),
                    "parts":[
                        {"x0": rw["x0"], "y0": rw["y0"], "x1": rw["x1"], "y1": rw["y1"]},
                        {"x0": nxt["x0"], "y0": nxt["y0"], "x1": nxt["x1"], "y1": nxt["y1"]},
                    ],
                })
                i = j + 1 
                found_continuation = True
                break

            if y_diff > line_height * 4: break

        if not found_continuation:
            merged.append(rw)
            i += 1

    tokens =[]
    for rw in merged:
        text = rw["text"]
        if not text: continue
        base = {k: rw[k] for k in ("page", "x0", "y0", "x1", "y1", "idx")}
        if "parts" in rw: base["parts"] = rw["parts"]

        if _is_initial(text): ttype = "initial"
        elif _is_word(text): ttype = "word"
        else: ttype = "junk"

        tokens.append({**base, "type": ttype, "text": text.upper(), "raw": rw["raw"]})

    return tokens


# =============================================================================
# СОПОСТАВЛЕНИЕ И ПРОСТРАНСТВЕННАЯ БЛИЗОСТЬ
# =============================================================================

def _surname_matches(token: dict, query: dict) -> bool:
    if not query["surname_norm"]:
        return False
        
    has_dash_token = bool(re.search(r"[-–—−]", token["raw"]))
    has_dash_query = bool(re.search(r"[-–—−]", query["surname"]))
    
    if has_dash_token and not has_dash_query:
        return False

    if normalize_surname(token["text"]) == query["surname_norm"]:
        return True
        
    token_forms = _all_surname_forms(token["text"])
    query_forms = _all_surname_forms(query["surname"])
    return bool(token_forms & query_forms)


def _name_matches(token: dict, query: dict, is_patr: bool = False) -> bool:
    initial = query["patronymic_initial"] if is_patr else query["name_initial"]
    full = query["patronymic_full"] if is_patr else query["name_full"]
    
    if initial is None: return False
    text = token["text"]
    if not text: return False
    
    # 1. Спец-проверка на склеенные инициалы (например "ПЕ", "ЮГ", "П.Е.")
    n_init = query["name_initial"]
    p_init = query["patronymic_initial"]
    if n_init and p_init:
        combined = (n_init + p_init).upper()
        text_cleaned = text.replace(".", "").upper()
        if text_cleaned == combined:
            return True
            
    # 2. Обычная проверка первой буквы
    if text[0].upper() != initial.upper(): return False

    if token["type"] == "initial": 
        return True

    # Игнорируем мета-слова таблиц и форм
    STOP_WORDS = {"ФАМИЛИЯ", "ИМЯ", "ОТЧЕСТВО", "ПОДПИСЬ", "ДАТА", "ДОЛЖНОСТЬ", "М.П.", "МП", "РУКОВОДИТЕЛЬ", "ДИРЕКТОР"}
    if text.upper() in STOP_WORDS: 
        return False

    # Если в запросе передано полное имя ("Эдуард")
    if full:
        if _USE_PYMORPHY:
            with _morph_lock:
                parsed_tok = _morph.parse(text.lower())
                parsed_full = _morph.parse(full.lower())
                tok_norms = {p.normal_form for p in parsed_tok}
                full_norms = {p.normal_form for p in parsed_full}
                if tok_norms & full_norms: 
                    return True
        
        # Запасные эвристики, если нет Pymorphy или формы не совпали
        if text.upper() == full.upper(): return True
        # Разрешаем "Эдуарда" -> "Эдуард"
        if len(full) > 3 and text.upper().startswith(full.upper()[:len(full)-2]): return True
        return False
        
    # Если в запросе ТОЛЬКО инициал ("Э."), а в тексте попалось целое слово
    else:
        if _USE_PYMORPHY:
            with _morph_lock:
                parsed = _morph.parse(text.lower())
                target_tag = "Patr" if is_patr else "Name"
                # Проверяем, является ли слово реальным именем/отчеством
                if any(target_tag in p.tag for p in parsed):
                    return True
                    
        # Запасная эвристика для отчеств (без словаря)
        if is_patr and text.upper().endswith(("ОВИЧ", "ЕВИЧ", "ИЧ", "ОВНА", "ЕВНА", "ИЧНА", "ИНИЧНА")):
            return True
            
        # Если нет pymorphy, то разрешаем слово
        if not _USE_PYMORPHY and not is_patr: 
            return True 
            
        return False

           
def _tokens_are_close(anchor: dict, candidate: dict) -> bool:
    if abs(anchor["y0"] - candidate["y0"]) > MAX_VERTICAL_DIST: return False
    gap = max(anchor["x0"], candidate["x0"]) - min(anchor["x1"], candidate["x1"])
    return gap <= MAX_HORIZONTAL_DIST


# =============================================================================
# ОСНОВНОЙ ПОИСК
# =============================================================================

def _search_by_surname_only(tokens: list, query: dict) -> list:
    results, seen =[], set()
    search_forms = _get_all_word_forms(query["surname"])

    for tok in tokens:
        if tok["type"] != "word": continue
        if "-" in tok["raw"] and "-" not in query["surname"]: continue

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
    results, seen = [], set()
    needs_name = query["name_initial"] is not None
    needs_patr = query["patronymic_initial"] is not None
    n = len(tokens)
    
    n_init = query["name_initial"] or ""
    p_init = query["patronymic_initial"] or ""
    combined_init = (n_init + p_init).upper()

    for i, tok in enumerate(tokens):
        text_cleaned = tok["text"].replace(".", "").upper()
        is_combined = (needs_name and needs_patr and text_cleaned == combined_init)
        
        if tok["type"] != "initial" and not is_combined: 
            continue

        name_tok, patr_tok, matched = None, None,[]

        if is_combined:
            name_tok = tok
            patr_tok = tok
            matched.append(tok)
        elif needs_name and tok["text"][0].upper() == query["name_initial"].upper():
            name_tok = tok
            matched.append(tok)
            if needs_patr:
                for j in range(i + 1, min(i + 1 + MAX_GAP, n)):
                    t = tokens[j]
                    if t["type"] == "junk": continue
                    if t["type"] != "initial": break
                    if not _tokens_are_close(tok, t): break
                    if t["text"][0].upper() == query["patronymic_initial"].upper():
                        patr_tok = t
                        matched.append(t)
                        break
        elif needs_patr and not needs_name and tok["text"][0].upper() == query["patronymic_initial"].upper():
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
    results, seen = [], set()
    needs_name = query["name_initial"] is not None
    needs_patr = query["patronymic_initial"] is not None

    tokens_by_page = {}
    for tok in tokens:
        tokens_by_page.setdefault(tok["page"],[]).append(tok)

    for i, tok in enumerate(tokens):
        if tok["type"] != "word" or not _surname_matches(tok, query):
            continue

        is_split_double = False
        if i + 1 < len(tokens):
            nxt = tokens[i+1]
            if re.search(r"^[-–—−]", nxt["raw"]):
                if (nxt["x0"] - tok["x1"]) < 40 and abs(nxt["y0"] - tok["y0"]) < 30:
                    is_split_double = True
        if i > 0:
            prv = tokens[i-1]
            if re.search(r"[-–—−]$", prv["raw"]):
                if (tok["x0"] - prv["x1"]) < 40 and abs(tok["y0"] - prv["y0"]) < 30:
                    is_split_double = True

        has_dash_query = bool(re.search(r"[-–—−]", query["surname"])) if query["surname"] else False
        if is_split_double and not has_dash_query:
            continue

        s_tok = tok
        page = s_tok["page"]
        page_tokens = tokens_by_page.get(page,[])
        
        lh = max(s_tok["y1"] - s_tok["y0"], 8)
        MAX_DIST_X = 60 * lh
        MAX_DIST_Y = 25 * lh

        n_cands = []
        p_cands =[]

        if needs_name:
            for t in page_tokens:
                if t == s_tok: continue
                if abs(t["x0"] - s_tok["x0"]) > MAX_DIST_X or abs(t["y0"] - s_tok["y0"]) > MAX_DIST_Y: continue
                if _name_matches(t, query, is_patr=False):
                    n_cands.append(t)

        if needs_patr:
            for t in page_tokens:
                if t == s_tok: continue
                if abs(t["x0"] - s_tok["x0"]) > MAX_DIST_X or abs(t["y0"] - s_tok["y0"]) > MAX_DIST_Y: continue
                if _name_matches(t, query, is_patr=True):
                    p_cands.append(t)

        best_combo = None
        best_score = float('inf')

        def calc_dist(t1, t2):
            cx1, cy1 = (t1["x0"] + t1["x1"]) / 2, (t1["y0"] + t1["y1"]) / 2
            cx2, cy2 = (t2["x0"] + t2["x1"]) / 2, (t2["y0"] + t2["y1"]) / 2
            dx = abs(cx1 - cx2)
            dy = abs(cy1 - cy2)
            if dy < 1.5 * lh: return dx
            return math.hypot(dx, dy * 4) 

        def get_layout_penalty(s, n, p):
            penalty = 0
            s_cx, s_cy = (s["x0"]+s["x1"])/2, (s["y0"]+s["y1"])/2
            if n: n_cx, n_cy = (n["x0"]+n["x1"])/2, (n["y0"]+n["y1"])/2
            if p: p_cx, p_cy = (p["x0"]+p["x1"])/2, (p["y0"]+p["y1"])/2

            if n and p:
                if n == p:
                    # Это склеенный токен (один за двоих)
                    is_horizontal = abs(s_cy - n_cy) < 1.5*lh
                    is_vertical = abs(s_cx - n_cx) < 5*lh
                    if not is_horizontal and not is_vertical: penalty += 50 * lh
                else:
                    is_horizontal = abs(s_cy - n_cy) < 1.5*lh and abs(n_cy - p_cy) < 1.5*lh
                    is_vertical = abs(s_cx - n_cx) < 5*lh and abs(n_cx - p_cx) < 5*lh
                    
                    if is_horizontal:
                        if not (s_cx < n_cx < p_cx or n_cx < p_cx < s_cx): penalty += 500 * lh
                    elif is_vertical:
                        if not (s_cy < n_cy < p_cy or n_cy < p_cy < s_cy): penalty += 500 * lh
                    else:
                        penalty += 100 * lh
            elif n:
                is_horizontal = abs(s_cy - n_cy) < 1.5*lh
                is_vertical = abs(s_cx - n_cx) < 5*lh
                if not is_horizontal and not is_vertical: penalty += 50 * lh
                
            return penalty

        if needs_name and needs_patr:
            for n in n_cands:
                for p in p_cands:
                    if n == p:
                        # Склеенный инициал: допускаем только если это действительно >=2 буквы (например "ПЕ")
                        text_cleaned = n["text"].replace(".", "").upper()
                        if len(text_cleaned) < 2:
                            continue
                    
                    score = calc_dist(s_tok, n) + calc_dist(n, p) + get_layout_penalty(s_tok, n, p)
                    if score < best_score:
                        best_score = score
                        best_combo = (n, p)
        elif needs_name:
            for n in n_cands:
                score = calc_dist(s_tok, n) + get_layout_penalty(s_tok, n, None)
                if score < best_score:
                    best_score = score
                    best_combo = (n, None)
        elif needs_patr:
            for p in p_cands:
                score = calc_dist(s_tok, p) + get_layout_penalty(s_tok, None, p)
                if score < best_score:
                    best_score = score
                    best_combo = (None, p)

        if (needs_name and not needs_patr and not best_combo): continue
        if (needs_patr and not needs_name and not best_combo): continue
        if (needs_name and needs_patr and (not best_combo or not best_combo[0] or not best_combo[1])): continue

        if best_combo and best_score < 1000 * lh:
            matched_tokens = [s_tok]
            if best_combo[0]: matched_tokens.append(best_combo[0])
            # Не добавляем токен второй раз, если он склеенный
            if best_combo[1] and best_combo[1] != best_combo[0]: matched_tokens.append(best_combo[1])

            for m in matched_tokens:
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