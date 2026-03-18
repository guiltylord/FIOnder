"""
Поиск ФИО в тексте, извлечённом OCR.

ПУБЛИЧНЫЙ API:
    search_in_text(words_with_coords, search_terms) -> list[dict]
"""

import re
import threading
import time

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

MAX_GAP             = 6
MAX_VERTICAL_DIST   = 80
MAX_HORIZONTAL_DIST = 400

# =============================================================================
# МОРФОЛОГИЯ
# =============================================================================

try:
    from pymorphy3 import MorphAnalyzer as _MorphAnalyzer
    _morph       = _MorphAnalyzer()
    _morph_lock  = threading.Lock()
    _norm_cache  = {}
    _forms_cache = {}
    _USE_PYMORPHY = True
    print("[search] Используется pymorphy3")
except ImportError:
    _USE_PYMORPHY = False
    print("[search] Используются встроенные правила (pymorphy3 не найден)")

_RULES = [
    ("СКОГО","СКИЙ"),("СКОМУ","СКИЙ"),("СКИМ","СКИЙ"),("ЦКОГО","ЦКИЙ"),
    ("ЦКОМУ","ЦКИЙ"),("ЦКИМ","ЦКИЙ"),("СКОЙ","СКИЙ"),("ЦКОЙ","ЦКИЙ"),
    ("ОВЫМ","ОВ"),("ЕВЫМ","ЕВ"),("ОВОМ","ОВ"),("ЕВОМ","ЕВ"),
    ("ОВОГО","ОВ"),("ЕВОГО","ЕВ"),("ОВОМУ","ОВ"),("ЕВОМУ","ЕВ"),
    ("ОВЫХ","ОВ"),("ЕВЫХ","ЕВ"),("ОВОЙ","ОВ"),("ЕВОЙ","ЕВ"),
    ("ОВУ","ОВ"),("ЕВУ","ЕВ"),("ОВЕ","ОВ"),("ЕВЕ","ЕВ"),
    ("ОВЫ","ОВ"),("ЕВЫ","ЕВ"),("ОВА","ОВ"),("ЕВА","ЕВ"),
    ("ИНЫМ","ИН"),("ИНОГО","ИН"),("ИНОМУ","ИН"),("ИНЫХ","ИН"),
    ("ИНОЙ","ИН"),("ИНУ","ИН"),("ИНЕ","ИН"),("ИНЫ","ИН"),("ИНА","ИН"),
]


def normalize_word(word: str) -> str:
    key = word.strip().upper()
    if not key:
        return ""
    if _USE_PYMORPHY:
        if key not in _norm_cache:
            with _morph_lock:
                parsed = _morph.parse(word.lower())
            for p in parsed:
                if "Surn" in str(p.tag):
                    _norm_cache[key] = p.normal_form.upper()
                    break
            else:
                _norm_cache[key] = parsed[0].normal_form.upper()
        return _norm_cache[key]
    for suffix, repl in _RULES:
        if key.endswith(suffix) and len(key) - len(suffix) >= 3:
            return key[:-len(suffix)] + repl
    return key


def _all_forms(word: str) -> frozenset:
    key = "F:" + word.strip().upper()
    if not _USE_PYMORPHY:
        return frozenset([normalize_word(word)])
    if key not in _forms_cache:
        with _morph_lock:
            parsed = _morph.parse(word.lower())
        _forms_cache[key] = frozenset(f.word.upper() for p in parsed for f in p.lexeme)
    return _forms_cache[key]


# =============================================================================
# ПОДГОТОВКА ТОКЕНОВ
# =============================================================================

def _is_initial(t: str) -> bool:
    return bool(re.fullmatch(r"[А-ЯЁA-Z]\.?", t.upper()))

def _is_word(t: str) -> bool:
    return bool(re.fullmatch(r"[А-ЯЁа-яёA-Za-z]+(?:-[А-ЯЁа-яёA-Za-z]+)?", t))


def prepare_tokens(words: list) -> list:
    # 1. Разворачиваем склейки («Т.В.Соколова» → «Т.», «В.», «Соколова»)
    expanded = []
    for w in words:
        text = w["text"].strip()
        if not text:
            continue
        s = re.sub(r"([А-ЯЁA-Z]\.)(?=[А-ЯЁA-Z]\.)", r"\1 ", text)
        s = re.sub(r"(?<=\.)(?=[А-ЯЁа-яёA-Za-z])", " ", s)
        s = re.sub(r"(?<=[А-ЯЁа-яёA-Za-z]{2})(?=[А-ЯЁA-Z]\.)", " ", s)
        parts = s.split()
        for p in parts:
            expanded.append({**w, "text": p})

    # 2. Очистка пунктуации и нумерации
    raw = []
    for i, w in enumerate(expanded):
        text = w["text"].strip()
        if not text:
            continue
        m = re.match(r"^\d+[.)\s]+", text)
        pfx = m.end() if m else 0
        text = text[pfx:]
        text = re.sub(r"^[^\wА-Яа-яA-Za-z]+", "", text)
        text = re.sub(r"[^\wА-Яа-яA-Za-z.]+$", "", text)
        if text.endswith(".") and not re.fullmatch(r"([А-Яа-яA-Za-z]\.)+", text):
            text = text[:-1]
        if not text:
            continue
        x0 = w["x0"] + (w["x1"] - w["x0"]) / max(len(w["text"]), 1) * pfx if pfx else w["x0"]
        raw.append({**w, "text": text, "x0": x0, "idx": i})

    # 3. Склейка переносов через дефис
    merged, skip = [], False
    for k, rw in enumerate(raw):
        if skip:
            skip = False
            continue
        if not rw["text"].endswith("-"):
            merged.append(rw)
            continue
        base = rw["text"][:-1]
        joined = False
        for j in range(k + 1, min(k + 5, len(raw))):
            nxt = raw[j]
            if not _is_word(nxt["text"]) or not _is_word(base):
                break
            lh = rw["y1"] - rw["y0"]
            yd = nxt["y0"] - rw["y0"]
            if j == k + 1 or (0 < yd < lh * 3 and abs(nxt["x0"] - rw["x0"]) < 100):
                merged.append({
                    **rw, "text": base + nxt["text"],
                    "x1": max(rw["x1"], nxt["x1"]),
                    "y1": max(rw["y1"], nxt["y1"]),
                    "parts": [
                        {"x0": rw["x0"],  "y0": rw["y0"],  "x1": rw["x1"],  "y1": rw["y1"]},
                        {"x0": nxt["x0"], "y0": nxt["y0"], "x1": nxt["x1"], "y1": nxt["y1"]},
                    ],
                })
                skip = (j == k + 1)
                joined = True
                break
            if yd > lh * 3:
                break
        if not joined:
            merged.append(rw)

    # 4. Классификация
    tokens = []
    for rw in merged:
        if not rw["text"]:
            continue
        t = rw["text"].upper()
        ttype = "initial" if _is_initial(t) else ("word" if _is_word(t) else "junk")
        base = {k: rw[k] for k in ("page", "x0", "y0", "x1", "y1", "idx")}
        if "parts" in rw:
            base["parts"] = rw["parts"]
        tokens.append({**base, "type": ttype, "text": t, "raw": rw["text"]})
    return tokens


# =============================================================================
# ПАРСИНГ ЗАПРОСА
# =============================================================================

def parse_query(query: str) -> dict:
    query = re.sub(r"([А-ЯЁа-яё]\.)([А-ЯЁа-яё]\.?)", r"\1 \2", query)
    parts = re.findall(r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?|[А-ЯЁ]\.", query)
    words, initials = [], []
    for p in parts:
        (initials if _is_initial(p) else words).append(p.upper() if not _is_initial(p) else p[0].upper())

    if not words:
        return {k: None for k in ("surname","surname_norm","name_initial","name_full","patronymic_initial","patronymic_full")}

    patr_sfx = ("ОВИЧ","ЕВИЧ","ИЧ","ОВНА","ЕВНА","ИЧНА","ИНИЧНА")
    surn_sfx = ("ОВ","ОВА","ЕВ","ЕВА","ИН","ИНА","СКИЙ","СКАЯ","ЦКИЙ","ЦКАЯ")
    best, best_s = 0, -1
    for idx, w in enumerate(words):
        s = 10 if w.endswith(surn_sfx) else 0
        if _USE_PYMORPHY:
            with _morph_lock:
                parsed = _morph.parse(w.lower())
            if any("Surn" in str(p.tag) for p in parsed): s += 5
            if any("Name" in str(p.tag) for p in parsed): s -= 5
        if idx == 0 and len(words) == 3 and any(words[1].endswith(x) for x in patr_sfx):
            s += 20
        if s > best_s:
            best_s, best = s, idx

    surname = words[best]
    rest    = [w for i, w in enumerate(words) if i != best]
    nf = rest[0] if rest else None
    pf = rest[1] if len(rest) >= 2 else None
    return {
        "surname":            surname,
        "surname_norm":       normalize_word(surname),
        "name_initial":       nf[0] if nf else (initials[0] if initials else None),
        "name_full":          nf,
        "patronymic_initial": pf[0] if pf else (initials[1] if len(initials) >= 2 else None),
        "patronymic_full":    pf,
    }


# =============================================================================
# СОПОСТАВЛЕНИЕ
# =============================================================================

_STOP = {"ФАМИЛИЯ","ИМЯ","ОТЧЕСТВО","ПОДПИСЬ","ДАТА","ДОЛЖНОСТЬ","М.П.","МП","РУКОВОДИТЕЛЬ","ДИРЕКТОР"}

def _surname_matches(tok: dict, q: dict) -> bool:
    if not q["surname_norm"]: return False
    if "-" in tok["text"] and "-" not in q["surname"]: return False
    if normalize_word(tok["text"]) == q["surname_norm"]: return True
    return bool(_all_forms(tok["text"]) & _all_forms(q["surname"]))

def _name_matches(tok: dict, initial, full) -> bool:
    if initial is None or tok["text"][0].upper() != initial: return False
    if tok["type"] == "initial": return True
    if full: return normalize_word(tok["text"]) == normalize_word(full)
    return tok["type"] == "word" and tok["text"] not in _STOP

def _close(a: dict, b: dict) -> bool:
    if abs(a["y0"] - b["y0"]) > MAX_VERTICAL_DIST: return False
    return max(a["x0"], b["x0"]) - min(a["x1"], b["x1"]) <= MAX_HORIZONTAL_DIST

def _build_result(tok: dict, term: str) -> list:
    parts = tok.get("parts") or [{"x0": tok["x0"], "y0": tok["y0"], "x1": tok["x1"], "y1": tok["y1"]}]
    return [{"search_term": term, "found_text": tok["raw"] if i == 0 else "",
             "page": tok["page"], **p} for i, p in enumerate(parts)]


# =============================================================================
# АЛГОРИТМЫ ПОИСКА
# =============================================================================

def _search_fio(tokens, q):
    results, seen = [], set()
    needs_n, needs_p = q["name_initial"] is not None, q["patronymic_initial"] is not None
    n, raw = len(tokens), q.get("_raw", "")

    for i, tok in enumerate(tokens):
        if tok["type"] != "word" or not _surname_matches(tok, q):
            continue
        name_tok = patr_tok = None
        matched  = [tok]

        for direction, start in [(+1, i + 1), (-1, i - 1)]:
            j, steps = start, 0
            while 0 <= j < n and steps < MAX_GAP * 2:
                t = tokens[j]; j += direction; steps += 1
                if t["page"] != tok["page"]: break
                if t["type"] == "junk" or not _close(tok, t): continue
                if needs_n and name_tok is None and _name_matches(t, q["name_initial"], q["name_full"]):
                    name_tok = t; matched.append(t)
                elif needs_p and patr_tok is None and _name_matches(t, q["patronymic_initial"], q["patronymic_full"]):
                    patr_tok = t; matched.append(t)
            if (not needs_n or name_tok) and (not needs_p or patr_tok):
                break

        if (needs_n and not name_tok) or (needs_p and not patr_tok):
            continue
        for m in matched:
            for e in _build_result(m, raw):
                key = (e["page"], round(e["x0"], 1), round(e["y0"], 1))
                if key not in seen:
                    seen.add(key); results.append(e)
    return results


def _search_surname(tokens, q):
    results, seen, raw = [], set(), q.get("_raw", "")
    for tok in tokens:
        if tok["type"] != "word" or not _surname_matches(tok, q): continue
        for e in _build_result(tok, raw):
            key = (e["page"], round(e["x0"], 1), round(e["y0"], 1))
            if key not in seen:
                seen.add(key); results.append(e)
    return results


def _search_initials(tokens, q):
    results, seen = [], set()
    needs_n, needs_p = q["name_initial"] is not None, q["patronymic_initial"] is not None
    n, raw = len(tokens), q.get("_raw", "")

    for i, tok in enumerate(tokens):
        if tok["type"] != "initial": continue
        name_tok = patr_tok = None; matched = []
        if needs_n and tok["text"][0] == q["name_initial"]:
            name_tok = tok; matched.append(tok)
            if needs_p:
                for j in range(i + 1, min(i + 1 + MAX_GAP, n)):
                    t = tokens[j]
                    if t["type"] == "junk": continue
                    if t["type"] != "initial" or not _close(tok, t): break
                    if t["text"][0] == q["patronymic_initial"]:
                        patr_tok = t; matched.append(t); break
        elif needs_p and not needs_n and tok["text"][0] == q["patronymic_initial"]:
            patr_tok = tok; matched.append(tok)

        if not ((not needs_n or name_tok) and (not needs_p or patr_tok)): continue
        for m in matched:
            for e in _build_result(m, raw):
                key = (e["page"], round(e["x0"], 1), round(e["y0"], 1))
                if key not in seen:
                    seen.add(key); results.append(e)
    return results


# =============================================================================
# ПУБЛИЧНЫЙ API
# =============================================================================

def search_in_text(words_with_coords: list, search_terms) -> list:
    start = time.time()
    terms = [t.strip() for t in (search_terms.split(",") if isinstance(search_terms, str) else search_terms) if t.strip()]
    tokens = prepare_tokens(words_with_coords)
    found  = []

    for term in terms:
        q = parse_query(term)
        q["_raw"] = term
        has_s = q["surname"] is not None
        has_i = q["name_initial"] is not None or q["patronymic_initial"] is not None

        if has_s and has_i:   found.extend(_search_fio(tokens, q))
        elif has_s:            found.extend(_search_surname(tokens, q))
        elif has_i:            found.extend(_search_initials(tokens, q))

    print(f"[TIME] Search: {time.time() - start:.2f}с | запросов: {len(terms)} | найдено: {len(found)}")
    return found