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
    word = word.strip().upper()
    if not word: return ""
    
    # Если фамилия уже заканчивается на стандартные ОВ, ЕВ, ИН, 
    # то, скорее всего, она уже в нормальной форме. Не трогаем её.
    if word.endswith(("ОВ", "ЕВ", "ИН", "СКИЙ", "ЦКИЙ")):
        return word

    if _USE_PYMORPHY:
        with _morph_lock:
            parsed = _morph.parse(word.lower())
        # Ищем вариант, который является фамилией
        for p in parsed:
            if "Surn" in p.tag:
                return p.normal_form.upper()
        return parsed[0].normal_form.upper()

    # Fallback на правила, если нет pymorphy
    for suffix, replacement in _RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def _all_surname_forms(word: str) -> frozenset:
    word = word.strip().upper()
    if not word: return frozenset()

    # Если есть pymorphy3, просто собираем ВСЕ падежи этого слова
    if _USE_PYMORPHY:
        cache_key = "SURN_FORMS:" + word
        if cache_key not in _forms_cache:
            with _morph_lock:
                parsed = _morph.parse(word.lower())
            all_forms = set()
            for p in parsed:
                # Берем все формы слова (склонения)
                for form in p.lexeme:
                    all_forms.add(form.word.upper())
            _forms_cache[cache_key] = frozenset(all_forms)
        return _forms_cache[cache_key]

    # ЕСЛИ БЕЗ PYMORPHY (Умный Fallback для женских и мужских фамилий)
    forms = {word}
    
    # Мужские: ОВ, ЕВ, ИН
    if word.endswith(("ОВ", "ЕВ", "ИН")):
        for ending in ["А", "У", "ОМ", "Е", "ЫМ", "ОГО", "ОМУ", "ЫХ"]: 
            forms.add(word + ending)
    # Женские: ОВА, ЕВА, ИНА
    elif word.endswith(("ОВА", "ЕВА", "ИНА")):
        base = word[:-1] # Убираем только 'А' для склонения женского рода
        for ending in ["ОЙ", "У", "Е", "ОЮ"]: 
            forms.add(base + ending)
    # Мужские СКИЙ / ЦКИЙ
    elif word.endswith(("СКИЙ", "ЦКИЙ")):
        base = word[:-2] 
        for ending in ["ОГО", "ОМУ", "ИМ", "ОМ", "ИЕ", "ИХ"]: 
            forms.add(base + ending)
    # Женские СКАЯ / ЦКАЯ
    elif word.endswith(("СКАЯ", "ЦКАЯ")):
        base = word[:-2] 
        for ending in ["ОЙ", "УЮ"]: 
            forms.add(base + ending)

    return frozenset(forms)


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
    # Очищаем базовый мусор из запроса
    q = query.replace(",", " ").replace(":", ".").replace(";", " ")
    q = re.sub(r"[–—−]", "-", q)
    
    tokens = q.split()
    words = []
    initials = []
    
    for t in tokens:
        # Если инициалы ввели как А.Б.
        m_dots = re.fullmatch(r"([А-ЯЁA-Zа-яёa-z])\.([А-ЯЁA-Zа-яёa-z])\.?", t)
        if m_dots:
            initials.extend([m_dots.group(1).upper(), m_dots.group(2).upper()])
            continue
            
        t_clean = re.sub(r"[^А-ЯЁа-яёA-Za-z\-]", "", t)
        if not t_clean: continue
            
        if len(t_clean) == 1:
            initials.append(t_clean.upper())
        elif len(t_clean) == 2 and t.isupper(): 
            # Если пользователь ввел капсом "ЮГ" или "ТВ" — это 100% инициалы
            _STOP = {"ЛИ", "ЯН", "АН", "УК", "КЮ", "ДО", "ИЗ", "ЗА", "НА", "ОТ", "ПО", "НЕ", "НИ"}
            if t_clean.upper() not in _STOP:
                initials.extend([t_clean[0].upper(), t_clean[1].upper()])
            else:
                words.append(t_clean)
        else:
            words.append(t_clean)
            
    # Скоринг для поиска фамилии
    best_surn_idx = 0
    max_score = -1

    patr_suffixes = ("ОВИЧ", "ЕВИЧ", "ИЧ", "ОВНА", "ЕВНА", "ИЧНА", "ИНИЧНА")
    surn_suffixes = ("ОВ", "ОВА", "ЕВ", "ЕВА", "ИН", "ИНА", "СКИЙ", "СКАЯ", "ЦКИЙ", "ЦКАЯ")

    if not words:
        return {"surname": None, "name_initial": None, "name_full": None, "patronymic_initial": None, "patronymic_full": None}

    for idx, w in enumerate(words):
        score = 0
        w_up = w.upper()
        if w_up.endswith(surn_suffixes): score += 10
        if _USE_PYMORPHY:
            with _morph_lock:
                parsed = _morph.parse(w.lower())
                if any("Surn" in p.tag for p in parsed): score += 5
                if any("Name" in p.tag for p in parsed): score -= 5
        
        if idx == 2 and len(words) == 3:
            if any(words[1].upper().endswith(s) for s in patr_suffixes): score += 20
        
        if score > max_score:
            max_score = score
            best_surn_idx = idx

    surname = words[best_surn_idx]
    remaining_words = [w for i, w in enumerate(words) if i != best_surn_idx]
    
    name_full = remaining_words[0] if len(remaining_words) >= 1 else None
    patr_full = remaining_words[1] if len(remaining_words) >= 2 else None

    name_initial = name_full[0].upper() if name_full else (initials[0] if len(initials) >= 1 else None)
    patr_initial = patr_full[0].upper() if patr_full else (initials[1] if len(initials) >= 2 else None)

    return {
        "surname": surname,
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
    # НОВОЕ: нормализуем двоеточие как точку для инициалов: Ю:Г: → Ю.Г.
    text = re.sub(r'([А-ЯЁA-Zа-яёa-z]):', r'\1.', text)

    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"^[^\wА-ЯЁа-яёA-Za-z\-]+", "", text)
    text = re.sub(r"[^\wА-ЯЁа-яёA-Za-z\.\-]+$", "", text)
    if text.endswith("."):
        if not re.fullmatch(r"([А-ЯЁа-яёA-Za-z]\.)+", text):
            text = text[:-1]
            text = re.sub(r"[^\wА-ЯЁа-яёA-Za-z\-]+$", "", text)
    return text

def prepare_tokens(words_with_coords: list) -> list:
    expanded_words = []
    
    # 1. РАСКЛЕЙКА СЛОЖНЫХ СЛУЧАЕВ
    for w in words_with_coords:
        text = re.sub(r"[–—−]", "-", w["text"].strip())
        if not text: continue
        
        # 1.1 Замена двоеточий и мусора между инициалами (Ю:Г: -> Ю.Г.)
        text = re.sub(r'([А-ЯЁA-Zа-яёa-z]):', r'\1.', text)
        
        # 1.2 Расклейка слипшихся спереди (ТВСоколова -> ТВ Соколова)
        text = re.sub(r'^([А-ЯЁA-Z]{2})([А-ЯЁA-Z][а-яёa-z]+)', r'\1 \2', text)
        
        # 1.3 Расклейка слипшихся сзади (АрхиповаЮГ -> Архипова ЮГ)
        text = re.sub(r'([А-ЯЁA-Z][а-яёa-z]+)([А-ЯЁA-Z]{2})$', r'\1 \2', text)

        # 1.4 Расклейка точек (Т.В.Соколова -> Т. В. Соколова)
        text = re.sub(r'([А-ЯЁA-Z]\.)(?=[А-ЯЁA-Z]\.)', r'\1 ', text)
        text = re.sub(r'(?<=\.)(?=[А-ЯЁа-яёA-Za-z])', ' ', text)
        
        # 1.5 Раздробление голых инициалов ("ЮГ" -> "Ю" "Г")
        _STOP = {"НА","ПО","ОТ","ИЗ","ДО","ЗА","ПРИ","НЕ","НИ","МЫ","ОН","ОНА","ОНИ","ИЛИ", "ФИО", "МП", "ИП"}
        
        parts = text.split()
        final_parts = []
        for p in parts:
            p_clean = re.sub(r"[^А-ЯЁа-яёA-Za-z]", "", p.upper())
            # Если это ровно 2-3 заглавные буквы и не предлог
            if len(p_clean) in (2, 3) and p.isupper() and p_clean not in _STOP:
                final_parts.extend(list(p_clean)) # "ЮГ" -> ["Ю", "Г"]
            else:
                final_parts.append(p)
                
        # Если слово было разбито на куски, распределяем координаты
        if len(final_parts) > 1:
            char_width = (w["x1"] - w["x0"]) / max(len(w["text"]), 1)
            curr_x = w["x0"]
            for p in final_parts:
                p_len = len(p)
                expanded_words.append({
                    "text": p, "raw": p, 
                    "page": w["page"], 
                    "x0": curr_x, "y0": w["y0"], 
                    "x1": curr_x + (char_width * p_len), "y1": w["y1"]
                })
                # Добавляем примерную ширину пробела
                curr_x += (char_width * p_len) + char_width 
        else:
            expanded_words.append({**w, "text": text, "raw": w["text"]})

    raw_tokens = []
    for i, w in enumerate(expanded_words):
        original = w["text"].strip()
        if not original: continue

        cleaned, prefix_len = _strip_numbering(original)
        if not cleaned: continue

        ends_with_dash = bool(re.search(r"-$", cleaned))
        
        # strip_punctuation обрезает любой мусор с краев слова, 
        # оставляя только чистые буквы.
        cleaned = _strip_punctuation(cleaned)
        if not cleaned: continue

        if ends_with_dash and cleaned and not cleaned.endswith("-"):
            cleaned = cleaned + "-"

        raw_tokens.append({
            "text": cleaned, "raw": w["raw"], "page": w["page"],
            "x0": w["x0"], "y0": w["y0"], "x1": w["x1"], "y1": w["y1"], "idx": i,
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
    t_text = token["text"].upper()
    q_text = query["surname"].upper()

    if not q_text: 
        return False

    # 1. Проверка на наличие дефиса (чтобы "Салтыков" не сматчился с "Салтыков-Щедрин")
    has_dash_token = bool(re.search(r"[-–—−]", token["raw"]))
    has_dash_query = bool(re.search(r"[-–—−]", query["surname"]))
    if has_dash_token and not has_dash_query:
        return False

    # 2. Идеальное точное совпадение (самый быстрый путь)
    if t_text == q_text:
        return True
        
    # 3. Умное сравнение через облако падежей
    # Берем все падежи слова из текста и все падежи слова из запроса
    token_forms = _all_surname_forms(t_text)
    query_forms = _all_surname_forms(q_text)
    
    # Если множества пересекаются (нашлась общая форма) — это 100% совпадение
    if bool(token_forms & query_forms):
        return True

    return False

def _name_matches(token: dict, initial: str, full: str, is_patr: bool = False) -> bool:
    """
    Умная проверка имени/отчества, которая позволяет:
    1. Искать полные имена ("Эдуард"), если в запросе был только инициал ("Э.").
    2. Искать инициалы ("Э."), если в запросе было полное имя ("Эдуард").
    """
    if initial is None: return False
    text = token["text"]
    if not text: return False
    
    # Первая буква всегда должна совпадать
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
            
        # Если нет pymorphy, то разрешаем слово (надеемся, что 2D-Scoring отсеет мусор)
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
    """
    Улучшенный алгоритм поиска ФИО с системой умных бонусов (Smart Scoring).
    Приоритезирует токены с пунктуацией и те, что находятся на одной горизонтали.
    """
    results, seen = [], set()
    needs_name = query["name_initial"] is not None
    needs_patr = query["patronymic_initial"] is not None

    # Группируем токены по страницам
    tokens_by_page = {}
    for tok in tokens:
        tokens_by_page.setdefault(tok["page"], []).append(tok)

    # Проходим по всем токенам в поисках Фамилии
    for i, tok in enumerate(tokens):
        if tok["type"] != "word" or not _surname_matches(tok, query):
            continue

        s_tok = tok
        page_tokens = tokens_by_page.get(s_tok["page"], [])
        
        # Высота буквы фамилии - базовая единица измерения
        lh = max(s_tok["y1"] - s_tok["y0"], 8)

        # Лимиты поиска (широко по горизонтали, узко по вертикали)
        LIMIT_X = 100 * lh 
        LIMIT_Y = 4 * lh 

        n_cands = []
        p_cands = []

        # Собираем кандидатов в радиусе
        for t in page_tokens:
            if t["idx"] == s_tok["idx"]: continue
            
            dx = abs(t["x0"] - s_tok["x0"])
            dy = abs(t["y0"] - s_tok["y0"])

            if dy > LIMIT_Y or dx > LIMIT_X:
                continue

            if needs_name and _name_matches(t, query["name_initial"], query["name_full"], is_patr=False):
                n_cands.append(t)
            
            if needs_patr and _name_matches(t, query["patronymic_initial"], query["patronymic_full"], is_patr=True):
                p_cands.append(t)

        best_combo = None
        best_score = float('inf')

        def calc_weighted_dist(t1, t2):
            """Расстояние, где вертикальный сдвиг в 2 раза 'дороже'."""
            dx = abs(((t1["x0"] + t1["x1"]) / 2) - ((t2["x0"] + t2["x1"]) / 2))
            dy = abs(((t1["y0"] + t1["y1"]) / 2) - ((t2["y0"] + t2["y1"]) / 2))
            return math.hypot(dx, dy * 2)

        def get_smart_score(s, n, p=None):
            """Система бонусов: чем меньше итоговый score, тем лучше кандидат."""
            # 1. Базовая геометрия
            score = calc_weighted_dist(s, n)
            if p: score += calc_weighted_dist(n, p)

            # 2. Бонус за точки (И. vs И)
            # Настоящий инициал с точкой получает огромную скидку
            if "." in n["raw"]: score -= 35 * lh
            if p and "." in p["raw"]: score -= 35 * lh
            
            # 3. Бонус за одну строку
            # Если Имя на одной линии с Фамилией
            if abs(s["y0"] - n["y0"]) < 1.2 * lh: score -= 20 * lh
            # Если Отчество на одной линии с Именем
            if p and abs(n["y0"] - p["y0"]) < 1.2 * lh: score -= 20 * lh

            # 4. Штраф за "мусорность"
            # Если токен слишком длинный для инициала, но не полное имя (ошибка расклейки)
            if n["type"] == "initial" and len(n["text"]) > 1: score += 10 * lh
            
            return score

        # Поиск лучшей связки (Имя + Отчество)
        if needs_name and needs_patr:
            for n in n_cands:
                for p in p_cands:
                    if n["idx"] == p["idx"]: continue
                    score = get_smart_score(s_tok, n, p)
                    if score < best_score:
                        best_score = score
                        best_combo = (n, p)
        
        # Поиск лучшей связки (Только Имя)
        elif needs_name:
            for n in n_cands:
                score = get_smart_score(s_tok, n)
                if score < best_score:
                    best_score = score
                    best_combo = (n, None)
        
        # Поиск лучшей связки (Только Отчество)
        elif needs_patr:
            for p in p_cands:
                score = get_smart_score(s_tok, p)
                if score < best_score:
                    best_score = score
                    best_combo = (None, p)

        # Валидация: проверяем, что нашли всё, что просил пользователь
        if needs_name and needs_patr:
            if not best_combo or not best_combo[0] or not best_combo[1]: continue
        elif needs_name:
            if not best_combo or not best_combo[0]: continue
        elif needs_patr:
            if not best_combo or not best_combo[1]: continue

        # Сохранение результатов
        if best_combo:
            matched = [s_tok]
            if best_combo[0]: matched.append(best_combo[0])
            if best_combo[1]: matched.append(best_combo[1])

            for m in matched:
                # Округляем координаты для исключения дубликатов в seen
                res_key = (m["page"], round(m["x0"], 1), round(m["y0"], 1))
                if res_key not in seen:
                    seen.add(res_key)
                    results.append({
                        "search_term": query.get("_raw", ""), 
                        "found_text": m["raw"],
                        "page": m["page"], 
                        "x0": m["x0"], "y0": m["y0"], 
                        "x1": m["x1"], "y1": m["y1"],
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