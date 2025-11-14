# src/runtime_translate_mapping_only.py
from pathlib import Path
import re, math, pandas as pd

from src.g2p_hu import hu_text_to_ipa

# If PanPhon distance is available in your project:
try:
    from src.phone_mapping import phone_distance     # symmetric IPA phone distance
    HAVE_PANPHON = True
except Exception:
    HAVE_PANPHON = False
    def phone_distance(a,b): return 0.0 if a == b else 1.0

# -------- IPA helpers (same vowel set as unit building) --------
VOWELS = {
    "i","ɪ","e","ɛ","æ","ɑ","a","ɒ","ʌ","ɝ","ə","o","ɔ","ʊ","u","y","ø",
    "eɪ","oʊ","aɪ","aʊ","ɔɪ",
    "i:","e:","a:","o:","u:","y:","ø:",
    "iː","eː","aː","oː","uː","yː","øː"
}
def is_vowel(p): return p in VOWELS

def extract_rhyme_units(phones, max_onset=1, max_coda=1, min_len=2, max_len=4):
    """Vowel-anchored units: [≤1 onset] + nucleus + [≤1 coda], 2–4 phones, exactly 1 vowel."""
    units = []
    i, N = 0, len(phones)
    while i < N:
        j = i
        while j < N and not is_vowel(phones[j]):
            j += 1
        if j >= N:
            break
        onset_start = max(i, j - max_onset)
        nucleus_end = j + 1
        coda_end    = min(N, nucleus_end + max_coda)
        candidates = []
        for start in range(onset_start, j+1):
            for end in range(nucleus_end, coda_end+1):
                seg = tuple(phones[start:end])
                if not (min_len <= len(seg) <= max_len): continue
                if sum(1 for p in seg if is_vowel(p)) != 1: continue
                candidates.append((start, end, seg))
        if not candidates:
            # fallback: vowel+next consonant if possible
            seg = (phones[j],)
            if j+1 < N and not is_vowel(phones[j+1]):
                seg = (phones[j], phones[j+1])
            units.append(tuple(seg))
            i = j + len(seg)
            continue
        start, end, seg = sorted(candidates, key=lambda se: (len(se[2]), se[0]))[0]
        units.append(seg)
        i = end
    return units

# -------- HU–HU unit distance for nearest-neighbour backoff --------
def hu_unit_distance(u: tuple[str,...], v: tuple[str,...], len_pen: float = 0.3, cv_pen: float = 0.6) -> float:
    """Distance between two HU units without using EN proxies.
       Align by position; add penalties for length mismatch and C/V mismatches."""
    m = min(len(u), len(v))
    if m == 0:
        return 2.0 + len_pen * abs(len(u) - len(v))
    base = 0.0
    for i in range(m):
        base += phone_distance(u[i], v[i]) + (cv_pen if (is_vowel(u[i]) != is_vowel(v[i])) else 0.0)
    base /= m
    return base + len_pen * abs(len(u) - len(v))

# -------- Load mapping: {HU_unit_tuple -> EN word} --------
def load_mapping(mapping_csv: str | Path) -> dict[tuple[str,...], str]:
    df = pd.read_csv(mapping_csv)
    mp = {}
    for _, row in df.iterrows():
        hu = tuple(str(row["hu_unit"]).split())
        en = str(row["en_word"]).strip()
        if en:
            mp[hu] = en
    return mp

# -------- Nearest neighbour over HU units already in the mapping --------
def nearest_from_mapping(hu_unit: tuple[str,...], mapping: dict[tuple[str,...], str],
                         max_len_diff: int = 1, max_cost: float = 1.6) -> str | None:
    best_w, best_c = None, float("inf")
    for key, word in mapping.items():
        if abs(len(key) - len(hu_unit)) > max_len_diff:
            continue
        c = hu_unit_distance(hu_unit, key)
        if c < best_c:
            best_c, best_w = c, word
    return best_w if (best_w and best_c <= max_cost) else None

# -------- Public API --------
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def translate(text: str, mapping_csv: str | Path,
              max_len_diff: int = 1, max_cost: float = 1.6,
              unk_token: str = "uh") -> str:
    """
    HU text -> English homophonic output using ONLY unit_to_word_compact.csv.
    - Keeps punctuation.
    - For unseen HU units, nearest-neighbour over existing HU keys in the mapping.
    """
    mp = load_mapping(mapping_csv)
    hu_keys = set(mp.keys())  # for quick exact hits

    out_tokens = []
    for tok in WORD_RE.findall(text):
        if not tok.strip() or not tok.isalpha():
            out_tokens.append(tok)
            continue

        phones = [p for p in hu_text_to_ipa(tok) if isinstance(p, str) and p]
        units = extract_rhyme_units(phones, max_onset=1, max_coda=1, min_len=2, max_len=4)

        words = []
        for u in units:
            if u in hu_keys:
                words.append(mp[u])
            else:
                w = nearest_from_mapping(u, mp, max_len_diff=max_len_diff, max_cost=max_cost)
                words.append(w if w else unk_token)
        out_tokens.append(" ".join(words))

    out = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(out_tokens))
    out = re.sub(r"\(\s+", "(", out); out = re.sub(r"\s+\)", ")", out)
    return out
