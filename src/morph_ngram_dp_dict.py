# src/morph_ngram_dp_dict.py
# ---------------------------------------------------------
# Build a HU-morph n-gram → EN-word dictionary using enPhon and your pure-DP search (search_dp.py)
# ---------------------------------------------------------
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Sequence
import re
import shlex
import subprocess
import pandas as pd
from wordfreq import top_n_list
from collections import Counter, defaultdict

# --- enPhon transcriber (uploaded in your repo) ---
from transcriber import Transcriber

# --- CMUdict loader (DO NOT wrap min_zipf; only use max_words as requested) ---
from .cmudict_utils import load_cmudict

# --- Your existing DP search utilities (pure per-phone cost) ---
# We assume search_dp.py exposes seq_cost_dp(en_seq, hu_seq, ins, dele) and phone_cost(en_p, hu_p)
from .search_dp import _edit_distance, _sub_cost_en_vs_hu
from .morph_splitting import refine_morph_units


# =========================
# 1) enPhon: segment to morphs + get IPA phones
# =========================

def _run(cmd: str, text: str) -> List[str]:
    p = subprocess.run(
        shlex.split(cmd),
        input=(text if text.endswith("\n") else text + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return p.stdout.decode("utf-8", errors="replace").splitlines()

def _emtsv_rows(text: str, emtsv_exec: str = "emtsv") -> List[Dict[str, str]]:
    tries = [
        f"{emtsv_exec} tok,emMorph --output-header",
        f"{emtsv_exec} emMorph --output-header",
        f"{str(__import__('os').path.expanduser('~/.local/bin/emtsv'))} tok,emMorph --output-header",
        f"{str(__import__('os').path.expanduser('~/.local/bin/emtsv'))} emMorph --output-header",
    ]
    last = None
    for cmd in tries:
        try:
            lines = _run(cmd, text)
            head = [h.strip() for h in lines[0].split("\t")]
            return [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln.strip()]
        except Exception as e:
            last = e
    raise RuntimeError(f"emtsv failed; last error: {last}")

def segment_morphs_enphon(text: str, emtsv_exec: str = "emtsv") -> List[str]:
    """
    emMorph segmentation + enPhon phonology (inner) with boundary markers,
    then split by markers to get 'finest' morphs (phono-aware).
    """
    tr_inner = Transcriber(ipaize=False)
    rows = _emtsv_rows(text, emtsv_exec=emtsv_exec)
    seg = tr_inner.segment(rows, {"form": "form", "anas": "anas"})    # inserts | # § ~ markers
    inner = Transcriber(ipaize=False)("".join(seg))                    # apply phonology, keep markers
    morphs: List[str] = []
    for w in inner.split():
        for p in re.split(r"[|#§~]+", w):
            p = p.strip()
            if p:
                morphs.append(p)
    return morphs

# Tokenize enPhon IPA robustly (no reliance on spaces)
_IPA_MULTI = ["tʃ", "dʒ", "eɪ", "oʊ", "aɪ", "aʊ", "ɔɪ", "juː", "iː", "uː", "ɔː", "ɑː", "ɜː"]
_IPA_MULTI.sort(key=len, reverse=True)

def _tokenize_ipa(s: str) -> List[str]:
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1; continue
        if ch == "ː":  # attach length to previous symbol
            if out: out[-1] += "ː"
            i += 1; continue
        matched = False
        for m in _IPA_MULTI:
            if s.startswith(m, i):
                out.append(m); i += len(m); matched = True; break
        if matched: continue
        out.append(ch); i += 1
    return out

_tr_ipa = Transcriber(ipaize=True)

def morphs_to_ipa_phones(morphs: Sequence[str]) -> List[List[str]]:
    """
    enPhon IPA for each morph, tokenized into phones.
    """
    phones_per_morph: List[List[str]] = []
    for m in morphs:
        s = _tr_ipa(m)               # inner → IPA string
        toks = _tokenize_ipa(s)
        if toks:
            phones_per_morph.append(toks)
    return phones_per_morph


# =========================
# 2) Build all 1/2/3-grams from morphs
# =========================

def ngrams_max3(morphs: Sequence[str]) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """
    Returns list of tuples: (morph_ngram, phones_ngram), where phones_ngram is
    the concatenated IPA phones for the morph sequence.
    """
    m2p = morphs_to_ipa_phones(morphs)
    out: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
    for i in range(len(morphs)):
        for n in (1, 2, 3):
            j = i + n
            if j > len(morphs): break
            morph_ng = tuple(morphs[i:j])
            phones_ng: List[str] = []
            for k in range(i, j):
                phones_ng.extend(m2p[k])
            out.append((morph_ng, tuple(phones_ng)))
    return out


# =========================
# 3) Load CMUdict top max_words and build a flat (word, pron) list
# =========================

def cmu_items_top(max_words=15000):
    cmu = load_cmudict(max_words=max_words)
    return [(w, tuple(prons[0])) for w, prons in cmu.items() if prons]

# =========================
# 4) Use search_dp.py (pure DP) to score best EN word for each n-gram
# =========================

def best_en_for_phones_dp(
    phones_ng: Tuple[str, ...],
    cmu_items: List[Tuple[str, Tuple[str, ...]]],
    ins: float = 1.0,
    dele: float = 1.0
) -> Tuple[str, Tuple[str, ...], float]:
    """
    Pure DP per-phone cost, exactly via search_dp internals.
    Normalized by len(phones_ng) to compare across lengths.
    """
    if not phones_ng:
        return "", tuple(), float("inf")

    best_w, best_p, best_c = "", tuple(), float("inf")
    for w, p in cmu_items:
        c = _edit_distance(
            list(p), list(phones_ng),
            sc=lambda e, h: 0.0 if e == h else _sub_cost_en_vs_hu(e, h),
            ins=ins,
            dele=dele,
        ) / max(1, len(phones_ng))
        if c < best_c:
            best_c, best_w, best_p = c, w, p
    return best_w, best_p, best_c

# =========================
# 5) Build the HU-morph n-gram dictionary
# =========================

def count_ngrams_from_top_hu_words(n_words: int = 20_000) -> Counter:
    """
    Builds frequency of HU morph n-grams (1..3) over the top-N Hungarian words.
    Counts are based on enPhon segmentation of each word.
    """
    words = [w for w in top_n_list("hu", n_words) if isinstance(w, str) and 2 <= len(w) <= 40]
    counts: Counter[Tuple[str, ...]] = Counter()
    for w in words:
        morphs = segment_morphs_enphon(w)          # enPhon + emMorph
        morphs = refine_morph_units(morphs)
        for morph_ng, _phones_ng in ngrams_max3(morphs):
            counts[morph_ng] += 1
    return counts


def build_ngram_dict_from_top_hu_words(test_n_words: int = 20000,
                                       cmu_max_words: int = 15000,
                                       min_test_count: int = 4,
                                       ins: float = 1.0, dele: float = 1.0) -> Dict[Tuple[str, ...], Dict[str, object]]:
    """
    Build HU-morph n-gram → EN mapping, but only for n-grams that occur >= min_test_count
    in the top-N Hungarian words list.
    """
    # 1) CMU pool
    cmu_items = cmu_items_top(max_words=cmu_max_words)
    # 2) count on top HU words
    test_counts = count_ngrams_from_top_hu_words(test_n_words)


    # 3) For each qualifying n-gram, compute best EN via pure-DP
    mapping: Dict[Tuple[str, ...], Dict[str, object]] = {}
    for ng, cnt in test_counts.items():
        if cnt < min_test_count:
            continue
        # phones for this n-gram (reconstruct from morphs)
        phones_ng: List[str] = []
        for m in ng:
            s = _tr_ipa(m)
            phones_ng.extend(_tokenize_ipa(s))
        w, p, c = best_en_for_phones_dp(tuple(phones_ng), cmu_items, ins=ins, dele=dele)
        mapping[ng] = {"en_word": w, "en_pron": p, "cost": c, "count": 0, "test_count": int(cnt)}
    return mapping

# =========================
# 6) Decode a morph sequence choosing tri vs (uni+bi) etc. by DP
# =========================

def decode_morphs_with_dict(morphs: Sequence[str],
                            mapping: Dict[Tuple[str, ...], Dict[str, object]],
                            max_n: int = 3) -> Tuple[List[str], float]:
    """
    Given a morph sequence and a n-gram→(word,cost) mapping,
    pick a word sequence with minimum total dp_cost (classic segmentation DP).
    Returns (list_of_words, total_cost).
    """
    n = len(morphs)
    # dp[i] = (cost, backpointer_len)
    dp: List[Tuple[float, int]] = [(0.0, 0)]
    dp += [(float("inf"), 0) for _ in range(n)]
    back: List[int] = [-1] * (n + 1)

    for i in range(n):
        base_cost, _ = dp[i]
        if base_cost == float("inf"):
            continue
        for L in range(1, max_n + 1):
            j = i + L
            if j > n: break
            key = tuple(morphs[i:j])
            if key not in mapping:
                continue
            c = mapping[key]["dp_cost"]
            new_cost = base_cost + float(c)
            if new_cost < dp[j][0]:
                dp[j] = (new_cost, L)
                back[j] = i

    # Reconstruct
    if dp[n][0] == float("inf"):
        return [], float("inf")
    words: List[str] = []
    pos = n
    while pos > 0:
        L = dp[pos][1]
        i = pos - L
        key = tuple(morphs[i:pos])
        words.append(mapping[key]["en_word"])
        pos = i
    words.reverse()
    return words, dp[n][0]


# =========================
# 7) Convenience helpers
# =========================

def dict_to_rows(mapping: Dict[Tuple[str, ...], Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for k, v in mapping.items():
        rows.append({
            "hu_ngram": " + ".join(k),
            "n": len(k),
            "count": v.get("count", 0),
            "en_word": v.get("en_word", ""),
            "en_pron": " ".join(v.get("en_pron", ())),
            "cost": v.get("dp_cost", float("inf")),
        })
    return rows

def save_mapping_csv(mapping, path: str, sep: str = ",") -> None:
    """
    Save HU-morph n-gram → EN mapping to CSV/TSV.
    Columns: hu_ngram, n, count, en_word, en_pron, dp_cost
    """
    df = pd.DataFrame(dict_to_rows(mapping))
    df.to_csv(path, index=False, sep=sep)

def load_mapping_csv(path: str, sep: str = ",") -> dict:
    """
    Load mapping saved by save_mapping_csv().
    Reconstructs tuple keys and EN pronunciation tuples.
    """
    df = pd.read_csv(path, sep=sep)
    # tolerate alternative column names
    col_ng = "hu_ngram" if "hu_ngram" in df.columns else ("unit" if "unit" in df.columns else None)
    if col_ng is None:
        raise ValueError(f"{path}: missing hu_ngram/unit column. Got: {list(df.columns)}")

    mapping = {}
    for _, r in df.iterrows():
        key = tuple(t.strip() for t in str(r[col_ng]).split(" + "))
        en_pron = tuple(str(r.get("en_pron", "")).split())
        # dp_cost can be 'inf' string; float() handles it
        dp_cost = float(r.get("cost", float("inf")))
        cnt = int(r.get("count", 0))
        mapping[key] = {
            "en_word": str(r.get("en_word", "")),
            "en_pron": en_pron,
            "cost": dp_cost,
            "count": cnt,
        }
    return mapping