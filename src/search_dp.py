# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
from collections import namedtuple

from wordfreq import zipf_frequency
from .metrics import _edit_distance
from .phone_mapping import phone_distance
from .ipa_map import proxies_for_hu_phone

Hyp = namedtuple("Hyp", ["cost", "words", "pos"])

# Treat diphthongs as single tokens (see ipa_utils patch below)
_VOWELS = {"i","ɪ","eɪ","ɛ","æ","ɑ","ʌ","ɝ","ə","oʊ","ɔ","ʊ","u","aɪ","aʊ","ɔɪ"}

def _is_vowel(p: str) -> bool:
    return p in _VOWELS

def _sub_cost_en_vs_hu(en_p: str, hu_p: str, alpha=1.5, cv_pen=1.2) -> float:
    """
    Cost of aligning an EN phone token (from CMUdict→IPA) to a HU target phone,
    consulting the §2.2 proxy list for hu_p and adding PanPhon distance.
    """
    best = 1e9
    for en_proxy, proxy_pen in proxies_for_hu_phone(hu_p):
        d = phone_distance(en_p, en_proxy)
        if _is_vowel(en_p) != _is_vowel(en_proxy):
            d += cv_pen
        cand = alpha * d + proxy_pen
        if cand < best:
            best = cand
    return best

def _best_segment_cost(en_phones: List[str], target: List[str], start: int,
                       stretch: int = 2, ins_cost=0.8, del_cost=0.8,
                       len_fit_beta=0.12) -> Tuple[float, int]:
    """
    Align EN pronunciation en_phones to a slice of target starting at `start`.
    Try slice lengths in [L-stretch, L+stretch], pick min-cost, and return (cost, new_pos).
    """
    N = len(target)
    L = len(en_phones)
    best_cost, best_end = 1e9, start

    # Try feasible slice lengths
    for s in range(max(1, L - stretch), min(N - start, L + stretch) + 1):
        end = start + s
        base = _edit_distance(
            en_phones, target[start:end],
            sc=lambda p, q: _sub_cost_en_vs_hu(p, q),
            ins=ins_cost, dele=del_cost
        )
        fit = len_fit_beta * abs(s - L)
        c = base + fit
        if c < best_cost:
            best_cost, best_end = c, end

    return best_cost, best_end

def decode_dp_beam(target_phones: List[str],
                   lexicon: Dict[str, List[List[str]]],
                   beam_size: int = 48,
                   rarity_lambda: float = 0.05,
                   ins_cost: float = 0.8,
                   del_cost: float = 0.8,
                   len_fit_beta: float = 0.12,
                   stretch: int = 2,
                   max_words: int = 40,
                   max_pron_len: int = 12) -> Tuple[str, List[str], float]:
    """
    Viterbi-style DP beam search over word pronunciations with local length fit and proxy-aware costs.
    """
    N = len(target_phones)
    charts: List[List[Hyp]] = [[] for _ in range(N + 1)]
    charts[0] = [Hyp(0.0, [], 0)]

    # Flatten lexicon to (word, pron) and filter unwieldy long prons
    word_prons: List[Tuple[str, List[str]]] = []
    for w, prons in lexicon.items():
        for p in prons:
            if max_pron_len is None or len(p) <= max_pron_len:
                word_prons.append((w, p))

    for t in range(N):
        if not charts[t]:
            continue
        charts[t].sort(key=lambda h: h.cost)
        fringe = charts[t][:beam_size]

        for hyp in fringe:
            if len(hyp.words) >= max_words:
                continue
            for w, pron in word_prons:
                seg_cost, new_pos = _best_segment_cost(
                    pron, target_phones, t,
                    stretch=stretch, ins_cost=ins_cost, del_cost=del_cost, len_fit_beta=len_fit_beta
                )
                # Rarity penalty: prefer common words
                z = zipf_frequency(w, "en")
                rarity_pen = max(0.0, 5.0 - z) * rarity_lambda

                charts[new_pos].append(Hyp(hyp.cost + seg_cost + rarity_pen,
                                           hyp.words + [w], new_pos))

        # Light pruning pass every few positions
        if (t % 4) == 0:
            for k in range(N + 1):
                if charts[k]:
                    charts[k].sort(key=lambda h: h.cost)
                    charts[k] = charts[k][:beam_size]

    # Select best hypothesis: prefer exact end; otherwise furthest coverage, then lowest cost
    if charts[N]:
        best = min(charts[N], key=lambda h: h.cost)
    else:
        allhyps = [h for lst in charts for h in lst]
        if not allhyps:
            return "", [], 0.0
        best = min(allhyps, key=lambda h: (-h.pos, h.cost))

    return " ".join(best.words), best.words, best.cost
