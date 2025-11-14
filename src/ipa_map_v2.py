# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple

HU_TO_EN_PROXIES: Dict[str, List[Tuple[str, float]]] = {
    # --- FRONT-ROUNDED VOWELS: preserve rounding first ---
    # ü /y/ ~ prefer jʊ (short) / juː (long); plain u only as fallback
    "y":  [("jʊ", 0.0), ("juː", 0.15), ("ʊ", 0.25), ("u", 0.35)],
    "y:": [("juː", 0.0), ("uː", 0.15), ("jʊ", 0.25)],

    # ö /ø/ ~ prefer oʊ (rounded diphthong) > ɔ (rounded monophthong) > ɜː (unrounded)
    "ø":  [("oʊ", 0.0), ("ɔ", 0.15), ("ɜː", 0.40)],
    "ø:": [("oʊ", 0.0), ("ɔː", 0.15), ("ɜː", 0.45)],

    # --- BACK ROUNDED ---
    "u":  [("ʊ", 0.0), ("u", 0.20)],
    "u:": [("uː", 0.0), ("ʊ", 0.20)],
    "o":  [("ɔ", 0.0), ("oʊ", 0.25)],      # avoid ɒ as primary (too unrounded/back)
    "o:": [("oʊ", 0.0), ("ɔː", 0.15)],

    # --- FRONT UNROUNDED (length pairing) ---
    "i":  [("ɪ", 0.0), ("i", 0.15)],
    "i:": [("iː", 0.0), ("i", 0.10)],

    # --- MID FRONT UNROUNDED (Hungarian 'e', 'é') ---
    "ɛ":  [("ɛ", 0.0), ("e", 0.20)],       # short e
    "e:": [("eɪ", 0.0), ("iː", 0.35)],     # long é → English FACE (/eɪ/) most robust in TTS

    # --- LOW/BACK ('a', 'á' in HU) ---
    "ɒ":  [("ɑ", 0.0), ("ʌ", 0.25)],       # HU 'a' → US LOT/STRUT
    "a:": [("ɑː", 0.0), ("aʊ", 0.35)],     # long á ~ PALM; aʊ only as last resort

    # --- PALATALS ---
    "c":  [("tʃ", 0.0), ("tj", 0.25)],     # ty → tʃ robustly realized
    "ɟ":  [("dʒ", 0.0), ("dj", 0.25)],     # gy → dʒ robustly realized
    "ɲ":  [("nj", 0.0), ("n", 0.60)],      # strongly prefer /nj/, but allow /n/ as costly fallback

    # --- SIBILANTS (keep identities) ---
    "s":  [("s", 0.0)],
    "ʃ":  [("ʃ", 0.0)],
    "z":  [("z", 0.0), ("ʒ", 0.45)],       # allow zh only very costly
    "ʒ":  [("ʒ", 0.0), ("z", 0.45)],
}

def proxies_for_hu_phone(hu_phone: str) -> List[Tuple[str, float]]:
    return HU_TO_EN_PROXIES.get(hu_phone, [(hu_phone, 0.0)])
