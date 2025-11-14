# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple

HU_TO_EN_PROXIES: Dict[str, List[Tuple[str, float]]] = {
    "y":  [("ju:", 0.0), ("u:", 0.2)],
    "y:": [("ju:", 0.0), ("u:", 0.2)],
    "ø":  [("ɜ:", 0.0), ("oʊ", 0.2)],
    "ø:": [("ɜ:", 0.0), ("oʊ", 0.2)],
    "i":  [("ɪ", 0.0), ("i", 0.15)],
    "i:": [("i:", 0.0), ("i", 0.15)],
    "u":  [("ʊ", 0.0), ("u", 0.15)],
    "u:": [("u:", 0.0), ("u", 0.15)],
    "o":  [("ɔ", 0.0), ("ɒ", 0.15)],
    "o:": [("oʊ", 0.0), ("ɔː", 0.2)],
    "c":  [("tʃ", 0.0), ("tj", 0.2)],
    "ɟ":  [("dʒ", 0.0), ("dj", 0.2)],
    "ɲ":  [("nj", 0.0)],
    "s":  [("s", 0.0)],
    "ʃ":  [("ʃ", 0.0)],
}

def proxies_for_hu_phone(hu_phone: str) -> List[Tuple[str, float]]:
    return HU_TO_EN_PROXIES.get(hu_phone, [(hu_phone, 0.0)])
