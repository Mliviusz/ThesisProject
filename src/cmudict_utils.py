# -*- coding: utf-8 -*-
from typing import Dict, List
from collections import defaultdict
import re, cmudict
from wordfreq import zipf_frequency
from .ipa_utils import arpabet_seq_to_ipa
_WORD_RE = re.compile(r"^([A-Za-z']+)(\(\d+\))?$")
def load_cmudict(min_zipf: float = 3.5, max_words: int = 10000) -> Dict[str, List[List[str]]]:
    raw = cmudict.dict()
    items = []
    for w, pron_list in raw.items():
        m = _WORD_RE.match(w)
        if not m: continue
        word = m.group(1).lower()
        if zipf_frequency(word, "en") < min_zipf: continue
        ipa_prons = [arpabet_seq_to_ipa(p) for p in pron_list]
        items.append((word, ipa_prons))
    items.sort(key=lambda x: zipf_frequency(x[0], "en"), reverse=True)
    if max_words is not None: items = items[:max_words]
    lex = defaultdict(list)
    for w, prons in items:
        lex[w].extend(prons)
    return dict(lex)
