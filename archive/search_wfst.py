# -*- coding: utf-8 -*-
"""
Minimal WFST decoder using Pynini.

Inputs:
  - target_phones: List[str]            (HU phones, already tokenized as phones)
  - lexicon: Dict[str, List[List[str]]] (EN word -> list of EN-IPA phone sequences)
Output:
  - (best_sentence: str, words: List[str], total_cost: float)

Design (byte tokenization everywhere to avoid symbol table mismatches):
  1) Encode each phone token to a single Private Use Area char (PUA).
  2) Build a Lexicon FST mapping Enc(EN-phone sequence) -> "word ".
     Take closure() so we can concatenate multiple words.
  3) Build a token-level edit transducer E over the encoded alphabet with:
       substitutions (a:b), insertions (:b), deletions (a:)
     Identity (a:a) cost = 0; otherwise uniform costs unless `pair_cost_fn` is provided.
  4) Compose: Enc(HU-target) @ E @ L  → shortest path → output words.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import pynini
from pynini.lib import pynutil


# -----------------------------
# Encoding utilities
# -----------------------------
def _collect_alphabet(lexicon: Dict[str, List[List[str]]], target_phones: List[str]) -> List[str]:
    phones = set(target_phones)
    for prons in lexicon.values():
        for p in prons:
            phones.update(p)
    return sorted(phones)


def _make_encoder(alphabet: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Map each phone to a single PUA char (U+E000..), and build the inverse.
    """
    base = 0xE000
    if len(alphabet) > 8000:
        raise ValueError("Too many distinct phones to encode.")
    enc, inv = {}, {}
    for i, ph in enumerate(alphabet):
        ch = chr(base + i)
        enc[ph] = ch
        inv[ch] = ph
    return enc, inv


def _encode_seq(seq: List[str], enc: Dict[str, str]) -> str:
    try:
        return "".join(enc[p] for p in seq)
    except KeyError as e:
        raise ValueError(f"Phone {e} not in encoder.") from e


# -----------------------------
# Lexicon FST: Enc(phones) -> "word "   (closure for multiword)
# -----------------------------
def _build_lexicon_fst(lexicon: Dict[str, List[List[str]]], enc: Dict[str, str]) -> pynini.Fst:
    pairs = []
    for w, prons in lexicon.items():
        out = w + " "
        for p in prons:
            pairs.append((_encode_seq(p, enc), out))
    if not pairs:
        raise ValueError("Lexicon has no pronunciations.")
    # Compile the single-word map, then allow concatenation via closure().
    L_word = pynini.string_map(pairs).optimize()
    return L_word.closure().optimize()


# -----------------------------
# Edit-distance transducer over encoded alphabet
# -----------------------------
def _build_edit_transducer(
    alphabet: List[str],
    sub_cost: float = 1.0,
    ins_cost: float = 0.7,
    del_cost: float = 0.7,
    pair_cost_fn: Optional[Callable[[str, str], float]] = None,
    inv: Optional[Dict[str, str]] = None,
) -> pynini.Fst:
    """
    E = (SUB ∪ INS ∪ DEL)* on the encoded alphabet (single PUA chars).
    If `pair_cost_fn` is provided, substitution cost for (en,hu) is evaluated in *phone space*.
    """
    ops = []

    # substitutions (including identity)
    for hu_ch in alphabet:
        for en_ch in alphabet:
            if pair_cost_fn is None:
                w = 0.0 if en_ch == hu_ch else float(sub_cost)
            else:
                if inv is None:
                    raise ValueError("inv (encoded_char -> phone) required when pair_cost_fn is used.")
                hu_ph = inv[hu_ch]
                en_ph = inv[en_ch]
                w = 0.0 if en_ph == hu_ph else float(pair_cost_fn(en_ph, hu_ph))
            ops.append(pynutil.add_weight(pynini.cross(hu_ch, en_ch), w))

    # insertions ε→en_ch and deletions hu_ch→ε
    for en_ch in alphabet:
        ops.append(pynutil.add_weight(pynini.cross("", en_ch), float(ins_cost)))
    for hu_ch in alphabet:
        ops.append(pynutil.add_weight(pynini.cross(hu_ch, ""), float(del_cost)))

    return pynini.union(*ops).closure().optimize()


# -----------------------------
# Public API
# -----------------------------
def decode_wfst(
    target_phones: List[str],
    lexicon: Dict[str, List[List[str]]],
    sub_cost: float = 1.0,
    ins_cost: float = 0.7,
    del_cost: float = 0.7,
    pair_cost_fn: Optional[Callable[[str, str], float]] = None,
) -> Tuple[str, List[str], float]:
    """
    Returns: (best_sentence, words, total_cost).  If no accepting path exists, ("", [], inf).
    """
    if not target_phones or not lexicon:
        return "", [], float("inf")

    # 1) Encode
    alpha_phones = _collect_alphabet(lexicon, target_phones)
    enc, inv = _make_encoder(alpha_phones)
    enc_target = _encode_seq(target_phones, enc)

    # 2) Build FSTs
    L = _build_lexicon_fst(lexicon, enc)  # Enc(phones)* -> "word "*
    E = _build_edit_transducer(
        alphabet=list(enc.values()),
        sub_cost=sub_cost, ins_cost=ins_cost, del_cost=del_cost,
        pair_cost_fn=pair_cost_fn,
        inv={v: k for k, v in enc.items()},
    )

    # 3) Compose (all byte-tokenized)
    inp = pynini.accep(enc_target)
    lat = (inp @ E @ L).rmepsilon().optimize()
    if lat.start() == -1 or lat.num_states() == 0:
        return "", [], float("inf")

    # 4) 1-best
    best = pynini.shortestpath(lat, nshortest=1, unique=True).rmepsilon().optimize()
    if best.start() == -1 or best.num_states() == 0:
        return "", [], float("inf")

    # 5) Extract output string
    try:
        out_iter = best.paths().ostrings()
        out_str = next(out_iter, "")
    except Exception:
        # Fallback: project to OUTPUT, then read acceptor strings
        try:
            proj = pynini.project(best, "output").optimize()
        except TypeError:
            from pynini import ProjectType
            best.project(ProjectType.OUTPUT)
            proj = best.optimize()
        out_iter = proj.paths().strings()
        out_str = next(out_iter, "")

    sentence = out_str.strip()
    words = sentence.split() if sentence else []

    # 6) Total cost
    sd = pynini.shortestdistance(lat, reverse=False)
    total_cost = float(sd[lat.start()])
    return sentence, words, total_cost
