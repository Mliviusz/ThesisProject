# -*- coding: utf-8 -*-
"""
Hungarian G2P using emPhon.
"""

from __future__ import annotations
from typing import List, Callable, Optional
import os, re, shlex, subprocess, importlib

# --------- helpers ---------

_WS = re.compile(r"\s+")
_PHON_LINE = re.compile(r"^#\s*phon\s*=\s*(.+)$")  # emPhon sentence-level IPA line

def _split_ipa(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return [t for t in _WS.split(s) if t]

def _ensure_nonempty(phones: List[str], src: str) -> List[str]:
    if not phones:
        raise RuntimeError(f"emPhon produced no phones via {src}.")
    return phones

# Multi-char phones first (longest match wins)
_MULTI = [
    "t͡ʃ", "d͡ʒ", "t͡s", "d͡z",   # tie-bar forms
    "tʃ",  "dʒ",  "ts",  "dz",   # plain digraphs
]
_SINGLE_SYMBOLS = set(list("pbtdkgɟcɲmnɴfvszʃʒhɫlrɾɹjwɰŋxɣçʝβθðɕʑʈɖɟɡqʔ") +
                      list("aɑɒeɛiɪoɔuʊyʏøœəɜɞɨʉɯɤ") + ["ʎ","ʀ","ʘ","ǀ","ǃ","ǂ","ǁ"])
_LEN_MARK = "ː"   # vowel length marker (keep vowel+ː together)
_PUNCT     = set(",.;:!?—–-()[]{}\"'")  # drop these

def _segment_word_ipa(word: str) -> list[str]:
    """Greedy longest-match segmentation of a single word's IPA into phones."""
    phones = []
    i = 0
    while i < len(word):
        ch = word[i]

        # Skip punctuation/whitespace just in case
        if ch.isspace() or ch in _PUNCT:
            i += 1
            continue

        # Affricates first (longest digraphs)
        matched = False
        for dig in _MULTI:
            L = len(dig)
            if word[i:i+L] == dig:
                phones.append(dig)
                i += L
                matched = True
                break
        if matched:
            continue

        # Vowel + length mark (e.g., aː, eː, øː, yː, oː, etc.)
        if i+1 < len(word) and word[i+1] == _LEN_MARK:
            phones.append(ch + _LEN_MARK)
            i += 2
            continue

        # Single symbol (+ possible combining diacritics)
        # Keep base char; attach following combining diacritics if present
        j = i + 1
        # Basic Unicode combining range: 0300–036F
        while j < len(word):
            code = ord(word[j])
            if 0x0300 <= code <= 0x036F:
                j += 1
            else:
                break
        phones.append(word[i:j])
        i = j

    return phones

def _segment_ipa_tokens(tokens: list[str]) -> list[str]:
    out = []
    for tok in tokens:
        out.extend(_segment_word_ipa(tok))
    return out

def _call_pipeline(text: str) -> Optional[List[str]]:
    """
    Run EMPHON_PIPE_CMD (e.g., 'emtsv -l hu emMorph emPhon [flags]')
    and parse the '# phon = ...' header emitted by emPhon.
    """
    cmd = os.environ.get("EMPHON_PIPE_CMD", "").strip()
    if not cmd:
        return None

    # Split the shell command safely
    args = shlex.split(cmd)
    try:
        proc = subprocess.run(
            args,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Pipeline failed ({cmd}): {e.stderr.decode('utf-8', 'ignore')}"
        ) from e

    out = proc.stdout.decode("utf-8", "replace").splitlines()
    # emPhon normally prepends a '# phon = ...' sentence IPA line
    for line in out:
        m = _PHON_LINE.match(line.strip())
        if m:
            raw = m.group(1).strip()
            # Always split into tokens (likely words and punctuation),
            # then segment each token into phones.
            tokens = [t for t in raw.split() if t]
            phones = []
            for tok in tokens:
                # skip punctuation tokens like "," "."
                if tok in {",", ".", ";", ":", "!", "?", "—", "–", "-", "(", ")", "[", "]", "{", "}", "\"", "'"}:
                    continue
                phones.extend(_segment_word_ipa(tok))
            return _ensure_nonempty(phones, f"pipeline: {cmd}")

    # Fallback: if there's no '# phon =' line, try to collect token-level column 2.
    toks: List[str] = []
    for line in out:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = _WS.split(line)
        if cols:
            ipa_tok = cols[-1]
            toks.append(ipa_tok)
    if toks:
        phones = _segment_ipa_tokens(toks)
        return _ensure_nonempty(phones, f"pipeline toks: {cmd}")


    raise RuntimeError(
        "Could not find sentence-level IPA in pipeline output (no '# phon =' line). "
        "Make sure emPhon runs in the emtsv chain and '--include-sentence' is enabled."
    )

# --------- B) python hook adapter ---------

def _call_emphon_python(text: str) -> Optional[List[str]]:
    spec = os.environ.get("EMPHON_PY_FUNC", "").strip()
    if not spec:
        return None
    if ":" not in spec:
        raise RuntimeError("EMPHON_PY_FUNC must look like 'package.module:function'")
    mod_name, func_name = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    func: Callable[[str], object] = getattr(module, func_name)
    out = func(text)
    if isinstance(out, list):
        phones = [str(x).strip() for x in out if str(x).strip()]
        return _ensure_nonempty(phones, f"python hook {spec}")
    elif isinstance(out, str):
        return _ensure_nonempty(_split_ipa(out), f"python hook {spec}")
    else:
        raise RuntimeError(
            f"emPhon python hook {spec} returned {type(out)}; expected List[str] or str."
        )

# --------- public API ---------

def hu_text_to_ipa(text: str) -> List[str]:
    """
    Hungarian text -> IPA phones via emPhon.
    Requires either:
      - EMPHON_PIPE_CMD (emtsv pipeline with emMorph->emPhon), or
      - EMPHON_PY_FUNC (direct Python function).
    """
    res = _call_pipeline(text)
    if res is not None:
        return res
    res = _call_emphon_python(text)
    if res is not None:
        print("python")
        return res
    raise RuntimeError(
        "emPhon is not configured.\n"
        "Set EMPHON_PIPE_CMD to an emtsv pipeline, e.g.:\n"
        "  export EMPHON_PIPE_CMD='emtsv -l hu emMorph emPhon'\n"
    )

if __name__ == "__main__":
    print(hu_text_to_ipa("Mit sütsz, kis szűcs"))
