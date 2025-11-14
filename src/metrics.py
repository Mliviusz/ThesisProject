# -*- coding: utf-8 -*-
from typing import List, Callable, Optional

def _edit_distance(a: List[str],
                   b: List[str],
                   sc: Optional[Callable[[str,str], float]] = None,
                   ins: float = 1.0,
                   dele: float = 1.0) -> float:
    n, m = len(a), len(b)
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i * dele
    for j in range(1, m+1):
        dp[0][j] = j * ins
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            bj = b[j-1]
            if ai == bj:
                sub = 0.0
            else:
                sub = sc(ai, bj) if sc is not None else 1.0
            dp[i][j] = min(
                dp[i-1][j] + dele,     # deletion
                dp[i][j-1] + ins,      # insertion
                dp[i-1][j-1] + sub     # substitution
            )
    return dp[n][m]

def wer(ref_words: List[str], hyp_words: List[str]) -> float:
    return _edit_distance(ref_words, hyp_words) / max(1, len(ref_words))

def cer(ref_chars: List[str], hyp_chars: List[str]) -> float:
    return _edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))
def per(r,h,sc=None, ins: float = 1.0, dele: float = 1.0): return _edit_distance(r,h,sc=sc, ins=ins, dele=dele)/max(1,len(r))
