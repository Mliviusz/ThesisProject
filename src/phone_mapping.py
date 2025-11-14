# -*- coding: utf-8 -*-
import panphon.distance
_dist = panphon.distance.Distance()
def phone_distance(p, q):
    try: return _dist.phoneme_distance(p, q)
    except Exception: return 2.0 if p!=q else 0.0
