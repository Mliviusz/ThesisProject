from typing import List

ARPABET_TO_IPA = {
    # Vowels
    "AA":"ɑ","AE":"æ","AH":"ʌ","AO":"ɔ",
    "AW":"aʊ","AY":"aɪ","EH":"ɛ","ER":"ɝ","EY":"eɪ",
    "IH":"ɪ","IY":"i","OW":"oʊ","OY":"ɔɪ","UH":"ʊ","UW":"u",
    # Consonants
    "P":"p","B":"b","T":"t","D":"d","K":"k","G":"g",
    "CH":"tʃ","JH":"dʒ","F":"f","V":"v","TH":"θ","DH":"ð",
    "S":"s","Z":"z","SH":"ʃ","ZH":"ʒ","HH":"h",
    "M":"m","N":"n","NG":"ŋ","L":"l","R":"ɹ","W":"w","Y":"j",
}

def strip_stress(p: str) -> str:
    while p and p[-1].isdigit():
        p = p[:-1]
    return p

def arpabet_seq_to_ipa(arpabet_phones: List[str]) -> List[str]:
    out = []
    for ph in arpabet_phones:
        key = strip_stress(ph)
        ipa = ARPABET_TO_IPA.get(key, key.lower())
        out.append(ipa)  # keep diphthongs as single tokens
    return out
