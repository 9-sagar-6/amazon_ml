import re
import unicodedata
import numpy as np
import pandas as pd

COMMON_LEGAL_SUFFIXES = {
    'corp': 'corporation',
    'inc': 'incorporated',
    'ltd': 'limited',
    'pvt': 'private',
    'co': 'company',
    'llc': 'limited liability company',
    'llp': 'limited liability partnership',
    'plc': 'public limited company',
    'dept': 'department',
    'div': 'division',
    'intl': 'international',
    'svc': 'services',
    'svcs': 'services',
    'tech': 'technologies',
    'mfg': 'manufacturing',
    'assoc': 'associates',
    'ctr': 'center',
    'cntr': 'center',
    'ste': 'suite',
    'apt': 'apartment',
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'blvd': 'boulevard',
    'dr': 'drive',
    'hwy': 'highway',
    'pkwy': 'parkway',
    'ln': 'lane',
    'ct': 'court',
    'pl': 'place',
    'way': 'way',
}

_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(k) for k in COMMON_LEGAL_SUFFIXES.keys()) + r')\b', flags=re.IGNORECASE)
_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
_MULTIPLE_SPACES = re.compile(r'\s+')

def _sub_func(match):
    return COMMON_LEGAL_SUFFIXES.get(match.group(1).lower(), match.group(1))

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().replace("&", " and ")
    text = _PATTERN.sub(_sub_func, text)
    text = _NON_ALPHANUM.sub(' ', text)
    text = _MULTIPLE_SPACES.sub(' ', text).strip()
    return text

def clean_text_list(items) -> list:
    """Fast, PyArrow-safe list comprehension text cleaning."""
    return [clean_text(s) if isinstance(s, str) else "" for s in items]

def extract_numbers(text: str) -> set:
    if not text or not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d+\b', text))

def compute_macro_f05(ground_truth_dict: dict, predictions_dict: dict) -> float:
    f05_scores = []
    for s1_id, true_matches in ground_truth_dict.items():
        pred_matches = set(predictions_dict.get(s1_id, []))
        if len(true_matches) == 0:
            if len(pred_matches) == 0:
                score = 1.0
            else:
                score = 0.0
        else:
            tp = len(true_matches & pred_matches)
            fp = len(pred_matches - true_matches)
            fn = len(true_matches - pred_matches)
            if tp == 0:
                score = 0.0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                score = (1.25 * precision * recall) / (0.25 * precision + recall)
        f05_scores.append(score)
    return float(np.mean(f05_scores))
