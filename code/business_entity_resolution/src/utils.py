import re
import unicodedata
import numpy as np
import pandas as pd

COMMON_LEGAL_SUFFIXES = {
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bltd\b': 'limited',
    r'\bpvt\b': 'private',
    r'\bco\b': 'company',
    r'\bllc\b': 'limited liability company',
    r'\bllp\b': 'limited liability partnership',
    r'\bplc\b': 'public limited company',
    r'\bdept\b': 'department',
    r'\bdiv\b': 'division',
    r'\bintl\b': 'international',
    r'\bsvc\b': 'services',
    r'\bsvcs\b': 'services',
    r'\btech\b': 'technologies',
    r'\bmfg\b': 'manufacturing',
    r'\bassoc\b': 'associates',
    r'\bctr\b': 'center',
    r'\bcntr\b': 'center',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bhwy\b': 'highway',
    r'\bpkwy\b': 'parkway',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bway\b': 'way',
}

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # NFKD unicode normalization
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    # Replace & with and
    text = text.replace("&", " and ")
    # Replace common abbreviations
    for pattern, repl in COMMON_LEGAL_SUFFIXES.items():
        text = re.sub(pattern, repl, text)
    # Remove non-alphanumeric chars except space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_numbers(text: str) -> set:
    if not text or not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d+\b', text))

def compute_macro_f05(ground_truth_dict: dict, predictions_dict: dict) -> float:
    """
    Computes macro-averaged F_0.5 score over all Source 1 entities in ground_truth_dict.
    ground_truth_dict: {s1_id: set(matched_s2_s3_ids)}
    predictions_dict: {s1_id: list/set(predicted_s2_s3_ids)}
    """
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
