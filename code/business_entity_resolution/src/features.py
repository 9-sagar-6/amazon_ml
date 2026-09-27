import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from joblib import Parallel, delayed
from utils import clean_text, extract_numbers

def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

def get_char_ngrams(text: str, n: int) -> set:
    if len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text)-n+1))

def extract_pair_features(s1_row: dict, cand_row: dict) -> dict:
    s1_name = s1_row.get("clean_name", "")
    cand_name = cand_row.get("clean_name", "")
    s1_addr = s1_row.get("clean_address", "")
    cand_addr = cand_row.get("clean_address", "")

    # String Similarities
    n_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    n_partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    n_token_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    n_token_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    n_wratio = fuzz.WRatio(s1_name, cand_name) / 100.0

    len1, len2 = len(s1_name), len(cand_name)
    n_len_diff = abs(len1 - len2) / max(len1, len2, 1)

    a_ratio = fuzz.ratio(s1_addr, cand_addr) / 100.0
    a_partial = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
    a_token_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
    a_token_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0

    # Overlaps
    s1_nums = s1_row.get("nums", set())
    cand_nums = cand_row.get("nums", set())
    num_jaccard = jaccard_similarity(s1_nums, cand_nums)

    s1_words = s1_row.get("words", set())
    cand_words = cand_row.get("words", set())
    word_1gram_jaccard = jaccard_similarity(s1_words, cand_words)

    s1_char3 = s1_row.get("char3", set())
    cand_char3 = cand_row.get("char3", set())
    char_3gram_jaccard = jaccard_similarity(s1_char3, cand_char3)

    cand_id = cand_row.get("entity_id", "")
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0

    return {
        "name_ratio": n_ratio,
        "name_partial_ratio": n_partial,
        "name_token_sort_ratio": n_token_sort,
        "name_token_set_ratio": n_token_set,
        "name_wratio": n_wratio,
        "name_len_diff": n_len_diff,
        "address_ratio": a_ratio,
        "address_partial_ratio": a_partial,
        "address_token_sort_ratio": a_token_sort,
        "address_token_set_ratio": a_token_set,
        "num_token_jaccard": num_jaccard,
        "word_1gram_jaccard": word_1gram_jaccard,
        "char_3gram_jaccard": char_3gram_jaccard,
        "is_source2": is_source2,
    }

def batch_extract_features(candidate_pairs: list, s1_dict: dict, pool_dict: dict, n_jobs: int = 4) -> pd.DataFrame:
    if not candidate_pairs:
        return pd.DataFrame()

    # Pre-populate helper sets
    for d in (s1_dict, pool_dict):
        for k, v in d.items():
            if "nums" not in v:
                v["nums"] = extract_numbers(v.get("business_address", "")) | extract_numbers(v.get("business_name", ""))
            if "words" not in v:
                v["words"] = set(v.get("clean_name", "").split()) | set(v.get("clean_address", "").split())
            if "char3" not in v:
                v["char3"] = get_char_ngrams(v.get("clean_name", ""), 3)

    def _pair_worker(pair):
        s1_id, cand_id = pair
        feat = extract_pair_features(s1_dict.get(s1_id, {}), pool_dict.get(cand_id, {}))
        feat["s1_id"] = s1_id
        feat["cand_id"] = cand_id
        return feat

    records = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=1000)(
        delayed(_pair_worker)(pair) for pair in candidate_pairs
    )
    return pd.DataFrame(records)
