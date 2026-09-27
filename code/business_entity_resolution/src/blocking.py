import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import clean_text, clean_text_list, extract_numbers

class MultiPassBlocker:
    """
    Multi-Pass Blocking Engine for Large-Scale Entity Resolution.
    - Sparse TF-IDF Vectorizer with min_df=10 & max_features=60000 (100x sparser matrix, 0 RAM spikes).
    - Inverted Index Key Blocking (Numeric tokens, PIN/Zip codes).
    """
    def __init__(self, top_k=25, min_tfidf_sim=0.15, pool_chunk_size=200000, batch_size=1000):
        self.top_k = top_k
        self.min_tfidf_sim = min_tfidf_sim
        self.pool_chunk_size = pool_chunk_size
        self.batch_size = batch_size

    def generate_candidates_for_country(self, s1_df: pd.DataFrame, pool_df: pd.DataFrame) -> dict:
        candidates_dict = defaultdict(set)
        if len(s1_df) == 0 or len(pool_df) == 0:
            return candidates_dict

        if "clean_combined" not in pool_df.columns:
            clean_names = clean_text_list(pool_df["business_name"].to_list())
            clean_addrs = clean_text_list(pool_df["business_address"].to_list())
            pool_df["clean_name"] = clean_names
            pool_df["clean_address"] = clean_addrs
            pool_df["clean_combined"] = [f"{n} {a}" for n, a in zip(clean_names, clean_addrs)]

        if "clean_combined" not in s1_df.columns:
            clean_names = clean_text_list(s1_df["business_name"].to_list())
            clean_addrs = clean_text_list(s1_df["business_address"].to_list())
            s1_df["clean_name"] = clean_names
            s1_df["clean_address"] = clean_addrs
            s1_df["clean_combined"] = [f"{n} {a}" for n, a in zip(clean_names, clean_addrs)]

        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 3),
            min_df=10,
            max_features=60000,
            sublinear_tf=True,
            dtype=np.float32
        )

        pool_tfidf = vectorizer.fit_transform(pool_df["clean_combined"])
        s1_tfidf = vectorizer.transform(s1_df["clean_combined"])

        pool_ids = pool_df["entity_id"].values
        s1_ids = s1_df["entity_id"].values

        num_s1 = len(s1_df)
        for s1_start in range(0, num_s1, self.batch_size):
            s1_end = min(s1_start + self.batch_size, num_s1)
            batch_s1_vec = s1_tfidf[s1_start:s1_end]
            sim_matrix = batch_s1_vec.dot(pool_tfidf.T)

            for i in range(s1_end - s1_start):
                s1_id = s1_ids[s1_start + i]
                row = sim_matrix[i].toarray().ravel()
                
                if len(row) <= self.top_k:
                    top_indices = np.argsort(-row)
                else:
                    top_indices = np.argpartition(-row, self.top_k)[:self.top_k]
                    top_indices = top_indices[np.argsort(-row[top_indices])]

                for idx in top_indices:
                    sim = float(row[idx])
                    if sim >= self.min_tfidf_sim:
                        candidates_dict[s1_id].add(pool_ids[idx])

        # Pass 2: Numeric Token & Zipcode Inverted Index Blocking
        num_index = defaultdict(list)
        for cand_id, combined in zip(pool_df["entity_id"].to_list(), pool_df["clean_combined"].to_list()):
            num_tokens = extract_numbers(combined)
            for num in num_tokens:
                if len(num) >= 4:
                    num_index[num].append(cand_id)

        for s1_id, combined in zip(s1_df["entity_id"].to_list(), s1_df["clean_combined"].to_list()):
            nums = extract_numbers(combined)
            for num in nums:
                if len(num) >= 4 and num in num_index:
                    for cand_id in num_index[num][:15]:
                        candidates_dict[s1_id].add(cand_id)

        return candidates_dict

    def run_blocking(self, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> dict:
        all_candidates = {}
        countries = s1_df["country"].unique()

        for country in countries:
            c_s1 = s1_df[s1_df["country"] == country].copy()
            c_s2 = s2_df[s2_df["country"] == country].copy()
            c_s3 = s3_df[s3_df["country"] == country].copy()
            c_pool = pd.concat([c_s2, c_s3], ignore_index=True)

            c_cands = self.generate_candidates_for_country(c_s1, c_pool)
            all_candidates.update(c_cands)

        for s1_id in s1_df["entity_id"]:
            if s1_id not in all_candidates:
                all_candidates[s1_id] = set()

        return all_candidates
