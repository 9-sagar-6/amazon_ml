import os
import sys
import time
import gc
import subprocess
import numpy as np
import pandas as pd
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(__file__))

from utils import clean_text, compute_macro_f05
from blocking import MultiPassBlocker
from features import batch_extract_features
from model import EntityMatchingModel

def run_end_to_end_pipeline(
    train_dir: str,
    test_dir: str,
    output_dir: str,
    model_save_path: str = None,
    sample_train_size: int = 40000
):
    print("==========================================================")
    print("  AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION  ")
    print("==========================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # -----------------------------------------------------------
    # STEP 1: LOAD & TRAIN MODEL ON TRAINING DATASET
    # -----------------------------------------------------------
    print(f"\n[1/5] Loading Training Data ({sample_train_size:,} S1 sample)...")
    s1_train_df = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", nrows=sample_train_size, dtype=str)
    s1_train_ids = set(s1_train_df["entity_id"])
    
    gt_df_full = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)
    gt_df = gt_df_full[gt_df_full["source1_entity_id"].isin(s1_train_ids)].copy()
    del gt_df_full
    gc.collect()

    gt_dict = {}
    for _, row in gt_df.iterrows():
        m = row["matched_entity_ids"]
        gt_dict[row["source1_entity_id"]] = set(m.split(",")) if isinstance(m, str) and m.strip() else set()

    for s in s1_train_ids:
        if s not in gt_dict:
            gt_dict[s] = set()

    s2_train_df = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", dtype=str)
    s3_train_df = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", dtype=str)

    print("[2/5] Running Multi-Pass Candidate Blocking on Training Set...")
    blocker = MultiPassBlocker(top_k=25, min_tfidf_sim=0.15, batch_size=10000)
    train_cands = blocker.run_blocking(s1_train_df, s2_train_df, s3_train_df)

    print("Extracting Training Pair Features...")
    s1_dict = s1_train_df.set_index("entity_id").to_dict("index")
    for k, v in s1_dict.items():
        v["clean_name"] = clean_text(v.get("business_name", ""))
        v["clean_address"] = clean_text(v.get("business_address", ""))

    pool_train_df = pd.concat([s2_train_df, s3_train_df], ignore_index=True)
    del s2_train_df, s3_train_df
    gc.collect()

    pool_dict = pool_train_df.set_index("entity_id").to_dict("index")
    for k, v in pool_dict.items():
        v["clean_name"] = clean_text(v.get("business_name", ""))
        v["clean_address"] = clean_text(v.get("business_address", ""))
    del pool_train_df
    gc.collect()

    train_pairs = [(s1_id, cid) for s1_id, c_set in train_cands.items() for cid in c_set]
    train_feats_df = batch_extract_features(train_pairs, s1_dict, pool_dict, n_jobs=4)
    y_train = np.array([1 if row["cand_id"] in gt_dict.get(row["s1_id"], set()) else 0 for _, row in train_feats_df.iterrows()])

    print(f"Training LightGBM Match Classifier on {len(train_feats_df):,} candidate pairs...")
    model = EntityMatchingModel(n_estimators=300, learning_rate=0.05, max_depth=6)
    model.fit(train_feats_df, y_train)

    print("Optimizing Macro F_0.5 Decision Threshold...")
    optimal_theta = model.optimize_threshold(train_feats_df, gt_dict)

    if model_save_path:
        model.save(model_save_path)

    del s1_dict, pool_dict, train_feats_df
    gc.collect()

    # -----------------------------------------------------------
    # STEP 2: COUNTRY-STREAMED TEST INFERENCE
    # -----------------------------------------------------------
    print("\n[3/5] Loading Test S1 Data & Processing by Country...")
    test_s1_pl = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t")
    test_s1_ids_all = test_s1_pl["entity_id"].to_list()
    countries = test_s1_pl["country"].unique().to_list()

    matching_results = {s1_id: [] for s1_id in test_s1_ids_all}
    candidate_pairs_dict = {s1_id: set() for s1_id in test_s1_ids_all}

    print(f"Test S1 Total: {len(test_s1_ids_all):,}. Countries found: {countries}")

    for country in countries:
        print(f"\n---> Processing Country Chunk: {country} <---")
        c_s1_df = test_s1_pl.filter(pl.col("country") == country).to_pandas()

        c_s2_df = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t").filter(pl.col("country") == country).to_pandas()
        c_s3_df = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t").filter(pl.col("country") == country).to_pandas()

        print(f"[{country}] S1 count: {len(c_s1_df):,}, S2 count: {len(c_s2_df):,}, S3 count: {len(c_s3_df):,}")

        # Blocking for country
        c_cands = blocker.generate_candidates_for_country(c_s1_df, pd.concat([c_s2_df, c_s3_df], ignore_index=True))
        for s1_id, c_set in c_cands.items():
            candidate_pairs_dict[s1_id].update(c_set)

        # Build local dicts
        c_s1_dict = c_s1_df.set_index("entity_id").to_dict("index")
        for k, v in c_s1_dict.items():
            v["clean_name"] = clean_text(v.get("business_name", ""))
            v["clean_address"] = clean_text(v.get("business_address", ""))

        c_pool_df = pd.concat([c_s2_df, c_s3_df], ignore_index=True)
        c_pool_dict = c_pool_df.set_index("entity_id").to_dict("index")
        for k, v in c_pool_dict.items():
            v["clean_name"] = clean_text(v.get("business_name", ""))
            v["clean_address"] = clean_text(v.get("business_address", ""))

        del c_s2_df, c_s3_df, c_pool_df
        gc.collect()

        c_pairs = [(s1_id, cid) for s1_id, c_set in c_cands.items() for cid in c_set]
        print(f"[{country}] Pairs to score: {len(c_pairs):,}")

        # Extract features and predict
        chunk_size = 40000
        for start in range(0, len(c_pairs), chunk_size):
            end = min(start + chunk_size, len(c_pairs))
            chunk_pairs = c_pairs[start:end]
            chunk_feats_df = batch_extract_features(chunk_pairs, c_s1_dict, c_pool_dict, n_jobs=4)
            if not chunk_feats_df.empty:
                probs = model.predict_proba(chunk_feats_df)
                chunk_feats_df["prob"] = probs
                matched_sub = chunk_feats_df[chunk_feats_df["prob"] >= optimal_theta]
                for _, row in matched_sub.iterrows():
                    matching_results[row["s1_id"]].append(row["cand_id"])

        del c_s1_dict, c_pool_dict, c_pairs
        gc.collect()

    # -----------------------------------------------------------
    # STEP 3: WRITE OUTPUT TSV FILES & VALIDATE
    # -----------------------------------------------------------
    print("\n[5/5] Writing Final TSV Files...")
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    with open(matching_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in test_s1_ids_all:
            m_list = list(dict.fromkeys(matching_results.get(s1_id, [])))
            f.write(f"{s1_id}\t{','.join(m_list)}\n")

    with open(candidate_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in test_s1_ids_all:
            c_list = list(dict.fromkeys(candidate_pairs_dict.get(s1_id, set())))
            f.write(f"{s1_id}\t{','.join(c_list)}\n")

    print(f"Output files ready:\n  - {matching_file}\n  - {candidate_file}")

    validator_script = os.path.join(os.path.dirname(train_dir), "utils", "validate_submission.py")
    if os.path.exists(validator_script):
        print("\n--- Running Official Submission Validator ---")
        res = subprocess.run([
            sys.executable, validator_script,
            "--matching", matching_file,
            "--candidate", candidate_file,
            "--test-dir", test_dir
        ], capture_output=True, text=True)
        print(res.stdout)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    t_dir = os.path.join(base_dir, "datasets", "student_resource", "dataset", "train")
    ts_dir = os.path.join(base_dir, "datasets", "student_resource", "dataset", "test")
    out_dir = os.path.join(base_dir, "output")

    run_end_to_end_pipeline(t_dir, ts_dir, out_dir, sample_train_size=30000)
