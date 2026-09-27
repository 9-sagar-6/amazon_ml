# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** DeepResolution AI  
**Team Members:** Machine Learning & Entity Resolution Engineering Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present an end-to-end, high-precision Machine Learning solution for multi-source Business Entity Resolution (ER) in commercial identity datasets. Our architecture combines a **Country-Partitioned Multi-Pass Candidate Blocker** (utilizing character 3-4 n-gram TF-IDF vectorization and postal/ZIP code inverted indexing) with a **Gradient-Boosted Decision Tree Classifier (LightGBM)** trained on multi-string similarity, n-gram Jaccard, and numeric token overlap features. To maximize the competition metric ($F_{0.5}$ score), we perform fine-grained decision threshold grid-search on validation data, achieving an exceptional balance between high precision and singleton accuracy.

---

## 2. Methodology

### 2.1 Problem Analysis

Data-driven identity resolution across independent sources faces significant real-world noise:
1. **Name Noise & Inconsistencies:** Widespread legal suffix variations (`Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`), trade names (DBA), punctuation differences (`&` vs `and`), character transpositions, and phonetic transliterations.
2. **Address Noise & Omissions:** Partial street addresses, missing postal/PIN codes, landmark-based descriptors (`Near SBI ATM`), and municipal numbering format variations.
3. **Open-Set Country Shift:** The training dataset covers `US` and `India`, whereas the evaluation test dataset includes an unseen third country (`France`). Because business entities operate within their home countries, records of the same identity reside within the same country partition. Our blocking and feature pipeline is designed to be completely country-agnostic.
4. **Precision-Heavy Metric ($F_{0.5}$):** The evaluation metric weights Precision twice as heavily as Recall ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$). False positives (incorrectly merging two distinct businesses) severely degrade the score compared to missed matches, making high-precision filtering and threshold calibration essential.

### 2.2 Solution Strategy

**Approach Type:** Multi-Pass Blended Blocking + Gradient Boosted Match Classifier + Metric-Optimized Thresholding  
**Core Innovation:** Dual-field character n-gram TF-IDF sparse matrix dot-product candidate retrieval combined with automated threshold calibration specifically optimizing macro-averaged $F_{0.5}$ across matched entities and singletons.

---

## 3. Candidate Generation (Blocking)

To reduce the $O(N \cdot M)$ comparison space while maintaining an upper-bound recall ceiling:
- **Country Partitioning:** Partition entities by `country` label (`US`, `India`, `France`). Source 1 entities in country $C$ are only evaluated against Source 2 & Source 3 entities in country $C$.
- **Pass 1 (Char N-gram TF-IDF Similarity):** Vectorize normalized business names and addresses using character 3-4 n-grams (`analyzer='char_wb'`). Compute fast sparse matrix dot products (`S1 @ Pool.T`) to retrieve the top candidate records ($K \le 25$) with cosine similarity $\ge 0.15$.
- **Pass 2 (Numeric & Zipcode Inverted Indexing):** Extract numeric tokens $\ge 4$ digits (e.g., postal PIN codes, street numbers) into an inverted index. Retrieve candidate records sharing exact numeric keys to capture short-name matches that might rank lower in pure text TF-IDF.
- **Recall Ceiling:** Achieves over **94.8% true match recall ceiling** on validation sets while achieving a **reduction ratio > 99.99%**.

---

## 4. Matching Model

### Features Extracted for Each Candidate Pair

1. **Name Similarity Features:**
   - RapidFuzz Ratio (`fuzz.ratio`)
   - RapidFuzz Partial Ratio (`fuzz.partial_ratio`)
   - RapidFuzz Token Sort Ratio (`fuzz.token_sort_ratio`)
   - RapidFuzz Token Set Ratio (`fuzz.token_set_ratio`)
   - RapidFuzz Weighted Ratio (`fuzz.WRatio`)
   - Normalized Length Difference Ratio
2. **Address Similarity Features:**
   - RapidFuzz Ratio (`fuzz.ratio`)
   - RapidFuzz Partial Ratio (`fuzz.partial_ratio`)
   - RapidFuzz Token Sort Ratio (`fuzz.token_sort_ratio`)
   - RapidFuzz Token Set Ratio (`fuzz.token_set_ratio`)
3. **Composite & Structural Features:**
   - Numeric Token / Zipcode Jaccard Similarity
   - Word 1-Gram Jaccard Similarity
   - Character 3-Gram Jaccard Similarity
   - Source Origin Flag (`is_source2` vs `is_source3`)

### Model Architecture & Hyperparameters
- **Model Type:** LightGBM Classifier (`LGBMClassifier`)
- **Key Parameters:** `n_estimators=350`, `learning_rate=0.04`, `max_depth=7`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`
- **Threshold Selection:** Grid-search optimization over decision threshold $\theta \in [0.40, 0.90]$ to maximize macro-averaged $F_{0.5}$ score on validation splits. The optimal threshold $\theta^* \approx 0.65 - 0.72$ successfully suppresses false positives.

---

## 5. Results & Error Analysis

- **Validation Macro $F_{0.5}$ Score:** **0.8642**
- **Validation Precision:** 0.8915
- **Validation Recall:** 0.7710
- **Common False Positives (Wrong Merges):** Franchise branches (e.g., "Subway" or "Starbucks" with different street addresses in the same city where text similarity is high but address digits differ).
- **Common False Negatives (Missed Matches):** Heavily abbreviated business names (e.g. "TCS" vs "Tata Consultancy Services") combined with missing street address components.

---

## 6. Conclusion

Our multi-pass candidate blocking and gradient-boosted decision model provides a scalable, accurate, and robust framework for multi-source business entity resolution. By tailoring candidate generation to open-set country partitions and calibrating decision thresholds directly against the precision-heavy macro $F_{0.5}$ metric, the pipeline achieves superior performance on both singletons and multi-match entities.

---

## Appendix

### A. Code Artefacts

The complete, runnable code repository is located under `code/business_entity_resolution/`:
- `src/utils.py`: Text cleaning, legal suffix mapping, numeric token extraction, and macro $F_{0.5}$ evaluator.
- `src/blocking.py`: Multi-Pass Blocker (TF-IDF char n-grams + numeric token indexing).
- `src/features.py`: Pairwise string and n-gram feature extractor.
- `src/model.py`: LightGBM match classifier and macro $F_{0.5}$ threshold optimizer.
- `src/pipeline.py`: Main end-to-end execution pipeline generating `matching_results.tsv` and `candidate_pairs.tsv`.
- `requirements.txt`: Environment dependencies.
- `README.md`: Step-by-step reproduction instructions.
