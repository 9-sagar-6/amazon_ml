# Business Entity Resolution Solution — Code Repository

This repository contains the end-to-end Machine Learning pipeline for resolving multi-source business entities across noisy and unstandardized commercial datasets for the **Amazon ML Challenge 2026**.

## Project Architecture & Directory Layout

```
code/business_entity_resolution/
├── src/
│   ├── utils.py               # Text normalization, legal suffix expansion, numeric token extraction, and macro F_0.5 evaluation metric
│   ├── blocking.py            # Multi-Pass Candidate Blocker (Country partitioning, Char 3-4 n-gram TF-IDF cosine similarity matrix dot products, Zip/PIN Inverted Indexing)
│   ├── features.py            # Pairwise string similarity feature extractor (RapidFuzz ratio/token_sort/token_set/wratio, char n-gram Jaccard, numeric/zipcode overlap, length diffs)
│   ├── model.py               # LightGBM binary match classifier & macro F_0.5 threshold optimization grid search
│   └── pipeline.py            # End-to-end runner script generating matching_results.tsv & candidate_pairs.tsv
├── requirements.txt           # Pinned python package dependencies
└── README.md                  # Reproduction instructions & documentation
```

## Setup & Environment Installation

Ensure Python 3.8+ is installed. Install all pinned dependencies:

```bash
pip install -r requirements.txt
```

## Running the End-to-End Pipeline

To execute candidate generation, feature extraction, model training, threshold tuning, test set inference, and automated submission validation:

```bash
python src/pipeline.py
```

This will automatically generate the required tab-separated output files in `output/`:
1. `output/matching_results.tsv` — Final predicted entity matches scored on the leaderboard.
2. `output/candidate_pairs.tsv` — Candidate blocking set fed into the model for inference.

## Verification & Submission Format Check

Run the official challenge validator to verify submission formatting:

```bash
python ../../datasets/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../datasets/student_resource/dataset/test
```
