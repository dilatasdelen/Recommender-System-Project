# Recommender Systems Project — Core Ranking (Popularity vs. MF/ALS)

A comparison of a non-personalized Popularity baseline against a Matrix Factorization
model trained with implicit ALS, evaluated on the Amazon Reviews 2023 (Movies_and_TV,
5-core) dataset under a leave-last-out protocol.

## Research Question
Does a Matrix Factorization / implicit ALS model improve ranking quality (NDCG@10)
over a Popularity baseline, and if so, at what cost in terms of catalogue coverage?

## Dataset
Amazon Reviews 2023 (McAuley Lab, UCSD), Movies_and_TV category, 5-core filtered.
Source: https://github.com/hyp1231/AmazonReviews2023

Included in `datasets/`: validation.parquet, test.parquet, popularity_recommendations.parquet,
als_recommendations.parquet, and the result CSVs (als_tuning_results.csv,
als_val_test_discrepancy.csv, evaluation_results.csv, genre_analysis_summary.csv,
genre_match_sample_users.csv).

Not included (exceed GitHub's 100 MB file limit, or are the original public
downloads): meta_Movies_and_TV.jsonl, Movies_and_TV.csv, merged_movies_tv.parquet,
train.parquet. To regenerate these: download the raw files from the link above,
then run `EDA.ipynb` (produces merged_movies_tv.parquet) followed by `split.ipynb`
(produces train/validation/test.parquet).

## Notebooks (run in this order)
1. `EDA.ipynb` — loads raw ratings + metadata, merges them, exploratory analysis
   (sparsity, rating distribution, metadata coverage by category).
2. `split.ipynb` — builds the leave-last-out train/validation/test split.
3. `popularity.ipynb` — builds the Popularity baseline and its recommendations.
4. `MFALS.ipynb` — builds the MF/ALS model, recommendations, hyperparameter tuning,
   and validation/test discrepancy check.
5. `evaluation.ipynb` — computes NDCG@10, Recall@10, Coverage for both models.
6. `genre_analysis.ipynb` — genre-level analysis of recommendations vs. user history.
7. `results_summary.ipynb` — compiles results from the above notebooks, statistical
   significance testing (Wilcoxon), and overall takeaways.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow matplotlib implicit scipy
```

## Report
See `final_report_tasdelen_duelger.docx` for the full write-up, including
methodology, results, discussion, limitations, and the AI-use disclosure.

## Authors
Dila Tasdelen and Havva Seda Duelger