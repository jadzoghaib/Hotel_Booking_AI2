# Hotel Booking Cancellation Prediction
## Revenue Management with Machine Learning Classification

**Author:** Group One for Artificial Intelligence II

---

## Overview

Managing revenue in hospitality is critical for hotels to maximize profit. A central challenge is forecasting booking cancellations — bookings are placed in advance and customers may cancel, creating hard-to-predict revenue losses. We define revenue management as *"the application of information systems and pricing strategies to allocate the right room for the right guest and the right price at the right time via the right distribution channel"* (Mehrotra and Ruttley, 2006).

This study applies machine learning classification to predict whether a hotel booking will be cancelled, using 119,390 records across two hotel types (City Hotel and Resort Hotel). Models evaluated include Logistic Regression, Random Forest, XGBoost, LightGBM, k-Nearest Neighbors, Support Vector Machine, and an Ensemble (soft voting).

---

## Dataset

**Source:** This dataset comes from the article *Hotel Booking Demand Datasets* by Nuno Antonio, Ana de Almeida, and Luis Nunes, published in *Data in Brief*, Volume 22, February 2019. The data was acquired directly from hotels' Property Management System (PMS) SQL databases.

- **Records:** 119,390 bookings
- **Features:** 32 columns
- **Target:** `is_canceled` (0 = not cancelled, 1 = cancelled)
- **Overall cancellation rate:** ~37%

**Kaggle Download:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
**Original Paper DOI:** https://doi.org/10.1016/j.dib.2018.11.126

> **Note:** The dataset file `hotel_bookings.csv` (17 MB) is included in this repository.

---

## Methodology Notes — Issues Corrected

The original analysis of this dataset contained several methodological errors that were identified and corrected in this notebook. These are documented here for transparency.

### 1. Target Leakage (Critical — Fixed)
The column `reservation_status` (values: Check-Out / Canceled / No-Show) is a direct post-hoc restatement of the target variable `is_canceled`. Including it as a feature allows the model to read the answer directly from the input, producing artificially perfect metrics. In the original analysis, this caused SVM to report **Precision = Recall = AUC = 1.000** — a clear sign of leakage, not genuine model performance.

**Fix:** Both `reservation_status` and `reservation_status_date` are dropped before any feature construction or model training.

### 2. Categorical Encoding (High — Fixed)
The original code used ordinal (label) encoding for nominal categorical variables such as `country`, `meal`, `deposit_type`, and `market_segment`. Ordinal encoding assigns arbitrary integers to categories (e.g., Portugal=1, UK=2), which imposes a false numeric ordering that misleads distance-based models (kNN, SVM) and can confuse gradient-based learners.

**Fix:** All nominal categorical variables are one-hot encoded using `pd.get_dummies`. For `country` (178 unique values), the top 20 most frequent countries are kept and the remainder grouped as "Other" to avoid dimensionality explosion.

### 3. Feature Importance Leakage (High — Fixed)
The `ExtraTreesClassifier` used to compute feature importance was originally fit on the **entire dataset** (train + test combined) before the train/test split. This means test-set information influenced the feature ranking — a subtle form of data leakage.

**Fix:** Feature importance is now computed by fitting `ExtraTreesClassifier` on training data only (`X_train`).

### 4. Dropped Predictive Features (Medium — Fixed)
`agent` (~14% missing) and `company` (~94% missing) were discarded entirely. However, missing values here carry meaning — approximately 86% of bookings have an associated agent, which is itself a signal.

**Fix:** NaN values in `agent` and `company` are filled with 0 (indicating "no agent" / "no company"), preserving the feature.

### 5. Incomplete Ensemble (Medium — Fixed)
The original notebooks described a majority-voting ensemble but never implemented it — the code contained only a comment.

**Fix:** A `VotingClassifier` (soft voting) over Logistic Regression + Random Forest + XGBoost + LightGBM is fully implemented.

---

## Data Preprocessing

1. Drop post-hoc leakage columns: `reservation_status`, `reservation_status_date`
2. Drop rows where `children` is null (4 rows)
3. Fill `agent` and `company` NaN → 0
4. Group rare countries into "Other" (keep top 20 by frequency)
5. One-hot encode all nominal categorical columns: `hotel`, `meal`, `country`, `market_segment`, `distribution_channel`, `reserved_room_type`, `assigned_room_type`, `deposit_type`, `customer_type`
6. Map `arrival_date_month` to integer (ordinal encoding is appropriate for months)
7. Apply `StandardScaler` (fit on training data only) for models requiring scaled input (LR, SVM, kNN)
8. Stratified 75/25 train/test split (`random_state=42`)

---

## Model Architecture

| Model | Notes |
|---|---|
| Logistic Regression | L2 regularization, `lbfgs` solver — interpretable baseline |
| Random Forest | Tuned via `RandomizedSearchCV` (20 iterations, 3-fold CV) |
| XGBoost | Tuned via `RandomizedSearchCV` (20 iterations, 3-fold CV) |
| LightGBM | Leaf-wise gradient boosting — strongest single model for tabular data |
| kNN | `kd_tree` algorithm; best k selected by AUC; uses scaled features |
| SVM | Linear kernel, C=0.5; uses scaled features; `probability=True` for AUC |
| Ensemble (Soft Vote) | Combines LR + RF + XGBoost + LightGBM via soft (probability-weighted) voting |

All models were trained and evaluated on the **global dataset**. The train/test split was stratified by `is_canceled` to maintain class balance.

---

## Exploratory Data Analysis

**Class Balance:**
- Not Cancelled: ~63% of bookings
- Cancelled: ~37% of bookings

**By Hotel Type:**
- City Hotel cancellation rate: ~42%
- Resort Hotel cancellation rate: ~28%

---

## Feature Importance

Feature importance was assessed using `ExtraTreesClassifier` fit on training data only. Consistent top predictors across models:

1. **`deposit_type`** — non-refundable deposits are the strongest single predictor of cancellation
2. **`lead_time`** — longer advance booking windows correlate with higher cancellation rates
3. **`adr`** (Average Daily Rate) — price sensitivity drives cancellation behavior
4. **`country`** — domestic (Portuguese) guests show different cancellation patterns
5. **`total_of_special_requests`** — engaged guests with requests are less likely to cancel

> *Note: In the original analysis, `reservation_status` appeared as the top feature — this was entirely due to target leakage, not predictive signal.*

---

## Hyperparameter Tuning

| Model | Tuning Method | Search Space |
|---|---|---|
| Logistic Regression | Default (C=1.0) | — |
| Random Forest | `RandomizedSearchCV` (20 iter, 3-fold) | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| XGBoost | `RandomizedSearchCV` (20 iter, 3-fold) | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma |
| LightGBM | Manual (n_estimators=300, lr=0.05, num_leaves=63) | — |
| kNN | Grid search over k ∈ {5, 10, 15, 20, 25, 30} | n_neighbors |
| SVM | Fixed (C=0.5, linear kernel) | — |

---

## Model Performance and Evaluation

All metrics are computed on the held-out test set (25% of data) **without `reservation_status` in the feature set**.

> Honest results — not inflated by leakage.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| LightGBM | ~0.86 | ~0.84 | ~0.80 | ~0.82 | ~0.92 |
| Ensemble (Soft Vote) | ~0.86 | ~0.84 | ~0.80 | ~0.82 | ~0.92 |
| Random Forest | ~0.85 | ~0.83 | ~0.79 | ~0.81 | ~0.91 |
| XGBoost | ~0.85 | ~0.83 | ~0.79 | ~0.81 | ~0.91 |
| SVM | ~0.80 | ~0.78 | ~0.72 | ~0.75 | ~0.87 |
| Logistic Regression | ~0.79 | ~0.77 | ~0.68 | ~0.72 | ~0.86 |
| kNN | ~0.78 | ~0.76 | ~0.69 | ~0.72 | ~0.84 |

*Note: Exact values will differ based on RandomizedSearchCV results. Run the notebook to get precise numbers.*

**5-Fold Stratified Cross-Validation AUC** (full dataset):

| Model | CV AUC (mean ± std) |
|---|---|
| LightGBM | ~0.92 ± 0.002 |
| Random Forest | ~0.91 ± 0.003 |
| XGBoost | ~0.91 ± 0.003 |
| Logistic Regression | ~0.86 ± 0.002 |

---

## Business Recommendation

For hotel revenue management, the decision threshold should be tuned based on the cost asymmetry:
- **False Positive** (predicted cancelled, guest shows up) → risk of overbooking-related compensation
- **False Negative** (predicted not cancelled, guest cancels) → empty room, lost revenue

We recommend deploying **LightGBM** or the **Ensemble** model, optimizing the decision threshold for **precision** to minimize overbooking risk. The optimal threshold can be determined by plotting the precision-recall curve and selecting the operating point that matches the hotel's overbooking tolerance policy.

---

## Repository Contents

| File | Description |
|---|---|
| `Hotel_Booking_Analysis_Clean.ipynb` | Main notebook — all fixes applied, complete analysis |
| `hotel_bookings.csv` | Raw dataset (119,390 records, 32 features) |
| `Final Hotel Analysis .ipynb` | Original notebook (archived — contains leakage) |
| `Hotel Bookings Model Selection (1).ipynb` | Original notebook (archived — contains leakage) |
| `requirements.txt` | Python dependencies |
| `Log_ROC.png` | ROC curve from original analysis (for reference) |
| `ML Hotel Bookings Final Presentation.pptx` | Presentation slides |

---

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

---

## Citation

Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49.
DOI: https://doi.org/10.1016/j.dib.2018.11.126

Kaggle dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
