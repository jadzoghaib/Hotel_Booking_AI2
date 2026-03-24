# Hotel Booking Cancellation Prediction & Overbooking Decision Tool

**Course:** Artificial Intelligence II
**Author:** Group 1 MiBA
**Dataset:** Hotel Booking Demand — Antonio, Almeida & Nunes (2019), *Data in Brief*, Vol. 22
**Kaggle:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

---

## 1. Business Problem

Hotels operate in an environment of chronic uncertainty: a significant fraction of confirmed reservations — roughly 37% in this dataset — are ultimately cancelled. This creates a direct revenue problem. Rooms left empty because of late cancellations cannot be re-sold on short notice. Hotels have historically responded by overbooking: accepting more reservations than they have rooms, on the assumption that some fraction will cancel.

The challenge is that overbooking blindly is just as bad as not overbooking at all. Accept too many bookings and you are walking guests — denying rooms to confirmed customers — which causes reputational damage, compensation costs, and regulatory exposure. Accept too few and you leave money on the table every night.

**The question this project answers:** given a new caller requesting a booking at a hotel that is already at full capacity, should the hotel accept the booking or reject it?

This is not a simple yes/no rule. It depends on:
- The cancellation likelihood of the new caller (predicted by a machine learning model)
- The cancellation likelihood of every existing booking already on the books (also from the model)
- The hotel's own risk tolerance — how much overbooking probability is acceptable?

The project is split into two components:
1. A **supervised ML notebook** that trains a cancellation probability model on 119,390 historical bookings
2. A **Streamlit decision-support app** that uses those probabilities to run a Monte Carlo simulation and produce a real-time ACCEPT / REJECT decision for each new booking request

---

## 2. The ML Notebook (`Hotel_Booking_Analysis_Clean.ipynb`)

### 2.1 Data

The dataset contains 119,390 hotel bookings from two Portuguese hotels (City Hotel and Resort Hotel) between 2015 and 2017. Each row is one booking with features describing the guest, the reservation, the stay, and the pricing. The target variable is `is_canceled` (1 = cancelled, 0 = showed up).

**Overall cancellation rate: ~37%**

### 2.2 Data Leakage — The Most Important Step

Three columns were dropped before any model sees the data:

| Column | Why it is leakage |
|---|---|
| `reservation_status` | Literally encodes whether the booking was cancelled — set after the fact |
| `reservation_status_date` | The date the status was updated — also post-hoc |
| `assigned_room_type` | Only assigned at check-in, not available at booking time |

Failing to drop these produces AUC scores in the 0.87–1.0 range that look impressive but are completely meaningless — the model is just reading the answer. Removing them gives honest AUC scores in the 0.88–0.96 range, which reflect actual predictive power at the time a booking decision needs to be made.

### 2.3 Feature Engineering

- **Missing values:** `agent` and `company` NaN → 0 (semantically correct: no agent, no company). Four rows with null `children` were dropped.
- **Rare countries:** Countries outside the top 20 by frequency were grouped into a single `Other` category to avoid high-cardinality one-hot explosion.
- **Month encoding:** Month names mapped to ordinal integers (January = 1 … December = 12) to preserve seasonal ordering.
- **One-hot encoding:** All nominal categorical columns (hotel, meal, country, market segment, distribution channel, room type, deposit type, customer type) were one-hot encoded.

### 2.4 Train / Test Split

80/20 stratified split. Stratification ensures the 37% cancellation rate is preserved in both partitions. The test set is held out completely and never used during training, hyperparameter search, or threshold selection.

### 2.5 Feature Scaling

`StandardScaler` fitted **only on the training set**, then applied to the test set. This prevents the scaler from seeing test-set distributional information — a subtle but real form of data leakage. Scaling is only required by Logistic Regression and the Neural Network (tree-based models are scale-invariant).

### 2.6 Feature Importance & Selection

A fast Random Forest (100 trees) was fitted on the training data to rank all ~78 post-encoding features by average impurity reduction (Gini importance). Rather than arbitrarily keeping a fixed top-N, we used a **95% cumulative importance threshold**: keep the minimum number of features whose combined importance sums to at least 95% of the total. This yielded **40 features** based on the actual data distribution.

`hotel_City Hotel` was forced into the feature set regardless of its importance rank, because hotel type is a required input in the decision app — dropping it would mean the app could not distinguish between hotel types at inference time.

**Top predictive features:**
- `lead_time` — longer lead times strongly correlate with cancellation (the guest has more time to change plans)
- `deposit_type_No Deposit` / `deposit_type_Non Refund` — deposit terms are among the strongest signals; non-refundable deposits massively reduce cancellation probability
- `adr` (average daily rate) — higher prices increase cancellation sensitivity
- `country_PRT` — domestic Portuguese bookings have distinct cancellation patterns
- `total_of_special_requests` — guests who made specific requests are more committed to showing up
- `agent` — bookings through certain agents have systematically different cancellation rates

### 2.7 Models Trained

Six models were trained on the same 80% training split and evaluated on the held-out 20% test set:

| Model | Why it was included |
|---|---|
| Logistic Regression | Linear baseline (S1 framework). Fully interpretable. Performance floor — any more complex model that fails to beat this is not worth the cost |
| Random Forest | Bagging ensemble with random feature subsets at each split (S2). Reduces variance without increasing bias. `max_features='sqrt'` as recommended in lectures |
| XGBoost | Gradient boosting — sequentially corrects residuals of previous trees (S2 extension). L2 regularisation and subsampling reduce overfitting |
| LightGBM | Microsoft's gradient boosting variant. Grows trees leaf-wise rather than level-wise — faster and often more accurate on large tabular datasets |
| Neural Network (MLP) | Two hidden layers (128 → 64 units), ReLU activations, L2 regularisation (α = 0.001), early stopping (S5/S6). Included as a flexible non-linear benchmark |
| Ensemble (Soft Voting) | Combines LR + RF + XGBoost + LightGBM via probability-weighted averaging. Diversity between a linear model and three tree-based models reduces variance beyond any single learner |

### 2.8 Hyperparameter Tuning

Random Forest and XGBoost were tuned with `RandomizedSearchCV` (20 random combinations, 3-fold stratified CV). Random search was preferred over grid search because it explores the hyperparameter space more efficiently — 20 random combinations from a large grid finds near-optimal parameters faster than exhaustive search.

**Random Forest search space:**
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, 30, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: fixed to `'sqrt'` (lecture recommendation)

**XGBoost search space:**
- `n_estimators`: [100, 200, 300]
- `max_depth`: uniform integer [3–7]
- `learning_rate`: uniform(0.01, 0.30)
- `subsample` / `colsample_bytree`: uniform(0.6, 1.0)
- `reg_lambda`: [0.5, 1, 2, 5]

### 2.9 Results

| Model | Accuracy | Precision | Recall | F1 | Test AUC | AUC Gap | CV AUC (5-fold) |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 0.889 | 0.879 | 0.811 | 0.844 | **0.958** | 0.039 | 0.960 |
| XGBoost | 0.879 | 0.857 | 0.807 | 0.831 | 0.952 | 0.017 | 0.944 |
| Ensemble (Soft Vote) | 0.878 | 0.876 | 0.780 | 0.826 | 0.951 | 0.032 | — |
| LightGBM | 0.877 | 0.859 | 0.798 | 0.827 | 0.951 | 0.012 | 0.947 |
| Neural Network (MLP) | 0.863 | 0.846 | 0.771 | 0.807 | 0.939 | 0.019 | 0.934 |
| Logistic Regression | 0.805 | 0.803 | 0.628 | 0.705 | 0.885 | 0.002 | 0.885 |

**Primary metric: Test AUC** (threshold-independent, robust to class imbalance).

Random Forest achieved the highest test AUC (0.958) and was selected as the saved model. The AUC gap (train AUC − test AUC = 0.039) is within acceptable bounds.

LightGBM achieved the lowest AUC gap (0.012), meaning it generalises most consistently. In a production deployment, LightGBM would be the preferred choice. For this project the highest test AUC was used as the selection criterion.

### 2.10 Threshold Selection

The default classification threshold of 0.5 is not the optimal operating point. For hotel cancellations, the two error types carry asymmetric business costs:

- **False Positive** (predict cancellation, guest shows up) → hotel oversells → guest denied room → reputational damage
- **False Negative** (predict no cancellation, guest cancels) → empty room → lost revenue

The F1-maximising threshold was identified on a **validation set carved from the training data** (never the test set). Using the test set for threshold selection would inflate reported performance.

### 2.11 Cross-Validation

5-fold stratified CV was run for the four main single models. The Ensemble and MLP were excluded due to computational cost. CV AUC mean ± std provides a more robust estimate than a single split and reveals stability across different data partitions.

### 2.12 Bias-Variance Analysis

| Model | Bias | Variance | Notes |
|---|---|---|---|
| Logistic Regression | High | Low | Linear boundary — underfits the non-linear patterns in this data |
| Random Forest | Low | Medium | Bagging reduces variance vs. a single tree; some residual overfitting |
| XGBoost | Low | Low-Medium | Regularisation controls variance well |
| LightGBM | Low | Low | Best bias-variance balance overall |
| Neural Network | Low | Medium | L2 + early stopping limit overfitting |
| Ensemble | Low | Low | Averaging diverse models further reduces variance |

### 2.13 Construct Gap

Following the S3 framework: `is_canceled` is the **observed label (Y)** but not the true **construct of interest (Yᶜ)**. The construct is *revenue impact* — but not all cancellations are equally costly:

- A cancellation 90 days in advance under a free-cancellation policy → room can be re-sold, cost ≈ 0
- A last-minute no-show on a non-refundable booking → direct revenue loss

The binary label flattens this distinction. A more precise model would weight cancellations by `lead_time` and `deposit_type` to reflect actual revenue exposure. The current model is a useful approximation but its performance metrics are an imperfect proxy for the hotel's true objective.

---

## 3. The Stochastic Overbooking Problem

### 3.1 Problem Formulation

At any given time, the hotel holds `N` existing bookings, each with a cancellation probability `pᵢ` from the ML model. Actual arrivals on check-in day are not known in advance — each booking is an independent coin flip:

```
guest i shows up  ~  Bernoulli(1 - pᵢ)
total arrivals    =  Σ Bernoulli(1 - pᵢ)   for i = 1 … N
```

This sum is a **Poisson Binomial distribution** — a sum of independent but non-identically distributed Bernoulli trials. It has no closed-form solution, so we solve it with Monte Carlo simulation.

### 3.2 Monte Carlo Simulation

For each decision, 10,000 scenarios are simulated:

```
For each of 10,000 simulations:
    For each booking i:
        flip a weighted coin: show up with probability (1 - pᵢ)
    Count total arrivals

P(overbooking) = simulations where arrivals > capacity / 10,000
```

This gives the full **arrival distribution** — not just an expected value, but the entire spread of possible outcomes — along with the probability of exceeding capacity.

### 3.3 Decision Rule

```
P(arrivals > capacity  WITH new booking) < risk_threshold  →  ACCEPT
P(arrivals > capacity  WITH new booking) ≥ risk_threshold  →  REJECT
```

The hotel manager sets the risk threshold manually (default: 5%). Two simulations run for every decision: one **without** the new booking (baseline risk) and one **with** it. The delta shows exactly how much this one booking increases overbooking exposure.

### 3.4 Why This Works

Most new callers at a fully-booked hotel have a non-zero cancellation probability. If the existing bookings have enough expected cancellations, accepting one more booking may carry negligible overbooking risk. The model makes this precise rather than leaving it to intuition.

---

## 4. The Streamlit App (`app.py`)

### 4.1 What It Does

On startup, the app automatically loads the hotel to **full capacity** using historical data. It samples exactly `capacity` bookings (225 for City Hotel, 190 for Resort Hotel), stratified 50/50 between cancelled and not-cancelled, runs them through the ML model to get cancellation probabilities, and displays the hotel's current expected occupancy.

The manager then fills in a form for the **new caller**. When submitted, the app:

1. Runs the new booking through the ML model to get its cancellation probability
2. Runs the Monte Carlo simulation with and without the new booking
3. Compares P(overbooking) to the risk threshold
4. Displays **ACCEPT** (green) or **REJECT** (red) with supporting metrics
5. Shows the full simulated arrival distribution as a histogram with a shaded overbooking zone

### 4.2 App Layout

**Sidebar:**
- Hotel type selector (City Hotel / Resort Hotel)
- Capacity input (pre-filled to 225 / 190, editable)
- Risk threshold slider (1%–20%, default 5%)

**Main panel — top:** Current booking status (total bookings, expected arrivals, expected occupancy %)

**Main panel — form:** New overbooking request with all 40 model features including arrival date, lead time, ADR, deposit type, market segment, customer type, country, meal plan, distribution channel, room type, agent, adults, children, stays, previous cancellations, previous bookings not cancelled, booking changes, parking spaces, and special requests.

**Main panel — output:**
- ACCEPT / REJECT badge
- Four metrics: cancellation probability of new booking, risk before, risk after, expected arrivals after
- Monte Carlo histogram with capacity line and shaded overbooking zone

---

## 5. Running the Project

### Prerequisites

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm joblib matplotlib
```

### Step 1 — Train the Model

Open `Hotel_Booking_Analysis_Clean.ipynb` in Jupyter and run **Kernel → Restart & Run All**. This trains all models, selects the best by test AUC, and saves it to `best_model.pkl`. Make sure `hotel_bookings.csv` is in the same directory.

### Step 2 — Launch the App

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### File Structure

```
hotel_booking_clean/
├── Hotel_Booking_Analysis_Clean.ipynb   # ML notebook
├── app.py                               # Streamlit app
├── preprocessing.py                     # Inference pipeline (mirrors notebook)
├── stochastic_model.py                  # Monte Carlo simulation
├── best_model.pkl                       # Saved model (generated by notebook)
├── hotel_bookings.csv                   # Raw dataset
└── README.md
```

---

## 6. Dataset Citation

Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
