# Hotel Booking Cancellation Prediction & Overbooking Decision Tool

**Course:** Artificial Intelligence II
**Author:** Group 1 MiBA
**Dataset:** Hotel Booking Demand — Antonio, Almeida & Nunes (2019), *Data in Brief*, Vol. 22
**Kaggle:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
**GitHub:** https://github.com/jadzoghaib/Hotel_Booking_AI2

> **Live demo:** [https://jadzoghaib-hotel-booking-ai2-app-my01c0.streamlit.app/](https://jadzoghaib-hotel-booking-ai2-app-my01c0.streamlit.app/)

---

## 1. Business Problem

Hotels operate in an environment of chronic uncertainty: a significant fraction of confirmed reservations — roughly 37% in this dataset — are ultimately cancelled. This creates a direct revenue problem. Rooms left empty because of late cancellations cannot be re-sold on short notice. Hotels have historically responded by overbooking: accepting more reservations than they have rooms, on the assumption that some fraction will cancel.

The challenge is that overbooking blindly is just as bad as not overbooking at all. Accept too many bookings and you are walking guests — denying rooms to confirmed customers — which causes reputational damage, compensation costs, and regulatory exposure. Accept too few and you leave money on the table every night.

**The question this project answers:** given a new caller requesting a booking at a hotel that is already at full capacity, should the hotel accept the booking or reject it?

This is not a simple yes/no rule. It depends on:
- The cancellation likelihood of the new caller (predicted by a machine learning model)
- The cancellation likelihood of every existing booking on the books for the same room type
- The hotel's own risk tolerance — how far above effective capacity expected arrivals may go

A fixed overbooking rule (e.g. "always overbook by 10%") ignores this heterogeneity entirely: a non-refundable direct booking and a free-cancellation online-TA booking carry vastly different cancellation risks. A guest who books months in advance through a third-party channel behaves very differently from one who books directly two days before arrival. By estimating per-booking cancellation probabilities, the system makes overbooking decisions conditional on the actual composition of the current booking portfolio, not a static guess.

The project is split into two components:
1. A **supervised ML notebook** that trains a cancellation probability model on 119,390 historical bookings
2. A **Streamlit decision-support app** that uses those probabilities to evaluate each new booking request in real time

**Full pipeline:** Preprocessing → Random Forest scoring → room-type expected arrivals → accept/reject decision (+ Monte Carlo reference histogram)

---

## 2. The ML Notebook (`notebooks/Hotel_Booking_Analysis_Clean.ipynb`)

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

### 2.3 Exploratory Data Analysis

EDA reveals several systematic patterns that directly shaped modelling decisions:

- **Lead time:** Cancellations are heavily concentrated among bookings with long lead times. The distribution for cancelled bookings is right-skewed and extends well beyond 200 days, while arrivals cluster near zero. This is the strongest single visual signal in the data.
- **ADR (average daily rate):** Cancellations spread across all price levels, but higher-priced bookings show a slight uptick in cancellation likelihood — price sensitivity increases with rate.
- **Previous cancellations:** Guests with any prior cancellation history cancel at dramatically higher rates. This is a near-binary behavioural signal.
- **Special requests:** Guests who make specific requests (bed preferences, dietary needs, etc.) are meaningfully less likely to cancel — they have invested effort in the stay.
- **Hotel type:** The City Hotel cancels at a higher rate than the Resort Hotel, reflecting differences in booking channel mix and guest profile.
- **Stay length:** Week-night and weekend-night stay distributions are similar between cancelled and non-cancelled groups — duration alone is not strongly predictive.

### 2.4 Feature Engineering

- **Missing values:** `agent` and `company` NaN → 0 (semantically correct: no agent, no company). Four rows with null `children` were dropped.
- **Rare countries:** Countries outside the top 20 by frequency were grouped into a single `Other` category to avoid high-cardinality one-hot explosion.
- **Month encoding:** Month names mapped to ordinal integers (January = 1 … December = 12) to preserve seasonal ordering.
- **Room type encoding:** Room types A–L mapped to ordinal integers (A = 1 … L = 9), reflecting an assumed quality hierarchy.
- **One-hot encoding:** All nominal categorical columns (hotel, meal, country, market segment, distribution channel, deposit type, customer type) were one-hot encoded.

### 2.5 Train / Test Split

80/20 stratified split. Stratification ensures the 37% cancellation rate is preserved in both partitions. The test set is held out completely and never used during training, hyperparameter search, or threshold selection.

### 2.6 Feature Scaling

`StandardScaler` fitted **only on the training set**, then applied to the test set. This prevents the scaler from seeing test-set distributional information — a subtle but real form of data leakage. Scaling is only required by Logistic Regression and the Neural Network (tree-based models are scale-invariant).

### 2.7 Feature Importance & Selection

A fast Random Forest (100 trees) was fitted on the training data to rank all ~78 post-encoding features by average impurity reduction (Gini importance). Rather than arbitrarily keeping a fixed top-N, we used a **95% cumulative importance threshold**: keep the minimum number of features whose combined importance sums to at least 95% of the total. This yielded **36 features** — roughly half the candidate set — with the importance curve visibly flattening beyond that point. This reduces inference latency and prevents overfitting to noise features without sacrificing predictive power, an important consideration for a live decision tool.

`hotel_City Hotel` was forced into the feature set regardless of its importance rank, because hotel type is a required input in the decision app — dropping it would mean the app could not distinguish between hotel types at inference time.

**Top predictive features:**
- `lead_time` — longer lead times strongly correlate with cancellation (the guest has more time to change plans)
- `deposit_type_No Deposit` / `deposit_type_Non Refund` — deposit terms are among the strongest signals; non-refundable deposits massively reduce cancellation probability
- `adr` (average daily rate) — higher prices increase cancellation sensitivity
- `country_PRT` — domestic Portuguese bookings have distinct cancellation patterns
- `total_of_special_requests` — guests who made specific requests are more committed to showing up
- `agent` — bookings through certain agents have systematically different cancellation rates

### 2.8 Models Trained

Six models were trained on the same 80/20 stratified split:

| Model | Why it was included |
|---|---|
| Logistic Regression | Interpretable linear baseline. Any more complex model that fails to beat this is not worth the cost |
| Random Forest | Handles mixed feature types and non-linear interactions without extra engineering. Robust to class imbalance and natively outputs the per-booking probabilities required by the Monte Carlo engine |
| XGBoost | Gradient boosting for structured tabular data with layered interactions (lead time, deposit type, market segment). L2 regularisation and subsampling control overfitting |
| LightGBM | Same boosting logic as XGBoost but trains significantly faster — important since in practice the model would be periodically retrained to adjust to evolving cancellation trends |
| Neural Network (MLP) | Two hidden layers (128 → 64 units), ReLU, L2 regularisation, early stopping. Tests whether a deep architecture captures high-order feature interactions that tree models miss |
| Ensemble (Soft Voting) | Averages probability outputs of the four non-linear models to reduce variance. Included to verify whether combining predictions outperforms the single best model |

### 2.9 Hyperparameter Tuning

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

### 2.10 Results

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

LightGBM achieved the lowest AUC gap (0.012), meaning it generalises most consistently. In a production deployment, LightGBM would be the preferred choice given its retraining speed. For this project the highest test AUC was used as the selection criterion.

**Confusion matrix — Random Forest (held-out test set):**

|  | Predicted: Not Cancelled | Predicted: Cancelled |
|---|---|---|
| **Actual: Not Cancelled** | 17,558 (TN) | 1,254 (FP) |
| **Actual: Cancelled** | 2,061 (FN) | 8,881 (TP) |

The model correctly identifies ~81% of actual cancellations (recall 0.811) while keeping false positives low. XGBoost, LightGBM, and the Ensemble show similar confusion structures — the consistent pattern across all tree-based models confirms that **non-linear feature interactions are important** in this setting, and that the logistic regression and MLP performance gaps are not noise.

### 2.11 Threshold Selection

The default classification threshold of 0.5 is not the optimal operating point. For hotel cancellations, the two error types carry asymmetric business costs:

- **False Positive** (predict cancellation, guest shows up) → hotel oversells → guest denied room → reputational damage
- **False Negative** (predict no cancellation, guest cancels) → empty room → lost revenue

The F1-maximising threshold was identified on a **validation set carved from the training data** (never the test set). Using the test set for threshold selection would inflate reported performance.

### 2.12 Cross-Validation

5-fold stratified CV was run for the four main single models. The Ensemble and MLP were excluded due to computational cost. CV AUC mean ± std provides a more robust estimate than a single split and reveals stability across different data partitions.

### 2.13 Bias-Variance Analysis

| Model | Bias | Variance | Notes |
|---|---|---|---|
| Logistic Regression | High | Low | Linear boundary — underfits the non-linear patterns in this data |
| Random Forest | Low | Medium | Bagging reduces variance vs. a single tree; some residual overfitting |
| XGBoost | Low | Low-Medium | Regularisation controls variance well |
| LightGBM | Low | Low | Best bias-variance balance overall |
| Neural Network | Low | Medium | L2 + early stopping limit overfitting |
| Ensemble | Low | Low | Averaging diverse models further reduces variance |

### 2.14 Construct Gap

Following the S3 framework: `is_canceled` is the **observed label (Y)** but not the true **construct of interest (Yᶜ)**. The construct is *revenue impact* — but not all cancellations are equally costly:

- A cancellation 90 days in advance under a free-cancellation policy → room can be re-sold, cost ≈ 0
- A last-minute no-show on a non-refundable booking → direct revenue loss

The binary label flattens this distinction. A more precise model would weight cancellations by `lead_time` and `deposit_type` to reflect actual revenue exposure. The current model is a useful approximation but its performance metrics are an imperfect proxy for the hotel's true objective.

---

## 3. The Stochastic Overbooking Problem

### 3.1 Problem Formulation

At any given time, the hotel holds existing bookings for each room type, each with a cancellation probability `pᵢ` from the ML model. Actual arrivals on check-in day are not known in advance — each booking is an independent coin flip:

```
guest i shows up  ~  Bernoulli(1 - pᵢ)
total arrivals    =  Σ Bernoulli(1 - pᵢ)   for i = 1 … N
```

This sum is a **Poisson Binomial distribution** — a sum of independent but non-identically distributed Bernoulli trials. It has no closed-form solution, so we approximate it with Monte Carlo simulation.

### 3.2 Room-Type Level Analysis

Decisions are made at the **room-type level**, not the hotel level. When a new caller requests Room Type X:

- Only existing bookings for Type X compete for the same inventory
- An **upgrade buffer** is added: guests can be upgraded to higher-tier room types if Type X is over capacity. The upgrade buffer equals the sum of all capacities for room types above X in the quality hierarchy (A < B < … < L)
- **Effective capacity** = Type X rooms + upgrade buffer

### 3.3 Decision Rule

```
expected_arrivals_after = Σ (1 - pᵢ) for all Type X bookings + (1 - p_new)
max_allowed             = effective_capacity × (1 + overbook_buffer)

expected_arrivals_after ≤ max_allowed  →  ACCEPT
expected_arrivals_after >  max_allowed  →  REJECT
```

The hotel manager sets the overbook buffer manually (0%–50%, default 10%). The overbook buffer controls how far above effective capacity expected arrivals are permitted to go. Setting it to 0% means expected arrivals must never exceed effective capacity; 10% allows up to 110% of effective capacity.

### 3.4 Monte Carlo Simulation (Reference)

For transparency, 10,000 Monte Carlo scenarios are also simulated for the competing room type (with the new booking included):

```
For each of 10,000 simulations:
    For each Type X booking i:
        flip a weighted coin: show up with probability (1 - pᵢ)
    Count total arrivals

P(overbooking) = simulations where arrivals > base Type X capacity / 10,000
```

The resulting arrival distribution is displayed as a histogram with capacity lines, giving the manager a visual sense of risk even when the expected-value decision is clear.

### 3.5 Why This Works

Most new callers at a fully-booked hotel have a non-zero cancellation probability. If the existing bookings have enough expected cancellations, accepting one more booking may carry negligible overbooking risk. The model makes this precise rather than leaving it to intuition, and the upgrade buffer adds a practical safety valve by accounting for the hotel's ability to move guests to better rooms.

---

## 4. The Streamlit App (`app.py`)

### 4.1 What It Does

On startup, the app automatically populates the hotel to **full capacity** using synthetic data: each room type is filled to its capacity with bookings assigned a uniform 30% cancellation probability. This represents a fully booked hotel in a neutral baseline state, with no dependency on the historical CSV at runtime.

The manager fills in a form for the **new caller**. When submitted, the app:

1. Runs the new booking through the ML model to get its cancellation probability
2. Identifies all existing bookings competing for the same room type
3. Computes expected arrivals before and after adding the new booking
4. Compares expected arrivals after to `effective_capacity × (1 + overbook_buffer)`
5. Displays **ACCEPT** (green) or **REJECT** (red) with supporting metrics
6. Shows the full simulated arrival distribution (10,000-run Monte Carlo) as a histogram with capacity lines

### 4.2 App Layout

**Sidebar:**
- Hotel type selector (City Hotel / Resort Hotel)
- Hotel capacity display (225 rooms / 190 rooms)
- Max Overbook Buffer slider (0%–50%, default 10%) — controls how far above effective capacity expected arrivals may go
- Room type capacities breakdown (expandable)

**Main panel — top:** Current booking status (total bookings, expected arrivals, expected occupancy %, overbook buffer)

**Main panel — form:** New overbooking request with all model features including arrival date, lead time, ADR, deposit type, market segment, customer type, country, meal plan, distribution channel, room type, agent, adults, children, stays, previous cancellations, previous bookings not cancelled, booking changes, parking spaces, and special requests.

**Main panel — output:**
- Cancellation likelihood warning for the new booking
- ACCEPT / REJECT badge with explanation
- Five metrics: cancellation probability, competing bookings count, expected arrivals before, expected arrivals after (with delta), effective capacity
- Monte Carlo histogram with base capacity line (red), effective capacity line (orange), and mean arrivals marker (green)

### 4.3 Room Type Capacities

Capacities are derived from dataset proportions:

| Room Type | City Hotel | Resort Hotel |
|---|---|---|
| A | 60 | 50 |
| B | 50 | 42 |
| C | 40 | 34 |
| D | 30 | 26 |
| E | 20 | 18 |
| F | 10 | 10 |
| G | 8 | 6 |
| H | 4 | 3 |
| L | 3 | 1 |

---

## 5. Running the Project

### Prerequisites

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm joblib matplotlib
```

Or restore the full Conda environment:

```bash
conda env create -f environment.yml
conda activate hotel_booking
pip install streamlit joblib
```

### Step 1 — Train the Model

Open `notebooks/Hotel_Booking_Analysis_Clean.ipynb` in Jupyter and run **Kernel → Restart & Run All**. This trains all models, selects the best by test AUC, and saves it to `best_model.pkl`. Make sure `hotel_bookings.csv` is in the project root.

### Step 2 — Compress the Model (optional)

```bash
python scripts/compress_simple.py
```

This produces `best_model_compressed.pkl`, which the app loads by default. Compression reduces file size while preserving prediction accuracy.

### Step 3 — Launch the App

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### File Structure

```
hotel_booking_clean/
├── notebooks/
│   └── Hotel_Booking_Analysis_Clean.ipynb   # ML notebook
├── scripts/
│   ├── compress_model.py                    # Model compression (joblib)
│   ├── compress_simple.py                   # Simplified compression script
│   └── generate_report.py                   # Generates .docx project report
├── docs/
│   ├── ML Hotel Bookings Final Presentation.pptx
│   └── Reference Study.pdf
├── app.py                                   # Streamlit app
├── preprocessing.py                         # Inference pipeline (mirrors notebook)
├── stochastic_model.py                      # Monte Carlo simulation
├── best_model.pkl                           # Saved model (generated by notebook)
├── best_model_compressed.pkl                # Compressed model (loaded by app)
├── hotel_bookings.csv                       # Raw dataset
├── requirements.txt                         # Pip dependencies
├── environment.yml                          # Conda environment
└── README.md
```

---

## 6. Limitations & Critical Reflection

### Incomplete cancellation definition

`is_canceled` captures formal cancellations but excludes **no-shows** — guests who neither cancel nor arrive. The model therefore underestimates true arrival risk: some bookings labelled "not cancelled" will still result in empty rooms. In a production system, no-show rates should be tracked separately and factored into the overbooking buffer.

### Data staleness

The training data spans 2015–2017. Cancellation behaviour has shifted materially since then — the post-pandemic period saw a rise in flexible bookings, last-minute planning, and OTA channel growth. The model will misestimate risk until retrained on more recent data. Because the hotel would need to retrain periodically, LightGBM's speed advantage over Random Forest becomes a practical argument for switching models in deployment.

### Proxy bias

Country of origin is one of the top predictive features, but it may proxy for demographic characteristics rather than booking intent. This introduces the risk of biased predictions across customer groups and warrants a fairness audit before any live deployment.

### Synthetic baseline population

The app initialises the hotel with a uniform 30% cancellation probability per booking rather than scoring actual historical reservations. This is a modelling simplification: the true baseline risk depends on the actual mix of bookings on a given night. Mis-specifying the baseline population biases expected occupancy estimates and downstream accept/reject decisions.

### Upgrade buffer assumptions

The room-type upgrade logic assumes perfect substitutability within the tier hierarchy — that any Type A guest can be moved to Type B if needed. In practice, guests have contractual and preference-based expectations about their room type. The upgrade buffer is a useful approximation but overstates the hotel's true flexibility.

---

## 7. Dataset Citation

Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
