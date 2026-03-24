"""
preprocessing.py
----------------
Replicates the notebook's feature engineering pipeline for a single booking
at inference time.

Usage
-----
    from preprocessing import preprocess_booking
    import joblib, pandas as pd

    model = joblib.load('best_model.pkl')
    booking = {
        'hotel': 'City Hotel',
        'lead_time': 45,
        'arrival_date_year': 2016,
        'arrival_date_month': 'July',
        ...
    }
    X = preprocess_booking(booking, model.feature_names_in_)
    prob_cancel = model.predict_proba(X)[0, 1]
"""

import pandas as pd
import numpy as np

# ── Constants (mirror the notebook) ──────────────────────────────────────────

TOP_COUNTRIES = [
    'PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL',
    'BRA', 'NLD', 'USA', 'CHE', 'CN',  'AUT', 'SWE', 'CHN',
    'POL', 'ISR', 'RUS', 'NOR',
]

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3,    'April': 4,
    'May': 5,     'June': 6,     'July': 7,      'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
}

NOMINAL_COLS = [
    'hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
    'deposit_type', 'customer_type',
]

# Ordinal encoding for room type (alphabetical = assumed quality order, P excluded)
ROOM_TYPE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'L': 9}


def preprocess_booking(booking: dict, expected_features) -> pd.DataFrame:
    """
    Transform a raw booking dict into a model-ready single-row DataFrame.

    Parameters
    ----------
    booking : dict
        Raw booking fields, e.g.:
        {
            'hotel': 'City Hotel',
            'lead_time': 45,
            'arrival_date_year': 2016,
            'arrival_date_month': 'July',   # string month name
            'arrival_date_week_number': 27,
            'arrival_date_day_of_month': 5,
            'stays_in_weekend_nights': 1,
            'stays_in_week_nights': 2,
            'meal': 'BB',
            'country': 'PRT',
            'market_segment': 'Online TA',
            'distribution_channel': 'TA/TO',
            'reserved_room_type': 'A',
            'deposit_type': 'No Deposit',
            'agent': 0,
            'customer_type': 'Transient',
            'adr': 85.0,
            'required_car_parking_spaces': 0,
            'total_of_special_requests': 1,
            'previous_cancellations': 0,
            'booking_changes': 0,
        }
    expected_features : array-like of str
        Exact feature names the model was trained on.
        Pass model.feature_names_in_ from the loaded RandomForest.

    Returns
    -------
    pd.DataFrame with one row and columns matching expected_features.
    Missing columns are filled with 0 (safe for one-hot encoded dummies).
    """
    row = booking.copy()

    # ── 1. Fill missing agent / company with 0 ────────────────────────────
    row.setdefault('agent', 0)
    row.setdefault('company', 0)
    if row['agent'] is None or (isinstance(row['agent'], float) and np.isnan(row['agent'])):
        row['agent'] = 0
    if row['company'] is None or (isinstance(row['company'], float) and np.isnan(row['company'])):
        row['company'] = 0

    # ── 2. Group rare countries → 'Other' ────────────────────────────────
    country = row.get('country', 'Other')
    if country not in TOP_COUNTRIES:
        country = 'Other'
    row['country'] = country

    # ── 3. Ordinal-encode month ───────────────────────────────────────────
    month_val = row.get('arrival_date_month', 1)
    if isinstance(month_val, str):
        row['arrival_date_month'] = MONTH_MAP.get(month_val, 1)

    # ── 4. Ordinal-encode room type ───────────────────────────────────────
    room_val = row.get('reserved_room_type', 'A')
    row['reserved_room_type'] = ROOM_TYPE_MAP.get(room_val, 1)

    # ── 5. Build single-row DataFrame ────────────────────────────────────
    df = pd.DataFrame([row])

    # ── 6. One-hot encode nominal columns (drop_first=False, same as notebook) ──
    cols_to_encode = [c for c in NOMINAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)

    # ── 6. Align to model's expected feature set ──────────────────────────
    # Add any missing columns as 0 (e.g. a country not seen in this booking)
    # Drop any extra columns not in the model
    df = df.reindex(columns=expected_features, fill_value=0)

    return df


def preprocess_batch(df_raw: pd.DataFrame, expected_features) -> pd.DataFrame:
    """
    Same pipeline applied to a DataFrame of multiple bookings (e.g. uploaded CSV).

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw bookings with the same columns as hotel_bookings.csv
        (minus the leakage columns — those will be ignored if present).
    expected_features : array-like of str

    Returns
    -------
    pd.DataFrame aligned to expected_features.
    """
    df = df_raw.copy()

    # Drop leakage columns if present
    leakage = ['reservation_status', 'reservation_status_date',
               'assigned_room_type', 'is_canceled']
    df.drop(columns=[c for c in leakage if c in df.columns], inplace=True)

    # Fill agent / company NaN
    df['agent']   = df['agent'].fillna(0)   if 'agent'   in df.columns else 0
    df['company'] = df['company'].fillna(0) if 'company' in df.columns else 0

    # Group rare countries
    if 'country' in df.columns:
        df['country'] = df['country'].apply(
            lambda x: x if x in TOP_COUNTRIES else 'Other'
        ).fillna('Other')

    # Ordinal-encode month
    if 'arrival_date_month' in df.columns:
        df['arrival_date_month'] = df['arrival_date_month'].map(MONTH_MAP).fillna(1)

    # Ordinal-encode room type
    if 'reserved_room_type' in df.columns:
        df['reserved_room_type'] = df['reserved_room_type'].map(ROOM_TYPE_MAP).fillna(1)

    # One-hot encode
    cols_to_encode = [c for c in NOMINAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)

    # Align to model features
    df = df.reindex(columns=expected_features, fill_value=0)

    return df
