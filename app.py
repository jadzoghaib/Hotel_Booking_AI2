"""
app.py
------
Streamlit app for hotel overbooking decision support.

Run with:
    streamlit run app.py
"""

import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from preprocessing import preprocess_booking, preprocess_batch, TOP_COUNTRIES, MONTH_MAP
from stochastic_model import simulate_arrivals

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Overbooking Decision Tool",
    page_icon="🏨",
    layout="wide",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()
EXPECTED_FEATURES = model.feature_names_in_.tolist()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏨 Hotel Setup")

hotel_type = st.sidebar.selectbox(
    "Hotel Type",
    options=["City Hotel", "Resort Hotel"],
)

HOTEL_CAPACITY = {"City Hotel": 225, "Resort Hotel": 190}

capacity = st.sidebar.number_input(
    "Hotel Capacity (rooms)",
    min_value=1, max_value=2000,
    value=HOTEL_CAPACITY[hotel_type],
    step=1,
)

overbook_buffer = st.sidebar.slider(
    "Max Overbook Buffer",
    min_value=0, max_value=50, value=10, step=1,
    format="%d%%",
    help=(
        "How far above capacity you are willing to push expected arrivals. "
        "0% = never exceed capacity. 10% = accept bookings until expected arrivals reach 110% of capacity."
    ),
) / 100.0

max_expected_arrivals = capacity * (1 + overbook_buffer)
st.sidebar.caption(
    f"Target: fill to **{capacity}** rooms (100% occupancy).  \n"
    f"Hard cap: expected arrivals ≤ **{max_expected_arrivals:.0f}** "
    f"({100 + overbook_buffer*100:.0f}% of capacity)."
)

# ── Auto-load existing bookings at full capacity ──────────────────────────────
@st.cache_data
def load_full_capacity_bookings(hotel: str, n_rooms: int, seed: int = 42):
    """
    Sample exactly n_rooms bookings from the dataset for the given hotel,
    stratified 50/50 between is_canceled=0 and is_canceled=1.
    Returns the raw DataFrame (leakage cols still present — preprocess_batch drops them).
    """
    df = pd.read_csv("hotel_bookings.csv").dropna(subset=["children"])
    df = df[df["hotel"] == hotel]

    n_cancel = n_rooms // 2
    n_show   = n_rooms - n_cancel

    cancelled = df[df["is_canceled"] == 1].sample(n=n_cancel, random_state=seed)
    showed_up = df[df["is_canceled"] == 0].sample(n=n_show,   random_state=seed)

    return pd.concat([cancelled, showed_up]).sample(frac=1, random_state=seed)

existing_bookings = load_full_capacity_bookings(hotel_type, int(capacity))
X_existing        = preprocess_batch(existing_bookings, EXPECTED_FEATURES)
existing_probs    = model.predict_proba(X_existing)[:, 1].tolist()

st.sidebar.markdown("---")
st.sidebar.success(
    f"Hotel loaded at full capacity: **{len(existing_probs)} bookings** "
    f"(50% cancelled / 50% arrived)"
)

# ── Baseline occupancy display ────────────────────────────────────────────────
st.title("Hotel Overbooking Decision Tool")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Bookings", len(existing_probs))
expected_now = sum(1 - p for p in existing_probs)
col2.metric("Expected Arrivals", f"{expected_now:.1f}")
col3.metric("Expected Occupancy", f"{expected_now / capacity * 100:.1f}%")
col4.metric("Overbook Cap", f"{max_expected_arrivals:.0f} rooms ({100 + overbook_buffer*100:.0f}%)")

st.markdown("---")

# ── New Booking Form ──────────────────────────────────────────────────────────
st.subheader("New Overbooking Request")

with st.form("booking_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        arrival_date = st.date_input(
            "Arrival Date",
            value=datetime.date.today() + datetime.timedelta(days=30),
        )
        lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=700, value=30)
        adr = st.number_input("Average Daily Rate (€)", min_value=0.0, max_value=5000.0, value=100.0, step=1.0)

    with c2:
        deposit_type = st.selectbox(
            "Deposit Type",
            options=["No Deposit", "Non Refund", "Refundable"],
        )
        market_segment = st.selectbox(
            "Market Segment",
            options=["Online TA", "Offline TA/TO", "Direct", "Corporate",
                     "Groups", "Complementary", "Aviation", "Undefined"],
        )
        customer_type = st.selectbox(
            "Customer Type",
            options=["Transient", "Transient-Party", "Contract", "Group"],
        )

    with c3:
        country = st.selectbox(
            "Country",
            options=TOP_COUNTRIES + ["Other"],
        )
        meal = st.selectbox(
            "Meal Plan",
            options=["BB", "HB", "FB", "SC", "Undefined"],
        )
        distribution_channel = st.selectbox(
            "Distribution Channel",
            options=["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        stays_week = st.number_input("Week Nights", min_value=0, max_value=30, value=2)
        stays_weekend = st.number_input("Weekend Nights", min_value=0, max_value=20, value=1)
        reserved_room = st.selectbox(
            "Reserved Room Type",
            options=["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
        )
    with c5:
        agent = st.number_input("Agent ID (0 = no agent)", min_value=0, value=0)
        previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
        booking_changes = st.number_input("Booking Changes", min_value=0, value=0)
    with c6:
        required_parking = st.number_input("Required Parking Spaces", min_value=0, max_value=8, value=0)
        special_requests = st.number_input("Total Special Requests", min_value=0, max_value=5, value=0)
        adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)

    submitted = st.form_submit_button("Evaluate Booking", use_container_width=True)

# ── Decision ──────────────────────────────────────────────────────────────────
if submitted:
    month_name  = arrival_date.strftime("%B")
    week_number = int(arrival_date.strftime("%W"))

    booking = {
        "hotel":                            hotel_type,
        "lead_time":                        lead_time,
        "arrival_date_year":                arrival_date.year,
        "arrival_date_month":               month_name,
        "arrival_date_week_number":         week_number,
        "arrival_date_day_of_month":        arrival_date.day,
        "stays_in_weekend_nights":          stays_weekend,
        "stays_in_week_nights":             stays_week,
        "meal":                             meal,
        "country":                          country,
        "market_segment":                   market_segment,
        "distribution_channel":             distribution_channel,
        "reserved_room_type":               reserved_room,
        "deposit_type":                     deposit_type,
        "agent":                            agent,
        "company":                          0,
        "customer_type":                    customer_type,
        "adr":                              adr,
        "required_car_parking_spaces":      required_parking,
        "total_of_special_requests":        special_requests,
        "previous_cancellations":           previous_cancellations,
        "booking_changes":                  booking_changes,
        "adults":                           adults,
        "children":                         children,
        "previous_bookings_not_canceled":   previous_bookings_not_canceled,
    }

    X_new        = preprocess_booking(booking, EXPECTED_FEATURES)
    p_cancel_new = float(model.predict_proba(X_new)[0, 1])

    # Expected arrivals before and after
    expected_before = sum(1 - p for p in existing_probs)
    expected_after  = expected_before + (1 - p_cancel_new)

    # Decision: ACCEPT if expected arrivals stay within the overbook cap
    accept = expected_after <= max_expected_arrivals

    # Monte Carlo simulation (for the histogram — context only)
    probs_with_new  = existing_probs + [p_cancel_new]
    arrivals_after  = simulate_arrivals(probs_with_new, n_simulations=10_000, seed=42)
    p_overbook      = float((arrivals_after > capacity).mean())

    # ── Output layout ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Decision")

    if accept:
        st.success(
            f"✅  ACCEPT  —  Expected arrivals ({expected_after:.1f}) "
            f"stay within your overbook cap ({max_expected_arrivals:.0f} rooms, "
            f"{100 + overbook_buffer*100:.0f}% of capacity)."
        )
    else:
        st.error(
            f"❌  REJECT  —  Expected arrivals ({expected_after:.1f}) "
            f"would exceed your overbook cap ({max_expected_arrivals:.0f} rooms, "
            f"{100 + overbook_buffer*100:.0f}% of capacity)."
        )

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cancellation Probability (new booking)", f"{p_cancel_new*100:.1f}%")
    m2.metric("Expected Arrivals — Before", f"{expected_before:.1f}")
    m3.metric("Expected Arrivals — After",  f"{expected_after:.1f}",
              delta=f"{expected_after - expected_before:+.2f}",
              delta_color="inverse")
    m4.metric("Expected Occupancy — After", f"{expected_after / capacity * 100:.1f}%")

    # Arrival distribution histogram
    st.markdown("#### Simulated Arrival Distribution (with new booking)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(arrivals_after, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(capacity, color="red", linewidth=2, linestyle="--",
               label=f"Capacity ({capacity})")
    ax.axvline(max_expected_arrivals, color="orange", linewidth=2, linestyle="--",
               label=f"Overbook cap ({max_expected_arrivals:.0f})")
    ax.axvline(arrivals_after.mean(), color="green", linewidth=1.5, linestyle=":",
               label=f"Mean arrivals ({arrivals_after.mean():.1f})")
    ax.axvspan(capacity + 1, arrivals_after.max() + 1, alpha=0.1, color="red")
    ax.set_xlabel("Number of Guests Arriving")
    ax.set_ylabel("Frequency (out of 10,000 simulations)")
    ax.set_title("Monte Carlo: Simulated Arrival Distribution (with new booking)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"Red dashed line = hotel capacity ({capacity} rooms).  "
        f"Orange dashed line = overbook cap ({max_expected_arrivals:.0f} rooms).  "
        f"Shaded area = scenarios where guests exceed capacity.  "
        f"P(arrivals > capacity) = {p_overbook*100:.1f}% (shown for reference only — "
        f"decision is based on expected arrivals vs. cap)."
    )
