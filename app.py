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

from preprocessing import preprocess_booking, TOP_COUNTRIES
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

# ── Constants ─────────────────────────────────────────────────────────────────
HOTEL_CAPACITY = {"City Hotel": 225, "Resort Hotel": 190}

ROOM_TYPES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L']

# Room capacities — uniform descending, sums match hotel capacity
ROOM_TYPE_DEFAULTS = {
    "City Hotel":   {'A': 60, 'B': 50, 'C': 40, 'D': 30, 'E': 20, 'F': 10, 'G': 8, 'H': 4, 'L': 3},
    "Resort Hotel": {'A': 50, 'B': 42, 'C': 34, 'D': 26, 'E': 18, 'F': 10, 'G': 6, 'H': 3, 'L': 1},
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏨 Hotel Setup")

hotel_type = st.sidebar.selectbox(
    "Hotel Type",
    options=["City Hotel", "Resort Hotel"],
)

capacity = HOTEL_CAPACITY[hotel_type]
st.sidebar.metric("Hotel Capacity (rooms)", capacity)

overbook_buffer = st.sidebar.slider(
    "Max Overbook Buffer",
    min_value=0, max_value=50, value=10, step=1,
    format="%d%%",
    help=(
        "How far above a room type's effective capacity you are willing to push expected arrivals. "
        "0% = never exceed capacity. 10% = accept until expected arrivals reach 110% of effective capacity."
    ),
) / 100.0

# ── Per-room-type capacities (fixed, from dataset proportions) ────────────────
room_capacity = ROOM_TYPE_DEFAULTS[hotel_type]

st.sidebar.markdown("---")
with st.sidebar.expander("Room Type Capacities"):
    st.caption("Based on dataset proportions.")
    for rt in ROOM_TYPES:
        st.markdown(f"**Type {rt}:** {room_capacity[rt]} rooms")


# ── Auto-load existing bookings at full capacity ──────────────────────────────
# ── Synthetic full house: each room type filled to its capacity, p_cancel = 0.30 ──
SYNTHETIC_CANCEL_PROB = 0.30

existing_bookings_rt = []
for rt in ROOM_TYPES:
    existing_bookings_rt.extend([rt] * room_capacity[rt])

existing_bookings = pd.DataFrame({'reserved_room_type': existing_bookings_rt})
existing_probs    = [SYNTHETIC_CANCEL_PROB] * len(existing_bookings)

st.sidebar.markdown("---")
st.sidebar.success(
    f"Hotel at full capacity: **{len(existing_probs)} bookings** "
    f"({int(SYNTHETIC_CANCEL_PROB*100)}% p_cancel per booking)"
)

# ── Baseline occupancy display ────────────────────────────────────────────────
st.title("Hotel Overbooking Decision Tool")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Bookings", len(existing_probs))
expected_now = sum(1 - p for p in existing_probs)
col2.metric("Expected Arrivals", f"{expected_now:.1f}")
col3.metric("Expected Occupancy", f"{expected_now / capacity * 100:.1f}%")
col4.metric("Overbook Buffer", f"{overbook_buffer*100:.0f}%")

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
            options=ROOM_TYPES,
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

    # ── Room-type level analysis ───────────────────────────────────────────
    # Filter existing bookings competing for the same room type
    room_mask        = existing_bookings['reserved_room_type'].values == reserved_room
    competing_probs  = [p for p, m in zip(existing_probs, room_mask) if m]
    n_competing      = len(competing_probs)

    # Upgrade buffer = total capacity of all room types higher than the reserved type
    rt_idx        = ROOM_TYPES.index(reserved_room)
    upgrade_buffer = sum(room_capacity[rt] for rt in ROOM_TYPES[rt_idx + 1:])

    # Effective capacity = this room type's capacity + upgrade buffer
    effective_cap    = room_capacity[reserved_room] + upgrade_buffer
    max_expected     = effective_cap * (1 + overbook_buffer)

    # Expected arrivals for this room type
    expected_competing_before = sum(1 - p for p in competing_probs)
    expected_competing_after  = expected_competing_before + (1 - p_cancel_new)

    # Decision
    accept = expected_competing_after <= max_expected

    # Monte Carlo on the competing room type only
    probs_with_new = competing_probs + [p_cancel_new]
    arrivals_after = simulate_arrivals(probs_with_new, n_simulations=10_000, seed=42)
    p_overbook     = float((arrivals_after > effective_cap).mean())

    # ── Output layout ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Decision — Room Type {reserved_room}")

    # Will this person cancel?
    will_cancel = p_cancel_new >= 0.5
    if will_cancel:
        st.warning(f"⚠️  This guest is **likely to cancel** (cancellation probability: {p_cancel_new*100:.1f}%)")
    else:
        st.info(f"✅  This guest is **likely to show up** (cancellation probability: {p_cancel_new*100:.1f}%)")

    if accept:
        st.success(
            f"✅  ACCEPT  —  Expected Type {reserved_room} arrivals ({expected_competing_after:.1f}) "
            f"stay within effective capacity ({effective_cap} rooms + {overbook_buffer*100:.0f}% buffer = {max_expected:.0f})."
        )
    else:
        st.error(
            f"❌  REJECT  —  Expected Type {reserved_room} arrivals ({expected_competing_after:.1f}) "
            f"would exceed effective capacity ({effective_cap} rooms + {overbook_buffer*100:.0f}% buffer = {max_expected:.0f})."
        )

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cancellation Probability", f"{p_cancel_new*100:.1f}%")
    m2.metric(f"Type {reserved_room} Bookings", n_competing)
    m3.metric("Expected Arrivals — Before", f"{expected_competing_before:.1f}")
    m4.metric("Expected Arrivals — After",  f"{expected_competing_after:.1f}",
              delta=f"{expected_competing_after - expected_competing_before:+.2f}",
              delta_color="inverse")
    m5.metric("Effective Capacity", f"{effective_cap} ({room_capacity[reserved_room]} + {upgrade_buffer})")

    # Arrival distribution histogram
    st.markdown(f"#### Simulated Arrival Distribution — Room Type {reserved_room} (with new booking)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(arrivals_after, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(room_capacity[reserved_room], color="red", linewidth=2, linestyle="--",
               label=f"Type {reserved_room} capacity ({room_capacity[reserved_room]})")
    ax.axvline(effective_cap, color="orange", linewidth=2, linestyle="--",
               label=f"Effective capacity with upgrades ({effective_cap})")
    ax.axvline(arrivals_after.mean(), color="green", linewidth=1.5, linestyle=":",
               label=f"Mean arrivals ({arrivals_after.mean():.1f})")
    ax.axvspan(room_capacity[reserved_room] + 1, arrivals_after.max() + 1, alpha=0.1, color="red")
    ax.set_xlabel("Number of Guests Arriving (Room Type " + reserved_room + ")")
    ax.set_ylabel("Frequency (out of 10,000 simulations)")
    ax.set_title(f"Monte Carlo: Simulated Arrivals for Room Type {reserved_room}")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"Red dashed = Type {reserved_room} base capacity ({room_capacity[reserved_room]}).  "
        f"Orange dashed = effective capacity including {upgrade_buffer} upgrade rooms ({effective_cap}).  "
        f"P(arrivals > base capacity) = {p_overbook*100:.1f}% (shown for reference)."
    )
