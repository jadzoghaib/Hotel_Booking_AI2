"""
stochastic_model.py
-------------------
Monte Carlo simulation for hotel overbooking risk.

Each booking has a cancellation probability from the ML model.
Actual arrivals = sum of independent Bernoulli(1 - p_cancel) trials.
We simulate 10,000 scenarios to get the arrival distribution and
compute P(arrivals > capacity).
"""

import numpy as np


def simulate_arrivals(cancellation_probs, n_simulations=10_000, seed=None):
    """
    Simulate the number of guests that actually show up.

    Parameters
    ----------
    cancellation_probs : list or array of floats
        ML model's predicted cancellation probability for each booking.
    n_simulations : int
        Number of Monte Carlo scenarios to run.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    arrivals : np.ndarray of shape (n_simulations,)
        Number of guests that showed up in each simulated scenario.
    """
    if seed is not None:
        np.random.seed(seed)

    show_up_probs = 1 - np.array(cancellation_probs, dtype=float)
    # Each row = one simulation, each col = one booking
    # binomial(1, p) is a Bernoulli(p) trial
    samples = np.random.binomial(
        1, show_up_probs,
        size=(n_simulations, len(show_up_probs))
    )
    return samples.sum(axis=1)


def overbooking_risk(cancellation_probs, capacity, n_simulations=10_000, seed=None):
    """
    Estimate P(arrivals > capacity) via Monte Carlo.

    Parameters
    ----------
    cancellation_probs : list or array of floats
    capacity : int
        Number of rooms available.
    n_simulations : int
    seed : int or None

    Returns
    -------
    float : estimated probability of overbooking
    """
    arrivals = simulate_arrivals(cancellation_probs, n_simulations, seed)
    return float((arrivals > capacity).mean())


def accept_decision(existing_probs, new_booking_prob, capacity,
                    risk_threshold, n_simulations=10_000, seed=None):
    """
    Decide whether to accept a new booking.

    Rule: ACCEPT if P(arrivals > capacity WITH new booking) < risk_threshold.

    Parameters
    ----------
    existing_probs : list of floats
        Cancellation probabilities for current bookings.
    new_booking_prob : float
        Cancellation probability for the new booking being considered.
    capacity : int
        Hotel room capacity.
    risk_threshold : float
        Maximum acceptable overbooking probability (e.g. 0.05 for 5%).
    n_simulations : int
    seed : int or None

    Returns
    -------
    dict with keys:
        risk_before  : P(overbook) without new booking
        risk_after   : P(overbook) with new booking
        accept       : bool — True if risk_after < risk_threshold
        expected_arrivals_before : float
        expected_arrivals_after  : float
        arrivals_distribution    : np.ndarray — raw simulation results (with new booking)
    """
    probs_with_new = list(existing_probs) + [new_booking_prob]

    arrivals_before = simulate_arrivals(existing_probs, n_simulations, seed)
    arrivals_after  = simulate_arrivals(probs_with_new, n_simulations, seed)

    risk_before = float((arrivals_before > capacity).mean())
    risk_after  = float((arrivals_after  > capacity).mean())

    return {
        'risk_before':               risk_before,
        'risk_after':                risk_after,
        'accept':                    risk_after < risk_threshold,
        'expected_arrivals_before':  float(sum(1 - p for p in existing_probs)),
        'expected_arrivals_after':   float(sum(1 - p for p in probs_with_new)),
        'arrivals_distribution':     arrivals_after,
    }
