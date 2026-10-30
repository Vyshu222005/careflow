import numpy as np
import pandas as pd
import pytest

import simulator
from simulator import ANTIBIOTICS, TRIAGE, Scenario, simulate


@pytest.fixture()
def hours():
    # hours since registration: triage then antibiotics; "d" was treated before sepsis triage, "e" has no antibiotics
    return pd.DataFrame({TRIAGE: [0.2, 0.5, 0.1, 0.3, 0.2], ANTIBIOTICS: [2.0, 0.8, 4.0, 0.1, np.nan]}, index=list("abcde"))


def test_segments_add_up_to_the_total(hours):
    parts = simulator.segments(hours)
    assert list(parts.index) == list("abcd")  # patient e has no antibiotics
    assert np.allclose(parts["segment1"] + parts["segment2"], parts["total"])
    assert parts.loc["d", "segment2"] == 0 and parts.loc["d", "segment1"] == pytest.approx(0.1)  # treated before sepsis triage


def test_baseline_scenario_changes_nothing(hours):
    result = simulate(hours, Scenario())
    assert np.allclose(result.simulated_hours, result.baseline_hours) and result.patients_changed == 0
    assert result.summary()["change"].abs().max() == pytest.approx(0.0)


def test_times_never_get_longer_and_never_negative(hours):
    for scenario in (Scenario(cut_segment1_pct=50, cut_segment2_pct=70, cut_segment2_minutes=200, adoption_pct=60), Scenario(cut_segment2_minutes=15)):
        result = simulate(hours, scenario)
        assert (result.simulated_hours <= result.baseline_hours + 1e-9).all() and (result.simulated_hours >= 0).all()


def test_segment2_floor_is_respected(hours):
    result = simulate(hours, Scenario(cut_segment2_minutes=600, floor_minutes=10))
    parts = simulator.segments(hours)
    expected = parts["segment1"] + np.minimum(parts["segment2"], 10 / 60).where(parts["segment2"] > 0, 0)
    assert np.allclose(result.simulated_hours, np.minimum(expected, result.baseline_hours))


def test_more_reduction_never_lowers_the_share_within_one_hour(hours):
    shares = [simulate(hours, Scenario(cut_segment2_minutes=m)).share_within(1) for m in range(0, 130, 10)]
    assert shares == sorted(shares)


def test_adoption_zero_changes_nobody_and_is_reproducible(hours):
    assert simulate(hours, Scenario(cut_segment2_minutes=60, adoption_pct=0)).patients_changed == 0
    a = simulate(hours, Scenario(cut_segment2_minutes=60, adoption_pct=50)).simulated_hours
    b = simulate(hours, Scenario(cut_segment2_minutes=60, adoption_pct=50)).simulated_hours
    assert a.equals(b)


def test_invalid_scenarios_and_inputs_are_rejected(hours):
    for bad in (Scenario(cut_segment1_pct=-1), Scenario(adoption_pct=101), Scenario(cut_segment2_minutes=-5)):
        with pytest.raises(ValueError):
            simulate(hours, bad)
    with pytest.raises(ValueError):
        simulator.segments(hours.drop(columns=ANTIBIOTICS))


def test_sensitivity_curve_is_monotonic(hours):
    sweep = simulator.sensitivity(hours, [0, 15, 30, 60, 120])
    assert sweep["within_1h_pct"].is_monotonic_increasing and sweep["median_hours"].is_monotonic_decreasing


def test_complete_bundle_only_changes_through_antibiotic_timing():
    hours = pd.DataFrame({TRIAGE: [0.1, 0.1], ANTIBIOTICS: [2.0, 0.5]}, index=["a", "b"])
    bundle = pd.DataFrame({"case:concept:name": ["a", "b", "c"], "lactate_1h": [True, True, True], "hypotension": [False, False, False],
                           "fluid_3h": [False, False, False], "high_lactate": [False, False, False], "repeat_lactate": [False, False, False],
                           "complete_bundle": [False, True, False]})
    base, sim = simulator.complete_bundle_share(bundle, simulate(hours, Scenario(cut_segment2_minutes=100)))
    assert base == pytest.approx(100 / 3) and sim == pytest.approx(200 / 3)  # patient c has no antibiotics, so never meets it
