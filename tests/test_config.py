"""Config invariants: the honesty knob must behave exactly as promised."""

from lifesim.config import Config


def test_n_endpoints_exact():
    assert Config(uncertainty=0.0).n_trajectories() == 200
    assert Config(uncertainty=1.0).n_trajectories() == 6000


def test_n_monotonic_in_u():
    us = [i / 20 for i in range(21)]
    ns = [Config(uncertainty=u).n_trajectories() for u in us]
    assert all(b >= a for a, b in zip(ns, ns[1:])), ns


def test_noise_strictly_positive_at_zero():
    # The cone never closes: noise > 0 even with the knob fully off.
    assert Config(uncertainty=0.0).noise_scale() > 0.0


def test_noise_positive_everywhere():
    for i in range(21):
        assert Config(uncertainty=i / 20).noise_scale() > 0.0


def test_pure_functions_no_rng():
    c = Config(uncertainty=0.37)
    assert c.n_trajectories() == c.n_trajectories()
    assert c.noise_scale() == c.noise_scale()


def test_u_clamped():
    assert Config(uncertainty=5.0).u() == 1.0
    assert Config(uncertainty=-2.0).u() == 0.0


def test_epistemic_floor_must_be_positive():
    try:
        Config(epistemic_floor=0.0)
    except ValueError:
        return
    raise AssertionError("epistemic_floor=0 should be rejected")
