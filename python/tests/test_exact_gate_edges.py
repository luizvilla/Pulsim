"""simulate(edge_fn=...): exact gate edges on the fixed-step engine.

Without it, switch_fn(t) at the END of a step sets the mask for the whole
step, so a PWM edge moves to a step boundary and the duty can only change
in steps of dt / T_pwm. With it, a step is split at its edges.

The circuit is a synchronous buck with body diodes and dead time (the
diodes carry the current during the dead time), 30 V in, 200 kHz, at a
coarse dt = 25 ns: 200 steps per period, a duty quantum of 0.5 %.
"""
import math

import numpy as np
import pytest

import pulsim as p

V_IN, F_SW, DEAD, DT = 30.0, 200e3, 100e-9, 25e-9
T = 1.0 / F_SW


def buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("V_in", "high", "0", V_IN)
    b.add_switch("S_hs", "high", "sw", 1e3, 1e-9)
    b.add_switch("S_ls", "sw", "0", 1e3, 1e-9)
    b.add_diode("D_hs", "sw", "high", 1e3, 1e-9, 0.7)
    b.add_diode("D_ls", "0", "sw", 1e3, 1e-9, 0.7)
    b.add_inductor("L1", "sw", "out", 33e-6)
    b.add_capacitor("C1", "out", "0", 61e-6)
    b.add_resistor("R_load", "out", "0", 33.0)
    return b


def pwm(b, duty):
    """switch_fn and edge_fn of a dead-time PWM pair at a fixed duty."""
    hs, ls = b.switch_index_of("S_hs"), b.switch_index_of("S_ls")
    mask = p.SwitchStateMask(b.graph.num_switches)

    def switch_fn(t):
        k = math.floor(t / T + 1e-9)
        phase = t - k * T
        mask.set(hs, DEAD <= phase < duty * T)
        mask.set(ls, duty * T + DEAD <= phase < T)
        return mask

    def edge_fn(t0, t1):
        k = math.floor(t0 / T + 1e-9)
        for kk in (k, k + 1):
            for e in (kk * T, kk * T + DEAD, kk * T + duty * T, kk * T + duty * T + DEAD):
                if t0 < e < t1:
                    return e
        return math.inf

    return switch_fn, edge_fn


def mean_out(duty, exact, t_end=0.02):
    b = buck()
    switch_fn, edge_fn = pwm(b, duty)
    extra = {"edge_fn": edge_fn} if exact else {}
    r = p.simulate(b, t_end=t_end, dt=DT, engine="pwl", switch_fn=switch_fn, store_every=4, **extra)
    t, v = np.asarray(r.times), np.asarray(r.v("out"))
    return float(v[t > t_end - 4e-3].mean()), r


def test_without_edges_the_duty_moves_in_quanta():
    # 0.3 and 0.3049 are within one 0.5 % quantum (0.15 V at 30 V): the same output, to far below it.
    assert mean_out(0.3, exact=False)[0] == pytest.approx(mean_out(0.3049, exact=False)[0], abs=1e-3)


def test_with_edges_the_output_follows_the_duty():
    v0, r = mean_out(0.3, exact=True)
    v1, _ = mean_out(0.3025, exact=True)
    v2, _ = mean_out(0.3049, exact=True)
    # dV/dD = V_in for a buck in continuous conduction.
    assert (v1 - v0) / 0.0025 == pytest.approx(V_IN, rel=0.02)
    assert (v2 - v0) / 0.0049 == pytest.approx(V_IN, rel=0.02)
    assert r.edge_split_steps > 0


def test_without_edge_fn_nothing_is_split():
    _, r = mean_out(0.3025, exact=False, t_end=1e-3)
    assert r.edge_split_steps == 0


def test_resumed_segments_give_the_same_result():
    """A run resumed in short segments ends where one long run does."""
    b = buck()
    switch_fn, edge_fn = pwm(b, 0.3025)
    whole = p.simulate(b, t_end=2e-3, dt=DT, engine="pwl", switch_fn=switch_fn, edge_fn=edge_fn)
    b2 = buck()
    switch_fn2, edge_fn2 = pwm(b2, 0.3025)
    t, snap = 0.0, None
    while t < 2e-3 - 1e-12:
        kwargs = {"resume_from": snap} if snap is not None else {}
        r = p.simulate(b2, t_start=t, t_end=t + 1e-4, dt=DT, engine="pwl", switch_fn=switch_fn2,
                       edge_fn=edge_fn2, **kwargs)
        snap, t = r.final_snapshot, t + 1e-4
    assert float(np.asarray(r.v("out"))[-1]) == pytest.approx(float(np.asarray(whole.v("out"))[-1]), abs=1e-6)


def test_refused_on_the_variable_step_engines():
    b = buck()
    switch_fn, edge_fn = pwm(b, 0.3)
    with pytest.raises(ValueError, match="edge_fn"):
        p.simulate(b, t_end=1e-4, engine="dsed", switch_fn=switch_fn, edge_fn=edge_fn)
