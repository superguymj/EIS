"""Geometry, source-port, streaming contract, and numerical regression checks."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .math3d import (unit_quaternions, from_rotvec, rotvec, multiply, conjugate,
                     angular_distance, matrices, slerp)
from .geometry import PinholeCamera, geometric_fov, coverage_margin, project_to_fov
from .gyroflow_global import SmoothingConfig, gyroflow_smooth, stabilize_global
from .sliding_window import WindowConfig, SlidingWindowStabilizer, stabilize_window
from .synthetic import make_motion


def scipy_reference_core(q, config, diagonal, feedback):
    """Separate scalar transcription using SciPy rotations, no production primitives."""
    rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    a = 1-np.exp(-1/(config.fps*config.slow_seconds))
    b = 1-np.exp(-1/(config.fps*config.fast_seconds))
    velocity = np.zeros(len(q))
    velocity[1:] = np.rad2deg((rotations[:-1].inv()*rotations[1:]).magnitude())*config.fps

    def scalar_fb(x):
        x = x.copy()
        for i in range(1, len(x)):
            x[i] = (1-b)*x[i-1] + b*x[i]
        for i in range(len(x)-2, -1, -1):
            x[i] = (1-b)*x[i+1] + b*x[i]
        return x

    def quaternion_fb(r, coefficients):
        outputs = []
        state = r[0]
        for item, coeff in zip(r, coefficients):
            state = state * Rotation.from_rotvec((state.inv()*item).as_rotvec()*coeff)
            outputs.append(state)
        backward = []
        state = outputs[-1]
        for item, coeff in zip(outputs[::-1], coefficients[::-1]):
            state = state * Rotation.from_rotvec((state.inv()*item).as_rotvec()*coeff)
            backward.append(state)
        return Rotation.concatenate(backward[::-1])

    ratio = scalar_fb(velocity)/(500*config.smoothness*(diagonal/120)*feedback*0.5)
    first = quaternion_fb(rotations, np.minimum(1, a+(b-a)*ratio))
    distance = (rotations.inv()*first).magnitude()
    if distance.max() < 1e-14:
        return first.as_quat()[:, [3, 0, 1, 2]]
    distance /= distance.max()
    distance[distance < 0.5] = 0
    distance = scalar_fb(distance)
    distance = (distance/distance.max()+1)/2
    result = quaternion_fb(first, np.minimum(1, a+(b-a)*ratio*distance))
    return result.as_quat()[:, [3, 0, 1, 2]]


@pytest.mark.parametrize('kind', ['static', 'pan', 'multi_axis', 'impulse', 'wrap'])
def test_core_against_independent_scipy_transcription(kind):
    motion = make_motion(kind, seed=14, seconds=4)
    config = SmoothingConfig()
    feedback = np.linspace(0.4, 1.0, len(motion.raw))
    actual, _ = gyroflow_smooth(motion.raw, config, 90, feedback)
    expected = scipy_reference_core(motion.raw, config, 90, feedback)
    assert np.max(angular_distance(actual, expected)) < 2e-8


def test_math_against_scipy():
    rng = np.random.default_rng(45)
    a, b = (unit_quaternions(rng.normal(size=(100, 4))) for _ in range(2))
    ref = Rotation.from_quat(a[:, [1,2,3,0]]) * Rotation.from_quat(b[:, [1,2,3,0]])
    assert np.allclose(matrices(multiply(a, b)), ref.as_matrix(), atol=1e-12)
    assert np.max(angular_distance(a, from_rotvec(rotvec(a)))) < 1e-12
    assert np.max(angular_distance(slerp(a, b, 0.0), a)) < 1e-12
    assert np.max(angular_distance(slerp(a, b, 1.0), b)) < 1e-12


@pytest.mark.parametrize('diagonal', [55, 100, 145])
def test_geometry_and_projection(diagonal):
    rng = np.random.default_rng(5)
    camera = PinholeCamera(diagonal)
    raw = unit_quaternions(rng.normal(size=(80, 4)))
    smooth = multiply(raw, from_rotvec(rng.normal(size=(80,3))*0.25))
    assert np.allclose(geometric_fov(raw, raw, camera), 1, atol=1e-12)
    fov = geometric_fov(raw, smooth, camera)
    valid = fov > 1e-8
    assert np.min(coverage_margin(raw[valid], smooth[valid], fov[valid], camera)) > -1e-10
    not_full = valid & (fov < 0.999)
    assert np.all(coverage_margin(raw[not_full], smooth[not_full], fov[not_full]+1e-5, camera) < 0)
    for floor in [0.5, 0.85, 0.98, 1.0]:
        projected, _ = project_to_fov(raw, smooth, floor, camera)
        assert np.min(geometric_fov(raw, projected, camera)) >= floor-1e-9
        assert np.min(coverage_margin(raw, projected, np.full(len(raw), floor), camera)) > -1e-8


@pytest.mark.parametrize('future', [0, 7, 30])
def test_stream_chunk_invariance_and_history(future):
    raw = make_motion('multi_axis', 90, seconds=5).raw
    config = SmoothingConfig(minimum_fov=0.96, max_iterations=2)
    window = WindowConfig(history=20, emit=6, future=future)
    reference = stabilize_window(raw, config, window, chunk_size=1)
    for chunk in [6, 19, len(raw)]:
        result = stabilize_window(raw, config, window, chunk_size=chunk)
        assert np.max(angular_distance(result.orientations, reference.orientations)) < 1e-10
        assert np.allclose(result.fov, reference.fov, atol=1e-12)
        assert result.diagnostics['max_buffered_frames'] <= 20+6+future
    assert len(reference.orientations) == len(raw)
    assert np.min(reference.geometric_fov) >= config.minimum_fov-1e-9
    assert np.min(reference.fov) >= config.minimum_fov-1e-9
    assert np.max(reference.fov-reference.geometric_fov) < 1e-9


def test_no_future_leakage_or_retroactive_mutation():
    raw = make_motion('pan', 39, seconds=5).raw
    cfg = SmoothingConfig(max_iterations=1)
    win = WindowConfig(history=20, emit=6, future=12)
    a, b = SlidingWindowStabilizer(cfg, win), SlidingWindowStabilizer(cfg, win)
    prefix = 18
    first_a, first_b = a.push(raw[:prefix])[0], b.push(raw[:prefix])[0]
    saved = first_a.orientations.copy()
    altered = multiply(from_rotvec(np.tile([1.1, -0.5, 0.8], (len(raw)-prefix,1))), raw[prefix:])
    a.push(raw[prefix:]); a.flush()
    b.push(altered); b.flush()
    assert np.array_equal(first_a.orientations, saved)
    assert np.array_equal(first_a.orientations, first_b.orientations)
    assert first_a.diagnostics['observed_through'] == 17
    assert len(a.window_diagnostics) == 0  # production does not retain an unbounded log
    with pytest.raises(RuntimeError):
        a.push(raw[:1])


@pytest.mark.parametrize('length', [0, 1, 2, 5, 7, 17])
def test_short_and_stationary(length):
    raw = np.tile([1., 0, 0, 0], (length, 1))
    cfg = SmoothingConfig(minimum_fov=1.0)
    result = stabilize_window(raw, cfg)
    assert result.orientations.shape == raw.shape
    assert np.isfinite(result.orientations).all()
    assert np.allclose(result.orientations, raw)
    assert np.allclose(result.fov, 1)
    global_result = stabilize_global(raw, cfg)
    assert np.allclose(global_result.orientations, raw)


def test_input_validation():
    with pytest.raises(ValueError):
        SmoothingConfig(fps=0)
    with pytest.raises(ValueError):
        WindowConfig(emit=0)
    with pytest.raises(ValueError):
        SlidingWindowStabilizer().push([[0,0,0,0]])
    with pytest.raises(ValueError):
        SlidingWindowStabilizer().push([[np.nan,0,0,0]])


def test_rotation_and_sign_invariance():
    raw = make_motion('return', 78, seconds=4).raw
    config = SmoothingConfig(minimum_fov=0.91, max_iterations=2)
    window = WindowConfig(history=24, emit=5, future=11,
                          prediction_guard_seconds=1.2, adaptive_boundary=True)
    change = from_rotvec(np.array([0.8, -0.4, 0.6]))
    result = stabilize_window(raw, config, window)
    changed = stabilize_window(multiply(change, raw), config, window)
    assert np.max(angular_distance(multiply(change, result.orientations), changed.orientations)) < 1e-8
    assert np.allclose(result.fov, changed.fov, atol=1e-8)
    flipped = raw.copy()
    flipped[::3] *= -1
    signed = stabilize_window(flipped, config, window)
    assert np.max(angular_distance(result.orientations, signed.orientations)) < 1e-9


def test_stream_indices_and_lookahead_budget():
    raw = make_motion('fast_turn', 61, seconds=4).raw
    config = SmoothingConfig(max_iterations=1)
    window = WindowConfig(history=10, emit=7, future=9)
    stream = SlidingWindowStabilizer(config, window)
    count = 0
    for arrived in range(len(raw)):
        for batch in stream.push(raw[arrived:arrived+1]):
            assert batch.start_index == count
            assert batch.diagnostics['observed_through'] == arrived
            assert arrived - (batch.start_index+len(batch.orientations)-1) == window.future
            count += len(batch.orientations)
    for batch in stream.flush():
        assert batch.start_index == count
        assert batch.diagnostics['observed_through'] == len(raw)-1
        count += len(batch.orientations)
    assert count == len(raw)
    assert stream.flush() == []


@pytest.mark.parametrize('use_numba', [True, False])
def test_single_file_bundle_parity(tmp_path, monkeypatch, use_numba):
    import importlib.util
    import sys
    from .build_bundle import build

    path = tmp_path/'gyroflow_sliding_window.py'
    build(path)
    raw = make_motion('walking', 52, seconds=1.2).raw
    config = SmoothingConfig(minimum_fov=0.96, max_iterations=1)
    window = WindowConfig(history=12, emit=5, future=7,
                          prediction_guard_seconds=1.2, adaptive_boundary=True)
    expected = stabilize_window(raw, config, window)
    if not use_numba:
        monkeypatch.setitem(sys.modules, 'numba', None)
    name = 'test_window_bundle_with_numba' if use_numba else 'test_window_bundle_without_numba'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    actual = module.stabilize_window(raw, config, window)
    assert np.max(angular_distance(actual.orientations, expected.orientations)) < 1e-9
    assert np.allclose(actual.fov, expected.fov, atol=1e-10)
    assert (tmp_path/'LICENSE').is_file()
