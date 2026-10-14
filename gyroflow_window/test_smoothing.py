"""Geometry, source-port, streaming contract, and numerical regression checks."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .math3d import (unit_quaternions, from_rotvec, rotvec, multiply, conjugate,
                     angular_distance, matrices, slerp)
from .geometry import PinholeCamera, geometric_fov, coverage_margin, project_to_fov
from .gyroflow_global import SmoothingConfig, gyroflow_smooth, stabilize_global, horizon_gain
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
@pytest.mark.parametrize('enablehorizon', [False, True])
def test_stream_chunk_invariance_and_history(future, enablehorizon):
    raw = make_motion('multi_axis', 90, seconds=5).raw
    config = SmoothingConfig(minimum_fov=0.96, max_iterations=2, enablehorizon=enablehorizon)
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


@pytest.mark.parametrize('enablehorizon', [False, True])
def test_no_future_leakage_or_retroactive_mutation(enablehorizon):
    raw = make_motion('pan', 39, seconds=5).raw
    cfg = SmoothingConfig(max_iterations=1, enablehorizon=enablehorizon)
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
@pytest.mark.parametrize('enablehorizon', [False, True])
def test_short_and_stationary(length, enablehorizon):
    raw = np.tile([1., 0, 0, 0], (length, 1))
    cfg = SmoothingConfig(minimum_fov=1.0, enablehorizon=enablehorizon)
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


@pytest.mark.parametrize('enablehorizon', [False, True])
def test_rotation_and_sign_invariance(enablehorizon):
    raw = make_motion('return', 78, seconds=4).raw
    config = SmoothingConfig(minimum_fov=0.91, max_iterations=2, enablehorizon=enablehorizon)
    window = WindowConfig(history=24, emit=5, future=11,
                          prediction_guard_seconds=1.2, adaptive_boundary=True)
    change = from_rotvec(np.array([0.8, -0.4, 0.6]))
    result = stabilize_window(raw, config, window)
    # A world-frame change must rotate the external vertical reference as well.
    changed_config = replace(config, horizon_up=matrices(change) @ np.array(config.horizon_up))
    changed = stabilize_window(multiply(change, raw), changed_config, window)
    assert np.max(angular_distance(multiply(change, result.orientations), changed.orientations)) < 1e-8
    assert np.allclose(result.fov, changed.fov, atol=1e-8)
    flipped = raw.copy()
    flipped[::3] *= -1
    signed = stabilize_window(flipped, config, window)
    assert np.max(angular_distance(result.orientations, signed.orientations)) < 1e-9


@pytest.mark.parametrize('enablehorizon', [False, True])
def test_stream_indices_and_lookahead_budget(enablehorizon):
    raw = make_motion('fast_turn', 61, seconds=4).raw
    config = SmoothingConfig(max_iterations=1, enablehorizon=enablehorizon)
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
@pytest.mark.parametrize('enablehorizon', [False, True])
def test_single_file_bundle_parity(tmp_path, monkeypatch, use_numba, enablehorizon):
    import importlib.util
    import sys
    from .build_bundle import build

    path = tmp_path/'gyroflow_sliding_window.py'
    build(path)
    raw = make_motion('walking', 52, seconds=1.2).raw
    config = SmoothingConfig(minimum_fov=0.96, max_iterations=1, enablehorizon=enablehorizon)
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


def horizon_error_deg(q, up=(0., 1., 0.)):
    """Independent SciPy measurement of image vertical, away from poles."""
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    vertical = rotation.inv().apply(np.broadcast_to(up, (len(q), 3)))
    return np.rad2deg(np.arctan2(vertical[:, 0], vertical[:, 1]))


@pytest.mark.parametrize('feedback', [None, 1.0, 0.5, 0.25, 0.0])
@pytest.mark.parametrize('length', [1, 24])
def test_horizon_stationary_tilt_and_feedback(feedback, length):
    # A stationary bank must be corrected even when the second pass exits early.
    raw = np.tile(from_rotvec(np.array([0., 0., np.deg2rad(12.)])), (length, 1))
    saved = raw.copy()
    config = SmoothingConfig(enablehorizon=True)
    actual, _ = gyroflow_smooth(raw, config, feedback=feedback)
    expected_roll = {None: 0., 1.0: 0., 0.5: 4., 0.25: 7.2, 0.0: 12.}[feedback]
    assert np.allclose(horizon_error_deg(actual), expected_roll, atol=1e-10)
    assert np.array_equal(raw, saved)


def test_horizon_per_frame_feedback_and_preserved_view():
    raw = make_motion('multi_axis', 82, seconds=3).raw
    feedback = np.linspace(0., 1., len(raw))
    base, base_info = gyroflow_smooth(raw, SmoothingConfig(), feedback=feedback)
    result, info = gyroflow_smooth(raw, SmoothingConfig(enablehorizon=True), feedback=feedback)
    assert np.allclose(matrices(result)[:, :, 2], matrices(base)[:, :, 2], atol=1e-12)
    before, after = horizon_error_deg(base), horizon_error_deg(result)
    # Check observable roll against the specified feedback mapping.
    assert np.allclose(after, before * (1-feedback)/(1+feedback), atol=1e-10)
    assert np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-12)
    assert np.array_equal(info['first_distance'], base_info['first_distance'])
    assert np.all(horizon_gain(feedback) >= feedback)
    assert np.all(np.diff(horizon_gain(feedback)) >= 0)
    assert np.allclose(horizon_gain([-1., 2.]), [0., 1.])
    with pytest.raises(ValueError, match='finite'):
        horizon_gain([np.nan])


def test_horizon_zoom_feedback_sees_corrected_poses():
    raw = np.tile(from_rotvec(np.array([0., 0., np.deg2rad(15.)])), (60, 1))
    config = SmoothingConfig(minimum_fov=0.9, enablehorizon=True)
    first = stabilize_global(raw, replace(config, max_iterations=0))
    final = stabilize_global(raw, config)
    assert first.fov.min() < config.minimum_fov
    assert final.iterations > 0
    assert np.all(final.diagnostics['feedback'] < 1)
    assert final.fov.min() > first.fov.min()
    assert np.max(np.abs(horizon_error_deg(final.orientations))) > 0.1
    # Global retains its finite soft feedback contract. The window adds hard FOV.
    windowed = stabilize_window(raw, config)
    assert windowed.fov.min() >= config.minimum_fov-1e-9
    assert coverage_margin(raw, windowed.orientations, windowed.fov, PinholeCamera()).min() >= -1e-8
    full_frame = stabilize_window(raw, replace(config, minimum_fov=1.0))
    assert np.max(angular_distance(full_frame.orientations, raw)) < 1e-10


@pytest.mark.parametrize('feedback', [0.25, 1.0])
def test_horizon_poles_and_half_turn_do_not_flip(feedback):
    config = SmoothingConfig(enablehorizon=True)
    for angles, axis in [([89.99, 90., 90.01], 0), ([-89.99, -90., -90.01], 0),
                         ([179.99, 180., 180.01], 2)]:
        vectors = np.zeros((3, 3))
        vectors[:, axis] = np.deg2rad(angles)
        raw = from_rotvec(vectors)
        # Independent single-frame calls expose singularities without hiding
        # them behind the temporal smoother or the final geometry projection.
        actual = np.concatenate([gyroflow_smooth(q[None], config, feedback=feedback)[0] for q in raw])
        assert np.isfinite(actual).all()
        assert np.max(np.rad2deg(angular_distance(actual[:-1], actual[1:]))) < 0.011
        assert np.allclose(matrices(actual)[:, :, 2], matrices(raw)[:, :, 2], atol=1e-12)


def test_horizon_degeneracy_tapers_without_a_hard_switch():
    config = SmoothingConfig(enablehorizon=True)
    corrected = []
    for pitch in [80., 82.5, 85.]:
        q = multiply(from_rotvec(np.deg2rad([pitch, 0., 0.])),
                     from_rotvec(np.deg2rad([0., 0., 20.])))
        out, _ = gyroflow_smooth(q[None], config)
        corrected.append(float(np.rad2deg(angular_distance(q, out[0]))))
    assert corrected[0] > corrected[1] > corrected[2]
    assert corrected[0] == pytest.approx(20., abs=1e-10)
    assert corrected[2] < 1e-10


@pytest.mark.parametrize('kwargs', [
    {'enablehorizon': 1}, {'enablehorizon': None},
    {'horizon_up': (0., 0., 0.)}, {'horizon_up': (0., 1.)},
    {'horizon_up': (0., np.nan, 0.)}, {'horizon_up': (0., np.inf, 0.)},
])
def test_horizon_config_validation(kwargs):
    with pytest.raises(ValueError):
        SmoothingConfig(**kwargs)


def test_horizon_reference_is_normalized_and_copied():
    up = [0., 2., 0.]
    config = SmoothingConfig(horizon_up=up)
    up[1] = 0
    assert config.horizon_up == (0., 1., 0.)
    assert SmoothingConfig(horizon_up=(0., 1e300, 0.)).horizon_up == config.horizon_up
    assert SmoothingConfig(horizon_up=(0., 1e-300, 0.)).horizon_up == config.horizon_up


def test_horizon_bundle_cli(tmp_path):
    import subprocess
    import sys
    from .build_bundle import build

    bundle, source, output = (tmp_path/name for name in ['stabilizer.py', 'input.npy', 'output.npz'])
    build(bundle)
    raw = np.tile(from_rotvec(np.deg2rad([0., 0., 5.])), (35, 1))
    np.save(source, raw)
    subprocess.run([sys.executable, str(bundle), str(source), '--output', str(output),
                    '--enablehorizon', '--horizon-up', '0', '2', '0', '--minimum-fov', '0.6'],
                   capture_output=True, text=True, check=True)
    with np.load(output) as result:
        assert result['orientations'].shape == raw.shape
        assert np.max(np.abs(horizon_error_deg(result['orientations']))) < 1e-8
        assert result['fov'].min() >= 0.6-1e-9
