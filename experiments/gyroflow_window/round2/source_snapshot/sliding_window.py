# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded-lookahead quaternion stabilization with immutable committed history.

Streaming contract: keep H past raw/committed frames, solve the next E with F
observed future frames, emit E once, and advance E. Synthetic tail prediction
uses only observed samples and never participates in the geometry guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

from .math3d import (unit_quaternions, conjugate, multiply, rotvec, from_rotvec,
                     angular_distance, world_velocity)
from .geometry import PinholeCamera, geometric_fov, project_to_fov
from .gyroflow_global import SmoothingConfig, StabilizationResult, gyroflow_smooth, zoom_curve


@dataclass(frozen=True)
class WindowConfig:
    history: int = 60
    emit: int = 6
    future: int = 15
    boundary_weight: float = 64.0
    prediction_seconds: float = 0.5
    prediction_decay_seconds: float = 0.6
    stable_distance_scale: bool = False
    initial_distance_deg: float = 3.0
    hard_fov: bool = True
    # Ablation only: disable boundary fit / predictor using weights of zero.

    def __post_init__(self):
        for name, lower in [('history', 2), ('emit', 1), ('future', 0)]:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < lower:
                raise ValueError(f'{name} must be an integer >= {lower}')
        for name in ['boundary_weight', 'prediction_seconds']:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ['prediction_decay_seconds', 'initial_distance_deg']:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


@dataclass
class FrameBatch:
    start_index: int
    orientations: np.ndarray
    fov: np.ndarray
    geometric_fov: np.ndarray
    diagnostics: dict = field(default_factory=dict)


@lru_cache(maxsize=128)
def _boundary_system(size, history_count, weight):
    """Quadratic second-difference system; history columns are eliminated."""
    total = size + history_count
    difference = np.diff(np.eye(total), n=2, axis=0)[:, history_count:]
    matrix = np.eye(size) + weight * (difference.T @ difference)
    band = np.zeros((3, size))
    band[0] = np.diag(matrix)
    if size > 1:
        band[1, :-1] = np.diag(matrix, -1)
    if size > 2:
        band[2, :-2] = np.diag(matrix, -2)
    return difference, cholesky_banded(band, lower=True, check_finite=False)


def fit_fixed_history(candidate, history, weight):
    """One tangent-space Gauss-Newton step, with committed history fixed.

    Minimize ||delta||^2 + weight*||angular_second_difference||^2.
    delta acts on the LEFT in a shared world frame. This is a local manifold
    approximation, not an exact nonlinear optimum or a hard jerk bound.
    """
    if weight <= 0 or len(candidate) < 1:
        return candidate.copy()
    previous = history[-2:]
    combined = np.concatenate([previous, candidate])
    if len(combined) < 3:
        return candidate.copy()
    velocity = rotvec(multiply(combined[1:], conjugate(combined[:-1])))
    acceleration = np.diff(velocity, axis=0)
    difference, factor = _boundary_system(len(candidate), len(previous), float(weight))
    rhs = -weight * (difference.T @ acceleration)
    correction = cho_solve_banded((factor, True), rhs, check_finite=False)
    return unit_quaternions(multiply(from_rotvec(correction), candidate))


def predict_tail(observed, config, window):
    count = int(round(window.prediction_seconds * config.fps))
    if count == 0 or len(observed) < 3:
        return np.empty((0, 4))
    fit_count = min(len(observed), max(4, int(round(0.3 * config.fps))))
    recent = observed[-fit_count:]
    positions = rotvec(multiply(recent, conjugate(recent[-1])))
    times = np.arange(fit_count, dtype=float) / config.fps
    centered_time = times - times.mean()
    centered_positions = positions - positions.mean(axis=0)
    velocity = centered_time @ centered_positions / (centered_time @ centered_time)
    residual = centered_positions - centered_time[:, None] * velocity
    energy = np.sum(centered_positions**2)
    confidence = np.clip(1 - np.sum(residual**2) / max(energy, 1e-12), 0, 1)
    # Regressed trend, damped when recent observations are poorly linear.
    velocity *= confidence
    speed = np.linalg.norm(velocity)
    if speed > np.deg2rad(180):
        velocity *= np.deg2rad(180) / speed
    future_t = np.arange(1, count+1) / config.fps
    tau = window.prediction_decay_seconds
    displacement = tau * (-np.expm1(-future_t / tau))[:, None] * velocity
    return multiply(from_rotvec(displacement), observed[-1])


class SlidingWindowStabilizer:
    """push(wxyz_batch) -> list[FrameBatch]; flush() finalizes a short tail.

    Input is uniform-rate, one orientation per video frame. No future data is
    read from outside the given chunks. Committed frames are never rewritten.
    A stream owns its own instance; instances are not shared across threads.
    """

    def __init__(self, config=None, window=None, camera=None, collect_diagnostics=False):
        self.config = config or SmoothingConfig()
        self.window = window or WindowConfig()
        self.camera = camera or PinholeCamera()
        self._raw = np.empty((0, 4))
        self._history = np.empty((0, 4))
        self._history_fov = np.empty(0)
        self._history_feedback = np.empty(0)
        self._distance_scale = np.deg2rad(self.window.initial_distance_deg)
        self._committed = 0
        self._closed = False
        self.max_buffered_frames = 0
        self.window_diagnostics = []
        self.collect_diagnostics = collect_diagnostics

    @property
    def committed_frames(self):
        return self._committed

    def push(self, orientations):
        if self._closed:
            raise RuntimeError('cannot push after flush')
        incoming = unit_quaternions(orientations)
        results = []
        position = 0
        needed_pending = self.window.emit + self.window.future
        # Consume big caller chunks incrementally, without buffering their suffix.
        while position < len(incoming):
            pending = len(self._raw) - len(self._history)
            take = min(len(incoming)-position, needed_pending-pending)
            self._raw = np.concatenate([self._raw, incoming[position:position+take]])
            position += take
            self.max_buffered_frames = max(self.max_buffered_frames, len(self._raw))
            if len(self._raw) - len(self._history) >= needed_pending:
                results.append(self._process(self.window.emit, eof=False))
        return results

    def flush(self):
        if self._closed:
            return []
        results = []
        while len(self._raw) > len(self._history):
            count = min(self.window.emit, len(self._raw)-len(self._history))
            results.append(self._process(count, eof=True))
        self._closed = True
        return results

    def _process(self, commit_count, eof):
        started = perf_counter()
        config, window = self.config, self.window
        raw = self._raw
        history_count = len(self._history)
        pending_raw = raw[history_count:]
        # Actual EOF uses the same finite-end boundary as the global baseline.
        prediction = np.empty((0, 4)) if eof else predict_tail(raw, config, window)
        extended = np.concatenate([raw, prediction])
        feedback = np.ones(len(raw))
        feedback[:history_count] = self._history_feedback
        distance_scale = self._distance_scale if window.stable_distance_scale else None
        iterations = 0
        thresholds = [0.95, 0.9, 0.85, 0.8]
        info = {}
        for iteration in range(config.max_iterations+1):
            extended_feedback = np.pad(feedback, (0, len(prediction)), mode='edge')
            candidate, info = gyroflow_smooth(
                extended, config, self.camera.diagonal_fov_deg, extended_feedback,
                distance_scale=distance_scale, normalization_end=len(raw))
            pending = fit_fixed_history(candidate[history_count:len(raw)], self._history, window.boundary_weight)
            all_poses = np.concatenate([self._history, pending])
            geometry = geometric_fov(raw, all_poses, self.camera)
            zoom = zoom_curve(geometry, config)
            bad = zoom[history_count:] < config.minimum_fov
            if not bad.any() or iteration == config.max_iterations:
                break
            indices = np.flatnonzero(bad) + history_count
            feedback[indices] *= np.minimum(zoom[indices] / config.minimum_fov, thresholds[min(iteration, 3)])
            iterations += 1

        projected = np.zeros(len(pending), dtype=bool)
        if window.hard_fov:
            pending, projected = project_to_fov(pending_raw, pending, config.minimum_fov, self.camera)
        all_poses = np.concatenate([self._history, pending])
        geometry = geometric_fov(raw, all_poses, self.camera)
        zoom = zoom_curve(geometry, config)[history_count:]
        # Fast crop-in / slow release, initialized only from committed zoom.
        if len(self._history_fov):
            state = self._history_fov[-1]
            alpha = -np.expm1(-1 / (config.fps * 0.2))
            for i in range(len(zoom)):
                state = min(zoom[i], (1-alpha)*state + alpha*zoom[i])
                zoom[i] = state
        final_geometry = geometry[history_count:]

        committed_poses = pending[:commit_count].copy()
        committed_fov = zoom[:commit_count].copy()
        if window.stable_distance_scale:
            errors = info.get('first_distance', np.zeros(len(raw)))[history_count:history_count+commit_count]
            observed_scale = max(np.deg2rad(0.2), float(np.max(errors, initial=0)))
            release = np.exp(-commit_count / (config.fps * 2.0))
            self._distance_scale = max(observed_scale, release * self._distance_scale)
        diagnostic = {
            'start': self._committed, 'count': commit_count,
            'observed_through': self._committed + len(pending_raw)-1,
            'actual_lookahead_last': len(pending_raw)-commit_count,
            'iterations': iterations,
            'projected_frames': int(np.sum(projected[:commit_count])),
            'distance_scale_deg': float(np.rad2deg(self._distance_scale)),
            'milliseconds': (perf_counter()-started)*1000,
            'eof': eof,
        }
        batch = FrameBatch(self._committed, committed_poses, committed_fov,
                           final_geometry[:commit_count].copy(), diagnostic)
        self._history = np.concatenate([self._history, committed_poses])[-window.history:]
        self._history_fov = np.concatenate([self._history_fov, committed_fov])[-window.history:]
        self._history_feedback = np.concatenate([self._history_feedback, feedback[history_count:history_count+commit_count]])[-window.history:]
        consumed = history_count + commit_count
        self._raw = np.concatenate([raw[max(0, consumed-window.history):consumed], raw[consumed:]])
        self._committed += commit_count
        if self.collect_diagnostics:
            self.window_diagnostics.append(diagnostic)
        return batch


def stabilize_window(orientations, config=None, window=None, camera=None, chunk_size=None):
    """Batch convenience wrapper that still uses the strict streaming API."""
    config = config or SmoothingConfig()
    window = window or WindowConfig()
    stream = SlidingWindowStabilizer(config, window, camera, collect_diagnostics=True)
    q = unit_quaternions(orientations)
    size = int(chunk_size or window.emit)
    if size <= 0:
        raise ValueError('chunk_size must be positive')
    batches = []
    for start in range(0, len(q), size):
        batches.extend(stream.push(q[start:start+size]))
    batches.extend(stream.flush())
    if not batches:
        return StabilizationResult(q.copy(), np.empty(0), np.empty(0), 0, {'windows': []})
    return StabilizationResult(
        np.concatenate([b.orientations for b in batches]),
        np.concatenate([b.fov for b in batches]),
        np.concatenate([b.geometric_fov for b in batches]),
        sum(b.diagnostics['iterations'] for b in batches),
        {'windows': stream.window_diagnostics, 'max_buffered_frames': stream.max_buffered_frames,
         'projected_frames': sum(b.diagnostics['projected_frames'] for b in batches)})
