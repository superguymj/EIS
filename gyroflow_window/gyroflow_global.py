# SPDX-License-Identifier: GPL-3.0-or-later
# Default algorithm derived from Gyroflow, Copyright 2021-2022 Aphobius.
# Envelope algorithm derived from Gyroflow, Copyright 2021 Marc Roeschlin,
# Adrian, Maik. Reference commit: 2331ebb3c253fc7cf357642518b0c8bf2937f815.
"""Source-faithful constant-parameter, non-per-axis Gyroflow baseline.

Includes DefaultAlgo's two forward/backward SLERP passes, distance weighting,
EnvelopeFollower zoom, and the original finite zoom-feedback iterations.
Optional horizon correction is a workspace extension, disabled by default.
Excludes lens distortion, rolling shutter, keyframes, retiming, and trim ranges.
The shared pinhole camera replaces Gyroflow's image geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .math3d import (unit_quaternions, angular_distance, scalar_forward_backward,
                     quaternion_forward_backward, envelope, matrices, multiply,
                     from_rotvec)
from .geometry import PinholeCamera, geometric_fov


@dataclass(frozen=True)
class SmoothingConfig:
    fps: float = 30.0
    smoothness: float = 0.5
    slow_seconds: float = 1.0
    fast_seconds: float = 0.1
    zoom_seconds: float = 4.0
    minimum_fov: float = 0.8
    max_iterations: int = 5
    enablehorizon: bool = False
    horizon_up: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self):
        for name in ['fps', 'smoothness', 'slow_seconds', 'fast_seconds', 'zoom_seconds']:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.fast_seconds > self.slow_seconds:
            raise ValueError('fast_seconds must not exceed slow_seconds')
        if not np.isfinite(self.minimum_fov) or not 0 < self.minimum_fov <= 1:
            raise ValueError('minimum_fov must be in (0, 1]')
        if not isinstance(self.max_iterations, int) or self.max_iterations < 0:
            raise ValueError('max_iterations must be a nonnegative integer')
        if not isinstance(self.enablehorizon, bool):
            raise ValueError('enablehorizon must be a boolean')
        up = np.asarray(self.horizon_up, dtype=np.float64)
        if up.shape != (3,) or not np.isfinite(up).all() or not np.any(up):
            raise ValueError('horizon_up must be a finite, nonzero 3-vector')
        # Normalize without overflowing on large, finite vectors; freeze a copy.
        up = up / np.max(np.abs(up))
        up /= np.linalg.norm(up)
        object.__setattr__(self, 'horizon_up', tuple(float(v) for v in up))


def horizon_gain(feedback):
    """Map the existing finite feedback to a bounded, soft horizon preference.

    h(c)=2c/(1+c) retains more correction than c inside (0,1), while still
    releasing it at zero. No separate horizon feedback state is maintained.
    """
    feedback = np.asarray(feedback, dtype=np.float64)
    if not np.isfinite(feedback).all():
        raise ValueError('horizon feedback must be finite')
    c = np.clip(feedback, 0.0, 1.0)
    return 2.0 * c / (1.0 + c)


def _apply_horizon(poses, config, feedback):
    """Correct image roll about local +Z, preserving the optical axis.

    Camera +Y should agree with the supplied world vertical. Taper the correction
    near a vertical optical axis and near an ambiguous half turn. This stateless
    best-effort policy does not attempt to make upside-down footage upright.
    """
    if not config.enablehorizon or len(poses) == 0:
        return poses
    gain = horizon_gain(feedback)
    vertical = np.einsum('nji,j->ni', matrices(poses), config.horizon_up)
    visibility = np.hypot(vertical[:, 0], vertical[:, 1])
    angle = np.arctan2(-vertical[:, 0], vertical[:, 1])

    # Suppress ill-defined leveling within 5 degrees of a pole; fully enable it
    # beyond 10 degrees. Smoothstep avoids a hard on/off threshold in the pose.
    pole = np.clip((visibility - np.sin(np.deg2rad(5.0))) /
                   (np.sin(np.deg2rad(10.0)) - np.sin(np.deg2rad(5.0))), 0.0, 1.0)
    pole = pole * pole * (3.0 - 2.0 * pole)
    # Partial shortest-arc correction has a branch cut at +/-180 degrees.
    # Fade between 150 and 180 degrees, so crossing the cut stays continuous.
    turn = np.clip((np.pi - np.abs(angle)) / np.deg2rad(30.0), 0.0, 1.0)
    turn = turn * turn * (3.0 - 2.0 * turn)
    correction = np.zeros((len(poses), 3))
    correction[:, 2] = angle * gain * pole * turn
    return multiply(poses, from_rotvec(correction))


@dataclass
class StabilizationResult:
    orientations: np.ndarray
    fov: np.ndarray
    geometric_fov: np.ndarray
    iterations: int
    diagnostics: dict


def gyroflow_smooth(q, config, diagonal_fov_deg=120.0, feedback=None, distance_scale=None, normalization_end=None):
    """DefaultAlgo port. None distance_scale selects exact global normalization.

    Returns (poses, diagnostics). Scalar/vector feedback corresponds to
    smoothing_fov_limit_per_frame, sampled here at the same fixed video FPS.
    duration_ms=N/fps*1000 reproduces the source's N/duration sample rate.
    Optional horizon correction shares this feedback and runs on every nonempty
    return path, before the caller evaluates geometry and updates feedback.
    """
    q = np.ascontiguousarray(q, dtype=np.float64)
    if len(q) == 0:
        return q.copy(), {'distance_max': 0.0}
    a_s = -np.expm1(-1 / (config.fps * config.slow_seconds))
    a_f = -np.expm1(-1 / (config.fps * config.fast_seconds))
    velocity = np.zeros(len(q))
    if len(q) > 1:
        velocity[1:] = np.rad2deg(angular_distance(q[:-1], q[1:])) * config.fps
    velocity = scalar_forward_backward(velocity, a_f)
    scalers = np.ones(len(q)) if feedback is None else np.broadcast_to(feedback, (len(q),))
    max_velocity = 500 * config.smoothness * (diagonal_fov_deg / 120.0) * scalers * 0.5
    ratio = velocity / np.maximum(max_velocity, 1e-15)
    alpha = np.minimum(1.0, a_s + (a_f-a_s) * ratio)
    first = quaternion_forward_backward(q, alpha)
    distance = angular_distance(q, first)
    first_distance = distance.copy()
    norm_end = len(q) if normalization_end is None else int(normalization_end)
    maximum = float(np.max(distance[:norm_end]))
    if distance_scale is None:
        if maximum == 0.0:
            # Rust 0/0 -> NaN; f64::min(NaN, 1) -> 1, so pass two is identity.
            return _apply_horizon(first, config, scalers), {'distance_max': maximum, 'first_distance': first_distance}
        distance = distance / maximum
    else:
        distance = np.clip(distance / max(float(distance_scale), 1e-8), 0, 1)
    distance[distance < 0.5] = 0.0
    distance = scalar_forward_backward(distance, a_f)
    if distance_scale is None:
        normalizer = float(np.max(distance[:norm_end]))
        if normalizer == 0.0:
            return _apply_horizon(first, config, scalers), {'distance_max': maximum, 'first_distance': first_distance}
        distance /= normalizer
    distance = (distance + 1) / 2
    alpha_second = np.minimum(1.0, a_s + (a_f-a_s) * ratio * distance)
    second = quaternion_forward_backward(first, alpha_second)
    return _apply_horizon(second, config, scalers), {'distance_max': maximum, 'first_distance': first_distance}


def zoom_curve(geometric, config):
    if len(geometric) == 0:
        return geometric.copy()
    alpha = -np.expm1(-1 / (config.fps * config.zoom_seconds))
    alpha_second = -np.expm1(-1 / (config.fps * 0.2))
    return envelope(envelope(geometric, alpha), alpha_second)


def stabilize_global(orientations, config=None, camera=None):
    config = config or SmoothingConfig()
    camera = camera or PinholeCamera()
    q = unit_quaternions(orientations)
    if len(q) == 0:
        return StabilizationResult(q, np.empty(0), np.empty(0), 0, {})
    feedback = np.ones(len(q))
    thresholds = [0.95, 0.9, 0.85, 0.8]
    iterations = 0
    smooth, info = gyroflow_smooth(q, config, camera.diagonal_fov_deg, feedback)
    geometric = geometric_fov(q, smooth, camera)
    fov = zoom_curve(geometric, config)
    for iteration in range(config.max_iterations):
        bad = fov < config.minimum_fov
        if not bad.any():
            break
        feedback[bad] *= np.minimum(fov[bad] / config.minimum_fov, thresholds[min(iteration, 3)])
        smooth, info = gyroflow_smooth(q, config, camera.diagonal_fov_deg, feedback)
        geometric = geometric_fov(q, smooth, camera)
        fov = zoom_curve(geometric, config)
        iterations += 1
    info = dict(info, feedback=feedback)
    return StabilizationResult(smooth, fov, geometric, iterations, info)
