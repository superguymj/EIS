# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact centered-rectangle coverage for an undistorted pinhole camera.

This is an explicit synthetic camera model, not Gyroflow's lens/rolling-shutter
engine. Both algorithms use exactly the same geometry in the experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .math3d import conjugate, multiply, matrices, slerp, from_rotvec


@dataclass(frozen=True)
class PinholeCamera:
    diagonal_fov_deg: float = 100.0
    aspect: float = 16 / 9

    def __post_init__(self):
        if not 0 < self.diagonal_fov_deg < 175 or not np.isfinite(self.diagonal_fov_deg):
            raise ValueError("diagonal FOV must be between 0 and 175 degrees")
        if not np.isfinite(self.aspect) or self.aspect <= 0:
            raise ValueError("aspect must be positive")

    @property
    def half_extents(self):
        diagonal = np.tan(np.deg2rad(self.diagonal_fov_deg) / 2)
        y = diagonal / np.sqrt(1 + self.aspect**2)
        return self.aspect * y, y


def geometric_fov(raw, smooth, camera):
    """Largest c in [0,1] whose output rectangle fits the original image.

    A ray at an output corner is [+-c*tx, +-c*ty, 1]. Mapping it through
    raw.inverse()*smooth gives a + c*b. All five input frustum inequalities
    are linear in c. Their intersection gives the exact centered crop bound.
    """
    raw, smooth = np.asarray(raw), np.asarray(smooth)
    if len(raw) == 0:
        return np.empty(0)
    tx, ty = camera.half_extents
    rotation = matrices(multiply(conjugate(raw), smooth))
    corners = np.array([[-tx, -ty, 0], [-tx, ty, 0], [tx, -ty, 0], [tx, ty, 0]])
    planes = np.array([[-1, 0, tx], [1, 0, tx], [0, -1, ty], [0, 1, ty], [0, 0, 1]])
    a = rotation[:, :, 2] @ planes.T
    b = np.einsum('nij,cj,pi->ncp', rotation, corners, planes)
    numerator = a[:, None, :]
    bounds = np.divide(numerator, -b, out=np.full_like(b, np.inf), where=b < -1e-14)
    result = np.minimum(1.0, np.min(bounds, axis=(1, 2)))
    invalid_center = np.any(a < -1e-12, axis=1) | (rotation[:, 2, 2] <= 0)
    result[invalid_center] = 0
    return np.maximum(0.0, result)


def coverage_margin(raw, smooth, fov, camera):
    """Independent corner reprojection check; >=0 means covered (normalized units)."""
    tx, ty = camera.half_extents
    c = np.asarray(fov)
    corners = np.array([[-tx, -ty], [-tx, ty], [tx, -ty], [tx, ty]])
    rays = np.ones((len(c), 4, 3))
    rays[:, :, :2] = c[:, None, None] * corners[None]
    rotation = matrices(multiply(conjugate(raw), smooth))
    projected = np.einsum('nij,ncj->nci', rotation, rays)
    z = projected[:, :, 2]
    x = projected[:, :, 0] / np.maximum(z, 1e-15)
    y = projected[:, :, 1] / np.maximum(z, 1e-15)
    return np.min(np.stack([tx-np.abs(x), ty-np.abs(y), z], axis=-1), axis=(1, 2))


def project_to_fov(raw, smooth, minimum_fov, camera, iterations=30):
    """Move infeasible poses toward raw along shortest arcs, retaining feasibility.

    The low endpoint always remains feasible. Bisection need not find every
    possible disconnected feasible interval; its final result is rechecked.
    """
    result = np.array(smooth, copy=True)
    if minimum_fov >= 1 - 1e-12:
        return np.array(raw, copy=True), np.ones(len(raw), dtype=bool)
    target = min(1.0, minimum_fov + 2e-8)
    bad = geometric_fov(raw, result, camera) < target
    if not np.any(bad):
        return result, bad
    q, s = raw[bad], result[bad]
    low, high = np.zeros(len(q)), np.ones(len(q))
    for _ in range(iterations):
        mid = (low + high) / 2
        test = slerp(q, s, mid)
        feasible = geometric_fov(q, test, camera) >= target
        low = np.where(feasible, mid, low)
        high = np.where(feasible, high, mid)
    result[bad] = slerp(q, s, low)
    # Numerical fallback preserves the advertised pinhole coverage invariant.
    still_bad = geometric_fov(raw, result, camera) < minimum_fov - 1e-10
    result[still_bad] = raw[still_bad]
    return result, bad


def horizon_reference(poses, up):
    """Local +Z leveling angle and reliability relative to a world vertical."""
    vertical = np.einsum('nji,j->ni', matrices(poses), up)
    visibility = np.hypot(vertical[:, 0], vertical[:, 1])
    angle = np.arctan2(-vertical[:, 0], vertical[:, 1])
    pole = np.clip((visibility - np.sin(np.deg2rad(5.0))) /
                   (np.sin(np.deg2rad(10.0)) - np.sin(np.deg2rad(5.0))), 0.0, 1.0)
    turn = np.clip((np.pi - np.abs(angle)) / np.deg2rad(30.0), 0.0, 1.0)
    reliability = pole * pole * (3 - 2 * pole) * turn * turn * (3 - 2 * turn)
    return angle, reliability


def level_horizon(poses, up):
    """Level about the optical axis; fade only at geometric degeneracies."""
    angle, reliability = horizon_reference(poses, up)
    correction = np.zeros((len(poses), 3))
    correction[:, 2] = angle * reliability
    return multiply(poses, from_rotvec(correction))


def project_horizon_to_fov(raw, candidate, minimum_fov, camera, up, iterations=30):
    """Keep leveling while repairing view motion within the exact crop bound.

    The regular path connects a feasible anchor to the leveled target, re-levels
    the interpolated view, and interpolates residual roll separately. A failed
    full-pose projection therefore need not undo roll and view motion together.
    Near poles/half turns, use the ordinary verified projection instead.
    """
    result, projected = project_to_fov(raw, candidate, minimum_fov, camera, iterations)
    if len(raw) == 0 or minimum_fov >= 1 - 1e-12:
        return result, projected
    _, raw_reliability = horizon_reference(raw, up)
    _, candidate_reliability = horizon_reference(candidate, up)
    regular = (raw_reliability >= 1 - 1e-12) & (candidate_reliability >= 1 - 1e-12)
    if not regular.any():
        return result, projected

    indices = np.flatnonzero(regular)
    q = raw[indices]
    target = level_horizon(candidate[indices], up)
    crop_bound = min(1.0, minimum_fov + 2e-8)
    full_level_fits = geometric_fov(q, target, camera) >= crop_bound
    result[indices[full_level_fits]] = target[full_level_fits]
    pending = ~full_level_fits
    if pending.any():
        selected = indices[pending]
        q, target = raw[selected], target[pending]
        level_raw = level_horizon(q, up)
        roll_anchor, _ = project_to_fov(q, level_raw, minimum_fov, camera, iterations)
        # Both anchors are feasible. Start from whichever is closer to level;
        # this also keeps a useful full-pose solution if view motion helps roll.
        fallback = result[selected]
        anchor_angle, _ = horizon_reference(roll_anchor, up)
        fallback_angle, _ = horizon_reference(fallback, up)
        use_fallback = np.abs(fallback_angle) < np.abs(anchor_angle) - 1e-12
        anchor = np.where(use_fallback[:, None], fallback, roll_anchor)
        anchor_angle, _ = horizon_reference(anchor, up)
        level_anchor = level_horizon(anchor, up)
        low, high = np.zeros(len(q)), np.ones(len(q))
        accepted = anchor.copy()
        for _ in range(iterations):
            fraction = (low + high) / 2
            view = slerp(level_anchor, target, fraction)
            _, reliability = horizon_reference(view, up)
            view = level_horizon(view, up)
            residual = np.zeros((len(q), 3))
            residual[:, 2] = -anchor_angle * (1 - fraction)
            trial = multiply(view, from_rotvec(residual))
            feasible = ((geometric_fov(q, trial, camera) >= crop_bound) &
                        (reliability >= 1 - 1e-12))
            accepted[feasible] = trial[feasible]
            low = np.where(feasible, fraction, low)
            high = np.where(feasible, high, fraction)
        result[selected] = accepted

    # Keep the original safety guarantee even on a nonmonotone/degenerate path.
    unsafe = geometric_fov(raw, result, camera) < minimum_fov - 1e-10
    result[unsafe] = raw[unsafe]
    return result, projected
