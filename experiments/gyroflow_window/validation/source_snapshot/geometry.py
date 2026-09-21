# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact centered-rectangle coverage for an undistorted pinhole camera.

This is an explicit synthetic camera model, not Gyroflow's lens/rolling-shutter
engine. Both algorithms use exactly the same geometry in the experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .math3d import conjugate, multiply, matrices, slerp


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
