# SPDX-License-Identifier: GPL-3.0-or-later
"""Quaternion primitives. Public convention: float64 [w, x, y, z], camera to world."""
from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


def unit_quaternions(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("orientations must have shape (N, 4), in wxyz order")
    if not np.isfinite(q).all():
        raise ValueError("orientations must be finite")
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("zero quaternion")
    return np.ascontiguousarray(q / norms[:, None])


def conjugate(q):
    result = np.array(q, dtype=np.float64, copy=True)
    result[..., 1:] *= -1
    return result


def multiply(a, b):
    a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    result = np.empty_like(a, dtype=np.float64)
    result[..., 0] = a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], axis=-1)
    result[..., 1:] = (a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:]
                       + np.cross(a[..., 1:], b[..., 1:]))
    return result


def rotvec(q):
    q = np.asarray(q, dtype=np.float64)
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    length = np.linalg.norm(q[..., 1:], axis=-1)
    angle = 2 * np.arctan2(length, np.maximum(q[..., 0], 0.0))
    scale = np.divide(angle, length, out=np.full_like(length, 2.0), where=length > 1e-12)
    return q[..., 1:] * scale[..., None]


def from_rotvec(v):
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1)
    scale = 0.5 * np.sinc(angle / (2 * np.pi))
    return np.concatenate([np.cos(angle / 2)[..., None], v * scale[..., None]], axis=-1)


def angular_distance(a, b):
    return np.linalg.norm(rotvec(multiply(conjugate(a), b)), axis=-1)


def world_velocity(q, fps):
    return rotvec(multiply(q[1:], conjugate(q[:-1]))) * fps


def slerp(a, b, weight):
    """Broadcasting shortest-arc SLERP; weight may be an N-vector."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    delta = rotvec(multiply(conjugate(a), b))
    w = np.asarray(weight)
    return multiply(a, from_rotvec(delta * (w if w.ndim == 0 else w[..., None])))


def matrices(q):
    w, x, y, z = np.moveaxis(np.asarray(q), -1, 0)
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0] = 1 - 2 * (y*y + z*z)
    r[..., 0, 1] = 2 * (x*y - z*w)
    r[..., 0, 2] = 2 * (x*z + y*w)
    r[..., 1, 0] = 2 * (x*y + z*w)
    r[..., 1, 1] = 1 - 2 * (x*x + z*z)
    r[..., 1, 2] = 2 * (y*z - x*w)
    r[..., 2, 0] = 2 * (x*z - y*w)
    r[..., 2, 1] = 2 * (y*z + x*w)
    r[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return r


@njit(cache=True)
def _slerp_one(a, b, weight):
    dot = np.dot(a, b)
    sign = 1.0
    if dot < 0.0:
        dot = -dot
        sign = -1.0
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1e-12:
        result = (1.0 - weight) * a + weight * sign * b
    else:
        theta = math.acos(dot)
        result = (math.sin((1-weight)*theta)*a + math.sin(weight*theta)*sign*b) / math.sin(theta)
    return result / math.sqrt(np.dot(result, result))


@njit(cache=True)
def quaternion_forward_backward(q, coefficients):
    result = q.copy()
    state = q[0].copy()
    for i in range(len(q)):
        state = _slerp_one(state, q[i], coefficients[i])
        result[i] = state
    state = result[-1].copy()
    for i in range(len(q)-1, -1, -1):
        state = _slerp_one(state, result[i], coefficients[i])
        result[i] = state
    return result


@njit(cache=True)
def scalar_forward_backward(values, coefficient):
    result = values.copy()
    state = result[0]
    for i in range(1, len(result)):
        state = (1-coefficient) * state + coefficient * result[i]
        result[i] = state
    for i in range(len(result)-2, -1, -1):
        state = (1-coefficient) * state + coefficient * result[i]
        result[i] = state
    return result


@njit(cache=True)
def envelope(values, coefficient):
    """Gyroflow envelope_follower: reverse then forward, both bounded by input."""
    result = values.copy()
    state = result[-1]
    for i in range(len(result)-1, -1, -1):
        state = min(result[i], coefficient*result[i] + (1-coefficient)*state)
        result[i] = state
    for i in range(len(result)):
        state = min(result[i], coefficient*result[i] + (1-coefficient)*state)
        result[i] = state
    return result
