"""Seeded clean camera paths plus independently generated rotational shake."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .math3d import from_rotvec, multiply, unit_quaternions


MOTION_KINDS = ('static', 'jitter', 'pan', 'stop_start', 'fast_turn', 'roll',
                'walking', 'impulse', 'chirp', 'multi_axis', 'late_outlier',
                'wrap', 'return', 'handheld')


@dataclass
class SyntheticMotion:
    raw: np.ndarray
    truth: np.ndarray
    time: np.ndarray
    name: str
    seed: int


def make_motion(kind, seed=0, fps=30.0, seconds=12.0):
    if kind not in MOTION_KINDS:
        raise ValueError(f'unknown motion: {kind}')
    rng = np.random.default_rng(seed)
    t = np.arange(max(1, int(round(fps*seconds)))) / fps
    n = len(t)
    angles = np.zeros((n, 3))  # pitch x, yaw y, roll z, in degrees
    phase = rng.uniform(-np.pi, np.pi, size=3)
    center = seconds * rng.uniform(0.40, 0.60)
    amplitude = 1.0
    shake = np.zeros_like(angles)
    if kind == 'static':
        amplitude = 0.0
        angles[:] = [11, 27, -8]
    elif kind == 'jitter':
        amplitude = 2.0
    elif kind == 'pan':
        angles[:, 1] = rng.uniform(7, 15) * t
        angles[:, 0] = 3 * np.sin(0.4*t)
    elif kind == 'stop_start':
        angles[:, 1] = 28*(1+np.tanh((t-seconds*0.25)/0.55)) + 20*(1+np.tanh((t-seconds*0.70)/0.4))
        amplitude = 1.5
    elif kind == 'fast_turn':
        angles[:, 1] = 55*(1+np.tanh((t-center)/0.12))
        angles[:, 0] = 15*(1+np.tanh((t-center-0.15)/0.3))
        amplitude = 0.8
    elif kind == 'roll':
        angles[:, 2] = 35*np.sin(0.7*t) + 55*np.exp(-((t-center)/0.6)**2)
        angles[:, 1] = 4*t
        amplitude = 2.0
    elif kind == 'walking':
        angles[:, 1] = 4*t
        angles[:, 0] = 6*np.sin(0.3*t)
        shake[:, 0] += 3.2*np.sin(2*np.pi*2.1*t + phase[0])
        shake[:, 1] += 1.8*np.sin(2*np.pi*1.05*t + phase[1])
        shake[:, 2] += 2.0*np.sin(2*np.pi*2.1*t + phase[2])
        amplitude = 0.45
    elif kind == 'impulse':
        angles[:, 1] = 2*t
        for moment in [0.2*seconds, center, 0.82*seconds]:
            pulse = np.exp(-0.5*((t-moment)/rng.uniform(0.02, 0.06))**2)
            shake += pulse[:, None]*rng.uniform(-18, 18, size=3)
        amplitude = 0.4
    elif kind == 'chirp':
        angles[:, 1] = 3*t
        envelope = 1+2*np.sin(np.pi*t/seconds)**2
        for axis in range(3):
            shake[:, axis] = envelope*np.sin(2*np.pi*(0.7*t + (5+axis)*t*t/(2*seconds)) + phase[axis])
        amplitude = 0.25
    elif kind == 'multi_axis':
        angles[:, 0] = 20*np.sin(0.6*t)
        angles[:, 1] = 35*np.sin(0.4*t) + 6*t
        angles[:, 2] = 15*np.sin(0.7*t+0.6)
        amplitude = 1.8
    elif kind == 'late_outlier':
        angles[:, 1] = 1.5*t
        moment = seconds*0.82
        shake[:, 0] += 30*np.exp(-0.5*((t-moment)/0.055)**2)
        shake[:, 2] += 22*np.exp(-0.5*((t-moment-0.05)/0.045)**2)
        amplitude = 0.6
    elif kind == 'wrap':
        angles[:, 1] = 40*t - 175
        angles[:, 2] = 12*np.sin(0.5*t)
        amplitude = 1.4
    elif kind == 'return':
        angles[:, 1] = 50*(np.tanh((t-center+1.1)/0.25)-np.tanh((t-center-1.1)/0.25))
        angles[:, 0] = 10*np.sin(0.45*t)
        amplitude = 1.0
    elif kind == 'handheld':
        velocity = np.zeros(3)
        for i in range(1, n):
            velocity = 0.985*velocity + rng.normal(size=3)*0.9
            angles[i] = angles[i-1] + velocity/fps
        amplitude = 2.4
    for axis in range(3):
        frequency = rng.uniform(4, min(10, fps*0.42))
        shake[:, axis] += amplitude*(0.7*np.sin(2*np.pi*frequency*t + phase[axis]) + 0.25*rng.normal(size=n))
    q_pitch = from_rotvec(np.deg2rad(angles[:, 0, None])*np.array([1, 0, 0]))
    q_yaw = from_rotvec(np.deg2rad(angles[:, 1, None])*np.array([0, 1, 0]))
    q_roll = from_rotvec(np.deg2rad(angles[:, 2, None])*np.array([0, 0, 1]))
    truth = multiply(multiply(q_yaw, q_pitch), q_roll)
    raw = multiply(truth, from_rotvec(np.deg2rad(shake)))
    # Quaternion sign is not physical orientation; algorithms must tolerate it.
    raw[rng.random(n) < 0.15] *= -1
    return SyntheticMotion(unit_quaternions(raw), unit_quaternions(truth), t, kind, seed)
