"""Reproduce FOV seam comparisons with fixed geometry and real pose inputs.

Run: python tools/fov_smoothing/compare_history.py
Real cases reuse the original poses already exported for 046 and 052.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gyroflow_window.geometry import PinholeCamera, coverage_margin
from gyroflow_window.gyroflow_global import SmoothingConfig, zoom_curve
from gyroflow_window.sliding_window import WindowConfig, fit_fov_history, stabilize_window

OUT = ROOT / 'experiments/fov_smoothing'


def metrics(fov, starts):
    first = np.diff(fov)
    second = np.diff(fov, n=2)
    # New first-frame velocity minus the last committed velocity.
    seams = np.asarray([start - 2 for start in starts if 2 <= start < len(fov)])
    return {
        'minimum_fov': float(np.min(fov)), 'mean_fov': float(np.mean(fov)),
        'max_frame_change': float(np.max(np.abs(first))),
        'second_difference_rms': float(np.sqrt(np.mean(second**2))),
        'seam_slope_change_rms': float(np.sqrt(np.mean(second[seams]**2))),
    }


def replay_fixed_geometry(geometry, config, window):
    """Isolate zoom with H/E/F visibility; no pose filter or feedback in this fixture."""
    output = []
    starts = []
    for start in range(0, len(geometry), window.emit):
        left = max(0, start - window.history)
        right = min(len(geometry), start + window.emit + window.future)
        target = zoom_curve(geometry[left:right], config)[start-left:]
        if output:
            state = output[-1]
            alpha = -np.expm1(-1 / (config.fps * 0.2))
            for i in range(len(target)):
                state = min(target[i], (1-alpha) * state + alpha * target[i])
                target[i] = state
        if window.fov_boundary_weight > 0:
            target, info = fit_fov_history(
                target, output[-2:], geometry[start:right],
                window.fov_boundary_weight * (config.fps / 30.)**4, config.minimum_fov)
            assert info['success']
        output.extend(target[:window.emit])
        starts.append(start)
    result = np.asarray(output)
    assert np.all(result <= geometry + 1e-12)
    assert np.all(result >= config.minimum_fov - 1e-12)
    return result, starts


def plot_comparison(name, time, geometry, old, new, zoom_center, title):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax in axes:
        ax.plot(time, geometry, color='#788b98', alpha=0.6, linewidth=1.1,
                linestyle='--', label='逐帧安全 FOV 上限')
        ax.plot(time, old, color='#de7525', linewidth=1.8, label='修改前：快速收紧 / 缓慢释放')
        ax.plot(time, new, color='#168575', linewidth=2, label='修改后：固定历史 + 覆盖约束拟合')
        ax.set_xlabel('时间（秒）')
        ax.set_ylabel('FOV 系数')
        ax.grid(alpha=0.2)
    axes[0].set_title(title)
    axes[0].legend(loc='best', fontsize=10)
    left, right = max(time[0], zoom_center - 0.8), min(time[-1], zoom_center + 1.0)
    axes[1].set_xlim(left, right)
    mask = (time >= left) & (time <= right)
    lo = min(np.min(old[mask]), np.min(new[mask]))
    # Magnify the actual output; the geometric ceiling may lie above this view.
    hi = max(np.max(old[mask]), np.max(new[mask]))
    pad = max(0.003, (hi-lo) * 0.08)
    axes[1].set_ylim(lo-pad, hi+pad)
    axes[1].set_title('修改前最大单帧变化附近的输出曲线放大图')
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.png', dpi=170)
    fig.savefig(OUT / f'{name}.svg')
    plt.close(fig)
    np.savetxt(OUT / f'{name}.csv', np.column_stack([time, geometry, old, new]),
               delimiter=',', comments='', header='time_s,geometric_fov,legacy_fov,history_fitted_fov')


def full_sequence_filters(config):
    time = np.arange(-30*30, 2*30+1) / 30
    geometry = np.where(time < 0, 0.95, 0.6)
    two = zoom_curve(geometry, config)
    four = zoom_curve(two, config)
    fig, ax = plt.subplots(figsize=(11, 4.7))
    ax.step(time, geometry, where='post', linestyle='--', color='#788b98', label='安全 FOV')
    ax.plot(time, two, color='#de7525', label='两轮：4 秒 + 0.2 秒')
    ax.plot(time, four, color='#168575', label='四轮：再重复 4 秒 + 0.2 秒')
    ax.set(xlim=(-24, 1), ylim=(0.585, 0.965), xlabel='距突降的时间（秒）', ylabel='FOV 系数',
           title='整段已知输入：增加滤波轮数会更早裁切；不增加实际滑窗前瞻')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'two_vs_four.png', dpi=170)
    plt.close(fig)
    report = {}
    for label, value in [('two', two), ('four', four)]:
        curvature = np.diff(value, n=2)
        index = np.flatnonzero((time[1:-1] < 0) & (curvature > 1e-12))[0]
        report[label] = {
            'positive_curvature_from_s': float(time[1:-1][index]),
            'fov_four_seconds_before_drop': float(value[np.flatnonzero(time == -4)[0]]),
        }
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11})
    config = SmoothingConfig(minimum_fov=0.6)
    report = {'full_sequence_filters': full_sequence_filters(config), 'synthetic': {}, 'real': {}}
    time = np.arange(360) / config.fps - 6.
    geometry = np.where((time >= 0) & (time < 2), 0.6, 0.95)
    for future in [0, 15, 30]:
        window = WindowConfig(future=future)
        old, starts = replay_fixed_geometry(geometry, config, replace(window, fov_boundary_weight=0))
        new, _ = replay_fixed_geometry(geometry, config, window)
        first_changed = int(np.flatnonzero(old < 0.95-1e-8)[0])
        report['synthetic'][str(future)] = {
            'window': asdict(window), 'first_changed_frame': first_changed,
            'first_change_old': float(old[first_changed]-old[first_changed-1]),
            'first_change_new': float(new[first_changed]-new[first_changed-1]),
            'old': metrics(old, starts), 'new': metrics(new, starts),
            'coverage_violation_frames': int(np.sum(new > geometry + 1e-9)),
        }
        if future == 0:
            assert old[180] == new[180] == 0.6
        if future == 15:
            assert abs(report['synthetic'][str(future)]['first_change_new']) < 0.015
            plot_comparison('step_history_comparison', time, geometry, old, new,
                            time[first_changed], '固定几何 FOV 滑窗重放：H/E/F=60/6/15，30 fps')

    for case in ['046', '052']:
        source = ROOT / 'experiments/horizon_video' / case
        meta = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
        with np.load(source / 'orientations.npz') as saved:
            raw, saved_fov = saved['raw'].copy(), saved['fov'].copy()
        cfg = SmoothingConfig(**meta['smoothing_config'])
        window = WindowConfig(**meta['window_config'])
        camera = PinholeCamera(meta['camera_model']['diagonal_fov_deg'], meta['camera_model']['aspect'])
        old = stabilize_window(raw, cfg, replace(window, fov_boundary_weight=0), camera)
        new = stabilize_window(raw, cfg, window, camera)
        saved_is_legacy = meta['window_config'].get('fov_boundary_weight', 0) == 0
        assert np.array_equal(old.fov if saved_is_legacy else new.fov, saved_fov)
        assert np.array_equal(old.orientations, new.orientations)
        assert np.array_equal(old.geometric_fov, new.geometric_fov)
        assert old.iterations == new.iterations
        assert [d['feedback_min'] for d in old.diagnostics['windows']] == [
            d['feedback_min'] for d in new.diagnostics['windows']]
        margin = coverage_margin(raw, new.orientations, new.fov, camera)
        assert margin.min() >= -1e-9
        assert new.fov.min() >= cfg.minimum_fov - 1e-9
        assert not any(d['fov_boundary_fallback'] for d in new.diagnostics['windows'])
        starts = [d['start'] for d in new.diagnostics['windows']]
        times = np.arange(len(raw)) / cfg.fps
        drop = int(np.argmax(np.abs(np.diff(old.fov))) + 1)
        plot_comparison(f'{case}_history_comparison', times, new.geometric_fov, old.fov, new.fov,
                        times[drop], f'{case} 真实姿态输入：feedback 开启，FOV 下限 0.6，H/E/F=60/6/15')
        np.savez_compressed(OUT / f'{case}_comparison.npz', raw=raw, orientations=new.orientations,
                            geometric_fov=new.geometric_fov, old_fov=old.fov, fov=new.fov)
        report['real'][case] = {
            'input': str(source.relative_to(ROOT)), 'video_sha256': meta['video_sha256'],
            'smoothing_config': asdict(cfg), 'window_config': asdict(window),
            'old': metrics(old.fov, starts), 'new': metrics(new.fov, starts),
            'orientations_bitwise_equal': True, 'geometry_bitwise_equal': True,
            'legacy_fov_matches_saved': bool(np.array_equal(old.fov, saved_fov)),
            'feedback_iterations': new.iterations,
            'feedback_minimum': min(d['feedback_min'] for d in new.diagnostics['windows']),
            'coverage_minimum_margin': float(margin.min()),
            'coverage_violation_frames': int(np.sum(margin < -1e-9)),
            'fov_floor_violation_frames': int(np.sum(new.fov < cfg.minimum_fov-1e-9)),
            'fov_solver_fallbacks': 0,
        }
    report['source_sha256'] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ['gyroflow_window/sliding_window.py', 'gyroflow_window/gyroflow_global.py',
                     'gyroflow_window/geometry.py', 'tools/fov_smoothing/compare_history.py']}
    (OUT / 'history_comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                                                encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
