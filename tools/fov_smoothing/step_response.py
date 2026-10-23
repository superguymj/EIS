"""Plot the current zoom_curve response to an observed future FOV step.

Run from the repository root:
    python tools/fov_smoothing/step_response.py

This isolates the FOV filter with the complete synthetic sequence available.
It is not an end-to-end bounded-lookahead stabilization experiment.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gyroflow_window.gyroflow_global import SmoothingConfig, zoom_curve


def main():
    destination = ROOT / 'experiments/fov_smoothing'
    destination.mkdir(parents=True, exist_ok=True)
    fps = 30
    time = np.arange(-8 * fps, 2 * fps + 1) / fps
    geometric = np.where(time < 0, 0.95, 0.60)
    config = SmoothingConfig(fps=fps, minimum_fov=0.6, zoom_seconds=4.0)
    smooth = zoom_curve(geometric, config)
    curvature = np.diff(smooth, n=2) * fps**2
    turning = np.flatnonzero((time[1:-1] < 0) & (curvature > 0))
    turn_time = float(time[1:-1][turning[0]])
    assert np.isfinite(smooth).all()
    assert np.all(smooth <= geometric + 1e-12)
    assert np.all(smooth >= 0.6 - 1e-12)
    assert np.all(np.diff(smooth) <= 1e-12)

    plt.rcParams.update({
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 11,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1.65, 1]})
    for ax in axes:
        ax.plot(time, geometric, color='#537a99', linestyle='--', linewidth=1.8,
                drawstyle='steps-post', label='逐帧安全 FOV')
        ax.plot(time, smooth, color='#df721d', linewidth=2.5, label='两轮包络平滑后的 FOV')
        ax.axvline(0, color='#9aa3ab', linewidth=0.8, alpha=0.8)
        ax.grid(alpha=0.2)
        ax.set_xlabel('距突降时刻的时间（秒）')
        ax.set_ylabel('FOV 系数')
    axes[0].set_xlim(-8, 1)
    axes[0].set_ylim(0.585, 0.97)
    axes[0].set_title('提前收紧：前段先缓后快')
    axes[0].legend(loc='upper right')
    axes[1].set_xlim(-1.1, 0.15)
    axes[1].set_ylim(0.596, 0.681)
    axes[1].set_title('末端放大：接近低点时再放缓')
    turn_index = int(np.argmin(np.abs(time - turn_time)))
    axes[1].scatter([turn_time], [smooth[turn_index]], color='#5a435a', s=35, zorder=4)
    axes[1].annotate(f'曲率变号约在 {turn_time:.1f} 秒',
                     xy=(turn_time, smooth[turn_index]), xytext=(-0.98, 0.615),
                     arrowprops={'arrowstyle': '->', 'color': '#5a435a'}, fontsize=10)
    fig.suptitle('合成示例：安全 FOV 从 0.95 突降至 0.60\n当前 zoom_curve，30 fps，时间常数 4 秒 + 0.2 秒', fontsize=14)
    fig.text(0.5, 0.025, '此图为整段已知输入的滤波响应；实际滑窗只能利用已观测的前瞻，不能预知窗口外的下降。',
             ha='center', fontsize=10, color='#555555')
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    fig.savefig(destination / 'step_response.png', dpi=180)
    fig.savefig(destination / 'step_response.svg')
    plt.close(fig)
    np.savetxt(destination / 'step_response.csv', np.column_stack([time, geometric, smooth]),
               delimiter=',', header='time_to_drop_seconds,geometric_fov,smoothed_fov', comments='')
    metadata = {
        'kind': 'synthetic full-sequence zoom_curve response',
        'fps': fps, 'zoom_seconds': config.zoom_seconds, 'second_stage_seconds': 0.2,
        'high_fov': 0.95, 'low_fov': 0.6,
        'first_positive_curvature_before_drop_seconds': turn_time,
        'max_coverage_bound_excess': float(np.max(smooth - geometric)),
        'plot': str(destination / 'step_response.png'),
    }
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
