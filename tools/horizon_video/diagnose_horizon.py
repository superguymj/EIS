"""Audit the horizon/crop tradeoff without changing the production algorithm."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

from plot_video_angles import ROOT, camera_euler
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.spatial.transform import Rotation

from gyroflow_window.geometry import PinholeCamera, geometric_fov
from gyroflow_window.gyroflow_global import SmoothingConfig
from compare_horizon import load_reference

# This script records the failure of the historical algorithm at FOV=.8.
# The corrected algorithm and FOV=.6 are evaluated by compare_horizon.py.
reference_core, sliding = load_reference()
gyroflow_smooth = reference_core.gyroflow_smooth
_apply_horizon = reference_core._apply_horizon


def trace_window(raw, config, window, camera):
    original_smooth = sliding.gyroflow_smooth
    original_fit = sliding.fit_fixed_history
    original_project = sliding.project_to_fov
    current = {}
    emitted = []

    def smooth(q, config, diagonal, feedback, **kwargs):
        before, _ = original_smooth(q, replace(config, enablehorizon=False), diagonal, feedback, **kwargs)
        after, info = original_smooth(q, config, diagonal, feedback, **kwargs)
        current.update(filtered=before.copy(), horizon=after.copy(), feedback=feedback.copy())
        return after, info

    def fit(candidate, history, weight):
        output = original_fit(candidate, history, weight)
        current['seam'] = output.copy()
        return output

    def project(raw, candidate, minimum_fov, camera):
        output, projected = original_project(raw, candidate, minimum_fov, camera)
        current['projected'] = projected.copy()
        return output, projected

    class TracedStream(sliding.SlidingWindowStabilizer):
        def _process(self, commit_count, eof):
            history_count = len(self._history)
            current.clear()
            batch = super()._process(commit_count, eof)
            entry = {
                stage: current[stage][history_count:history_count + commit_count].copy()
                for stage in ['filtered', 'horizon', 'feedback']
            }
            entry.update(seam=current['seam'][:commit_count].copy(), output=batch.orientations.copy())
            entry['projected'] = current.get('projected', np.zeros(commit_count, dtype=bool))[:commit_count]
            emitted.append(entry)
            return batch

    stream = TracedStream(config, window, camera, collect_diagnostics=True)
    with patch.object(sliding, 'gyroflow_smooth', smooth), \
         patch.object(sliding, 'fit_fixed_history', fit), \
         patch.object(sliding, 'project_to_fov', project):
        stream.push(raw)
        stream.flush()
    arrays = {stage: np.concatenate([entry[stage] for entry in emitted]) for stage in emitted[0]}
    arrays['iterations'] = np.array([d['iterations'] for d in stream.window_diagnostics])
    return arrays


def euler(q):
    return camera_euler(Rotation.from_quat(q[:, [1, 2, 3, 0]]))


def main():
    directory = ROOT / 'experiments/horizon_video/052'
    frames = json.loads((directory / 'original_camera.json').read_text(encoding='utf-8-sig'))
    source = np.array([f['org_quat'] for f in frames])
    rotation = Rotation.from_euler('x', -90, degrees=True) * Rotation.from_quat(source[:, [1, 2, 3, 0]])
    raw = rotation.as_quat()[:, [3, 0, 1, 2]]
    times = np.array([f['timestamp_ms'] for f in frames]) / 1000
    times -= times[0]
    config = SmoothingConfig(fps=float(1 / np.median(np.diff(times))), minimum_fov=0.8)
    window, camera = sliding.WindowConfig(), PinholeCamera()
    runs = {
        'horizon_off': trace_window(raw, config, window, camera),
        'horizon_on': trace_window(raw, replace(config, enablehorizon=True), window, camera),
        'unconstrained_diagnostic': trace_window(raw, replace(config, enablehorizon=True, max_iterations=0),
                                               replace(window, hard_fov=False), camera),
    }
    raw_angles = euler(raw)
    summary = {}
    csv_columns = [times, *raw_angles.T]
    csv_names = ['time_s', 'raw_pitch_deg', 'raw_yaw_deg', 'raw_roll_deg']
    for name, run in runs.items():
        stages = {}
        for stage in ['filtered', 'horizon', 'seam', 'output']:
            angles = euler(run[stage])
            stages[stage] = {
                'roll_rms_deg': float(np.sqrt(np.mean(angles[:, 2] ** 2))),
                'pitch_yaw_delta_from_raw_rms_deg': np.sqrt(np.mean((angles[:, :2] - raw_angles[:, :2]) ** 2, axis=0)).tolist(),
                'pitch_yaw_second_difference_rms_deg': np.sqrt(np.mean(np.diff(angles[:, :2], n=2, axis=0) ** 2, axis=0)).tolist(),
                'minimum_geometric_fov': float(geometric_fov(raw, run[stage], camera).min()),
            }
        output_angles = euler(run['output'])
        summary[name] = {
            'feedback_percentiles_0_50_100': np.percentile(run['feedback'], [0, 50, 100]).tolist(),
            'projected_frames': int(np.sum(run['projected'])),
            'total_frames': len(raw), 'stages': stages,
        }
        csv_columns.extend([*output_angles.T, run['feedback'], run['projected']])
        csv_names.extend([f'{name}_{axis}' for axis in ['pitch_deg', 'yaw_deg', 'roll_deg', 'feedback', 'projected']])

    # Isolate the horizon operation with identical input/feedback, so a change
    # caused by crop feedback is not incorrectly attributed to quaternion axes.
    before, _ = gyroflow_smooth(raw, config, camera.diagonal_fov_deg)
    after, _ = gyroflow_smooth(raw, replace(config, enablehorizon=True), camera.diagonal_fov_deg)
    delta = np.max(np.abs(euler(after)[:, :2] - euler(before)[:, :2]), axis=0)
    leveled_raw = _apply_horizon(raw, replace(config, enablehorizon=True), np.ones(len(raw)))
    level_fov = geometric_fov(raw, leveled_raw, camera)
    summary['isolated_horizon'] = {
        'pitch_yaw_max_change_deg': delta.tolist(),
        'roll_rms_deg': float(np.sqrt(np.mean(euler(after)[:, 2] ** 2))),
        'fully_level_original_view_minimum_fov': float(level_fov.min()),
        'fully_level_original_view_frames_below_08': int(np.sum(level_fov < 0.8)),
    }
    assert np.max(delta) < 1e-10
    assert summary['isolated_horizon']['roll_rms_deg'] < 1e-10
    # Compare with the uninstrumented historical solver. The local angles.csv
    # may now contain newer output at FOV=.6, which is not this .8 reference.
    reference = sliding.stabilize_window(raw, replace(config, enablehorizon=True), window, camera)
    np.testing.assert_allclose(euler(runs['horizon_on']['output']), euler(reference.orientations), atol=1e-8)

    np.savetxt(directory / 'horizon_diagnosis.csv', np.column_stack(csv_columns), delimiter=',',
               header=','.join(csv_names), comments='', fmt='%.12g')
    (directory / 'horizon_diagnosis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(directory / 'horizon_stages.npz', raw=raw, time_s=times,
                        **{f'{name}_{stage}': value for name, run in runs.items() for stage, value in run.items()})

    font_file = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font_file))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font_file).get_name(),
                         'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(3, 3, figsize=(17.5, 10), sharex=True, sharey='row')
    fig.patch.set_facecolor('#f8fafc')
    titles = ['① 原有窗口平滑：horizon 关闭\nFOV 下限 0.80',
              '② 修复前（288a63a）：horizon 开启\nFOV 下限 0.80（原问题图）',
              '③ 诊断候选：平滑 + 水平矫正\n关闭裁剪反馈和硬投影（非交付结果）']
    for col, (name, run) in enumerate(runs.items()):
        angles = euler(run['output'])
        for row, axis_name in enumerate(['Pitch / 俯仰', 'Yaw / 偏航', 'Roll / 横滚']):
            axis = axes[row, col]
            axis.plot(times, raw_angles[:, row], color='#8095b5', linewidth=1.0, label='原始角度')
            axis.plot(times, angles[:, row], color='#df5b32', linewidth=1.65, label='处理后角度')
            axis.grid(True, color='#e3e8ef', linewidth=0.65)
            axis.spines[['top', 'right']].set_visible(False)
            axis.margins(x=0, y=0.12)
            if row == 0:
                axis.set_title(titles[col], fontsize=12, loc='left', pad=16, color='#25334b', fontweight='bold')
                axis.legend(loc='upper right', fontsize=9)
            if col == 0:
                axis.set_ylabel(f'{axis_name}（°）')
            if row == 2:
                axis.set_ylim(-21, 3)
                axis.set_xlabel('时间（秒）')
                rms = summary[name]['stages']['output']['roll_rms_deg']
                axis.text(0.03, 0.07, f'Roll RMS = {rms:.3f}°', transform=axis.transAxes,
                          fontsize=11, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})
    fig.suptitle('同一段真实视频：水平矫正被裁剪约束抵消了多少？', x=0.065, y=0.98,
                 ha='left', fontsize=20, fontweight='bold', color='#17253c')
    fig.text(0.065, 0.927, '视频 052  ·  同一份原始四元数  ·  29.97 fps  ·  H/E/F = 60/6/15  ·  仅第③列取消裁剪约束', color='#52617a')
    needed = summary['unconstrained_diagnostic']['stages']['output']['minimum_geometric_fov']
    fig.text(0.065, 0.026,
             f'第③列的 pitch/yaw 保留明显平滑，roll 接近 0；但该候选最少只能保留 {needed:.3f} 的成像平面尺度。\n'
             '旧代码优先保证 FOV ≥ 0.80：共用反馈削弱三个轴的平滑，最终整姿态投影又把角度拉回原始值。',
             color='#52617a', fontsize=11, linespacing=1.8)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.13, wspace=0.1, hspace=0.2)
    fig.savefig(directory / 'horizon_diagnosis.png', dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    for name, values in summary.items():
        if name != 'isolated_horizon':
            print(name, json.dumps({k: v for k, v in values.items() if k != 'stages'}))
            for stage, metrics in values['stages'].items():
                print(' ', stage, json.dumps(metrics))
        else:
            print(name, json.dumps(values))


if __name__ == '__main__':
    main()
