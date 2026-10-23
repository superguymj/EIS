"""Reproduce the old/new comparison with the same data and FOV=.6.

The reference code is read from commit 288a63a, not reimplemented here.
The later FOV history fit is disabled to isolate the historical horizon change.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import types

from plot_video_angles import ROOT, camera_euler
import numpy as np
from scipy.spatial.transform import Rotation
from gyroflow_window.geometry import PinholeCamera, coverage_margin
from gyroflow_window.gyroflow_global import SmoothingConfig
from gyroflow_window.sliding_window import WindowConfig, stabilize_window


def load_reference():
    modules = []
    for stem in ['gyroflow_global', 'sliding_window']:
        source = subprocess.run(
            ['git', 'show', f'288a63a:gyroflow_window/{stem}.py'], cwd=ROOT,
            capture_output=True, text=True, check=True).stdout
        name = f'gyroflow_window._reference_{stem}'
        module = types.ModuleType(name)
        module.__package__ = 'gyroflow_window'
        sys.modules[name] = module
        exec(compile(source, f'288a63a:{stem}.py', 'exec'), module.__dict__)
        modules.append(module)
    modules[1].gyroflow_smooth = modules[0].gyroflow_smooth
    return modules


def main():
    reference_core, reference_window = load_reference()
    report = {'reference_commit': '288a63a', 'minimum_fov': 0.6, 'videos': {}}
    for case in ['046', '052_feedback06']:
        directory = ROOT / 'experiments/horizon_video' / case
        if case == '052_feedback06' and not directory.is_dir():
            directory = ROOT / 'experiments/horizon_video/052'
        with np.load(directory / 'orientations.npz') as data:
            raw = data['raw'].copy()
            saved_output = data['orientations'].copy()
        metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        config = SmoothingConfig(**metadata['smoothing_config'])
        window = replace(WindowConfig(**metadata['window_config']), fov_boundary_weight=0)
        camera = PinholeCamera()
        old = reference_window.stabilize_window(raw, config, window, camera)
        current = stabilize_window(raw, config, window, camera)
        np.testing.assert_array_equal(saved_output, current.orientations)
        baseline = stabilize_window(raw, replace(config, enablehorizon=False), window, camera)
        old_baseline = reference_window.stabilize_window(raw, replace(config, enablehorizon=False), window, camera)
        assert np.array_equal(baseline.orientations, old_baseline.orientations)
        assert np.array_equal(baseline.fov, old_baseline.fov)
        # The old and new core receive the same varying feedback in this check.
        coefficients = np.linspace(0.1, 1, len(raw))
        current_core = reference_core.gyroflow_smooth(raw, replace(config, enablehorizon=False),
                                                    camera.diagonal_fov_deg, coefficients)[0]
        from gyroflow_window.gyroflow_global import gyroflow_smooth
        checked_core = gyroflow_smooth(raw, replace(config, enablehorizon=False),
                                       camera.diagonal_fov_deg, coefficients)[0]
        assert np.array_equal(current_core, checked_core)
        results = {}
        for name, result in [('raw', None), ('horizon_off', baseline), ('before_fix', old), ('after_fix', current)]:
            q = raw if result is None else result.orientations
            angles = camera_euler(Rotation.from_quat(q[:, [1, 2, 3, 0]]))
            results[name] = {
                'roll_rms_deg': float(np.sqrt(np.mean(angles[:, 2]**2))),
                'maximum_absolute_roll_deg': float(np.max(np.abs(angles[:, 2]))),
                'pitch_yaw_second_difference_rms_deg': np.sqrt(np.mean(np.diff(angles[:, :2], n=2, axis=0)**2, axis=0)).tolist(),
            }
            if result is not None:
                margin = coverage_margin(raw, q, result.fov, camera)
                results[name].update(
                    minimum_fov=float(np.min(result.fov)), iterations=result.iterations,
                    projected_frames=result.diagnostics['projected_frames'],
                    coverage_violation_frames=int(np.sum(margin < -1e-8)),
                    fov_floor_violation_frames=int(np.sum(result.fov < config.minimum_fov - 1e-9)),
                )
                assert results[name]['coverage_violation_frames'] == 0
                assert results[name]['fov_floor_violation_frames'] == 0
        report['videos'][case] = {
            'video': metadata['video'], 'frame_count': len(raw),
            'horizon_off_bitwise_compatible': True, 'results': results,
        }
    target = ROOT / 'experiments/horizon_video/comparison_feedback06.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
