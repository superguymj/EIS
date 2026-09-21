"""Seeded, reproducible accuracy / continuity / FOV / runtime experiments.

Example:
python -m gyroflow_window.benchmark --output experiments/gyroflow_window/round1 \
    --motions jitter,pan,impulse,multi_axis --futures 6,15 --variants naive_safe,anchored,predictive,stable_scale
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .math3d import angular_distance, world_velocity, from_rotvec
from .geometry import PinholeCamera, geometric_fov, coverage_margin, project_to_fov
from .gyroflow_global import SmoothingConfig, StabilizationResult, stabilize_global, zoom_curve
from .sliding_window import WindowConfig, stabilize_window
from .synthetic import make_motion, MOTION_KINDS


def metric_row(result, global_result, motion, config, camera, emit):
    q = result.orientations
    errors = np.rad2deg(angular_distance(q, global_result.orientations))
    feasible_reference = global_result.diagnostics.get('feasible_reference', global_result.orientations)
    feasible_errors = np.rad2deg(angular_distance(q, feasible_reference))
    truth_errors = np.rad2deg(angular_distance(q, motion.truth))
    accel = np.diff(world_velocity(q, config.fps), axis=0)*config.fps
    accel_norm = np.rad2deg(np.linalg.norm(accel, axis=1))
    # Acceleration indexed by its third frame; include first two frames of each block.
    indices = np.arange(2, len(q))
    seam = (indices % emit <= 1) & (indices >= emit)
    interior = errors[min(len(errors)//4, int(config.fps)):max(min(len(errors)//4, int(config.fps))+1, len(errors)-int(config.fps))]
    geometry = geometric_fov(motion.raw, q, camera)
    coverage = coverage_margin(motion.raw, q, result.fov, camera)
    diagnostics = result.diagnostics
    windows = diagnostics.get('windows', [])
    timings = np.array([w['milliseconds'] for w in windows if not w['eof']])
    return {
        'global_rmse_deg': float(np.sqrt(np.mean(errors**2))),
        'feasible_global_rmse_deg': float(np.sqrt(np.mean(feasible_errors**2))),
        'global_p95_deg': float(np.percentile(errors, 95)),
        'global_max_deg': float(np.max(errors)),
        'global_interior_rmse_deg': float(np.sqrt(np.mean(interior**2))),
        'truth_rmse_deg': float(np.sqrt(np.mean(truth_errors**2))),
        'accel_rms_deg_s2': float(np.sqrt(np.mean(accel_norm**2))) if len(accel_norm) else 0,
        'seam_accel_rms_deg_s2': float(np.sqrt(np.mean(accel_norm[seam]**2))) if seam.any() else 0,
        'nonseam_accel_rms_deg_s2': float(np.sqrt(np.mean(accel_norm[~seam]**2))) if (~seam).any() else 0,
        'pose_fov_violations': int(np.sum(geometry < config.minimum_fov-1e-8)),
        'zoom_floor_violations': int(np.sum(result.fov < config.minimum_fov-1e-8)),
        'floor_max_shortfall': float(np.max(np.maximum(config.minimum_fov-result.fov, 0))),
        'floor_mean_shortfall': float(np.mean(np.maximum(config.minimum_fov-result.fov, 0))),
        'coverage_violations': int(np.sum(coverage < -1e-8)),
        'minimum_geometric_fov': float(np.min(geometry)),
        'minimum_output_fov': float(np.min(result.fov)),
        'fov_rmse': float(np.sqrt(np.mean((result.fov-global_result.fov)**2))),
        'projected_frames': int(diagnostics.get('projected_frames', 0)),
        'window_p95_ms': float(np.percentile(timings, 95)) if len(timings) else 0,
        'window_max_ms': float(np.max(timings)) if len(timings) else 0,
        'max_buffered_frames': int(diagnostics.get('max_buffered_frames', 0)),
    }


def variants(args, future):
    base = WindowConfig(history=args.history, emit=args.emit, future=future)
    choices = {
        'naive': replace(base, boundary_weight=0, prediction_seconds=0, hard_fov=False),
        'naive_safe': replace(base, boundary_weight=0, prediction_seconds=0),
        'anchored': replace(base, boundary_weight=64, prediction_seconds=0),
        'predictive': replace(base, boundary_weight=64, prediction_seconds=0.5),
        'stable_scale': replace(base, boundary_weight=64, prediction_seconds=0.5, stable_distance_scale=True),
        'tuned': replace(base, boundary_weight=1, prediction_seconds=1.5, prediction_endpoint_fit=False),
        'denoised': replace(base, boundary_weight=1, prediction_seconds=1.5, prediction_endpoint_fit=True),
        'final': base,
    }
    if args.weights is not None:
        if args.decays is not None:
            return {f'p{p:g}_b{w:g}_t{d:g}': replace(base, prediction_seconds=p, boundary_weight=w, prediction_decay_seconds=d)
                    for p, w, d in itertools.product(args.predictions, args.weights, args.decays)}
        return {f'p{p:g}_b{w:g}': replace(base, prediction_seconds=p, boundary_weight=w)
                for p, w in itertools.product(args.predictions, args.weights)}
    return {name: choices[name] for name in args.variants}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summaries(rows):
    output = []
    for name in sorted({r['algorithm'] for r in rows}):
        selected = [r for r in rows if r['algorithm'] == name]
        row = {'algorithm': name, 'runs': len(selected), 'frames': sum(r['frames'] for r in selected)}
        for field in ['global_rmse_deg', 'feasible_global_rmse_deg', 'global_p95_deg', 'truth_rmse_deg', 'accel_rms_deg_s2',
                      'seam_accel_rms_deg_s2', 'nonseam_accel_rms_deg_s2', 'fov_rmse', 'elapsed_seconds', 'window_p95_ms', 'floor_mean_shortfall']:
            row[field] = float(np.mean([r[field] for r in selected]))
        for field in ['pose_fov_violations', 'zoom_floor_violations', 'coverage_violations', 'projected_frames']:
            row[field] = int(sum(r[field] for r in selected))
        row['worst_global_rmse_deg'] = max(r['global_rmse_deg'] for r in selected)
        row['worst_floor_shortfall'] = max(r['floor_max_shortfall'] for r in selected)
        output.append(row)
    return output


def make_plots(output, rows, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [r for r in summary if r['algorithm'] not in ('raw', 'global', 'global_safe')]
    if not selected:
        return
    names = [r['algorithm'] for r in selected]
    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(names)*1.3), 4.5))
    for ax, field, title in zip(axes, ['global_rmse_deg', 'truth_rmse_deg', 'seam_accel_rms_deg_s2'],
                               ['Angular RMSE to global (deg)', 'Angular RMSE to synthetic intent (deg)', 'Boundary acceleration RMS (deg/s^2)']):
        ax.bar(names, [r[field] for r in selected])
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output/'summary.png', dpi=150)
    plt.close(fig)
    futures = sorted({r['future'] for r in rows if r['future'] >= 0})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in names:
        subset = [r for r in rows if r['algorithm'] == name]
        values = [np.mean([r['global_rmse_deg'] for r in subset if r['future'] == f]) for f in futures]
        axes[0].plot(futures, values, 'o-', label=name)
        for floor in sorted({r['minimum_fov'] for r in subset}):
            values = [np.mean([r['global_rmse_deg'] for r in subset if r['future'] == f and r['minimum_fov'] == floor]) for f in futures]
            if name == names[-1]:
                axes[1].plot(futures, values, 'o-', label=f'{name}, floor={floor:g}')
    axes[0].set_title('Lookahead / fidelity tradeoff')
    axes[1].set_title('Crop constraint / fidelity tradeoff')
    for ax in axes:
        ax.set_xlabel('Observed future frames F')
        ax.set_ylabel('Angular RMSE to global (deg)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'lookahead.png', dpi=150)
    plt.close(fig)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_directory = output/'source_snapshot'
    snapshot_directory.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, snapshot_directory/source.name)
    if args.save_trajectories:
        (output/'trajectories').mkdir(exist_ok=True)
    # Compilation is outside measured algorithm runtime; record its elapsed time.
    started = perf_counter()
    warm = make_motion('jitter', 1, fps=args.fps, seconds=2)
    stabilize_global(warm.raw)
    stabilize_window(warm.raw)
    warmup_seconds = perf_counter()-started
    metadata = vars(args).copy()
    metadata.update(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                    source_commit='2331ebb3c253fc7cf357642518b0c8bf2937f815', warmup_seconds=warmup_seconds,
                    geometry='centered undistorted pinhole; same input/output aspect and dimensions',
                    quaternion_convention='wxyz; camera-to-world; one sample per video frame')
    metadata['code_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    metadata['window_variants'] = {name: asdict(window) for name, window in variants(args, args.futures[0]).items()}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    rows = []
    cases = list(itertools.product(args.motions, args.seeds, args.lenses, args.floors))
    start_all = perf_counter()
    for case_index, (kind, seed, diagonal, floor) in enumerate(cases):
        motion = make_motion(kind, seed, args.fps, args.seconds)
        config = SmoothingConfig(fps=args.fps, minimum_fov=floor, max_iterations=args.iterations)
        camera = PinholeCamera(diagonal)
        case_name = f'{kind}_s{seed}_d{diagonal:g}_c{floor:g}'
        base = {'case': case_name, 'motion': kind, 'seed': seed, 'diagonal_fov': diagonal,
                'minimum_fov': floor, 'fps': args.fps, 'frames': len(motion.raw),
                'history': args.history, 'emit': args.emit}
        started = perf_counter()
        reference = stabilize_global(motion.raw, config, camera)
        elapsed_global = perf_counter()-started
        feasible_q, _ = project_to_fov(motion.raw, reference.orientations, floor, camera)
        reference.diagnostics['feasible_reference'] = feasible_q
        feasible_geometry = geometric_fov(motion.raw, feasible_q, camera)
        feasible_result = StabilizationResult(feasible_q, zoom_curve(feasible_geometry, config), feasible_geometry, reference.iterations, {})
        identity = np.ones(len(motion.raw))
        raw_result = StabilizationResult(motion.raw, identity, identity, 0, {})
        for name, result, elapsed in [('raw', raw_result, 0), ('global', reference, elapsed_global), ('global_safe', feasible_result, elapsed_global)]:
            row = dict(base, algorithm=name, future=-1, elapsed_seconds=elapsed)
            row.update(metric_row(result, reference, motion, config, camera, args.emit))
            rows.append(row)
        arrays = {'raw': motion.raw, 'truth': motion.truth, 'global_q': reference.orientations,
                  'global_fov': reference.fov, 'time': motion.time}
        for future in args.futures:
            for name, window in variants(args, future).items():
                started = perf_counter()
                result = stabilize_window(motion.raw, config, window, camera)
                elapsed = perf_counter()-started
                row = dict(base, algorithm=name, future=future, elapsed_seconds=elapsed)
                row.update(metric_row(result, reference, motion, config, camera, args.emit))
                rows.append(row)
                if args.save_trajectories:
                    arrays[f'{name}_F{future}_q'] = result.orientations
                    arrays[f'{name}_F{future}_fov'] = result.fov
        if args.save_trajectories:
            np.savez_compressed(output/'trajectories'/f'{case_name}.npz', **arrays)
        # Persist progress after every complete case, so long runs are inspectable.
        write_csv(output/'metrics.csv', rows)
        print(f'{case_index+1}/{len(cases)} {case_name} elapsed={perf_counter()-start_all:.1f}s', flush=True)
    summary = summaries(rows)
    write_csv(output/'summary.csv', summary)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not args.no_plots:
        make_plots(output, rows, summary)
    for row in summary:
        print(f"{row['algorithm']:18s} global={row['global_rmse_deg']:.4f} deg truth={row['truth_rmse_deg']:.4f} deg "
              f"seam={row['seam_accel_rms_deg_s2']:.2f} floor_bad={row['zoom_floor_violations']} coverage_bad={row['coverage_violations']}")


def csv_values(text, converter):
    return [converter(v) for v in text.split(',')]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--motions', type=lambda x: csv_values(x, str), default=['jitter','pan','stop_start','impulse','multi_axis'])
    p.add_argument('--seeds', type=lambda x: csv_values(x, int), default=[11])
    p.add_argument('--lenses', type=lambda x: csv_values(x, float), default=[70,120])
    p.add_argument('--floors', type=lambda x: csv_values(x, float), default=[0.6,0.9,0.97])
    p.add_argument('--futures', type=lambda x: csv_values(x, int), default=[6,15,30])
    p.add_argument('--variants', type=lambda x: csv_values(x, str), default=['naive_safe','anchored','predictive','stable_scale'])
    p.add_argument('--weights', type=lambda x: csv_values(x, float))
    p.add_argument('--predictions', type=lambda x: csv_values(x, float), default=[0,0.25,0.5,1.0])
    p.add_argument('--decays', type=lambda x: csv_values(x, float))
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--history', type=int, default=60)
    p.add_argument('--emit', type=int, default=6)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--save-trajectories', action='store_true')
    p.add_argument('--no-plots', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
