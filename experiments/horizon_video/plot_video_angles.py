"""Plot real-video camera angles before and after the horizon window stabilizer.

Run from the repository root with:
    python experiments/horizon_video/plot_video_angles.py

The Gyroflow executable only supplies the original camera orientations. All
smoothed orientations come from this repository's current sliding-window code.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gyroflow_window.geometry import PinholeCamera, coverage_margin
from gyroflow_window.gyroflow_global import SmoothingConfig
from gyroflow_window.sliding_window import WindowConfig, stabilize_window


def export_original_poses(video: Path, destination: Path) -> None:
    fields = {
        "export_all_samples": False,
        "original": {"quaternion": True, "euler_angles": True},
        "stabilized": {"quaternion": False, "euler_angles": False},
        "zooming": {"fovs": False, "minimal_fovs": False, "focal_length": False},
    }
    executable = ROOT / "dataset/tools/gyroflow/Gyroflow.exe"
    command = [str(executable), str(video), "--export-metadata",
               f"3:{destination}", "--export-metadata-fields", json.dumps(fields)]
    print(f"Exporting original camera orientations from {video.name}", flush=True)
    process = subprocess.run(
        command, cwd=executable.parent, capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=120,
    )
    destination.with_suffix(".log").write_bytes(process.stdout + process.stderr)
    if process.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"Camera export failed; see {destination.with_suffix('.log')}")


def camera_euler(rotation: Rotation) -> np.ndarray:
    # R = Ry(yaw) Rx(pitch) Rz(roll); local Z is the camera optical axis.
    angles = rotation.as_euler("YXZ", degrees=False)[:, [1, 0, 2]]
    return np.rad2deg(np.unwrap(angles, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / (
        "dataset/insta360/20260826_112713/PRO_VID_20260826_112713_00_046.mp4"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "046")
    parser.add_argument("--minimum-fov", type=float, default=0.6)
    args = parser.parse_args()
    video, output = args.video.resolve(), args.output.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    output.mkdir(parents=True, exist_ok=True)
    original_file = output / "original_camera.json"
    export_original_poses(video, original_file)
    frames = json.loads(original_file.read_text(encoding="utf-8-sig"))
    timestamps = np.array([frame["timestamp_ms"] for frame in frames])
    frame_indices = np.array([frame["frame"] for frame in frames])
    original = np.array([frame["org_quat"] for frame in frames])
    assert len(frames) > 1 and np.array_equal(frame_indices, np.arange(len(frames)))
    assert np.isfinite(original).all() and np.all(np.diff(timestamps) > 0)
    input_norm_error = float(np.max(np.abs(np.linalg.norm(original, axis=1) - 1)))
    assert input_norm_error < 1e-6  # SciPy normalizes the small export roundoff.
    spacing_ms = float(np.median(np.diff(timestamps)))
    assert np.max(np.abs(np.diff(timestamps) - spacing_ms)) < 1e-6
    fps = 1000.0 / spacing_ms
    time = (timestamps - timestamps[0]) / 1000.0

    # Gyroflow's integrated orientation uses world +Z as vertical. One fixed
    # world-basis rotation maps that direction to the package's world +Y.
    # Camera-local axes and all relative camera rotations remain unchanged.
    world_change = Rotation.from_euler("x", -90, degrees=True)
    raw_rotation = world_change * Rotation.from_quat(original[:, [1, 2, 3, 0]])
    raw = raw_rotation.as_quat()[:, [3, 0, 1, 2]]
    config = SmoothingConfig(fps=fps, enablehorizon=True, minimum_fov=args.minimum_fov)
    window = WindowConfig(history=60, emit=6, future=15)
    camera = PinholeCamera(diagonal_fov_deg=100.0, aspect=16 / 9)
    result = stabilize_window(raw, config, window, camera)
    smooth_rotation = Rotation.from_quat(result.orientations[:, [1, 2, 3, 0]])
    raw_angles, smooth_angles = camera_euler(raw_rotation), camera_euler(smooth_rotation)
    # Align only equivalent 360-degree branches; do not independently zero curves.
    smooth_angles += 360 * np.round((raw_angles[0] - smooth_angles[0]) / 360)

    reconstruction_error = 0.0
    roll_measurement_error = 0.0
    for rotation, angles in [(raw_rotation, raw_angles), (smooth_rotation, smooth_angles)]:
        rebuilt = Rotation.from_euler("YXZ", angles[:, [1, 0, 2]], degrees=True)
        error = float(np.max((rebuilt.inv() * rotation).magnitude()))
        reconstruction_error = max(reconstruction_error, error)
        assert error < 1e-10
        vertical = rotation.inv().apply(np.tile([0., 1., 0.], (len(frames), 1)))
        horizon_angle = np.rad2deg(np.arctan2(vertical[:, 0], vertical[:, 1]))
        difference = (angles[:, 2] - horizon_angle + 180) % 360 - 180
        roll_measurement_error = max(roll_measurement_error, float(np.max(np.abs(difference))))
    assert roll_measurement_error < 1e-10
    margin = coverage_margin(raw, result.orientations, result.fov, camera)
    assert np.isfinite(smooth_angles).all()
    assert np.min(result.fov) >= config.minimum_fov - 1e-9
    assert np.min(margin) >= -1e-8

    csv_data = np.column_stack([
        frame_indices, timestamps, time, raw_angles, smooth_angles, raw,
        result.orientations, result.fov, result.geometric_fov,
    ])
    csv_header = ("frame,timestamp_ms,time_s,raw_pitch_deg,raw_yaw_deg,raw_roll_deg,"
                  "smoothed_pitch_deg,smoothed_yaw_deg,smoothed_roll_deg,"
                  "raw_qw,raw_qx,raw_qy,raw_qz,smoothed_qw,smoothed_qx,smoothed_qy,"
                  "smoothed_qz,fov,geometric_fov")
    np.savetxt(output / "angles.csv", csv_data, delimiter=",", header=csv_header,
               comments="", fmt="%.12g")
    np.savez_compressed(output / 'orientations.npz', raw=raw, orientations=result.orientations,
                        fov=result.fov, geometric_fov=result.geometric_fov)
    roll_rms = np.sqrt(np.mean(np.stack([raw_angles[:, 2], smooth_angles[:, 2]]) ** 2, axis=1))
    with video.open("rb") as video_file:
        video_digest = hashlib.file_digest(video_file, "sha256").hexdigest()
    metadata = {
        "video": video.relative_to(ROOT).as_posix(),
        "video_sha256": video_digest,
        "original_pose_source": "Gyroflow original camera export from source MP4 IMU",
        "smoothed_pose_source": "gyroflow_window.sliding_window.stabilize_window",
        "frame_count": len(frames), "fps": fps, "plotted_duration_s": float(time[-1]),
        "world_basis_change": "Q = Rx(-90 degrees) * Q_gyroflow; maps world +Z to +Y",
        "euler_convention": "intrinsic YXZ: Q = Ry(yaw) Rx(pitch) Rz(roll); degrees",
        "curve_processing": "360-degree unwrap only; no extra filtering or independent zeroing",
        "smoothing_config": asdict(config), "window_config": asdict(window),
        "camera_model": {**asdict(camera), "note": "package pinhole model; not calibrated lens rendering"},
        "roll_rms_deg": {"raw": float(roll_rms[0]), "smoothed": float(roll_rms[1])},
        "pitch_yaw_second_difference_rms_deg": {
            "raw": np.sqrt(np.mean(np.diff(raw_angles[:, :2], n=2, axis=0)**2, axis=0)).tolist(),
            "smoothed": np.sqrt(np.mean(np.diff(smooth_angles[:, :2], n=2, axis=0)**2, axis=0)).tolist(),
        },
        "feedback": {"enabled": True, "iterations": result.iterations,
                     "minimum": min(d['feedback_min'] for d in result.diagnostics['windows'])},
        "projected_frames": result.diagnostics['projected_frames'],
        "validation": {
            "max_exported_quaternion_norm_error": input_norm_error,
            "max_euler_reconstruction_error_rad": reconstruction_error,
            "max_euler_roll_vs_projected_vertical_error_deg": roll_measurement_error,
            "minimum_output_fov": float(np.min(result.fov)),
            "minimum_coverage_margin": float(np.min(margin)),
            "maximum_absolute_roll_deg": float(np.max(np.abs(smooth_angles[:, 2]))),
            "fov_floor_violation_frames": int(np.sum(result.fov < config.minimum_fov-1e-9)),
            "coverage_violation_frames": int(np.sum(margin < -1e-8)),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    chinese_font = Path("C:/Windows/Fonts/msyh.ttc")
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=chinese_font).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "svg.fonttype": "path"})
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    titles = ["Pitch / 俯仰角（X）", "Yaw / 偏航角（Y）", "Roll / 横滚角（光轴 Z）"]
    for index, (axis, title) in enumerate(zip(axes, titles)):
        axis.plot(time, raw_angles[:, index], color="#8095b5", linewidth=1.25,
                  label="原始角度", alpha=0.95)
        axis.plot(time, smooth_angles[:, index], color="#df5b32", linewidth=2.0,
                  label="平滑后角度（水平矫正开启）")
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#25334b", pad=9)
        axis.set_ylabel("角度（°）")
        axis.grid(True, color="#e5eaf0", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#cdd5df")
        axis.legend(loc="best", frameon=True, facecolor="white", edgecolor="#e5eaf0", fontsize=10)
        axis.margins(x=0, y=0.15)
    axes[2].set_ylim(min(raw_angles[:, 2].min(), smooth_angles[:, 2].min(), 0) - 1.5,
                     max(raw_angles[:, 2].max(), smooth_angles[:, 2].max(), 0) + 1.5)
    axes[-1].set_xlabel("时间（秒）")
    fig.suptitle("真实视频姿态：原始角度与平滑后角度", x=0.08, y=0.975,
                 ha="left", fontsize=19, fontweight="bold", color="#17253c")
    fig.text(0.08, 0.929, f"{video.name}  |  {len(frames)} 帧  |  {fps:.2f} fps",
             fontsize=11, color="#52617a")
    fig.text(0.08, 0.018,
             f"窗口 H/E/F = 60/6/15  ·  FOV 下限 = {config.minimum_fov:.2f}（针孔模型）  ·  "
             f"Roll RMS：{roll_rms[0]:.2f}° → {roll_rms[1]:.2e}°\n"
             f"feedback 保留：{result.iterations} 次反馈迭代  ·  "
             "YXZ 内禀欧拉角；共同竖直参考 +Y；曲线未分别归零。",
             fontsize=10, color="#52617a", linespacing=1.8)
    fig.subplots_adjust(left=0.08, right=0.975, top=0.87, bottom=0.11, hspace=0.38)
    fig.savefig(output / "pitch_yaw_roll.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(output / "pitch_yaw_roll.svg", facecolor=fig.get_facecolor())
    # A fixed four-second detail makes yaw shake visible beside a large pan.
    zoom_start = float(time[-1] / 4)
    zoom_end = min(float(time[-1]), zoom_start + 4.0)
    visible = (time >= zoom_start) & (time <= zoom_end)
    if np.sum(visible) >= 3:
        for index, axis in enumerate(axes):
            axis.set_xlim(zoom_start, zoom_end)
            values = np.concatenate([raw_angles[visible, index], smooth_angles[visible, index]])
            pad = max(float(np.ptp(values)) * 0.12, 0.5)
            axis.set_ylim(float(values.min() - pad), float(values.max() + pad))
        fig.suptitle(f"真实视频姿态：局部放大 {zoom_start:.1f}–{zoom_end:.1f} 秒",
                     x=0.08, y=0.975, ha="left", fontsize=19, fontweight="bold", color="#17253c")
        fig.savefig(output / "pitch_yaw_roll_zoom.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({"figure": str(output / "pitch_yaw_roll.png"),
                      "frames": len(frames), "roll_rms_deg": metadata["roll_rms_deg"],
                      "validation": metadata["validation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
