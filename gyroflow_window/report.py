"""Rebuild the measured report and figures from saved, completed experiments.

Run from the workspace root: python -m gyroflow_window.report
This does not rerun or tune any stabilization algorithm.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from .math3d import angular_distance, world_velocity


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT/'experiments/gyroflow_window'
DOCS = ROOT/'docs'
ASSETS = DOCS/'assets/gyroflow-sliding-window'
FIELDS = ['global_rmse_deg', 'feasible_global_rmse_deg', 'truth_rmse_deg',
          'seam_accel_rms_deg_s2', 'nonseam_accel_rms_deg_s2', 'accel_rms_deg_s2',
          'fov_rmse', 'elapsed_seconds', 'window_p95_ms']


def read_rows(directory):
    with (EXPERIMENTS/directory/'metrics.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in row:
            if key not in ('case', 'motion', 'algorithm'):
                row[key] = float(row[key])
    return rows


def selected(rows, **filters):
    return [r for r in rows if all(r[k] == v for k, v in filters.items())]


def average(rows, field):
    return mean(r[field] for r in rows)


def table(headers, values):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
    lines.extend('| ' + ' | '.join(str(v) for v in row) + ' |' for row in values)
    return '\n'.join(lines)


def plot_summary(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    futures = sorted({r['future'] for r in rows if r['future'] >= 0})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, label in [('naive_safe', 'Direct windows + hard FOV'), ('final', 'Final sliding window')]:
        for ax, field in zip(axes, [FIELDS[0], FIELDS[1], FIELDS[3]]):
            values = [average(selected(rows, algorithm=name, future=f), field) for f in futures]
            ax.plot(futures, values, 'o-', label=label)
    titles = ['RMSE to original global (deg)', 'RMSE to feasible global (deg)', 'Seam acceleration RMS (deg/s^2)']
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Observed future frames F (30 fps)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS/'lookahead.png', dpi=150)
    plt.close(fig)

    floors = sorted({r['minimum_fov'] for r in rows})
    grids = [np.array([[average(selected(rows, algorithm='final', future=f, minimum_fov=c), field)
                       for f in futures] for c in floors]) for field in FIELDS[:2]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, grid, title in zip(axes, grids, ['Final vs original global', 'Final vs feasible global']):
        picture = ax.imshow(grid, vmin=0, vmax=max(g.max() for g in grids), cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(futures)), [str(int(f)) for f in futures])
        ax.set_yticks(range(len(floors)), [f'{c:g}' for c in floors])
        ax.set_xlabel('Observed future frames F')
        ax.set_ylabel('Minimum FOV coefficient')
        ax.set_title(title)
        for i in range(len(floors)):
            for j in range(len(futures)):
                color = 'white' if grid[i,j] > max(g.max() for g in grids)*0.6 else 'black'
                ax.text(j, i, f'{grid[i,j]:.3f}', ha='center', va='center', color=color)
    fig.colorbar(picture, ax=axes, label='Mean per-configuration angular RMSE (deg)', shrink=0.8)
    fig.savefig(ASSETS/'fov_matrix.png', dpi=150)
    plt.close(fig)


def plot_case(row, output_name):
    import matplotlib.pyplot as plt

    case, future = row['case'], int(row['future'])
    with np.load(EXPERIMENTS/'validation_final/trajectories'/f'{case}.npz') as data:
        t = data['time']
        global_q = data['global_q']
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for name, label in [('naive_safe', 'Direct windows'), ('final', 'Final')]:
            q = data[f'{name}_F{future}_q']
            error = np.rad2deg(angular_distance(q, global_q))
            axes[0].plot(t, error, label=label, linewidth=1.2)
            axes[1].plot(t, data[f'{name}_F{future}_fov'], label=label, linewidth=1.2)
            acceleration = np.diff(world_velocity(q, row['fps']), axis=0)*row['fps']
            axes[2].plot(t[2:], np.rad2deg(np.linalg.norm(acceleration, axis=1)), label=label, linewidth=0.9)
        axes[1].plot(t, data['global_fov'], '--', label='Original global', linewidth=1)
        axes[1].axhline(row['minimum_fov'], color='black', linestyle=':', label='FOV floor')
        axes[0].set_ylabel('Error to global (deg)')
        axes[1].set_ylabel('FOV coefficient')
        axes[2].set_ylabel('Acceleration (deg/s^2)')
        axes[2].set_xlabel('Time (s)')
        axes[0].set_title(f'{case}; F={future}, H={int(row["history"])}, E={int(row["emit"])}')
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ASSETS/output_name, dpi=150)
        plt.close(fig)


def main():
    rows = read_rows('validation_final')
    metadata = json.loads((EXPERIMENTS/'validation_final/metadata.json').read_text(encoding='utf-8'))
    expected_cases = len(metadata['motions'])*len(metadata['seeds'])*len(metadata['lenses'])*len(metadata['floors'])
    expected_rows = expected_cases*(3+len(metadata['futures'])*len(metadata['variants']))
    if len(rows) != expected_rows:
        raise RuntimeError('The final benchmark is incomplete; refusing to generate a partial report')
    final = selected(rows, algorithm='final')
    naive = selected(rows, algorithm='naive_safe')
    global_rows = selected(rows, algorithm='global')
    naive_by_key = {(r['case'], r['future']): r for r in naive}
    counterpart = lambda r: naive_by_key[(r['case'], r['future'])]
    ASSETS.mkdir(parents=True, exist_ok=True)
    plot_summary(rows)
    worst = max(final, key=lambda r: r[FIELDS[0]])
    regression = max(final, key=lambda r: r[FIELDS[0]]-counterpart(r)[FIELDS[0]])
    improvement = min(final, key=lambda r: r[FIELDS[0]]-counterpart(r)[FIELDS[0]])
    plot_case(worst, 'worst_absolute.png')
    plot_case(regression, 'worst_regression.png')
    plot_case(improvement, 'best_improvement.png')

    lines = ['# Gyroflow 滑动窗口：迭代开发与实测结果', '',
             '本文件由 `python -m gyroflow_window.report` 从已完成实验的 CSV 生成。'
             '算法细节、接口及边界条件见 [实现说明](gyroflow-sliding-window.md)。', '',
             '## 1. 结论与比较口径', '',
             f'最终版本在 {len(final)} 组窗口配置中，相比相同 H/E/F、同样强制 FOV 底线的直接分窗对照：', '']
    labels = ['相对原版 Global 的姿态 RMSE', '相对可行 Global 的姿态 RMSE', '相对合成干净轨迹的 RMSE',
              '接缝角加速度 RMS', '非接缝角加速度 RMS', '全段角加速度 RMS', '缩放曲线 RMSE']
    lines.append(table(['指标', '直接分窗', '最终窗口', '均值下降比例'],
                       [[label, f'{average(naive,field):.4f}', f'{average(final,field):.4f}',
                         f'{100*(1-average(final,field)/average(naive,field)):.2f}%']
                        for label, field in zip(labels, FIELDS)]))
    lines += ['', '前三项角度单位为度；角加速度单位为 °/s²；缩放为无量纲系数。'
              '“均值”先对每个配置算 RMSE/RMS，再对配置等权平均，不是把所有帧混在一起计算一次 RMSE。'
              '降低均值不意味着每个运动、每个 F 都改善，退化配置见后文。', '',
              '指标定义：姿态误差采用最短旋转角 `||Log(G_i⁻¹ S_i)||`，其中 Log 是 SO(3) 旋转向量，'
              '先算弧度再转成度。角速度采用世界坐标系增量 `ω_i = fps · Log(S_i S_(i-1)⁻¹)`，'
              '角加速度为 `a_i = fps · (ω_i - ω_(i-1))`。'
              '接缝统计取每个 E 帧输出块的前两帧，使二阶差分跨过提交边界，首块除外；非接缝统计其余位置。'
              '加速度既包含抖动，也包含真实有意运动，因此它是平顺性指标，不是单独的画质评分。', '',
              '## 2. 验证数据与基线', '',
              f'- 最终验证 seed={metadata["seeds"]}，运动类型 {len(metadata["motions"])} 种：'
              + '、'.join(metadata['motions']) + '。',
              f'- 镜头对角视场角 {metadata["lenses"]}°；FOV 底线 {metadata["floors"]}；前瞻 F={metadata["futures"]}。',
              f'- H={metadata["history"]}，E={metadata["emit"]}，{metadata["fps"]} fps，每段 {metadata["seconds"]:g} 秒。',
              f'- {expected_cases} 组运动/镜头/底线组合，乘以 {len(metadata["futures"])} 档 F 后，每个窗口算法 {len(final)} 组。'
              f'最终窗口累计检查 {int(sum(r["frames"] for r in final)):,} 个“配置-帧”。',
              '- 同一个运动随机种子的轨迹会被不同镜头、底线和 F 复用；这些配置不是独立拍摄的视频样本。',
              '- Global 是固定参数、不分轴的源码逻辑 Python 移植，不是编译运行完整 Rust 引擎；全部比较共享理想针孔几何。', '',
              '`global_safe` 是在 Global 输出后附加同样的逐帧 FOV 可行性投影，不是官方 Gyroflow 算法，'
              '也不是受约束全局最优解。`naive_safe` 关闭预测和边界修正，但保留同样的局部缩放迭代、'
              '硬 FOV 投影及缩放衔接。', '']
    baseline_table = []
    for name in ['raw', 'global', 'global_safe', 'naive_safe', 'final']:
        group = selected(rows, algorithm=name)
        baseline_table.append([name, len(group), f'{average(group,FIELDS[0]):.4f}',
                               f'{average(group,FIELDS[1]):.4f}', f'{average(group,FIELDS[2]):.4f}',
                               int(sum(r['zoom_floor_violations'] for r in group)),
                               int(sum(r['coverage_violations'] for r in group))])
    lines += [table(['算法', '配置数', 'Global RMSE/°', '可行 Global RMSE/°', '干净轨迹 RMSE/°', 'FOV 底线违约帧', '覆盖违约帧'], baseline_table), '',
              '全段基线不依赖 F，因此只计算一次，没有人为重复五份；其镜头/运动/底线权重与窗口表一致。', '',
              f'原版 Global 的最大底线缺口为 {max(r["floor_max_shortfall"] for r in global_rows):.5f}，'
              f'逐帧缺口均值为 {average(global_rows,"floor_mean_shortfall"):.5f}。'
              '底线违约指裁剪超过所允许的最大放大倍率，不等于画面露黑；两者单独统计。'
              '原版外层迭代是有限次软反馈，不能把它误当作严格底线保证。', '',
              '## 3. 六轮开发与被否定的方案', '',
              '不同轮次的运动、F 和镜头矩阵不同，只能在同一轮内比较，不应把各轮均值直接串成下降曲线。'
              'round1–4 使用 seed=11；随后 seed=101 的扩展诊断发现周期运动误预测和长 F 过度修正，'
              '因此该诊断不再算独立验证，round5–6 使用它继续开发。最终冻结后才使用 seed=509。', '']
    rounds = {
        'round1': ['naive_safe', 'anchored', 'predictive', 'stable_scale'],
        'round2': ['p0_b0', 'p1.5_b0', 'p1.5_b1'],
        'round3': ['p1.5_b1_t0.3', 'p1.5_b1_t0.6', 'p1.5_b1_t1.2', 'p1.5_b1_t3'],
        'round4': ['naive_safe', 'tuned', 'denoised'],
        'round5': ['naive_safe', 'denoised', 'adaptive06', 'adaptive09', 'adaptive12'],
        'round6': ['naive_safe', 'adaptive06', 'guard09', 'guard12', 'guarded06'],
    }
    development = []
    for directory, variants in rounds.items():
        source = read_rows(directory)
        for name in variants:
            group = selected(source, algorithm=name)
            if not group:
                raise RuntimeError(f'Missing development variant {directory}/{name}')
            development.append([directory, name, len(group), f'{average(group,FIELDS[0]):.4f}',
                                f'{average(group,FIELDS[2]):.4f}', f'{average(group,FIELDS[3]):.2f}'])
    lines += [table(['轮次', '方案', '配置数', 'Global RMSE/°', '干净轨迹 RMSE/°', '接缝 RMS/(°/s²)'], development), '',
              '- round1：强边界权重 64 虽压低接缝加速度，却明显偏离 Global；跨窗口最大偏差尺度没有稳定收益，未设为默认。',
              '- round2–3：降低边界权重、加长虚拟尾部；在有限网格上比较 0.3/0.6/1.2/3 秒预测衰减，保留 0.6 秒。',
              '- round4：用回归末端代替直接复制最后一帧，减少噪声向虚拟尾部传播。',
              '- 扩展诊断：原 0.3 秒趋势会把行走摆动误判成持续转动；长 F 下额外修正有时降低轨迹精度。原结果保留在 `validation`。',
              '- round5：比较 0.6/0.9/1.2 秒趋势拟合，并使边界权重随真实 F 衰减。更长拟合虽抑制摆动，也更容易滞后突然转向。',
              '- round6：比较长短两尺度可信度组合，最终保留 `guarded06`：0.6 秒短拟合、1.2 秒长期检查、自适应边界权重。', '',
              '最终参数：H=60、E=6、F=15；边界基础权重 1；虚拟尾部 1.5 秒；'
              '预测速度衰减 0.6 秒；末端修正最多 2°；角速度上限 180°/s；'
              '缩放最多反馈 5 次后再做硬投影。全部场景使用同一套规则，没有按运动类别分支。', '',
              '## 4. 真实前瞻 F 与精度/延迟', '']
    fov_futures = sorted({r['future'] for r in final})
    future_table = []
    for f in fov_futures:
        a, b = selected(naive, future=f), selected(final, future=f)
        future_table.append([int(f), f'{f/metadata["fps"]:.3f}–{(f+metadata["emit"]-1)/metadata["fps"]:.3f}',
                             f'{average(a,FIELDS[0]):.4f}', f'{average(b,FIELDS[0]):.4f}',
                             f'{average(b,FIELDS[1]):.4f}', f'{average(a,FIELDS[3]):.1f}', f'{average(b,FIELDS[3]):.1f}'])
    lines += [table(['F/帧', '批内等待范围/s', '直接分窗 RMSE/°', '最终 RMSE/°', '最终可行参考 RMSE/°', '直接接缝 RMS', '最终接缝 RMS'], future_table), '',
              '等待范围从批内最后一帧到最早一帧，不含计算耗时。F=0 仍可能因 E 帧组批产生等待。'
              '虚拟尾部只用已观测数据预测，不增加真实前瞻或等待时间。', '',
              '此矩阵中收益集中在较短前瞻；F=60 时最终方案的平均轨迹误差略高于直接分窗，'
              '但接缝角加速度仍较低。因此不能把增加预测/边界处理理解为所有 F 下都提高姿态精度。', '',
              '![前瞻与误差/接缝](assets/gyroflow-sliding-window/lookahead.png)', '',
              '## 5. FOV 底线与精度', '',
              '以下是最终窗口“相对原版 Global RMSE / 相对可行 Global RMSE”，单位为度。'
              '不能只看高底线时对原版 Global 的误差：该参考本身可能使用了不允许的裁剪量。', '']
    matrix = []
    for floor in sorted({r['minimum_fov'] for r in final}):
        values = []
        for f in fov_futures:
            group = selected(final, minimum_fov=floor, future=f)
            values.append(f'{average(group,FIELDS[0]):.4f} / {average(group,FIELDS[1]):.4f}')
        matrix.append([f'{floor:g}']+values)
    lines += [table(['FOV 底线']+[f'F={int(f)}' for f in fov_futures], matrix), '',
              '![FOV 与误差矩阵](assets/gyroflow-sliding-window/fov_matrix.png)', '',
              '0.60、0.85、0.97 的底线在同尺寸模型中分别对应最大约 1.667×、1.176×、1.031× 放大。'
              '高底线留给旋转补偿的余量很小，因此更多原始抖动必须保留。', '',
              '## 6. 不隐瞒退化：逐配置与最差案例', '']
    pair_rows = []
    for label, field in zip(labels[:6], FIELDS[:6]):
        differences = [r[field]-counterpart(r)[field] for r in final]
        pair_rows.append([label, sum(d < -1e-8 for d in differences),
                          sum(abs(d) <= 1e-8 for d in differences), sum(d > 1e-8 for d in differences)])
    lines += [table(['指标', '改善配置数', '相同配置数', '退化配置数'], pair_rows), '',
              '相同包含无抖动静止场景。判定容差为 1e-8，对所有配置等权计数；'
              '这些数量不是独立样本显著性检验，也不能用均值改善替代逐场景结论。', '',
              '按最终姿态误差排序的五个最差配置：', '']
    worst_rows = sorted(final, key=lambda r: r[FIELDS[0]], reverse=True)[:5]
    lines.append(table(['案例', 'F', '直接分窗/°', '最终/°', '最终可行参考/°', '最终逐帧最大误差/°'],
                       [[r['case'], int(r['future']), f'{counterpart(r)[FIELDS[0]]:.4f}',
                         f'{r[FIELDS[0]]:.4f}', f'{r[FIELDS[1]]:.4f}', f'{r["global_max_deg"]:.4f}'] for r in worst_rows]))
    lines += ['', f'最大绝对误差配置：`{worst["case"]}`，F={int(worst["future"])}。', '',
              '![最大绝对误差](assets/gyroflow-sliding-window/worst_absolute.png)', '',
              f'相对直接分窗退化最多：`{regression["case"]}`，F={int(regression["future"])}；'
              f'RMSE 从 {counterpart(regression)[FIELDS[0]]:.4f}° 变为 {regression[FIELDS[0]]:.4f}°。', '',
              '![最明显退化](assets/gyroflow-sliding-window/worst_regression.png)', '',
              f'相对直接分窗改善最多：`{improvement["case"]}`，F={int(improvement["future"])}；'
              f'RMSE 从 {counterpart(improvement)[FIELDS[0]]:.4f}° 变为 {improvement[FIELDS[0]]:.4f}°。', '',
              '![最明显改善](assets/gyroflow-sliding-window/best_improvement.png)', '',
              '预测不能知道观测区外的突然转向和停止；F 很小时，趋势延续有可能预测错误。'
              '此外，固定历史、局部归一化和最终几何投影都会使窗口结果不同于离线 Global。'
              '硬投影能确保模型可行性，但不保证角速度、角加速度或 jerk 的硬上限。', '',
              '## 7. 性能、缓冲与额外设置验证', '',
              f'平台：{metadata["platform"]}；Python {metadata["python"]}；NumPy {metadata["numpy"]}。'
              f'首轮预热记录 {metadata["warmup_seconds"]:.3f} 秒，未计入以下算法运行耗时。', '']
    runtime = []
    for name in ['naive_safe', 'final']:
        group = selected(rows, algorithm=name)
        runtime.append([name, f'{average(group,"elapsed_seconds"):.3f}',
                        f'{average(group,"window_p95_ms"):.3f}', f'{max(r["window_max_ms"] for r in group):.3f}',
                        int(max(r['max_buffered_frames'] for r in group))])
    lines += [table(['算法', '每 12 秒片段平均计算/s', '各配置窗口 P95 均值/ms', '实测最慢窗口/ms', '真实帧缓冲峰值'], runtime), '',
              '这是桌面预热后的 Python/Numba 实测，不是手机耗时承诺；P95 一列是各配置 P95 的均值，不是所有窗口的联合 P95。'
              '126 帧对应 H=60、E=6、F=60 的最大配置；默认 F=15 时上限为 81 帧。'
              '额外内存还包括虚拟尾部、差分矩阵、临时数组、调用者输入批次及输出。', '']
    extra_rows = []
    for directory in ['robustness_final_60fps', 'robustness_final_short_h']:
        extra = read_rows(directory)
        extra_metadata = json.loads((EXPERIMENTS/directory/'metadata.json').read_text(encoding='utf-8'))
        for name in ['naive_safe', 'final']:
            group = selected(extra, algorithm=name)
            extra_rows.append([directory, name,
                               f'{extra_metadata["fps"]:g}/{extra_metadata["history"]}/{extra_metadata["emit"]}',
                               len(group), f'{average(group,FIELDS[0]):.4f}', f'{average(group,FIELDS[3]):.1f}',
                               int(sum(r['zoom_floor_violations'] for r in group)),
                               int(sum(r['coverage_violations'] for r in group))])
    lines += [table(['补充实验', '算法', 'fps/H/E', '配置数', 'Global RMSE/°', '接缝 RMS', '底线违约', '覆盖违约'], extra_rows), '',
              '额外实验同样使用冻结参数；其 F、镜头、种子和运动定义以对应 metadata.json 为准。'
              '不同 fps、H、E 的矩阵不能直接用均值比较优劣。', '',
              '## 8. 复现、原始证据与适用边界', '',
              '- [最终逐配置 CSV](../experiments/gyroflow_window/validation_final/metrics.csv)',
              '- [最终参数、代码 SHA256 与环境](../experiments/gyroflow_window/validation_final/metadata.json)',
              '- [最终源码快照](../experiments/gyroflow_window/validation_final/source_snapshot/)',
              '- [最终逐帧输入/输出轨迹 NPZ](../experiments/gyroflow_window/validation_final/trajectories/)',
              '- [60 fps 补充实验](../experiments/gyroflow_window/robustness_final_60fps/metrics.csv)',
              '- [短 H/大 E 补充实验](../experiments/gyroflow_window/robustness_final_short_h/metrics.csv)', '',
              '```powershell\npython -m pytest gyroflow_window/test_smoothing.py -q\n'
              'python -m gyroflow_window.build_bundle\npython -m gyroflow_window.report\n```', '',
              '完整主验证命令见 [实现说明](gyroflow-sliding-window.md#72-正式验证矩阵)。各轮 metadata.json 保存全部生成参数，'
              'round2 起还保存运行时源码快照；修改参数后请使用新的输出目录，避免覆盖原始证据。', '',
              '约束检查容差为 1e-8；可行性投影还留有浮点安全余量。单元测试覆盖独立 SciPy 核心核对、四元数符号/坐标系不变性、'
              '几何覆盖、FOV=1、空/短片段、任意拆包、输出不可变、真实前瞻预算、收尾，以及单文件有/无 Numba 的一致性。', '',
              '所有“无覆盖违约”结论仅适用于此纯旋转、无畸变、无滚动快门的针孔模型。'
              '没有真实图像、镜头标定、平移/视差、运动模糊或主观播放评测；不能据此声称真实 App 已经达到相同画质或不会露黑。'
              'Global 基线经过源码移植和独立计算核对，尚未与编译后的完整 Gyroflow 引擎逐帧比较。', '']
    target = DOCS/'gyroflow-sliding-window-results.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
