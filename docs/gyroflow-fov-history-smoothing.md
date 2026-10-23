# 固定历史 FOV 的滑窗衔接平滑：代码修改与验证

更新日期：2026-10-23。

本次解决滑动窗口刚看到未来裁切需求时，FOV 在接缝处一开始就快速下降的问题。新增一个带上下界的标量轨迹拟合：固定最后两帧已提交 FOV，联合求解本次未提交的 E+F 帧，尽量衔接此前的缩放速度。姿态滤波、horizon、feedback 以及针孔覆盖模型继续使用原有计算。

主代码是 [sliding_window.py](../gyroflow_window/sliding_window.py)，单文件交付是 [dist/gyroflow_sliding_window.py](../dist/gyroflow_sliding_window.py)。本次完整测试 **86 passed**；真实输入 046、052 的几何覆盖和 FOV 下限违例均为 0。

仓库管理：整个 `experiments/` 目录已加入 `.gitignore`，原先跟踪的实验产物从索引移除，本地文件保留。图、CSV、NPZ、JSON 和实验源码快照均为本地产物；下面用路径标明，不随本次提交上传。可复现脚本迁移到受版本管理的 `tools/`。

## 1. 旧实现为什么会一开始下降很快

原来的 FOV 输出有两个步骤：

1. `zoom_curve()` 对几何 FOV 做两轮包络滤波，时间常数为 `zoom_seconds`（默认 4 秒）和 0.2 秒，每轮先反向、再正向。
2. 滑窗从最后一帧已提交 FOV 初始化状态，以 0.2 秒时间常数做快速收紧、缓慢释放：

```python
state = min(target[i], (1-alpha)*state + alpha*target[i])
```

当新的 `target[i]` 小于上一帧 `state`，加权平均位于两者之间，`min()` 会直接选中较小的目标。因而这一步只延缓视野恢复，并没有平滑向下的突变，也没有参考此前的下降速度。

此外，窗口内的包络滤波只能看到当前 H/E/F 的真实数据。一处低 FOV 首次进入前瞻范围后，整段窗口的包络目标可能一起下降；历史部分重新算出的包络也不等于已经提交的历史 FOV。即使整段已知输入的包络曲线呈现先缓、后快、再缓的形状，逐窗输出仍可能先跳到低值，再缓慢下降。

例如，30 fps、H/E/F=60/6/15，安全 FOV 从 0.95 突降到 0.6：突降进入窗口时，旧输出第一帧从 0.95 降至约 0.635。增大 `zoom_seconds` 或重复包络滤波不会建立已提交历史的边界约束。

## 2. 新增的拟合目标与覆盖约束

### 2.1 固定历史值和历史速度

记最近两帧已提交 FOV 为 `h[-2]`、`h[-1]`，本窗口有 n 个未提交的真实帧。正常情况下 n=E+F，`flush()` 时按实际剩余帧数缩短。

原包络滤波和快速收紧/缓慢释放仍生成目标 `z`。新增拟合只求解未来变量 `f[0:n]`：

$$
\min_f\quad \sum_{i=0}^{n-1}(f_i-z_i)^2
  +\lambda\left\|D_2\,[h_{-2},h_{-1},f_0,\ldots,f_{n-1}]^T\right\|^2.
$$

其中 `D2` 是逐帧二阶差分算子，表示相邻缩放速度的变化。跨接缝最前面的两个惩罚项为：

```text
f[0] - 2*h[-1] + h[-2]
f[1] - 2*f[0] + h[-1]
```

第一个项将新一帧的速度衔接到已提交的速度；第二个项抑制进入本窗口后立即改变速度。后续项平滑本窗口内部。历史 FOV 是常量，不参与优化；只提交前 E 帧，剩余 F 帧在下次窗口仍可重算。

这与姿态部分 `fit_fixed_history()` 使用固定历史、惩罚二阶变化的思路一致。FOV 是标量，采用带逐帧上下界的线性最小二乘，不需要四元数切空间近似。

### 2.2 上界必须是实际几何 FOV

默认 `hard_fov=True` 时，约束为：

$$
F_{\min}\le f_i\le G_i,
$$

`G_i` 是最终姿态投影后重新计算的 `geometric_fov[i]`。在允许误差范围内，这保证输出不突破裁切下限，也不会超出当前针孔模型允许的画面覆盖范围。

**几何上界不是包络目标 `z_i`。** 新结果可以高于包络目标，只要仍小于真实安全上界。否则包络一旦提前降至 0.635，拟合也被强制限制在 0.635 以下，仍然无法从历史的 0.95 平顺衔接。包络目标在目标函数里负责牵引轨迹，几何上界负责覆盖。

关闭 `hard_fov` 的消融模式使用 `0 <= f_i <= G_i`，保留覆盖约束，但不承诺 `minimum_fov` 下限。默认模式先对姿态执行硬投影，使 FOV 区间可行。对于几何计算的微小舍入误差，实际下界取 `min(minimum_fov, G_i)`，优先不突破几何上界；验收按 1e-9 容差检查下限。

### 2.3 权重和帧率

新增配置：

```python
WindowConfig(fov_boundary_weight=1000.0)
```

求解时使用：

```text
lambda = fov_boundary_weight * (fps / 30)**4
```

`fov_boundary_weight` 是 30 fps 下的二阶差分权重。二阶差分约随采样间隔的平方变化，因此使用 fps 的四次方缩放，使加速度惩罚在不同帧率下具有可比较的强度。这不保证整个系统在不同帧率下完全相同：H/E/F 仍以帧为单位。

权重越大，越重视缩放速度的衔接，也可能更偏离包络目标、延缓视野恢复或增加临时裁切。参数必须有限且非负。设为 **0** 会跳过新拟合，恢复旧版 FOV 输出；本次两段真实输入已与保存的旧 FOV 逐位核对。

`boundary_weight` 仍控制姿态接缝，两者分别配置。`zoom_seconds` 控制包络目标的时间尺度；`minimum_fov` 控制裁切底线，均不是新接缝权重。

## 3. 具体修改了哪些代码

| 文件 | 修改内容 |
|---|---|
| [gyroflow_window/sliding_window.py](../gyroflow_window/sliding_window.py) | 增加配置、矩阵缓存、带界拟合函数、最终 FOV 接入点和诊断字段 |
| [gyroflow_window/build_bundle.py](../gyroflow_window/build_bundle.py) | 添加 `--fov-boundary-weight` CLI 参数，并传给 `WindowConfig` |
| [dist/gyroflow_sliding_window.py](../dist/gyroflow_sliding_window.py) | 由构建程序重新生成，包含相同拟合与 CLI |
| [gyroflow_window/test_smoothing.py](../gyroflow_window/test_smoothing.py) | 新增优化、覆盖、突降和端到端回归；加强 FOV 历史不可变与 CLI 验收 |
| [tools/fov_smoothing/compare_history.py](../tools/fov_smoothing/compare_history.py) | 生成固定几何滑窗重放、046/052 实测、两轮/四轮滤波对比与 JSON/CSV/NPZ |
| [tools/fov_smoothing/step_response.py](../tools/fov_smoothing/step_response.py) | 保留此前整段已知输入的包络响应实验，明确其与实际滑窗的区别 |
| [tools/horizon_video/compare_horizon.py](../tools/horizon_video/compare_horizon.py) | 复现历史 horizon 对比时显式关闭新 FOV 拟合，隔离两次算法修改；兼容当前 052 数据目录 |
| [tools/horizon_video/plot_video_angles.py](../tools/horizon_video/plot_video_angles.py)、[diagnose_horizon.py](../tools/horizon_video/diagnose_horizon.py) | 从实验目录迁移到工具目录，输入输出仍指向本地 `experiments/`；旧诊断与独立历史求解器核对，避免误用新版 CSV |
| [.gitignore](../.gitignore) | 忽略整个 `experiments/`，实验文件从 Git 索引移除，保留本地副本 |
| [NOTICE](../gyroflow_window/NOTICE.md)、[dist/NOTICE.md](../dist/NOTICE.md) | 将带覆盖约束的 FOV 历史拟合注明为本仓库扩展 |
| [滑窗接入文档](gyroflow-sliding-window.md)、[horizon 文档](gyroflow-horizon.md) | 添加新算法入口、参数和历史结果适用范围说明 |

### 3.1 `WindowConfig`

在现有字段末尾增加 `fov_boundary_weight=1000.0`，避免改变旧位置参数对应的字段；在 `__post_init__()` 中检查负值、NaN 和无穷值。

### 3.2 `_fov_boundary_system()`

新增带 `lru_cache(maxsize=128)` 的矩阵构造函数。根据未提交帧数、可用历史帧数和有效权重，生成：

```text
A = [I; sqrt(lambda) * D_pending]
H = sqrt(lambda) * D_history
```

求解右侧为 `[z; -H @ previous]`。历史最多两帧；相同尺寸与权重复用矩阵。矩阵缓存不保存未来帧内容。

### 3.3 `fit_fov_history()`

该函数输入候选 FOV、已提交历史、逐帧几何上界、有效权重和下限，返回拟合结果及求解信息：

1. 构造上下界和经过限幅的安全候选。
2. 无历史、无待求解帧或权重为 0 时直接返回安全候选。只有一帧历史时使用已有信息，不伪造更早的速度。
3. 将上下界相等的变量先固定并移出求解，处理 `minimum_fov=1` 和恰好到达下限的帧。
4. 使用已有 SciPy 依赖中的 `lsq_linear(method='bvls')` 求解带界最小二乘，容差 1e-11，迭代上限为 `max(50, 5*n)`。
5. 对结果做数值限幅。求解未成功或返回非有限值时，使用第 1 步的安全候选，并记录回退标记。

二次目标含单位阵保真项，未固定变量的目标严格凸。回归测试通过独立构造目标梯度、检查边界 KKT 符号验证解，不只核对求解器返回的成功标记。

### 3.4 `SlidingWindowStabilizer._process()`

新增调用位于最终姿态投影之后、复制 `committed_fov` 之前。完整顺序是：

```text
原有姿态平滑、horizon、历史姿态接缝
    -> 几何 FOV + 原包络目标
    -> 原 feedback 迭代
    -> 最终姿态覆盖投影
    -> 重新计算几何 FOV、包络和快速收紧/缓慢释放目标
    -> 新增：固定历史 FOV 的带界拟合
    -> 只提交前 E 帧并保存已提交 FOV
```

feedback 仍检查原包络 FOV，不改成拟合结果，也没有增加或取消反馈轮次。新拟合只改变最后的缩放输出，已经提交的姿态和 FOV 不会回写。046、052 的修改前后姿态、几何 FOV 逐位一致，反馈轮次和反馈最小值一致。

诊断增加：

- `fov_boundary_iterations`：本窗口 BVLS 求解迭代数；跳过或全部变量固定时为 0。
- `fov_boundary_fallback`：求解失败而使用安全候选时为 `True`。

### 3.5 单文件与兼容性

构建脚本将上述源码合并到单文件，不手工修改生成文件。CLI 新增 `--fov-boundary-weight`，默认 1000。原有输入格式、输出的三个数组以及 `push()` / `flush()` 协议保持兼容。

`gyroflow_global.py`、`math3d.py`、`geometry.py` 本次没有改动；Global 没有滑窗提交接缝，继续使用原来的包络 FOV。窗口默认输出的 FOV 会发生变化，因此原先的 FOV 数值和耗时报告属于历史结果；需要复现旧缩放时显式设置权重为 0。

## 4. 验证结果与曲线

### 4.1 固定几何 FOV 的滑窗重放

此实验固定安全 FOV 序列为 `0.95 -> 0.6 -> 0.95`，仅隔离缩放环节。它没有运行姿态滤波和 feedback，不能代替下面的真实姿态端到端验证。

| 前瞻 F | 开始收紧那一帧的下降量：旧 | 新 | 全程最大单帧变化：旧 | 新 |
|---:|---:|---:|---:|---:|
| 0 | 0.350000 | 0.350000 | 0.350000 | 0.350000 |
| 15 | 0.314977 | 0.009514 | 0.314977 | 0.028548 |
| 30 | 0.285652 | 0.008463 | 0.285652 | 0.025366 |

F=15 时，旧方法首次看到未来低值便直接大幅收紧；新方法利用仍然安全的较大 FOV 空间分摊变化。F=0 且突降恰好位于新批次第一帧时，两种方法都必须立即到 0.6：历史已经提交，没有可用的提前量，不能为了曲线平顺保留不安全的 FOV。

`experiments/fov_smoothing/step_history_comparison.png`?固定几何 FOV 的滑窗对比??????

### 4.2 046、052 真实姿态输入

复用此前从真实 MP4 导出的原始姿态，运行完整 `stabilize_window()`。两段都开启 horizon 和 feedback，`minimum_fov=0.6`、H/E/F=60/6/15、约 29.97 fps、100° 对角 FOV 与 16:9 针孔模型。旧版使用 `fov_boundary_weight=0`，新版使用默认 1000。

表中差分单位分别为 FOV/帧和 FOV/帧²。接缝指标使用第一帧新速度与上一帧已提交速度之差，即 `f[k]-2*f[k-1]+f[k-2]`。

| 视频 | 指标 | 旧输出 | 新输出 |
|---|---|---:|---:|
| 046 | 最大单帧变化 | 0.035312 | 0.013153 |
| 046 | 全程二阶差分 RMS | 0.003937 | 0.000812 |
| 046 | 接缝速度变化 RMS | 0.006890 | 0.001748 |
| 046 | 最小 FOV | 0.672226 | 0.649533 |
| 046 | 平均 FOV | 0.752005 | 0.750111 |
| 052 | 最大单帧变化 | 0.005970 | 0.000711 |
| 052 | 全程二阶差分 RMS | 0.00041223 | 0.00004770 |
| 052 | 接缝速度变化 RMS | 0.00094812 | 0.00006588 |
| 052 | 最小 FOV | 0.60000002 | 0.60000000 |
| 052 | 平均 FOV | 0.605091 | 0.604923 |

两段覆盖违例、下限违例、求解回退均为 0。046 未触发反馈下降；052 累计反馈重算 115 次、最小反馈系数约 0.272842，与修改前一致。输出姿态没有变化，FOV 平滑不改变此前的 pitch/yaw/roll 曲线。

046 的最小 FOV 比旧输出更小：为了减少速度突变，新曲线在候选已经走平时仍可能短暂继续下降，再回升。这是更平顺与额外裁切的取舍，当前目标没有保证单调性或禁止低于包络目标；不能仅凭二阶差分下降宣称所有方面都改善。

- `experiments/fov_smoothing/046_history_comparison.png`?046 FOV 对比图??????、`experiments/fov_smoothing/046_history_comparison.csv`?CSV??????。
- `experiments/fov_smoothing/052_history_comparison.png`?052 FOV 对比图??????、`experiments/fov_smoothing/052_history_comparison.csv`?CSV??????。
- `experiments/fov_smoothing/history_comparison.json`?完整参数、数值和源码散列??????。

这些结果验证姿态算法输出和针孔覆盖，尚未对真实镜头标定、滚动快门和重映射后的视频逐像素验收。

### 4.3 多加两轮包络滤波的对照

对整段已知的 `0.95 -> 0.6` 输入，重复一组 4 秒 + 0.2 秒包络后，仍存在先缓、后快、再缓的拐点，但曲线更早收紧：拐点从突降前约 0.6 秒移到约 4.37 秒；突降前 4 秒的 FOV 从约 0.815 降到约 0.681。

`experiments/fov_smoothing/two_vs_four.png`?两轮与四轮对比图??????。该对照不增加真实滑窗前瞻，也没有替代已提交历史的接缝约束；正式算法本次没有增加包络轮数。

## 5. 测试覆盖

```powershell
python -m pytest gyroflow_window/test_smoothing.py -q
```

本次完整结果：**86 passed in 65.40s**。

- 可行匀速缩放轨迹保持原样，输入候选和历史数组不被修改。
- 已看到未来突降但当前仍有几何余量时，首帧不再直接坠落到包络目标。
- 1、2、21、64 个变量，1/2 帧历史，含固定边界的 8 组问题，独立检查 KKT 条件和覆盖上下界。
- 突降已到当前帧时覆盖优先；无待处理帧、单帧尾部、上下界完全相等均能处理。
- 模拟求解失败，确认回退保持有限、可覆盖的输出。
- 完整窗口流程开启/关闭 `hard_fov` 的对照，确认 FOV 粗糙度改善、姿态和反馈保持一致。
- 将已有“未来不泄漏、历史不回写”测试扩展到已有历史的窗口和 FOV 数组。
- 保留不同前瞻、不同外部拆包、短视频、朝天/朝地、horizon、覆盖测试。
- 单文件有/无 Numba 与源模块一致；CLI 使用非默认权重 37 与源码结果核对。

实际生成的 `dist/gyroflow_sliding_window.py` 另通过 CLI 处理了 046、052 的完整真实输入，`orientations`、`fov` 和 `geometric_fov` 均与源模块逐位一致，记录见 `experiments/fov_smoothing/bundle_parity.json`?bundle_parity.json??????。历史 horizon 比较脚本也已运行通过，确认关闭新拟合后可复现原来的兼容性结论。

## 6. 使用与复现

```python
from gyroflow_window.gyroflow_global import SmoothingConfig
from gyroflow_window.sliding_window import WindowConfig, stabilize_window

config = SmoothingConfig(fps=30, minimum_fov=0.6, enablehorizon=True)
window = WindowConfig(history=60, emit=6, future=15, fov_boundary_weight=1000)
result = stabilize_window(raw, config, window)
```

同一 `WindowConfig` 可传给流式 `SlidingWindowStabilizer`。旧输出对照使用 `fov_boundary_weight=0`。

```powershell
python -m gyroflow_window.build_bundle
python dist/gyroflow_sliding_window.py input.npy --output output.npz --fps 30 --minimum-fov 0.6 --enable-horizon --fov-boundary-weight 1000
python tools/fov_smoothing/compare_history.py
```

最后一条会重建实验目录里的合成与真实对比图、CSV、NPZ 和 JSON。真实实验输入为本地 `experiments/horizon_video/046/`、`052/` 的 `orientations.npz` 与 `metadata.json`，不重新导出 MP4。输出集中在被忽略的 `experiments/fov_smoothing/`。新克隆的仓库需要本地输入数据；可先运行 `tools/horizon_video/plot_video_angles.py` 从本地 MP4 导出姿态。

## 7. 适用边界

本次提供的是覆盖约束下的软加速度惩罚，没有承诺严格 C1 连续、最大缩放速度或 jerk 上限。新约束与历史运动冲突时，覆盖优先。更多前瞻能提供更多提前过渡的空间，但会增加等待时间；增大拟合权重不能替代未知的未来数据。

优化对象是 FOV 系数，不是放大倍率 `1/FOV`；两者的速度、加速度指标不同。当前实验也没有优化视频内容、镜头畸变或运动模糊。

求解仅针对当前 E+F 帧，并复用最多两帧已提交 FOV；未增加真实输入前瞻。矩阵缓存有界，但带界最小二乘增加了计算工作量，长前瞻、大批次及目标移动平台的运行耗时仍应另行测量。
