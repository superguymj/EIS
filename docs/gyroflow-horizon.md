# Gyroflow horizon 功能修改说明

`enablehorizon` 为候选相机姿态增加水平偏好：运行原有两轮四元数平滑后，绕相机光轴修正画面倾斜，再让外层计算 FOV 并继续反馈迭代。水平矫正与原有平滑共用逐帧 `feedback`，通过 `h(c)=2c/(1+c)` 转换成矫正力度。

开关默认关闭，在一段视频或一个流实例创建时固定。算法主体集中在 `gyroflow_window/gyroflow_global.py`；窗口调度、接缝修正、原有几何检查和硬 FOV 投影继续使用现有实现。

## 1 改动文件与具体职责

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| [gyroflow_global.py](../gyroflow_window/gyroflow_global.py) | 模块说明和导入 | 说明可选 horizon 是本项目扩展；复用 `math3d` 已有的 `matrices`、`multiply`、`from_rotvec` |
| [gyroflow_global.py](../gyroflow_window/gyroflow_global.py) | `SmoothingConfig` | 新增 `enablehorizon` 和 `horizon_up`，校验并归一化竖直参考 |
| [gyroflow_global.py](../gyroflow_window/gyroflow_global.py) | `horizon_gain` | 将同一份反馈系数映射为水平矫正力度，不新增反馈状态 |
| [gyroflow_global.py](../gyroflow_window/gyroflow_global.py) | `_apply_horizon` | 计算当前图像倾斜角，处理方向退化，右乘绕局部 Z 轴的矫正四元数 |
| [gyroflow_global.py](../gyroflow_window/gyroflow_global.py) | `gyroflow_smooth` | 在两个提前返回 `first` 的分支和最终返回 `second` 的分支统一应用 horizon |
| [build_bundle.py](../gyroflow_window/build_bundle.py) | `CLI` 模板 | 新增 `--enablehorizon`、别名 `--enable-horizon` 和 `--horizon-up X Y Z`，传给 `SmoothingConfig` |
| [test_smoothing.py](../gyroflow_window/test_smoothing.py) | 现有流式和单文件测试 | 将相关测试扩展到 horizon 开启和关闭两种状态；坐标变换时同时变换竖直参考 |
| [test_smoothing.py](../gyroflow_window/test_smoothing.py) | 新增 horizon 测试 | 检查静止倾斜、反馈力度、视线保持、闭环反馈、退化方向、参数和命令行入口 |
| [NOTICE.md](../gyroflow_window/NOTICE.md) | 实现范围 | 标明反馈控制的 horizon 矫正属于本项目扩展，不包含 IMU 重力估计或完整 Gyroflow horizon 行为 |
| [gyroflow-sliding-window.md](gyroflow-sliding-window.md) | 入口说明、范围和参数表 | 增加功能入口及配置说明，明确旧实验关闭此功能 |
| [gyroflow_sliding_window.py](../dist/gyroflow_sliding_window.py) | 自动生成的交付文件 | 由构建程序重新合并源模块，包含新配置、辅助函数和命令行选项 |
| [dist/NOTICE.md](../dist/NOTICE.md) | 构建时复制 | 与源目录的实现范围说明保持一致 |

`math3d.py` 提供的基础运算已经足够，本次没有修改其运算实现。`sliding_window.py`、`geometry.py` 和预测算法的源代码也没有修改：调用者已经将同一个 `SmoothingConfig` 和反馈数组传给 `gyroflow_smooth()`，因此新功能能进入现有闭环。

## 2 配置接口

在 `SmoothingConfig` 末尾增加：

```python
enablehorizon: bool = False
horizon_up: tuple[float, float, float] = (0.0, 1.0, 0.0)
```

`enablehorizon` 必须为布尔值。它属于共享的平滑配置，因此 Global 和窗口模式采用相同的水平矫正规则。`WindowConfig` 仍描述 H/E/F 等窗口行为。

`horizon_up` 必须是有限、非零的三维向量。构造配置时先除以最大绝对分量，再进行单位化，避免大数归一化时溢出；最后复制为不可变 tuple。调用方传入列表后再修改原列表，不会改变已经创建的配置。

`SmoothingConfig` 继续使用 `frozen=True`。每条流应在创建时确定是否启用 horizon 和竖直参考，处理中保持同一份配置，不提供运行中切换或渐变开关的接口。

### 2.1 竖直参考的含义

项目的姿态约定是 `wxyz`、相机到世界，光轴为相机局部 `+Z`。本功能期望相机局部 `+Y` 与给定世界竖直方向在图像平面上的投影一致。默认 `(0, 1, 0)` 与当前合成轨迹的约定相符。

真实数据接入时必须确认坐标约定。如果外部相机坐标的 `+Y` 指向图像下方，应统一坐标系，或提供与该 `+Y` 对应的世界竖直方向。该功能不会从第一帧推断真实水平，也不会从陀螺仪或加速度计样本估计重力。

如果对所有相机姿态施加世界坐标变换 `W`，应同时变换参考：

```text
Q_new = W * Q
up_new = R(W) * up
```

否则改变的是水平目标本身，不能期待相同的防抖结果。测试中的旋转不变性检查已按此约定更新。

## 3 共享 feedback 的转换函数

原有的 `feedback` 是逐帧缩放反馈系数，记为 `c_i`，初始为 1。它仍按原有方式影响两轮平滑：

```python
max_velocity = base_velocity * c_i
ratio = velocity / max_velocity
alpha = min(1, a_s + (a_f - a_s) * ratio)
```

新增 `horizon_gain()` 对同一个 `c_i` 计算：

\[
\bar c_i = \operatorname{clip}(c_i, 0, 1), \qquad
h_i = \frac{2\bar c_i}{1+\bar c_i}
\]

| 原有反馈 c | 水平矫正力度 h |
|---:|---:|
| 1.00 | 1.0000 |
| 0.80 | 0.8889 |
| 0.50 | 0.6667 |
| 0.25 | 0.4000 |
| 0.10 | 0.1818 |
| 0.00 | 0.0000 |

在 0 到 1 之间，`h(c) >= c`，所以相比直接把 `c` 当作水平矫正力度，该映射会保留更多水平矫正。映射单调递增，在两端分别为 0 和 1，导数 `2/(1+c)^2` 有界。

这是可解释的软优先策略，不是保证 roll 优先于其他轴的约束优化器。它也不是对 `c` 的放大倍率估计；`c` 和 `h(c)` 都是控制系数。没有新增第二套 horizon 反馈数组、历史状态或独立外层迭代。

该函数支持标量和数组，拒绝 NaN/Inf。裁剪到 `[0, 1]` 只作用于 horizon 力度，原有滤波公式对传入反馈的使用没有改动。`feedback=None` 时，原有代码创建全 1 的 `scalers`，因此 horizon 采用满力度，再受下一节的方向可靠性衰减影响。

## 4 水平矫正的具体计算

设两轮平滑后的候选姿态为 `S`，世界竖直单位向量为 `u`。先把它变换到候选相机坐标系：

\[
g = R(S)^T u
\]

图像平面的竖直投影为 `(g_x, g_y)`，需要的局部 roll 矫正角为：

\[
\theta = \operatorname{atan2}(-g_x, g_y)
\]

没有方向退化时，最终姿态为：

\[
S_{out} = S \cdot \operatorname{Exp}([0,0,h(c)\theta])
\]

实现直接构造旋转向量并右乘四元数。这与沿最短旋转弧向完全拉平姿态做 SLERP 等价。右乘表示绕相机局部 Z 轴旋转，因此该步骤保持 `R(S)[0,0,1]^T`，即保持候选姿态的视线方向。

例如，固定 12° roll、没有其他运动时：`c=1` 输出约 0°；`c=0.5` 时 `h=2/3`，输出约 4°；`c=0.25` 时 `h=0.4`，输出约 7.2°。这些例子处于正常方向范围，未触发以下衰减。

### 4.1 朝天和朝地时的处理

光轴接近世界竖直方向时，图像里的竖直投影接近零，roll 的水平目标变得不可靠。用：

\[
p = \sqrt{g_x^2 + g_y^2}
\]

计算可靠性。光轴距世界竖直或其反向不超过 5° 时，停止附加水平矫正；距离达到 10° 时采用全部力度；中间按 `smoothstep(t)=t^2(3-2t)` 过渡。

代码里的归一化过渡变量为：

\[
t_p = \operatorname{clip}\left(
\frac{p-\sin 5^\circ}{\sin 10^\circ-\sin 5^\circ}, 0, 1\right)
\]

### 4.2 接近上下倒置时的处理

`atan2` 在 `-180°` 与 `+180°` 之间有分支切换。部分矫正时，如果直接将这个角度乘以力度，会把一个很小的姿态变化放大成明显跳变。

本次采用无额外状态的退让策略：所需矫正角的绝对值不超过 150° 时不衰减；从 150° 到 180°，按 smoothstep 将力度衰减到零。对应：

\[
t_b = \operatorname{clip}\left(\frac{\pi-|\theta|}{\pi/6},0,1\right)
\]

最终使用的力度为：

\[
h_{effective}=h(c)\,\operatorname{smoothstep}(t_p)\,
\operatorname{smoothstep}(t_b)
\]

因此，完全倒置的画面会保留倒置状态；本功能不承诺将 360° 翻滚中的所有画面恢复为正立。上述衰减用于避免明确的方向分支跳变，不提供角速度、角加速度或 jerk 上界。阈值是实现中的固定保守规则，不是新增的用户反馈参数。

## 5 在 gyroflow_smooth 中的插入位置

原有计算仍按以下顺序进行：

1. 由原始姿态计算角速度，并使用现有 `feedback` 得到第一轮系数。
2. 第一轮前向和后向 SLERP，得到 `first`。
3. 计算原始姿态与 `first` 的偏差，执行原有归一化与偏差平滑。
4. 使用 `alpha_second` 执行第二轮前向和后向 SLERP，得到 `second`。
5. 由 `_apply_horizon()` 修正候选姿态，再返回给调用者。

水平矫正不改变第一轮、第二轮的系数公式，也不把已矫正的输出作为原始输入传回滤波。`first_distance`、`distance_max` 仍描述原有第一轮偏差，不能当作最终水平误差。

### 5.1 两个提前返回分支

原实现有两个不经过最终 `return second` 的非空分支：第一轮最大偏差为零，以及偏差平滑后的归一化最大值为零。这两个分支都会直接返回 `first`。

现在三个非空返回点统一改为：

```python
return _apply_horizon(first_or_second, config, scalers), diagnostics
```

静止倾斜和单帧输入也能获得水平矫正。空输入继续直接返回空数组。关闭开关时，辅助函数直接返回传入的姿态数组，不执行矩阵变换或重新归一化。

## 6 外层反馈闭环

Global 和窗口模式的 horizon 都在每一轮 FOV 检查之前执行。以窗口为例，完整顺序是：

```text
原始 H/E/F 姿态与预测尾部
    ↓
gyroflow_smooth
    原有 first pass
    原有 second pass
    使用 h(feedback) 做 horizon 矫正
    ↓
固定历史的接缝修正
    ↓
使用原始姿态检查几何 FOV 并生成缩放曲线
    ↓
若超限，更新同一套 feedback，重新运行上述步骤
    ↓
迭代结束后执行现有硬 FOV 投影与缩放衔接
    ↓
提交 E 帧
```

反馈仍按原来的规则只在超限位置下降：

\[
c_j \leftarrow c_j\min(F_j/F_{min},t_k)
\]

其中反馈轮次阈值依次为 `0.95, 0.90, 0.85, 0.80`，后续保持 `0.80`。下一轮既通过 `c_j` 减弱原有平滑，也通过 `h(c_j)` 释放部分水平矫正。每一轮始终重新使用同一份原始姿态，几何检查也始终相对于原始姿态进行。

因此最终输出未必 roll 为零：接缝修正和裁剪约束可能使候选再次倾斜。窗口仍通过已有硬投影保证针孔模型下的 FOV 底线；Global 仍是有限轮次的软反馈，不增加硬底线保证。

H/E/F 的含义、已提交姿态不可修改、真实前瞻预算、`push()` 拆包和 `flush()` 协议保持原有实现。新增计算只使用当前调用提供的姿态和反馈，不读取额外真实未来帧。

## 7 使用方式

### 7.1 Python 接口

```python
import numpy as np
from gyroflow_window.gyroflow_global import SmoothingConfig, stabilize_global
from gyroflow_window.sliding_window import WindowConfig, stabilize_window

raw = np.load("input.npy")  # (N, 4), wxyz, camera-to-world
config = SmoothingConfig(
    fps=30,
    minimum_fov=0.85,
    enablehorizon=True,
    horizon_up=(0, 1, 0),
)

window_result = stabilize_window(
    raw, config, WindowConfig(history=60, emit=6, future=15)
)
global_result = stabilize_global(raw, config)
```

持续输入时，将同一个 `config` 传给 `SlidingWindowStabilizer(config=...)` 即可。处理另一条流时新建实例，不在原实例处理中替换配置。

### 7.2 单文件命令行

```powershell
python dist/gyroflow_sliding_window.py input.npy --output output.npz --fps 30 --minimum-fov 0.85 --enablehorizon --horizon-up 0 1 0
```

`--enable-horizon` 是 `--enablehorizon` 的别名。不传开关时保留原有行为。单文件通过下面的命令生成，后续修改应先改源模块，再重新构建：

```powershell
python -m gyroflow_window.build_bundle
```

## 8 测试覆盖与结果解释

运行：

```powershell
python -m pytest gyroflow_window/test_smoothing.py -q
```

新增测试与主要检查项如下：

| 测试 | 检查内容 |
|---|---|
| `test_horizon_stationary_tilt_and_feedback` | 单帧及静止序列能绕过提前返回遗漏；不同反馈对应预期的剩余倾斜；输入不被改写 |
| `test_horizon_per_frame_feedback_and_preserved_view` | 支持逐帧反馈；保持光轴和单位四元数；偏差诊断仍对应原滤波；转换函数有界且单调 |
| `test_horizon_zoom_feedback_sees_corrected_poses` | 水平矫正会触发外层反馈，反馈下降后缩放需求减小；窗口硬 FOV 和 FOV=1 行为保持 |
| `test_horizon_poles_and_half_turn_do_not_flip` | 单独经过朝天、朝地和半圈分支附近时，不把微小变化变成 180° 跳变 |
| `test_horizon_degeneracy_tapers_without_a_hard_switch` | 方向可靠性衰减具有中间力度 |
| `test_horizon_config_validation` | 拒绝非布尔开关和非法竖直参考 |
| `test_horizon_reference_is_normalized_and_copied` | 正确处理很大和很小的有限参考向量，并隔离外部可变列表 |
| `test_horizon_bundle_cli` | 自动生成单文件的命令行确实开启功能，并传入归一化的竖直参考 |

已有的拆包一致性、历史不可变、真实前瞻预算、空输入与短片段、旋转和符号不变性、单文件有无 Numba 一致性测试，均覆盖开启和关闭两种配置。

### 8.1 本次验证记录

- `python -m pytest gyroflow_window/test_smoothing.py -q`：61 项测试通过。
- 关闭开关的兼容性：与修改前提交 `0c2b6bc` 对比，使用 `static`、`pan`、`walking`、`multi_axis`、`roll`、`wrap` 六种轨迹，种子为 1031、每段 3 秒，分别设置 `minimum_fov=0.6/0.85/0.97`，共 18 组。核心检查使用从 0.1 到 1 的逐帧反馈。核心姿态和 `first_distance`、Global 姿态和 FOV 数组逐位一致，Global 迭代次数相同。
- 实际交付文件：重新生成 `dist/gyroflow_sliding_window.py` 后，以 `multi_axis`、种子 703、2 秒、H/E/F 为 20/6/15、FOV 底线为 0.9 验证。开关开启和关闭时，单文件与源模块的姿态、FOV、几何 FOV 均逐位一致。
- 实际命令行：使用交付文件的 `--enable-horizon` 别名和 `--horizon-up 0 2 0` 运行同一段输入，结果与 Python 接口逐位一致。标准 `--enablehorizon` 拼写由上述自动化测试覆盖。
- 交付目录的 LICENSE、NOTICE 与源目录一致；本文中的相对文件链接均可解析。

旧的 [实验结果](gyroflow-sliding-window-results.md) 对应 `enablehorizon=False`，不能作为新功能开启后的效果数据。评估开启状态时，应增加水平误差，并同时检查视线方向误差、接缝加速度和裁剪；原有干净轨迹可能包含有意 roll，不能仅凭相对旧 Global 或旧干净轨迹的完整姿态 RMSE 判断功能优劣。
