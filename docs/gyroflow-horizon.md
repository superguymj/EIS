# 水平矫正与 FOV feedback 协同算法

更新日期：2026-10-19。本文对应修复后的源代码与 `dist/gyroflow_sliding_window.py`；本次文档补充明确修复范围和三轴平滑的取舍，不代表又实现了新的联合优化算法。

本次修改保留 FOV feedback、窗口提交协议和画面覆盖检查。`enablehorizon=True` 时，原有两轮平滑继续接受逐帧反馈；水平步骤生成 roll 接近 0 的目标；接缝拟合后恢复水平目标；最终通过保持水平的可行性搜索修正视线运动。搜索的每个可接受姿态都必须满足原来的 FOV 底线。

本次真实视频实验使用 `minimum_fov=0.6`。库和命令行的默认值仍为 `0.8`，调用时显式传入 `0.6`。没有通过关闭 feedback、关闭 `hard_fov` 或在画图后把 roll 数值写成零来获得结果。

后续版本说明：2026-10-23 增加了 [固定历史 FOV 的衔接平滑](gyroflow-fov-history-smoothing.md)。本文第 6 节和 72 项测试记录对应 horizon 修复阶段；新的缩放拟合会改变 FOV 曲线，不改变这里验证的姿态输出。复现本文历史 FOV 数据时使用 `WindowConfig(fov_boundary_weight=0)`，历史比较脚本已显式设置该参数。

`experiments/` 已整体忽略，本文中的图和数据路径仅指本地产物；复现脚本迁至 `tools/horizon_video/`，不再随实验产物一起忽略。

## 1. 原问题与修改范围

旧版 `288a63a` 将同一反馈系数 `c` 用在两处：原有滤波的速度阈值，以及水平力度 `h(c)=2c/(1+c)`。开启水平后裁剪需求增大，反馈下降，pitch/yaw 的平滑与水平矫正一起减弱。之后的接缝拟合可能重新引入 roll；最终整姿态 SLERP 投影又会同时撤销视线调整与 roll 矫正。

因此，旧测试中“反馈系数满足公式”和“FOV 不越界”通过，不能说明最终提交的相机姿态已经水平。本次增加了有实际反馈压力的端到端水平验收。

此前实现的缺陷是：水平目标会被反馈耦合、接缝拟合和整姿态回退削弱，最终输出没有达到预期的水平效果，而验收没有完整检查这个输出链路。当前代码已修正这条链路，并记录最终姿态的水平误差和覆盖结果。修复并不等于三轴平滑已经达到最佳平衡；当前策略优先保持水平，尚未提供可调的 roll 与 pitch/yaw 联合优化权重。

| 文件 | 本次职责 |
|---|---|
| [gyroflow_global.py](../gyroflow_window/gyroflow_global.py) | 保留反馈控制的双向平滑；生成完整水平目标；开启 horizon 时在 Global 反馈后增加覆盖约束投影 |
| [geometry.py](../gyroflow_window/geometry.py) | 提取水平角及方向可靠性；绕光轴拉平；增加水平与视线分开处理的 FOV 可行性搜索 |
| [sliding_window.py](../gyroflow_window/sliding_window.py) | 接缝后恢复可观测的水平目标；保留反馈更新；最终选择 horizon 专用投影；记录反馈与残余水平误差 |
| [test_smoothing.py](../gyroflow_window/test_smoothing.py) | 检查最终 roll、平滑效果、活跃反馈、覆盖、流式协议和单文件一致性 |
| [build_bundle.py](../gyroflow_window/build_bundle.py) | 更新命令行说明，重新生成交付文件 |
| [plot_video_angles.py](../tools/horizon_video/plot_video_angles.py) | 从真实 MP4 导出原始姿态，运行正式窗口算法，保存三轴曲线、CSV、姿态与验证指标 |
| [compare_horizon.py](../tools/horizon_video/compare_horizon.py) | 直接读取旧提交的实现，在同一数据、同一 FOV 参数下比较修复前后 |

就本次 horizon 修复而言，`enablehorizon=False` 的计算路径保持原有行为。隔离后续 FOV 拟合后，两段真实视频核对中关闭 horizon 的核心、窗口姿态与 FOV 均与旧版逐位一致。

## 2. 坐标与配置

输入仍为 `(N, 4)`、`wxyz`、相机到世界四元数，每个视频帧一个姿态。相机光轴为局部 `+Z`，`horizon_up` 是期望与相机局部 `+Y` 对齐的世界竖直参考，默认 `(0, 1, 0)`。

```python
config = SmoothingConfig(
    fps=29.97002997002997,
    minimum_fov=0.6,
    enablehorizon=True,
    horizon_up=(0, 1, 0),
)
```

开关必须为布尔值。竖直参考必须是有限非零三维向量，构造时安全归一化并复制为不可变 tuple。配置在创建流时确定。

若世界坐标变换为 `Q_new = W * Q`，参考也必须变换为 `up_new = R(W) * up`。本功能不根据首帧猜水平，也不承担 IMU 重力估计。

## 3. 平滑与水平目标

### 3.1 feedback 继续控制原有平滑

两轮前后向四元数滤波、角速度计算、偏差归一化和滤波系数公式保持原样。逐帧反馈 `c_i` 继续进入：

```text
max_velocity_i = base_velocity * c_i
ratio_i = velocity_i / max_velocity_i
alpha_i = min(1, a_s + (a_f - a_s) * ratio_i)
```

反馈下降仍会使滤波更多跟随原始运动，释放裁剪空间。没有另建 horizon 反馈数组，也没有把已矫正姿态重新当作原始传感器输入。

### 3.2 水平目标由几何约束限幅

设滤波候选为 `S`，世界竖直单位向量为 `u`：

```text
g = R(S)^T u
theta = atan2(-g_x, g_y)
S_level = S * Exp([0, 0, reliability * theta])
```

正常方向下 `reliability=1`，目标 roll 为 0。右乘绕相机局部 Z 轴的旋转保持候选光轴不变，因此在 `Q=Ry(yaw)Rx(pitch)Rz(roll)` 的欧拉角约定下，这一步保持 pitch/yaw。

水平目标不再乘 `h(c)`。是否能提交这个目标，由真实原始姿态对应的几何 FOV、feedback 重算以及最终可行性搜索决定。低层 `gyroflow_smooth()` 只生成目标，不单独承诺覆盖；应通过 `stabilize_window()` 或 `stabilize_global()` 获得经过覆盖检查的结果。

`horizon_gain()` 为兼容旧调用者而保留原公式，但已不参与正式算法。旧版“`c=0.5` 时固定 12° 倾斜输出 4°”不再是当前算法的验收规则。

### 3.3 方向退化保护

`horizon_reference()` 保留两个 smoothstep 衰减：

- 光轴距世界竖直或其反向小于等于 5° 时不附加拉平；达到 10° 时采用完整目标。
- 所需局部 roll 矫正角从 150° 接近 180° 时逐渐关闭，避免半圈分支跳变。

完全倒置画面不承诺恢复正立。退化范围不参与保持水平的正常方向搜索，而使用原有覆盖投影。该保护不提供角速度、加速度或 jerk 的硬上界。

## 4. 保留 feedback 的完整窗口流程

```text
真实 H/E/F 原始姿态 + 只依赖已观测数据的预测尾部
    ↓
gyroflow_smooth：原有两轮平滑，逐帧 feedback 仍然有效
    ↓
生成水平目标
    ↓
固定历史接缝拟合
    ↓
正常方向重新拉平，避免接缝引入 roll
    ↓
相对于原始姿态计算几何 FOV 和缩放包络
    ↓
超限时降低同一份 feedback，重新计算原始窗口
    ↓
反馈轮次结束，执行 horizon 专用 FOV 可行性搜索
    ↓
重新计算几何与缩放、衔接历史缩放，只提交 E 帧
```

更新规则仍为：

```text
c_j ← c_j * min(F_j / minimum_fov, threshold_k)
threshold_k = 0.95, 0.90, 0.85, 0.80，后续保持 0.80
```

默认最多 5 次反馈重算，`minimum_fov` 仍是反馈的目标与默认硬投影的下限。只有未提交帧的反馈会更新；历史输出、真实前瞻预算、拆包一致性及 `flush()` 协议保持不变。

## 5. 水平与视线分开的覆盖投影

`project_horizon_to_fov()` 替代开启 horizon 时的最终整姿态回退：

1. 计算原有 `project_to_fov()` 的可行结果，作为安全候选。
2. 正常方向下，若完整水平目标满足 FOV 下限，直接接受。
3. 否则计算“保持原始视线、尽量拉平”的可行锚点。它与第 1 步的候选都经过几何验证；选择残余水平角较小的锚点。
4. 将锚点与目标的视线姿态插值，每次试探重新拉平，再单独插值锚点的残余 roll。
5. 最多二分 30 次，只保留满足 FOV 下限并处于可靠方向的试探结果，最后再次检查。数值失败时回到原始姿态，保证覆盖。

若原始视线本身存在可行的完整水平姿态，锚点残余 roll 为 0，整条搜索路径保持 roll 为 0；FOV 不足时主要退让视线平滑量，而不会因整个四元数一起回退而重新带入原始 roll。

若完整水平与 FOV 底线冲突，仍允许残余 roll。这个一维可行路径搜索不承诺找到所有姿态中的全局最优解。无论是否水平，最终都要通过几何检查，不能在输出后单独把欧拉角 roll 改为 0。

窗口默认 `hard_fov=True`。原有 `hard_fov=False` 仍仅用于消融实验，不提供硬底线保证。本次正式实验一直使用默认 `True`。

Global 开启 horizon 后也增加相同的最终覆盖投影。关闭 horizon 时 Global 仍保留原有有限轮次软反馈行为。

### 5.1 水平矫正与三轴平滑的取舍（trade-off）

在固定相机模型和 `minimum_fov` 下，roll 矫正与 pitch/yaw 平滑共享有限的裁切余量。余量充足时可以同时改善；约束吃紧时，要求 roll 更接近 0，可能迫使 pitch/yaw 更多跟随原始运动。这个取舍来自三轴旋转共同影响画面覆盖，并不意味着每个场景都必须牺牲某一轴。

当前实现采用水平优先的可行路径：只要原始视线存在可行的水平锚点，搜索就保持 roll 为 0，主要退让视线平滑量。052 在 FOV 下限 0.6 时展示了这种代价，具体指标见第 6.2 节。与此同时，最终投影逐帧决定姿态，未联合约束投影后的时间连续性；它也可能增加曲线粗糙度。因此，不能把所有尖峰都归为不可避免的物理取舍，也不能仅以 roll 为零判定整体防抖通过。

后续改进方向是保留 feedback 和覆盖硬约束，将 pitch/yaw 平滑、roll 接近 0 以及 roll 自身的时间平滑共同纳入目标。允许少量、缓慢变化的残余倾斜，以换取更平顺的视线运动。这个可调的联合优化目前尚未实现，现有接口也没有对应权重参数。

评估该改进时，应在同一视频、FOV 下限和窗口参数下，同时比较水平误差、三轴时间粗糙度与覆盖违例，使用最终提交的四元数生成曲线，不对画图数据另做平滑或强制归零。

## 6. 真实视频实验

### 6.1 数据与测量

按用户要求，将实验下限从 0.8 放宽为 0.6，增加较平缓的 `PRO_VID_20260826_112713_00_046.mp4`：732 帧、29.97 fps，约 24.39 秒。它的原始四元数角速度 RMS 约 31.1°/s；原 052 视频约 164.8°/s。选择依据是原始运动量，不是处理后的水平误差。

仍保留 052 作为反馈压力回归。两个实验使用同样的 H/E/F=60/6/15、默认最多 5 次反馈、100° 对角 FOV、16:9 针孔相机模型。

Gyroflow 仅从原始 MP4 提供逐帧原始相机姿态；平滑姿态来自本仓库正式算法。其积分坐标的世界竖直是 +Z，实验统一左乘 `Rx(-90°)`，映射到世界 +Y。图中欧拉角为 SciPy 的内禀 `YXZ`，排列为 pitch/yaw/roll；只做等价的 360° 展开，不对曲线分别归零，也不额外滤波。

同时使用 `atan2((R(Q)^T up)_x, (R(Q)^T up)_y)` 独立测量画面水平角，与欧拉 roll 核对；通过欧拉角重建四元数核对分解；用角点重投影检查最终覆盖。

### 6.2 正式结果

下表的修复前后均为 **同一视频、同一 `minimum_fov=0.6`**。修复前代码直接读取提交 `288a63a`，避免与上一次 0.8 的图混为一谈。

| 视频 | 原始 roll RMS | 修复前 roll RMS | 修复后 roll RMS | 修复后最小 FOV | 反馈迭代累计 / 投影帧数 | 覆盖违例 |
|---|---:|---:|---:|---:|---:|---:|
| 046，732 帧 | 4.746325° | 0.000389° | < 1e-8° | 0.672226 | 0 / 0 | 0 |
| 052，220 帧 | 11.016813° | 3.509434° | < 1e-8° | 0.60000002 | 115 / 16 | 0 |

046 的 feedback 机制保持启用，但没有超限，无需下降；052 在 37 个窗口中累计执行 115 次反馈重算，说明压力场景没有跳过反馈。最终最大绝对 roll 分别约 `1.27e-14°` 与 `2.54e-14°`，属于浮点零值。

用逐帧欧拉角二阶差分 RMS 记录曲线粗糙度，单位为度/帧²：

| 视频与处理 | Pitch | Yaw |
|---|---:|---:|
| 046 原始 | 0.370668 | 0.321541 |
| 046 horizon 关闭 | 0.094394 | 0.247522 |
| 046 修复后开启 | 0.094450 | 0.247499 |
| 052 原始 | 4.882754 | 4.829048 |
| 052 修复前开启，FOV=0.6 | 1.614331 | 1.413928 |
| 052 修复后开启，FOV=0.6 | 1.976057 | 1.750749 |

046 保留原有 pitch/yaw 平滑，roll 达到水平。052 的 pitch/yaw 相对原始仍明显平滑，但在同样裁剪预算下，相比旧版允许残余 roll 的结果更粗糙。这说明当前水平优先策略存在视线平滑代价，不能宣称三个轴在所有视频上都同时改善；这组结果也没有将裁切约束的必然取舍与逐帧投影带来的额外粗糙度分别量化。该指标不等于相对于真实运动轨迹的误差。

这是姿态算法与针孔覆盖验证；尚未将这些输出四元数接入真实镜头标定、滚动快门和视频重映射后进行逐像素黑边验收。

### 6.3 图和数据

- 046：`experiments/horizon_video/046/pitch_yaw_roll.png`?完整三轴曲线??????、`experiments/horizon_video/046/pitch_yaw_roll_zoom.png`?局部放大??????、`experiments/horizon_video/046/angles.csv`?CSV??????、`experiments/horizon_video/046/metadata.json`?参数与验证??????。
- 052：`experiments/horizon_video/052/pitch_yaw_roll.png`?完整三轴曲线??????、`experiments/horizon_video/052/pitch_yaw_roll_zoom.png`?局部放大??????、`experiments/horizon_video/052/angles.csv`?CSV??????、`experiments/horizon_video/052/metadata.json`?参数与验证??????。
- `experiments/horizon_video/comparison_feedback06.json`?同参数旧版 / 新版 / 关闭开关比较数据??????。

当前 `experiments/horizon_video/052/` 保存 horizon 修复后 FOV=0.6 的数据；参数以该目录的 `metadata.json` 为准。`diagnose_horizon.py` 固定读取旧提交，仅用于复现原问题，其历史诊断不能代替当前结果。

## 7. 使用与复现

```python
from gyroflow_window.gyroflow_global import SmoothingConfig, stabilize_global
from gyroflow_window.sliding_window import WindowConfig, stabilize_window

config = SmoothingConfig(fps=30, minimum_fov=0.6, enablehorizon=True)
result = stabilize_window(raw, config, WindowConfig(history=60, emit=6, future=15))
# 持续输入：SlidingWindowStabilizer(config, window)，按原协议 push / flush。
global_result = stabilize_global(raw, config)
```

生成单文件与运行命令行：

```powershell
python -m gyroflow_window.build_bundle
python dist/gyroflow_sliding_window.py input.npy --output output.npz --fps 30 --minimum-fov 0.6 --enable-horizon --horizon-up 0 1 0
```

`--enablehorizon` 与 `--enable-horizon` 为同一个开关。无需新增反馈配置。

真实数据位于本地 `dataset/`，图的复现需要其中的 MP4 与便携 Gyroflow：

```powershell
python tools/horizon_video/plot_video_angles.py
python tools/horizon_video/plot_video_angles.py --video dataset/insta360/20260826_195214/PRO_VID_20260826_195214_00_052.mp4 --output experiments/horizon_video/052 --minimum-fov 0.6
python tools/horizon_video/compare_horizon.py
```

第一条默认处理 046、FOV=0.6，同时生成全程与固定时间段局部放大图。绘图脚本使用当前窗口默认配置，包含新增的 FOV 拟合；历史比较脚本关闭该拟合以隔离 horizon 修改，并需要 Git 中存在 `288a63a`。

## 8. 验收与兼容性

```powershell
python -m pytest gyroflow_window/test_smoothing.py -q
```

本次完整运行：**72 passed**。重点验收包括：

- 单帧及静止倾斜的非空提前返回都生成水平目标；逐帧 feedback 仍会改变滤波后的视线。
- 构造“视线转动超限、存在可行水平锚点”的独立场景：普通整姿态投影会恢复 roll，新投影保持水平并满足 FOV。
- 9 组不同 FOV 底线与镜头视角的随机姿态，使用角点重投影独立检查覆盖与单位四元数；FOV=1 时返回原始姿态。
- 有冲击运动、feedback 实际下降的完整窗口流程，检查最终提交 roll 接近 0、pitch/yaw 二阶差分降低及画面覆盖。
- 保留朝天、朝地、半圈、参考归一化、旋转与符号不变性测试。
- 保留拆包一致性、历史不可变、前瞻预算、短片段、EOF、单文件有无 Numba 与 CLI 一致性测试。

关闭开关的真实数据兼容性、修复前后同参数比较由 `compare_horizon.py` 中的逐位断言额外检查。实际生成的 `dist` 文件也通过命令行处理这两段真实输入，姿态、FOV、几何 FOV 均与源模块逐位一致，记录于 `experiments/horizon_video/bundle_parity.json`?bundle_parity.json??????。原有关闭 horizon 的合成实验报告继续有效；开启后的效果应以上面的新验收与结果为准。
