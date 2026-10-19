# Source and scope

The default quaternion smoothing and envelope-filter implementations are Python
translations of Gyroflow commit `2331ebb3c253fc7cf357642518b0c8bf2937f815`.

- Default smoothing: Copyright 2021–2022 Aphobius.
- Dynamic zoom envelope: Copyright 2021 Marc Roeschlin, Adrian, Maik.
- Upstream project: https://github.com/gyroflow/gyroflow
- Source license: GPL-3.0-or-later. The derivative Python implementation and
  generated single-file distribution carry the same SPDX identifier.
  See the accompanying LICENSE file for the full license text.

The synthetic pinhole geometry, bounded streaming adapter, boundary fitting,
tail prediction, optional coverage-constrained horizon correction, synthetic data,
and experiments are additions in this workspace.
They are not an official Gyroflow implementation of streaming stabilization.

The baseline supports constant parameters and the non-per-axis default algorithm.
It does not reproduce Gyroflow's complete calibrated image transform, rolling
shutter correction, integration, full horizon-lock behavior, keyframes, retiming,
or trim logic. The optional horizon correction uses an explicit world vertical,
retains the existing zoom feedback, and adds a level-preserving FOV projection.
It does not estimate gravity from IMU samples.
