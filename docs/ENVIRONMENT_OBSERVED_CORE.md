# 已观察环境快照

观察日期：2026-10-19（Asia/Singapore；对应host的2026-10-18 UTC）。只读pyvenv.cfg及dist-info/METADATA，没有启动仿真、导入训练模型或执行pip。以下为本地behavior-venv当前安装，不等同于所有历史采集/训练运行的锁定环境；OpenPI训练环境应另取对应运行证据。

| 组件 | 已观察版本 |
|---|---|
| Python | 3.10.12 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| Isaac Sim | 4.5.0.0 |
| OmniGibson | 3.7.2（editable） |
| BDDL | 3.7.0（editable） |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| h5py | 3.16.0 |
| opencv-python | 4.11.0.86 |
| PyYAML | 6.0.3 |
| imageio-ffmpeg | 0.6.0 |
| telemoma | 0.3.0 |

旧requirements快照的h5py3.15.1与当前安装不一致；README中Python3.11/Isaac5.1/OG3.8叙述也不能代表本地已观察安装。更完整的源码commit、版权与版本边界见reports/DEPENDENCY_AUDIT.md。该文档仅记录事实，不建议在原项目环境中直接执行安装或升级。
