# COWMATA 牛尾环 IMU 算法工程

`20260818` 是面向后续算法迭代的干净基线：以模型为中心的 Python API、统一 CLI、按牛划分的数据契约，以及可追溯的训练与推理产物。

仓库采用 Ultralytics YOLO26 式的“统一入口、示例、测试、CI 和清晰贡献规范”，但只保留 COWMATA 当前真实需要的结构。当前版本为 `v0.1.0`，建议先维护为公司私有仓库。

## 快速开始

```powershell
cd <COWMATA 仓库路径>
conda env create -f environment.yml
conda activate cowmata
# 按训练机 CUDA 版本安装 PyTorch，再执行：
pip install -e . --no-deps
cowmata check-data
cowmata predict --cache-key <cache_key> --out runs/predict
```

不放入本地大数据也能跑通 GitHub 内置的 60 秒真实数据演示：

```powershell
cowmata predict --cache-key demo_session_60s `
  --data-root examples/demo_data --out runs/demo
```

Python API 保持“加载一次、多次预测”：

```python
from cowmata import COWMATA

model = COWMATA("weights/deploy/gbdt_full.joblib")
result = model.predict("<cache_key>", project="runs/predict")
print(result.dense.head())
print(result.candidates)
```

## 常用命令

```powershell
# 数据、标签、缓存和按牛分割预检
cowmata check-data

# CPU 安装烟雾测试；正式深度训练前请再测 CUDA
cowmata check-env --device cpu --precision fp32
cowmata check-env --device cuda --precision auto

# 生成数据诊断报告
cowmata diagnose

# 显式选择阶段；不会覆盖旧运行结果
cowmata pipeline -- --stages diagnose,features,feature_model

# 训练新的全量候选挖掘 GBDT，默认输出时间戳目录，不覆盖部署权重
python -m scripts.train_full_gbdt --feature-table runs/feature_table/feature_table.parquet
```

`environment.yml` 故意不自动安装 PyTorch，避免覆盖训练机上的 CUDA 版。请先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与本机匹配的 CUDA wheel，然后使用 `--no-deps` 安装本项目。

## 目录

```text
cattle_imu/              稳定算法核心（保留序列化模型兼容性）
cowmata/                 面向用户的 API 和 CLI
scripts/                 数据、训练、评价和候选挖掘入口
configs/                 数据契约配置
datasets/cowmata_imu/    小型标签/分割 + 本机监督缓存（大文件 Git 忽略）
examples/demo_data/     60 秒真实演示缓存，克隆后可直接推理
weights/deploy/          可用的正式推理权重（进入 Git）
weights/checkpoints/     开发/续训检查点（非正式部署模型）
runs/                    所有新实验输出（Git 忽略）
tests/                   数据与模型合同测试
docs/                    数据契约、迁移说明和算法设计文档
```

## 不可破坏的算法约束

- 原始九轴 IMU 按 50 Hz 连续保存，训练时再生成窗口。
- 传感器和视频统一到时间轴，标签保留事件起止时刻。
- 训练、验证、测试按 `cow_id` 组织，不以重叠滑窗数量充当独立样本数。
- 共享时序编码器输出姿态、行走和多事件头；站卧状态由事件与状态机共同确定。
- 评价报告独立牛和独立事件的 Precision/Recall/F1、误报/牛/24 h 和定位误差。

更详细的数据边界见 [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md)，全量缓存交付见 [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md)，从 `20260816` 到本版的取舍见 [`docs/MIGRATION_20260818.md`](docs/MIGRATION_20260818.md)，GitHub/Claude 使用方式见 [`docs/GITHUB_AND_CLAUDE.md`](docs/GITHUB_AND_CLAUDE.md)。
