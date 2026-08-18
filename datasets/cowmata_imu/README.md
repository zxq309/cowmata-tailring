# 数据与 Git 边界

本目录的标签、会话表和按牛分割清单进入 Git。以下派生数据仅在本机保留，由 `.gitignore` 排除：

- `supervised_cache/samples.csv`：344,287 个训练中心点及母标签索引，约 60 MB。
- `supervised_cache/session_cache/`：132 个 50 Hz、13 通道的监督会话数组，约 1.23 GB。

它们用于重建特征表、重训 GBDT/TCN 和运行真实数据诊断，不是可随意删除的旧输出。GitHub 克隆后可用 `examples/demo_data/` 完成端到端推理烟雾测试；正式重训需恢复上述本地数据。外置交付与恢复流程见 [`../../docs/DATA_ACCESS.md`](../../docs/DATA_ACCESS.md)。
