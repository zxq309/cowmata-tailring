# COWMATA IMU 数据契约

## 母数据

- 九轴 IMU 原始采样率为 50 Hz，连续数据永久保留；不在采集或原始标注阶段固定切窗。
- `features.npy` 每行 13 个通道，顺序和单位由缓存生成器及 `metadata.json` 共同约束。
- 连续段由 `metadata.json/segments` 定义；窗口不得跨越数据缺口。
- 视频和传感器对齐到绝对时间轴，事件标注保留起止时刻。

## 标签

- 持续状态：`STANDING` / `LYING` / `WALKING`；历史 `FEEDING` 可读取但归入 `UPRIGHT` 辅助语义。
- 状态转移：`STANDING_UP` / `LYING_DOWN`。
- 尾部与排泄事件：`URINATION` / `DEFECATION` / `TAIL_RAISED` / `TAIL_WAGGING`。
- 事件可与持续状态重叠，不强制收缩为单一互斥 Softmax 类别。

## 分割与统计

- 正式泛化评价按 `cow_id` 划分，同一头牛不能跨训练集与测试集。
- 归一化统计、阈值选择和早停不得使用测试牛。
- 事件样本量按独立牛、独立事件和困难负样本统计；重叠窗口数不代表独立样本。

## 目录映射

唯一项目数据根为 `datasets/cowmata_imu/`，机器可读路径集中在 `configs/dataset.yaml`。数据不进入 Git，但本地工作基线保留必要缓存以保证算法可直接验证。
