# GitHub 维护与 Claude 上传

## 分层原则

- GitHub：代码、配置、测试、小型标签/分割、两个权重和 60 秒演示数据。
- 本地数据：`supervised_cache/samples.csv` 和 `supervised_cache/session_cache/`，不进 Git。
- `runs/`：实验产物，不进 Git；验证后仅把定稿权重和报告提升进仓库。

当前两个权重分别约 6.7 MB 和 8.5 MB，可直接进 Git。[GitHub 官方文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) 说明普通 Git 单文件超过 50 MiB 会警告、超过 100 MiB 会拒绝；未来单个权重超过该阈值时再用 Git LFS，不要现在引入。

## 首次推送

```powershell
git config user.name "<your-name>"
git config user.email "<your-github-email>"
git add .
git commit -m "Initial clean COWMATA algorithm baseline 20260818"
git remote add origin https://github.com/<your-account>/<your-repo>.git
git push -u origin main
```

建议先建私有仓库，确认数据与设备信息可公开后再转为公开。

推荐仓库名：`cowmata-tailring`。本版对应 Python 包版本和 Git 标签 `v0.1.0`；当天的下一版继续在同一仓库提交并升级版本，不再复制整个 GitHub 仓库。

## `dist/` 是否保留

上传 GitHub 后，不保留 `dist/`：GitHub 已提供克隆和 Download ZIP，重复归档容易和源码失去同步。发布文件应由确定的 Git 标签按需生成，不把手工 ZIP 当作源代码版本。

## Claude 网页版

[Claude 官方帮助](https://support.anthropic.com/en/articles/8241126-what-kinds-of-documents-can-i-upload-to-claude-ai) 列出每个文件 30 MB 的限制。优先让 Claude 阅读 GitHub 仓库；若私有仓库无法授权访问，只上传源码、配置、测试、README 和权重清单，不上传二进制权重和训练缓存。临时分析包放在 `runs/share/` 并按需重建，用完删除，不进入 Git。
