# External Project Reference Radar

Last verified: 2026-08-18.

This watchlist is for model design, benchmarking, engineering practice, and experiment ideas. A link here does not make the project a dependency and does not authorize code copying. Before adopting an implementation, review its license, data assumptions, maintenance status, input shape, leakage risk, and reproducibility on COWMATA's cow-level splits.

## Priority watchlist

| Project | Why it matters to COWMATA | What to watch |
|---|---|---|
| [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library) | Unified implementations for classification, anomaly detection, imputation, and forecasting | New classification baselines, benchmark caveats, general time-series architectures |
| [timeseriesAI/tsai](https://github.com/timeseriesAI/tsai) | Broad PyTorch/fastai time-series classification library | InceptionTime/ResNet/TCN baselines, augmentation, training recipes, interpretability |
| [aeon-toolkit/aeon](https://github.com/aeon-toolkit/aeon) | Actively maintained time-series ML and deep-learning toolkit | Multivariate classification benchmarks, evaluation utilities, dataset interfaces |
| [sktime/sktime](https://github.com/sktime/sktime) | Unified time-series framework with mature estimator conventions | Reproducible pipelines, model selection, evaluation design |
| [tslearn-team/tslearn](https://github.com/tslearn-team/tslearn) | Classical time-series learning and similarity methods | DTW-based retrieval, clustering, prototype analysis, small-data baselines |

## Temporal convolution and event segmentation

| Project | Relevance | Caution |
|---|---|---|
| [paul-krug/pytorch-tcn](https://github.com/paul-krug/pytorch-tcn) | Current TCN dependency; causal/non-causal and streaming temporal convolution | Verify receptive field, padding, latency, and version compatibility |
| [yabufarha/ms-tcn](https://github.com/yabufarha/ms-tcn) | Canonical multi-stage temporal action segmentation | Older environment; adapt ideas rather than importing the training stack |
| [dipika-singhania/C2F-TCN](https://github.com/dipika-singhania/C2F-TCN) | Multi-resolution and semi-supervised temporal action segmentation | Video-action assumptions differ from continuous IMU events |
| [ChinaYi/ASFormer](https://github.com/ChinaYi/ASFormer) | Transformer architecture for temporal action segmentation | Benchmark boundary quality and computational cost on 50 Hz sensor streams |

## Representation learning and wearables

| Project | Relevance | Caution |
|---|---|---|
| [zhihanyue/ts2vec](https://github.com/zhihanyue/ts2vec) | General unsupervised time-series representations | Require a clear downstream gain over supervised baselines before restoring an SSL branch |
| [emadeldeen24/TS-TCC](https://github.com/emadeldeen24/TS-TCC) | Temporal/contextual contrastive representation learning | Reproduce with cow-independent pretraining and evaluation boundaries |
| [OxWearables/ssl-wearables](https://github.com/OxWearables/ssl-wearables) | Large-scale self-supervised wearable accelerometer learning | Human-wearable domain and channel layout differ from cattle tail sensing |

## Rare-event and anomaly candidate mining

| Project | Relevance | Caution |
|---|---|---|
| [sintel-dev/Orion](https://github.com/sintel-dev/Orion) | Unsupervised anomaly pipelines for ranking rare temporal patterns | Anomaly scores are review candidates, not behavior labels |
| [decisionintelligence/TAB](https://github.com/decisionintelligence/TAB) | Benchmarking framework for time-series anomaly detection | Validate event-level metrics and threshold protocols against COWMATA needs |

## Data and experiment engineering

| Project | Relevance | Adoption trigger |
|---|---|---|
| [DVC](https://github.com/iterative/dvc) | Dataset/model version pointers without committing large arrays to Git | Adopt when private object storage and multiple dataset versions become routine |
| [MLflow](https://github.com/mlflow/mlflow) | Experiment, model, and artifact tracking | Adopt when manual `runs/` manifests become a coordination bottleneck |

## COWMATA adoption checklist

Before an external idea enters the main algorithm:

1. define the exact task and current baseline;
2. confirm license compatibility;
3. preserve continuous raw data and train-time windowing;
4. keep train/validation/test separated by cow;
5. compare independent-event metrics and false alarms per cow per 24 hours;
6. report compute, latency, memory, and field-deployment implications;
7. add a small reproducible experiment and tests;
8. retain the change only if it produces a defensible improvement.
