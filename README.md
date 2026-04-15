# LiDAR Point Cloud Classification on Databricks

GPU-accelerated 6-class semantic segmentation of aerial LiDAR point clouds using **U-Next** (RandLA-Net backbone + UNet++ decoder), trained and tracked end-to-end on Databricks with MLflow and Unity Catalog.

## Classes

| ID | Label |
|----|-------|
| 2  | Ground |
| 3  | Low Vegetation |
| 4  | Medium Vegetation |
| 5  | High Vegetation |
| 6  | Building |
| 9  | Water |

## Benchmark Results

Trained on a `g4dn.12xlarge` cluster (4x NVIDIA T4) with Databricks ML 15.4 LTS GPU runtime.

| Configuration | Wall Time | Accuracy | Cost/run |
|---|---|---|---|
| 1 node, 1 GPU | 46s | 90.1% | ~$0.006 |
| 1 node, 4 GPU (DDP) | 11s | 89.6% | ~$0.006 |
| Multi-node (2x 4 GPU) | ~30s | ~96.6% | ~$0.036 |

4-GPU DDP delivers a **4.2x speedup at the same cost** as single GPU. See [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for the full analysis.

## Notebooks

| Notebook | Description |
|---|---|
| `01_ingest_and_train.py` | LAS ingestion, Delta table creation, XGBoost baseline |
| `02_visualize.py` | 3D point cloud visualization |
| `03_hyperopt_multigpu.py` | Hyperparameter search (Optuna / Hyperopt) |
| `04_unext_singlegpu.py` | U-Next single-GPU training |
| `04_unext_4gpu.py` | U-Next 4-GPU DDP training |
| `04_unext_multigpu.py` | U-Next multi-node DDP training |
| `04_unext_multinode.py` | Multi-node distributed training |
| `04_unext_experiments.py` | Training experiments and ablations |
| `05_benchmark_comparison.py` | Full benchmark (all configurations, MLflow tracking) |

## Infrastructure

```
Catalog:   chlor
Schema:    lidar_schema
Volume:    chlor.lidar_schema.raw_las        -- LAS tile storage
Table:     chlor.lidar_schema.point_cloud    -- Delta table
Model:     chlor.lidar_schema.lidar_unext    -- Unity Catalog registered model
```

## Getting Started

1. Upload LAS files to the Unity Catalog Volume `chlor.lidar_schema.raw_las`
2. Run `generate_dataset.py` to create synthetic data (optional, for testing)
3. Run notebooks `01` through `05` in order on a Databricks GPU cluster

## Tech Stack

- **Compute:** Databricks ML Runtime 15.4 LTS GPU
- **Model:** U-Next (PyTorch)
- **Distributed Training:** PyTorch DDP via `TorchDistributor`
- **Tracking:** MLflow
- **Storage:** Delta Lake + Unity Catalog
