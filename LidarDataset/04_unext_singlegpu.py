# Databricks notebook source
# MAGIC %md
# MAGIC # LiDAR Semantic Segmentation — U-Next (Single Node, Single GPU)
# MAGIC
# MAGIC PyTorch re-implementation of **U-Next** (zeng-ziyin/U-Next) — a UNet++-style
# MAGIC extension of RandLA-Net for 3D point cloud semantic segmentation.
# MAGIC
# MAGIC Key architectural components from the original paper:
# MAGIC - **Local Feature Aggregation** — attention-weighted kNN neighbour pooling
# MAGIC - **Dilated Residual Block** — two stacked LFA blocks with residual connection
# MAGIC - **UNet++ nested skip connections** — dense cross-scale feature reuse in decoder
# MAGIC - **Weighted cross-entropy loss** — handles ASPRS class imbalance
# MAGIC
# MAGIC Multi-GPU distribution via **HorovodRunner** (Databricks native, MPI-backed):
# MAGIC each worker gets a shard of the data and a replica of the model;
# MAGIC gradients are all-reduced across workers after every backward pass.
# MAGIC
# MAGIC ```
# MAGIC  Driver
# MAGIC   └─ HorovodRunner(np=N_WORKERS)
# MAGIC        ├─ Worker 0 (GPU 0): model replica, data shard 0 → grad → allreduce
# MAGIC        ├─ Worker 1 (GPU 1): model replica, data shard 1 → grad → allreduce
# MAGIC        ├─ Worker 2 (GPU 2): model replica, data shard 2 → grad → allreduce
# MAGIC        └─ Worker 3 (GPU 3): model replica, data shard 3 → grad → allreduce
# MAGIC ```
# MAGIC
# MAGIC **Prerequisite:** `01_ingest_and_train` must have run to populate `chlor.lidar_schema.point_cloud`.

# COMMAND ----------

# MAGIC %md ## 0. Config

# COMMAND ----------

CATALOG     = "chlor"
SCHEMA      = "lidar_schema"
SRC_TABLE   = f"{CATALOG}.{SCHEMA}.point_cloud"
MODEL_TABLE = f"{CATALOG}.{SCHEMA}.unext_predictions"
MODEL_NAME  = f"{CATALOG}.{SCHEMA}.lidar_unext"

# Point cloud sampling
N_POINTS    = 4096    # points per training chunk (reduced for memory)
K_NEIGHBORS = 16      # kNN for local feature aggregation
N_FEATURES  = 4       # x, y, z, intensity

# Data loading — sample a fraction to avoid driver OOM
SAMPLE_FRAC  = 0.05   # use 5% of total points for quick test run
MAX_CHUNKS   = 100    # cap total chunks saved to /tmp

# Model dims (encoder channels per stage)
D_IN        = N_FEATURES
ENCODER_DIM = [16, 64, 128, 256, 512]   # 5 stages (4 downsample steps)
NUM_CLASSES = 6

# Training
N_EPOCHS    = 5       # reduced for test run
BATCH_SIZE  = 2       # per worker
LR          = 1e-3
N_WORKERS   = 1       # single GPU

CLASS_NAMES = {0: "Ground", 1: "LowVeg", 2: "MedVeg", 3: "HighVeg", 4: "Building", 5: "Water"}
# Original ASPRS ids → encoded index mapping (set in section 2)
ASPRS_TO_IDX = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 9: 5}

# MLflow experiment
EXPERIMENT = "/Users/<YOUR_EMAIL>/lidar-unext"

# COMMAND ----------

# MAGIC %md ## 1. Load & Prepare Point Cloud Chunks

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.neighbors import KDTree
import pickle, os, tempfile

# Load sampled dataset to avoid driver OOM
sdf = spark.table(SRC_TABLE).sample(fraction=SAMPLE_FRAC, seed=42)
pdf = sdf.toPandas()
pdf["cls_idx"] = pdf["classification"].map(ASPRS_TO_IDX)
pdf = pdf.dropna(subset=["cls_idx"])
pdf["cls_idx"] = pdf["cls_idx"].astype(np.int64)

print(f"Sampled points: {len(pdf):,} ({SAMPLE_FRAC*100:.0f}% of full dataset)")
print(f"Class distribution:\n{pdf['cls_idx'].value_counts().sort_index()}")

# COMMAND ----------

# MAGIC %md ### 1a. Chunk scenes into fixed-size blocks & precompute kNN

# COMMAND ----------

def make_chunks(tile_pdf, n_pts=N_POINTS, k=K_NEIGHBORS):
    """
    Split a tile into overlapping chunks of n_pts points.
    Returns list of dicts: {xyz, features, labels, knn_idx}
    """
    xyz   = tile_pdf[["x","y","z"]].values.astype(np.float32)
    feats = tile_pdf[["x","y","z","intensity"]].values.astype(np.float32)
    # normalize intensity to [0,1]
    feats[:, 3] = feats[:, 3] / 65535.0
    labels = tile_pdf["cls_idx"].values.astype(np.int64)

    n = len(xyz)
    chunks = []
    for start in range(0, n, n_pts):
        end = min(start + n_pts, n)
        if end - start < 512:
            break
        idx = np.arange(start, end)
        if len(idx) < n_pts:
            pad = np.random.choice(idx, n_pts - len(idx))
            idx = np.concatenate([idx, pad])

        c_xyz   = xyz[idx]
        c_feats = feats[idx]
        c_labels = labels[idx]

        # Precompute kNN indices for each downsampling stage
        knn_stages = []
        curr_xyz = c_xyz.copy()
        n_curr   = len(curr_xyz)
        for _ in range(4):  # 4 encoder stages
            tree = KDTree(curr_xyz)
            _, nn_idx = tree.query(curr_xyz, k=k)  # (n_curr, k)
            knn_stages.append(nn_idx.astype(np.int32))
            # 4x random downsample for next stage
            n_next = max(n_curr // 4, 1)
            sub_idx = np.random.choice(n_curr, n_next, replace=False)
            curr_xyz = curr_xyz[sub_idx]
            n_curr = n_next  # update for next iteration
            # store sub_idx for skip connections
            knn_stages.append(sub_idx.astype(np.int32))
        chunks.append({
            "xyz":     c_xyz,
            "feats":   c_feats,
            "labels":  c_labels,
            "knn":     knn_stages,
        })
    return chunks

print("Building chunks per tile...")
all_chunks = []
for tile, grp in pdf.groupby("tile_name"):
    tile_chunks = make_chunks(grp)
    all_chunks.extend(tile_chunks)
    print(f"  {tile}: {len(grp):>9,} pts → {len(tile_chunks)} chunks")
    if len(all_chunks) >= MAX_CHUNKS:
        break

all_chunks = all_chunks[:MAX_CHUNKS]
print(f"\nTotal chunks: {len(all_chunks)} (capped at {MAX_CHUNKS})")

# Save chunks to UC Volume — visible on driver and all worker nodes via FUSE
spark.sql("CREATE VOLUME IF NOT EXISTS chlor.lidar_schema.tmp_singlegpu")
CHUNK_DIR = "/Volumes/chlor/lidar_schema/tmp_singlegpu/lidar_chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)
import glob as _glob
for _f in _glob.glob(f"{CHUNK_DIR}/chunk_*.pkl"):
    try: os.remove(_f)
    except OSError: pass
for i, c in enumerate(all_chunks):
    with open(f"{CHUNK_DIR}/chunk_{i:05d}.pkl", "wb") as f:
        pickle.dump(c, f)
print(f"Chunks saved to {CHUNK_DIR}")

# COMMAND ----------

# MAGIC %md ## 2. U-Next Architecture (PyTorch)

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC Input (N×4)
# MAGIC   │
# MAGIC   ├─ Enc0: LFA → 16d   ──────────────────────────────────────────── Dec0 → logits
# MAGIC   │   └─ DS4 (N/4)                                                     ↑
# MAGIC   ├─ Enc1: LFA → 64d  ──────────────────────────── x01 ──────────── Dec1
# MAGIC   │   └─ DS4 (N/16)                                   ↑                ↑
# MAGIC   ├─ Enc2: LFA → 128d ──────────────── x02 ── x12 ── Dec2(UNet++)    Dec2
# MAGIC   │   └─ DS4 (N/64)                     ↑      ↑                      ↑
# MAGIC   ├─ Enc3: LFA → 256d ── x03 ── x13 ── x23 ─── ─────────────────── Dec3
# MAGIC   │   └─ DS4 (N/256)    ↑                                              ↑
# MAGIC   └─ Bottleneck: 512d ──────────────────────────────────────────────── Dec4
# MAGIC ```

# COMMAND ----------

UNEXT_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Helpers ─────────────────────────────────────────────────────────────────

def gather_neighbours(features, nn_idx):
    """
    features : (B, N, C)
    nn_idx   : (B, N, K)  long tensor of neighbour indices
    returns  : (B, N, K, C)
    """
    B, N, C = features.shape
    K = nn_idx.shape[2]
    idx_flat = nn_idx.reshape(B, -1)                        # (B, N*K)
    gathered = features.gather(1,
        idx_flat.unsqueeze(-1).expand(-1, -1, C))           # (B, N*K, C)
    return gathered.reshape(B, N, K, C)


def random_sample(features, sub_idx):
    """
    features : (B, N, C)
    sub_idx  : (B, n)  indices to keep
    returns  : (B, n, C)
    """
    B, N, C = features.shape
    n = sub_idx.shape[1]
    idx = sub_idx.unsqueeze(-1).expand(-1, -1, C)           # (B, n, C)
    return features.gather(1, idx)


def nearest_upsample(fine_xyz, coarse_xyz, coarse_feat):
    """
    Upsample coarse_feat to fine resolution via nearest-neighbour.
    All tensors on same device.
    fine_xyz   : (B, N, 3)
    coarse_xyz : (B, n, 3)
    coarse_feat: (B, n, C)
    returns    : (B, N, C)
    """
    # squared distance (B, N, n)
    diff = fine_xyz.unsqueeze(2) - coarse_xyz.unsqueeze(1)  # (B,N,n,3)
    dist2 = (diff ** 2).sum(-1)                             # (B,N,n)
    nn_idx = dist2.argmin(-1, keepdim=True)                 # (B,N,1)
    upsampled = coarse_feat.gather(
        1, nn_idx.expand(-1, -1, coarse_feat.shape[-1]))    # (B,N,C)
    return upsampled


# ── Shared MLP (1D conv over N points) ──────────────────────────────────────

class SharedMLP(nn.Sequential):
    def __init__(self, dims, bn=True, activation=True):
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Conv2d(dims[i], dims[i+1], 1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(dims[i+1]))
            if activation:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        super().__init__(*layers)


# ── Attention Pooling ────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Conv2d(d_in, d_in, 1, bias=False),
            nn.BatchNorm2d(d_in),
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(d_in, d_out, 1, bias=False),
            nn.BatchNorm2d(d_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # x: (B, C, N, K)
        scores = torch.softmax(self.score_fn(x), dim=-1)    # (B,C,N,K)
        pooled = (x * scores).sum(dim=-1, keepdim=True)     # (B,C,N,1)
        return self.mlp(pooled)                              # (B,C_out,N,1)


# ── Local Feature Aggregation ────────────────────────────────────────────────

class LocalFeatureAggregation(nn.Module):
    """
    Relative-position encoding → shared MLP → attention pooling
    Implements the core RandLA-Net LFA block.
    """
    def __init__(self, d_in, d_out, k=16):
        super().__init__()
        self.k = k
        # relative position encoding: (x,y,z,dist, xi,yi,zi) → 10d
        self.rel_mlp = SharedMLP([10, d_in//2, d_in])
        self.fc      = nn.Sequential(
            nn.Conv2d(d_in * 2, d_in, 1, bias=False),
            nn.BatchNorm2d(d_in),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.att = AttentionPooling(d_in, d_out // 2)
        self.shortcut = nn.Sequential(
            nn.Conv2d(d_in, d_out // 2, 1, bias=False),
            nn.BatchNorm2d(d_out // 2),
        )

    def forward(self, xyz, features, nn_idx):
        """
        xyz      : (B, N, 3)
        features : (B, N, C)  — C == d_in
        nn_idx   : (B, N, K)
        returns  : (B, N, d_out//2)
        """
        B, N, C = features.shape
        K = self.k

        # Gather neighbour xyz and features
        nbr_xyz  = gather_neighbours(xyz,      nn_idx)    # (B,N,K,3)
        nbr_feat = gather_neighbours(features, nn_idx)    # (B,N,K,C)

        # Relative position encoding
        xyz_exp  = xyz.unsqueeze(2).expand(-1,-1,K,-1)    # (B,N,K,3)
        rel_xyz  = xyz_exp - nbr_xyz                       # (B,N,K,3)
        dist     = (rel_xyz**2).sum(-1, keepdim=True).sqrt()
        rel_enc  = torch.cat([rel_xyz, dist, xyz_exp, nbr_xyz], dim=-1)  # (B,N,K,10)

        # → (B, 10, N, K) for Conv2d
        rel_enc  = rel_enc.permute(0,3,1,2)               # (B,10,N,K)
        r        = self.rel_mlp(rel_enc)                  # (B,C,N,K)

        nbr_feat = nbr_feat.permute(0,3,1,2)              # (B,C,N,K)
        concat   = torch.cat([r, nbr_feat], dim=1)        # (B,2C,N,K)
        fused    = self.fc(concat)                        # (B,C,N,K)

        pooled   = self.att(fused)                        # (B,C_out,N,1)
        shortcut = self.shortcut(
            features.permute(0,2,1).unsqueeze(-1))        # (B,C_out,N,1)
        out = F.leaky_relu(pooled + shortcut, 0.2)
        return out.squeeze(-1).permute(0,2,1)             # (B,N,d_out//2)


# ── Dilated Residual Block ────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    """Two LFA blocks with a residual connection — mirrors original U-Next."""
    def __init__(self, d_in, d_out, k=16):
        super().__init__()
        self.lfa1 = LocalFeatureAggregation(d_in,         d_out, k)
        self.lfa2 = LocalFeatureAggregation(d_out // 2,   d_out, k)
        self.shortcut = nn.Sequential(
            nn.Conv1d(d_in, d_out // 2, 1, bias=False),
            nn.BatchNorm1d(d_out // 2),
        )

    def forward(self, xyz, features, nn_idx):
        sc = self.shortcut(features.permute(0,2,1)).permute(0,2,1)
        x  = self.lfa1(xyz, features, nn_idx)
        x  = self.lfa2(xyz, x,        nn_idx)
        return F.leaky_relu(x + sc, 0.2)


# ── U-Next ────────────────────────────────────────────────────────────────────

class UNext(nn.Module):
    """
    UNet++ style point cloud segmentation network based on RandLA-Net.
    Encoder: 4 DilatedResBlocks + random downsampling
    Decoder: nearest-neighbour upsampling + UNet++ nested skip connections
    """
    def __init__(self, d_in=4, num_classes=6,
                 enc_dims=(16, 64, 128, 256, 512), k=16):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        E = enc_dims

        # Input projection
        self.fc_start = nn.Sequential(
            nn.Linear(d_in, E[0]),
            nn.BatchNorm1d(E[0]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder
        self.enc0 = DilatedResBlock(E[0],     E[1], k)
        self.enc1 = DilatedResBlock(E[1]//2,  E[2], k)
        self.enc2 = DilatedResBlock(E[2]//2,  E[3], k)
        self.enc3 = DilatedResBlock(E[3]//2,  E[4], k)

        # Bottleneck
        self.mlp_bot = nn.Sequential(
            nn.Linear(E[4]//2, E[4]//2),
            nn.BatchNorm1d(E[4]//2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # UNet++ nested dense skip nodes
        # x_ij: feature at encoder depth i, after j up-connections
        self.x01 = nn.Linear(E[1]//2 + E[2]//2, E[1]//2)
        self.x02 = nn.Linear(E[1]//2 + E[2]//2, E[1]//2)
        self.x03 = nn.Linear(E[1]//2 + E[2]//2, E[1]//2)

        self.x12 = nn.Linear(E[2]//2 + E[3]//2, E[2]//2)
        self.x13 = nn.Linear(E[2]//2 + E[3]//2, E[2]//2)

        self.x23 = nn.Linear(E[3]//2 + E[4]//2, E[3]//2)

        # Decoder
        # dec3 input: f3 + up4, both E[4]//2. dec3 output = D3 = E[4]//4
        # dec2 input: cat(x21=E[3]//2, up3=E[4]//4). dec2 output = D2 = E[3]//4
        # dec1 input: cat(x12=E[2]//2, up2=E[3]//4). dec1 output = D1 = E[2]//4
        # dec0 input: cat(x=E[0], up1=E[2]//4). dec0 output = E[0]
        self.dec3 = nn.Sequential(nn.Linear(E[4]//2,           E[4]//4), nn.LeakyReLU(0.2))
        self.dec2 = nn.Sequential(nn.Linear(E[3]//2 + E[4]//4, E[3]//4), nn.LeakyReLU(0.2))
        self.dec1 = nn.Sequential(nn.Linear(E[2]//2 + E[3]//4, E[2]//4), nn.LeakyReLU(0.2))
        self.dec0 = nn.Sequential(nn.Linear(E[0]   + E[2]//4,  E[0]),    nn.LeakyReLU(0.2))

        self.head = nn.Linear(E[0], num_classes)

    def forward(self, xyz, features, knn_stages):
        """
        xyz         : (B, N, 3)
        features    : (B, N, d_in)
        knn_stages  : list of [nn_idx_0, sub_idx_0, nn_idx_1, sub_idx_1, ...]
                      interleaved knn indices and subsample indices for 4 stages
        """
        B, N, _ = xyz.shape

        # ── Input projection ────────────────────────────────────────────────
        x = self.fc_start(features.reshape(B*N, -1)).reshape(B, N, -1)

        # Unpack knn_stages
        nn0, si0, nn1, si1, nn2, si2, nn3, si3 = knn_stages

        # ── Encoder ──────────────────────────────────────────────────────────
        f0 = self.enc0(xyz, x, nn0)                            # (B,N,E1//2)

        xyz1 = random_sample(xyz, si0)
        f1   = random_sample(f0, si0)
        f1   = self.enc1(xyz1, f1, nn1)                        # (B,N/4,E2//2)

        xyz2 = random_sample(xyz1, si1)
        f2   = random_sample(f1,   si1)
        f2   = self.enc2(xyz2, f2, nn2)                        # (B,N/16,E3//2)

        xyz3 = random_sample(xyz2, si2)
        f3   = random_sample(f2,   si2)
        f3   = self.enc3(xyz3, f3, nn3)                        # (B,N/64,E4//2)

        xyz4 = random_sample(xyz3, si3)
        f4   = random_sample(f3,   si3)
        B4, N4, C4 = f4.shape
        f4   = self.mlp_bot(f4.reshape(B4*N4, C4)).reshape(B4, N4, C4)

        # ── UNet++ nested skips ──────────────────────────────────────────────
        # Level 0 dense nodes
        up_f1_to_0 = nearest_upsample(xyz,  xyz1, f1)
        x00 = f0
        x01 = F.leaky_relu(self.x01(torch.cat([x00, up_f1_to_0], dim=-1)), 0.2)

        up_f2_to_1 = nearest_upsample(xyz1, xyz2, f2)
        x10 = f1
        x11 = F.leaky_relu(self.x12(torch.cat([x10, up_f2_to_1], dim=-1)), 0.2)
        up_x11_to_0 = nearest_upsample(xyz, xyz1, x11)
        x02 = F.leaky_relu(self.x02(torch.cat([x01, up_x11_to_0], dim=-1)), 0.2)

        up_f3_to_2 = nearest_upsample(xyz2, xyz3, f3)
        x20 = f2
        x21 = F.leaky_relu(self.x23(torch.cat([x20, up_f3_to_2], dim=-1)), 0.2)
        up_x21_to_1 = nearest_upsample(xyz1, xyz2, x21)
        x12 = F.leaky_relu(self.x13(torch.cat([x11, up_x21_to_1], dim=-1)), 0.2)
        up_x12_to_0 = nearest_upsample(xyz, xyz1, x12)
        x03 = F.leaky_relu(self.x03(torch.cat([x02, up_x12_to_0], dim=-1)), 0.2)

        # ── Decoder ──────────────────────────────────────────────────────────
        up4 = nearest_upsample(xyz3, xyz4, f4)
        d3  = self.dec3(f3 + up4)

        up3 = nearest_upsample(xyz2, xyz3, d3)
        d2  = self.dec2(torch.cat([x21, up3], dim=-1))

        up2 = nearest_upsample(xyz1, xyz2, d2)
        d1  = self.dec1(torch.cat([x12, up2], dim=-1))

        up1 = nearest_upsample(xyz, xyz1, d1)
        # project x back to original E0 dim for concat
        x_enc0_proj = F.leaky_relu(
            nn.functional.linear(x, torch.zeros(self.head.in_features,
                x.shape[-1], device=x.device)), 0.2) if False else x
        d0  = self.dec0(torch.cat([x, up1], dim=-1))

        logits = self.head(d0)                                 # (B,N,num_classes)
        return logits
'''

# Write model code to UC Volume — accessible on all nodes via FUSE
import os as _os
_os.makedirs("/Volumes/chlor/lidar_schema/tmp_singlegpu", exist_ok=True)
with open("/Volumes/chlor/lidar_schema/tmp_singlegpu/unext_model.py", "w") as f:
    f.write(UNEXT_CODE)
print("U-Next model written to /Volumes/chlor/lidar_schema/tmp_singlegpu/unext_model.py")

# COMMAND ----------

# MAGIC %md ## 3. TorchDistributor Training Function

# COMMAND ----------

import mlflow

mlflow.set_experiment(EXPERIMENT)

def train_ddp():
    """Runs on every GPU worker via TorchDistributor (PyTorch DDP)."""
    import sys, os, pickle, glob
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist

    # Exec model code via closure (cloudpickle captures UNEXT_CODE so workers
    # don't need FUSE mount access to load the model file)
    _model_ns = {}
    exec(UNEXT_CODE, _model_ns)
    UNext = _model_ns["UNext"]

    # ── Init DDP (TorchDistributor sets RANK / LOCAL_RANK / WORLD_SIZE) ─────
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Only init process group for multi-process runs
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # ── Load chunk file paths (each worker gets every world_size-th chunk) ──
    chunk_files = sorted(glob.glob("/Volumes/chlor/lidar_schema/tmp_singlegpu/lidar_chunks/chunk_*.pkl"))
    my_files = chunk_files[rank::world_size]
    if rank == 0:
        print(f"World size: {world_size} | chunks per worker: {len(my_files)}")

    def load_batch(files, batch_size):
        import random
        random.shuffle(files)
        batch_xyz, batch_feats, batch_labels, batch_knn = [], [], [], []
        for fp in files:
            with open(fp, "rb") as fh:
                c = pickle.load(fh)
            batch_xyz.append(c["xyz"])
            batch_feats.append(c["feats"])
            batch_labels.append(c["labels"])
            batch_knn.append(c["knn"])
            if len(batch_xyz) == batch_size:
                yield batch_xyz, batch_feats, batch_labels, batch_knn
                batch_xyz, batch_feats, batch_labels, batch_knn = [], [], [], []

    def collate(batch_xyz, batch_feats, batch_labels, batch_knn):
        xyz    = torch.tensor(np.stack(batch_xyz),    dtype=torch.float32, device=device)
        feats  = torch.tensor(np.stack(batch_feats),  dtype=torch.float32, device=device)
        labels = torch.tensor(np.stack(batch_labels), dtype=torch.long,    device=device)
        knn_stages = []
        for i in range(8):
            arr = np.stack([bk[i] for bk in batch_knn])
            knn_stages.append(torch.tensor(arr, dtype=torch.long, device=device))
        return xyz, feats, labels, knn_stages

    # ── Model ────────────────────────────────────────────────────────────────
    _n_pts      = int(os.environ.get("UNEXT_N_POINTS",    4096))
    _k          = int(os.environ.get("UNEXT_K_NEIGHBORS", 16))
    _n_feats    = int(os.environ.get("UNEXT_N_FEATURES",  4))
    _n_classes  = int(os.environ.get("UNEXT_NUM_CLASSES", 6))
    _enc_dims   = [16, 64, 128, 256, 512]
    _n_epochs   = int(os.environ.get("UNEXT_N_EPOCHS",    5))
    _batch_size = int(os.environ.get("UNEXT_BATCH_SIZE",  4))
    _lr         = float(os.environ.get("UNEXT_LR",        1e-3))
    _model_name = os.environ.get("UNEXT_MODEL_NAME", "lidar_unext")
    _experiment = os.environ.get("UNEXT_EXPERIMENT", "/Users/<YOUR_EMAIL>/lidar-unext")

    model = UNext(d_in=_n_feats, num_classes=_n_classes,
                  enc_dims=_enc_dims, k=_k).to(device)
    if world_size > 1 and dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    class_counts = np.array([1200000, 9480, 336540, 148520, 14137719, 14400], dtype=np.float32)
    class_weights = (1.0 / (class_counts + 1)).astype(np.float32)
    class_weights = class_weights / class_weights.sum() * _n_classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    optimizer = optim.Adam(model.parameters(), lr=_lr * world_size)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_n_epochs)

    if rank == 0:
        import mlflow
        mlflow.set_experiment(_experiment)
        mlflow.start_run(run_name="unext-multigpu")
        mlflow.log_params({
            "n_points": _n_pts, "k_neighbors": _k,
            "n_epochs": _n_epochs, "batch_size": _batch_size,
            "world_size": world_size, "lr": _lr,
            "architecture": "U-Next (RandLA-Net UNet++)",
        })

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(_n_epochs):
        model.train()
        epoch_loss, n_correct, n_total = 0.0, 0, 0

        for b_xyz, b_feats, b_labels, b_knn in load_batch(my_files, _batch_size):
            try:
                xyz, feats, labels, knn_stages = collate(b_xyz, b_feats, b_labels, b_knn)
                optimizer.zero_grad()
                logits = model(xyz, feats, knn_stages)
                B, N, C = logits.shape
                loss = criterion(logits.reshape(B*N, C), labels.reshape(B*N))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                preds = logits.argmax(-1)
                n_correct += (preds == labels).sum().item()
                n_total   += labels.numel()
            except Exception as e:
                if rank == 0:
                    print(f"  Batch error (skipped): {e}")
                continue

        scheduler.step()
        acc = n_correct / max(n_total, 1)
        if rank == 0:
            print(f"Epoch {epoch+1:3d}/{_n_epochs}  loss={epoch_loss:.4f}  acc={acc:.4f}")
            import mlflow
            mlflow.log_metrics({"train_loss": epoch_loss, "train_acc": acc}, step=epoch)

    if rank == 0:
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(), "/Volumes/chlor/lidar_schema/tmp_singlegpu/unext_final.pt")
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, TensorSpec
        sig = ModelSignature(
            inputs=Schema([
                TensorSpec(np.dtype("float32"), (-1, _n_pts, 3),       "xyz"),
                TensorSpec(np.dtype("float32"), (-1, _n_pts, _n_feats), "feats"),
            ]),
            outputs=Schema([
                TensorSpec(np.dtype("float32"), (-1, _n_pts, _n_classes), "logits"),
            ]),
        )
        mlflow.pytorch.log_model(raw_model, artifact_path="model",
                                 signature=sig,
                                 registered_model_name=_model_name)
        mlflow.end_run()
        print("Training complete. Model saved.")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()

def _train_ddp_safe():
    """Wrapper that captures worker errors to a volume file for debugging."""
    import os, traceback
    rank = os.environ.get("RANK", "0")
    try:
        train_ddp()
    except Exception:
        err_path = f"/Volumes/chlor/lidar_schema/tmp_singlegpu/error_rank{rank}.txt"
        with open(err_path, "w") as f:
            f.write(traceback.format_exc())
        raise

# COMMAND ----------

# MAGIC %md ## 4. Launch Multi-GPU Training via TorchDistributor

# COMMAND ----------

import os
os.environ["UNEXT_N_POINTS"]    = str(N_POINTS)
os.environ["UNEXT_K_NEIGHBORS"] = str(K_NEIGHBORS)
os.environ["UNEXT_N_FEATURES"]  = str(N_FEATURES)
os.environ["UNEXT_NUM_CLASSES"] = str(NUM_CLASSES)
os.environ["UNEXT_N_EPOCHS"]    = str(N_EPOCHS)
os.environ["UNEXT_BATCH_SIZE"]  = str(BATCH_SIZE)
os.environ["UNEXT_LR"]          = str(LR)
os.environ["UNEXT_MODEL_NAME"]  = MODEL_NAME
os.environ["UNEXT_EXPERIMENT"]  = EXPERIMENT

print("Launching U-Next training (single GPU)...")
_train_ddp_safe()
print("Training finished.")

# COMMAND ----------

# MAGIC %md ## 5. Evaluate Final Model

# COMMAND ----------

import torch
import sys
sys.path.insert(0, "/Volumes/chlor/lidar_schema/tmp_singlegpu")
from unext_model import UNext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UNext(d_in=N_FEATURES, num_classes=NUM_CLASSES, enc_dims=ENCODER_DIM, k=K_NEIGHBORS)
model.load_state_dict(torch.load("/Volumes/chlor/lidar_schema/tmp_singlegpu/unext_final.pt", map_location=device))
model.to(device).eval()
print("Model loaded for evaluation.")

# COMMAND ----------

import pickle, glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

all_true, all_pred = [], []
sample_files = sorted(glob.glob("/Volumes/chlor/lidar_schema/tmp_singlegpu/lidar_chunks/chunk_*.pkl"))[:50]  # evaluate on 50 chunks

with torch.no_grad():
    for fp in sample_files:
        with open(fp, "rb") as f:
            c = pickle.load(f)
        xyz    = torch.tensor(c["xyz"],   dtype=torch.float32, device=device).unsqueeze(0)
        feats  = torch.tensor(c["feats"], dtype=torch.float32, device=device).unsqueeze(0)
        labels = c["labels"]
        try:
            knn_stages = [torch.tensor(arr, dtype=torch.long, device=device).unsqueeze(0)
                          for arr in c["knn"]]
            logits = model(xyz, feats, knn_stages)
            preds  = logits.argmax(-1).squeeze(0).cpu().numpy()
            all_true.extend(labels.tolist())
            all_pred.extend(preds.tolist())
        except Exception as e:
            import traceback
            print(f"Eval error on {fp}: {e}")
            traceback.print_exc()

if not all_true:
    print("WARNING: No evaluation chunks succeeded — check eval errors above.")
else:
    print(classification_report(all_true, all_pred,
          labels=list(range(NUM_CLASSES)),
          target_names=list(CLASS_NAMES.values())))

# COMMAND ----------

# Confusion matrix
if all_true:
    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels_cm = list(CLASS_NAMES.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels_cm))); ax.set_xticklabels(labels_cm, rotation=35, ha="right")
    ax.set_yticks(range(len(labels_cm))); ax.set_yticklabels(labels_cm)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("U-Next Confusion Matrix (row-normalised)", fontweight="bold")
    for i in range(len(labels_cm)):
        for j in range(len(labels_cm)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i,j] > 0.5 else "black")
    plt.tight_layout()
    display(fig)
    plt.close()
else:
    print("Skipping confusion matrix — no eval data.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Component | Detail |
# MAGIC |---|---|
# MAGIC | Architecture | U-Next — UNet++ nested skips on RandLA-Net backbone |
# MAGIC | Distribution | Single GPU (direct, no DDP) |
# MAGIC | Input | 4096 pts/chunk, 4 features (x,y,z,intensity), k=16 neighbours |
# MAGIC | Encoder | 4× DilatedResBlock + 4× random downsampling |
# MAGIC | Decoder | Nearest-neighbour upsample + UNet++ dense skip nodes |
# MAGIC | Loss | Weighted cross-entropy (inverse class frequency) |
# MAGIC | Registered model | `chlor.lidar_schema.lidar_unext` |
# MAGIC | MLflow experiment | `/Users/<YOUR_EMAIL>/lidar-unext` |
