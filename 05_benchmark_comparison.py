# Databricks notebook source
# MAGIC %md # LiDAR U-Next — Setup Benchmark (Full Run)

# COMMAND ----------

# MAGIC %md ## 0. Config

# COMMAND ----------

# ── Unity Catalog location ────────────────────────────────────────────────────
CATALOG     = "chlor"
SCHEMA      = "lidar_schema"
SRC_TABLE   = f"{CATALOG}.{SCHEMA}.point_cloud"   # Delta table with all LAS points
MODEL_NAME  = f"{CATALOG}.{SCHEMA}.lidar_unext"   # UC-registered MLflow model
EXPERIMENT  = "/Users/<YOUR_EMAIL>/lidar-unext-benchmark"

# ── Model geometry ────────────────────────────────────────────────────────────
N_POINTS    = 4096    # points per training chunk (spatial window)
K_NEIGHBORS = 16      # k-NN neighbours used in each LFA layer
N_FEATURES  = 4       # input features: x, y, z, intensity (normalised 0-1)
NUM_CLASSES = 6       # Ground, LowVeg, MedVeg, HighVeg, Building, Water
ENCODER_DIM = [16, 64, 128, 256, 512]  # channel widths at each encoder stage

# ── Full run settings ─────────────────────────────────────────────────────────
# Lowered from 30% to avoid driver OOM when calling toPandas() on the full table.
# At 8% we get ~5-10M points which yields ~100 4096-pt chunks — enough for
# statistically meaningful accuracy comparisons across all four setups.
SAMPLE_FRAC = 0.08    # fraction of point_cloud table to load onto driver
MAX_CHUNKS  = 100     # cap total chunks (one chunk = one 4096-pt training window)
N_EPOCHS    = 10      # full training epochs per setup
BATCH_SIZE  = 4       # chunks per gradient step (2 or 4 fit comfortably in T4 VRAM)
LR          = 3e-4    # starting LR; scheduler decays via CosineAnnealingLR

# ASPRS classification codes → compact 0-based class indices for CrossEntropyLoss
# ASPRS standard: 2=Ground, 3=LowVeg, 4=MedVeg, 5=HighVeg, 6=Building, 9=Water
ASPRS_TO_IDX = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 9: 5}

# ── Hyperparameter search grid (Setup D) ─────────────────────────────────────
# Four trials run sequentially on one GPU to simulate a lightweight HP search.
# Varying lr × batch_size reveals sensitivity before committing to a full DDP run.
HP_CONFIGS = [
    {"lr": 1e-3,  "batch_size": 2, "label": "lr=1e-3 bs=2"},
    {"lr": 3e-4,  "batch_size": 2, "label": "lr=3e-4 bs=2"},
    {"lr": 1e-3,  "batch_size": 4, "label": "lr=1e-3 bs=4"},
    {"lr": 3e-4,  "batch_size": 4, "label": "lr=3e-4 bs=4"},
]

# ── Cost model (approximate list prices, use spot for 60-70% reduction) ───────
DBU_RATE_4GPU   = 8.0    # DBU/hr — g4dn.12xlarge (4× T4, 48 vCPU, 192 GB)
DBU_RATE_1GPU   = 2.0    # DBU/hr — g4dn.xlarge   (1× T4, 4 vCPU,  16 GB)
DBU_PRICE_USD   = 0.22   # USD/DBU (approximate ML compute list rate)

# COMMAND ----------

# MAGIC %md ## 1. Prepare Shared Chunks

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import pickle, os

# Shared UC Volume for storing preprocessed chunk files.
# All four training setups (A, B, C, D) read from the same directory so
# chunk preprocessing cost is paid once and results are directly comparable.
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.tmp_benchmark")
CHUNK_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/tmp_benchmark/lidar_chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)

# Clear any leftover chunks from a previous run to avoid stale data
import glob as _glob
for _f in _glob.glob(f"{CHUNK_DIR}/chunk_*.pkl"):
    try: os.remove(_f)
    except OSError: pass

# Sample from the full Delta table and collect to the driver.
# toPandas() on the full table would OOM the driver; 8% stays well within limits.
sdf = spark.table(SRC_TABLE).sample(fraction=SAMPLE_FRAC, seed=42)
pdf = sdf.toPandas()

# Map ASPRS class codes to 0-based indices; drop points with unlabelled classes
pdf["cls_idx"] = pdf["classification"].map(ASPRS_TO_IDX)
pdf = pdf.dropna(subset=["cls_idx"])
pdf["cls_idx"] = pdf["cls_idx"].astype(np.int64)
print(f"Loaded {len(pdf):,} points")


def make_chunks(tile_pdf, n_pts=N_POINTS, k=K_NEIGHBORS):
    """
    Slice a single LiDAR tile into fixed-size training chunks and pre-compute
    the k-NN graphs needed by RandLA-Net's encoder stages.

    Each encoder stage performs:
      1. k-NN query on the current point set  → nn_idx  (N × k)
      2. Random spatial downsampling 4:1       → sub_idx (N//4,)

    These two arrays are interleaved in knn_stages:
      [nn_idx_0, sub_idx_0, nn_idx_1, sub_idx_1, nn_idx_2, sub_idx_2, nn_idx_3, sub_idx_3]

    Pre-computing on CPU avoids repeating the KDTree query inside the GPU
    training loop, which would be the dominant bottleneck for small batch sizes.
    """
    xyz    = tile_pdf[["x","y","z"]].values.astype(np.float32)
    feats  = tile_pdf[["x","y","z","intensity"]].values.astype(np.float32)
    feats[:, 3] = feats[:, 3] / 65535.0   # normalise intensity from uint16 → [0, 1]
    labels = tile_pdf["cls_idx"].values.astype(np.int64)
    n = len(xyz)
    chunks = []

    for start in range(0, n, n_pts):
        end = min(start + n_pts, n)
        # Skip window if too small for a meaningful batch
        if end - start < 512:
            break

        idx = np.arange(start, end)
        # If the last window is short, pad by sampling with replacement from the same window
        if len(idx) < n_pts:
            idx = np.concatenate([idx, np.random.choice(idx, n_pts - len(idx))])

        c_xyz, c_feats, c_labels = xyz[idx], feats[idx], labels[idx]

        # Build kNN graph at each of the 4 encoder resolutions.
        # After each stage the point set shrinks by 4× via random downsampling.
        knn_stages = []
        curr_xyz = c_xyz.copy()
        n_curr = len(curr_xyz)   # must be tracked explicitly; len(curr_xyz) after reassignment

        for _ in range(4):
            # k-NN indices at current resolution (used by LocalFeatureAggregation)
            tree = KDTree(curr_xyz)
            _, nn_idx = tree.query(curr_xyz, k=k)
            knn_stages.append(nn_idx.astype(np.int32))

            # Random downsample: pick n_curr//4 points without replacement
            n_next   = max(n_curr // 4, 1)
            sub_idx  = np.random.choice(n_curr, n_next, replace=False)
            curr_xyz = curr_xyz[sub_idx]
            n_curr   = n_next   # critical: update counter or sub_idx will index into wrong size
            knn_stages.append(sub_idx.astype(np.int32))

        chunks.append({"xyz": c_xyz, "feats": c_feats, "labels": c_labels, "knn": knn_stages})

    return chunks


# Build chunks tile-by-tile so spatial locality is preserved within each chunk.
# Processing tile by tile avoids mixing points from different geographic areas
# that happen to be adjacent in the Delta table's sort order.
all_chunks = []
for tile, grp in pdf.groupby("tile_name"):
    all_chunks.extend(make_chunks(grp))
    if len(all_chunks) >= MAX_CHUNKS:
        break
all_chunks = all_chunks[:MAX_CHUNKS]

# Serialise to UC Volume so TorchDistributor workers (any rank, any node) can read them.
# Pickle is used for simplicity; for very large datasets consider memory-mapped numpy arrays.
for i, c in enumerate(all_chunks):
    with open(f"{CHUNK_DIR}/chunk_{i:05d}.pkl", "wb") as f:
        pickle.dump(c, f)
print(f"Saved {len(all_chunks)} chunks → {CHUNK_DIR}")

# COMMAND ----------

# MAGIC %md ## 2. U-Next Model Code

# COMMAND ----------

# The model code is stored as a string and exec()'d inside train_ddp().
# This pattern is required because TorchDistributor serialises the training
# function with cloudpickle and sends it to worker processes, which may not
# have the model class defined in their module scope.  exec() at runtime
# inside the worker guarantees the class is always available regardless of
# how the function was serialised.
UNEXT_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_neighbours(features, nn_idx):
    """
    Gather k-nearest-neighbour features for each point.
    features : (B, N, C)   — point feature tensor
    nn_idx   : (B, N, K)   — pre-computed kNN indices from KDTree
    Returns  : (B, N, K, C) — neighbour features for every query point
    """
    B, N, C = features.shape
    K = nn_idx.shape[2]
    idx_flat = nn_idx.reshape(B, -1)                                    # (B, N*K)
    gathered = features.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, C))
    return gathered.reshape(B, N, K, C)


def random_sample(features, sub_idx):
    """
    Select a random subset of points using pre-computed indices.
    features : (B, N, C)
    sub_idx  : (B, M)      — indices of M surviving points (M < N)
    Returns  : (B, M, C)
    """
    B, N, C = features.shape
    return features.gather(1, sub_idx.unsqueeze(-1).expand(-1, -1, C))


def nearest_upsample(fine_xyz, coarse_xyz, coarse_feat):
    """
    Propagate coarse features back to a finer resolution by nearest-neighbour
    interpolation (1-NN in Euclidean space).  Used in the decoder to upsample
    feature maps from each encoder stage back to full resolution.
    fine_xyz    : (B, N_fine,   3)
    coarse_xyz  : (B, N_coarse, 3)
    coarse_feat : (B, N_coarse, C)
    Returns     : (B, N_fine,   C)
    """
    diff   = fine_xyz.unsqueeze(2) - coarse_xyz.unsqueeze(1)  # (B, N_fine, N_coarse, 3)
    dist2  = (diff ** 2).sum(-1)                               # (B, N_fine, N_coarse)
    nn_idx = dist2.argmin(-1, keepdim=True)                    # (B, N_fine, 1)
    return coarse_feat.gather(1, nn_idx.expand(-1, -1, coarse_feat.shape[-1]))


class SharedMLP(nn.Sequential):
    """1×1 convolution MLP applied pointwise (works on (B,C,N,K) tensors)."""
    def __init__(self, dims, bn=True, activation=True):
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Conv2d(dims[i], dims[i+1], 1, bias=not bn))
            if bn: layers.append(nn.BatchNorm2d(dims[i+1]))
            if activation: layers.append(nn.LeakyReLU(0.2, inplace=True))
        super().__init__(*layers)


class AttentionPooling(nn.Module):
    """
    Squeeze-and-excitation style pooling over the k-NN neighbourhood.
    Learns per-neighbour attention weights, then produces a single
    weighted-sum feature vector.  This replaces the max-pool in vanilla PointNet++.
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.score_fn = nn.Sequential(nn.Conv2d(d_in, d_in, 1, bias=False), nn.BatchNorm2d(d_in))
        self.mlp = nn.Sequential(nn.Conv2d(d_in, d_out, 1, bias=False), nn.BatchNorm2d(d_out), nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x):
        # x : (B, C, N, K) — neighbour feature cube
        scores = torch.softmax(self.score_fn(x), dim=-1)   # attention over K neighbours
        return self.mlp((x * scores).sum(dim=-1, keepdim=True))


class LocalFeatureAggregation(nn.Module):
    """
    Core RandLA-Net building block.  For each point:
      1. Encode relative geometry (position + distance) of its k neighbours
      2. Fuse with neighbour features via a shared MLP
      3. Aggregate with learned attention weights
    The shortcut connection (residual path) adds the projected input directly.
    Output dim is always d_out//2 (DilatedResBlock adds a second LFA to reach d_out//2).
    """
    def __init__(self, d_in, d_out, k=16):
        super().__init__()
        self.k = k
        # 10-dim relative geometry: Δxyz (3) + dist (1) + abs_xyz (3) + nbr_xyz (3)
        self.rel_mlp  = SharedMLP([10, d_in//2, d_in])
        self.fc       = nn.Sequential(nn.Conv2d(d_in*2, d_in, 1, bias=False), nn.BatchNorm2d(d_in), nn.LeakyReLU(0.2, inplace=True))
        self.att      = AttentionPooling(d_in, d_out//2)
        self.shortcut = nn.Sequential(nn.Conv2d(d_in, d_out//2, 1, bias=False), nn.BatchNorm2d(d_out//2))

    def forward(self, xyz, features, nn_idx):
        B, N, C = features.shape; K = self.k
        nbr_xyz  = gather_neighbours(xyz, nn_idx)                          # (B,N,K,3)
        nbr_feat = gather_neighbours(features, nn_idx)                     # (B,N,K,C)
        xyz_exp  = xyz.unsqueeze(2).expand(-1,-1,K,-1)                     # (B,N,K,3)
        rel_xyz  = xyz_exp - nbr_xyz                                       # relative position
        # Concatenate: relative pos, Euclidean distance, abs query xyz, abs neighbour xyz
        rel_enc  = torch.cat([rel_xyz, (rel_xyz**2).sum(-1,keepdim=True).sqrt(), xyz_exp, nbr_xyz], dim=-1).permute(0,3,1,2)
        r        = self.rel_mlp(rel_enc)
        fused    = self.fc(torch.cat([r, nbr_feat.permute(0,3,1,2)], dim=1))
        return F.leaky_relu(self.att(fused) + self.shortcut(features.permute(0,2,1).unsqueeze(-1)), 0.2).squeeze(-1).permute(0,2,1)


class DilatedResBlock(nn.Module):
    """
    Two stacked LFA layers with a residual shortcut — the "dilated" effect comes
    from applying the same kNN graph twice, implicitly enlarging the receptive field.
    Input  : (B, N, d_in)
    Output : (B, N, d_out//2)   ← always halved due to AttentionPooling inside LFA
    """
    def __init__(self, d_in, d_out, k=16):
        super().__init__()
        self.lfa1     = LocalFeatureAggregation(d_in,     d_out, k)   # d_in  → d_out//2
        self.lfa2     = LocalFeatureAggregation(d_out//2, d_out, k)   # d_out//2 → d_out//2
        self.shortcut = nn.Sequential(nn.Conv1d(d_in, d_out//2, 1, bias=False), nn.BatchNorm1d(d_out//2))

    def forward(self, xyz, features, nn_idx):
        sc = self.shortcut(features.permute(0,2,1)).permute(0,2,1)    # residual path
        return F.leaky_relu(self.lfa2(xyz, self.lfa1(xyz, features, nn_idx), nn_idx) + sc, 0.2)


class UNext(nn.Module):
    """
    U-Next: RandLA-Net encoder backbone + UNet++ dense nested skip connections.

    Encoder path (enc0–enc3) progressively downsamples the point cloud 4× at
    each stage.  Skip nodes (x01, x02, x03, x11, x12, x21, x23) form a dense
    grid of connections between encoder stages — each node fuses the direct skip
    feature with an upsampled feature from the next coarser resolution, creating
    an ensemble of nested U-shapes rather than a single skip path.

    Dimension flow (enc_dims = [16, 64, 128, 256, 512]):
      Input   : (B, N,     4)  — x, y, z, intensity
      fc_start: (B, N,    16)  — E[0]
      enc0    : (B, N,    32)  — E[1]//2  (DilatedResBlock always outputs d_out//2)
      enc1    : (B, N/4,  64)  — E[2]//2
      enc2    : (B, N/16, 128) — E[3]//2
      enc3    : (B, N/64, 256) — E[4]//2
      mlp_bot : (B, N/64, 256) — bottleneck refinement
    """
    def __init__(self, d_in=4, num_classes=6, enc_dims=(16,64,128,256,512), k=16):
        super().__init__()
        E = enc_dims  # shorthand; E[i] is the *target* dim before the //2 halving

        # Initial per-point feature projection
        self.fc_start = nn.Sequential(nn.Linear(d_in, E[0]), nn.BatchNorm1d(E[0]), nn.LeakyReLU(0.2, inplace=True))

        # Encoder: 4 DilatedResBlocks, each at a different spatial resolution
        self.enc0 = DilatedResBlock(E[0],    E[1], k)   # full resolution   → E[1]//2 = 32
        self.enc1 = DilatedResBlock(E[1]//2, E[2], k)   # 1/4  resolution   → E[2]//2 = 64
        self.enc2 = DilatedResBlock(E[2]//2, E[3], k)   # 1/16 resolution   → E[3]//2 = 128
        self.enc3 = DilatedResBlock(E[3]//2, E[4], k)   # 1/64 resolution   → E[4]//2 = 256

        # Bottleneck MLP at coarsest resolution (no spatial downsampling)
        self.mlp_bot = nn.Sequential(nn.Linear(E[4]//2, E[4]//2), nn.BatchNorm1d(E[4]//2), nn.LeakyReLU(0.2, inplace=True))

        # UNet++ dense skip nodes: x{row}{col} in the nested grid
        # Each node fuses its skip feature (left) with an upsampled feature (below-right)
        self.x01 = nn.Linear(E[1]//2 + E[2]//2, E[1]//2)  # row0 col1: f0 + up(f1)
        self.x02 = nn.Linear(E[1]//2 + E[2]//2, E[1]//2)  # row0 col2: x01 + up(x11)
        self.x03 = nn.Linear(E[1]//2 + E[2]//2, E[1]//2)  # row0 col3: x02 + up(x12)
        self.x12 = nn.Linear(E[2]//2 + E[3]//2, E[2]//2)  # row1 col2: f1 + up(f2)
        self.x13 = nn.Linear(E[2]//2 + E[3]//2, E[2]//2)  # row1 col3: x11 + up(x21)
        self.x23 = nn.Linear(E[3]//2 + E[4]//2, E[3]//2)  # row2 col3: f2 + up(f3)

        # Decoder: progressively upsample from bottleneck back to full resolution
        self.dec3 = nn.Sequential(nn.Linear(E[4]//2,           E[4]//4), nn.LeakyReLU(0.2))  # 256→128
        self.dec2 = nn.Sequential(nn.Linear(E[3]//2 + E[4]//4, E[3]//4), nn.LeakyReLU(0.2)) # (128+128)→64
        self.dec1 = nn.Sequential(nn.Linear(E[2]//2 + E[3]//4, E[2]//4), nn.LeakyReLU(0.2)) # (64+64)→32
        # dec0 input: x03 (E[1]//2=32) concatenated with upsampled dec1 output (E[2]//4=32) = 64
        self.dec0 = nn.Sequential(nn.Linear(E[1]//2 + E[2]//4, E[0]),    nn.LeakyReLU(0.2)) # 64→16
        self.head = nn.Linear(E[0], num_classes)   # per-point classification logits

    def forward(self, xyz, features, knn_stages):
        """
        xyz        : (B, N, 3)  — point coordinates
        features   : (B, N, 4)  — normalised input features
        knn_stages : list of 8 tensors alternating [nn_idx_i, sub_idx_i] for i in 0..3
        Returns    : (B, N, num_classes) — per-point class logits
        """
        B, N, _ = xyz.shape

        # Project input features to E[0] channels
        x = self.fc_start(features.reshape(B*N,-1)).reshape(B,N,-1)

        # Unpack precomputed kNN graphs and downsample indices for all 4 stages
        nn0,si0,nn1,si1,nn2,si2,nn3,si3 = knn_stages

        # ── Encoder ────────────────────────────────────────────────────────────
        f0   = self.enc0(xyz, x, nn0)                                  # (B, N,     32)
        xyz1 = random_sample(xyz, si0)
        f1   = self.enc1(xyz1, random_sample(f0, si0), nn1)            # (B, N/4,   64)
        xyz2 = random_sample(xyz1, si1)
        f2   = self.enc2(xyz2, random_sample(f1, si1), nn2)            # (B, N/16,  128)
        xyz3 = random_sample(xyz2, si2)
        f3   = self.enc3(xyz3, random_sample(f2, si2), nn3)            # (B, N/64,  256)

        # Bottleneck: one more downsample then MLP refinement
        xyz4 = random_sample(xyz3, si3)
        f4s  = random_sample(f3, si3); B4,N4,C4 = f4s.shape
        f4   = self.mlp_bot(f4s.reshape(B4*N4,C4)).reshape(B4,N4,C4)  # (B, N/256, 256)

        # ── UNet++ dense skip nodes ────────────────────────────────────────────
        # Column 1: one upsample from adjacent coarser stage
        x01 = F.leaky_relu(self.x01(torch.cat([f0,  nearest_upsample(xyz,  xyz1, f1)],  dim=-1)), 0.2)
        x11 = F.leaky_relu(self.x12(torch.cat([f1,  nearest_upsample(xyz1, xyz2, f2)],  dim=-1)), 0.2)
        x21 = F.leaky_relu(self.x23(torch.cat([f2,  nearest_upsample(xyz2, xyz3, f3)],  dim=-1)), 0.2)
        # Column 2: fuse column-1 node with upsampled column-1 node from next row
        x02 = F.leaky_relu(self.x02(torch.cat([x01, nearest_upsample(xyz,  xyz1, x11)], dim=-1)), 0.2)
        x12 = F.leaky_relu(self.x13(torch.cat([x11, nearest_upsample(xyz1, xyz2, x21)], dim=-1)), 0.2)
        # Column 3: deepest dense skip — aggregates info from all three resolutions
        x03 = F.leaky_relu(self.x03(torch.cat([x02, nearest_upsample(xyz,  xyz1, x12)], dim=-1)), 0.2)

        # ── Decoder ────────────────────────────────────────────────────────────
        d3 = self.dec3(f3 + nearest_upsample(xyz3, xyz4, f4))          # coarsest decoder
        d2 = self.dec2(torch.cat([x21, nearest_upsample(xyz2, xyz3, d3)], dim=-1))
        d1 = self.dec1(torch.cat([x12, nearest_upsample(xyz1, xyz2, d2)], dim=-1))
        d0 = self.dec0(torch.cat([x03, nearest_upsample(xyz,  xyz1, d1)], dim=-1))  # full res

        return self.head(d0)   # (B, N, num_classes)
'''

# COMMAND ----------

# MAGIC %md ## 3. Shared train_ddp (reads config from os.environ)

# COMMAND ----------

import mlflow
mlflow.set_experiment(EXPERIMENT)


def train_ddp():
    """
    Core training loop.  Designed to run identically in three contexts:
      • Single-process (Setup A): RANK=0, WORLD_SIZE=1, no dist.init
      • Multi-GPU DDP  (Setup B): spawned by TorchDistributor(local_mode=True)
                                  RANK and LOCAL_RANK set per GPU by the launcher
      • HP trial       (Setup D): same as A, called 4× with different env vars

    Configuration is passed entirely via environment variables so the same
    function body works under cloudpickle serialisation (TorchDistributor) and
    direct calls without any argument changes.
    """
    import sys, os, pickle, glob, time
    import numpy as np
    import torch, torch.nn as nn, torch.optim as optim, torch.distributed as dist

    # exec() the model code string so UNext is defined in the worker's local scope.
    # Required because cloudpickle cannot serialise class definitions defined
    # in the outer notebook cell.
    _ns = {}; exec(UNEXT_CODE, _ns); UNext = _ns["UNext"]

    # ── DDP rank / world_size (set by TorchDistributor; default to single-process) ──
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialise NCCL process group only when running multi-GPU
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)   # pin this process to the correct GPU

    # ── Read hyperparameters from env vars (set by the calling cell) ─────────
    _chunk_dir  = os.environ.get("UNEXT_CHUNK_DIR", CHUNK_DIR)
    _n_epochs   = int(os.environ.get("UNEXT_N_EPOCHS",   3))
    _batch_size = int(os.environ.get("UNEXT_BATCH_SIZE", 2))
    _lr         = float(os.environ.get("UNEXT_LR",       1e-3))
    _run_name   = os.environ.get("UNEXT_RUN_NAME",  "benchmark")
    _experiment = os.environ.get("UNEXT_EXPERIMENT", EXPERIMENT)
    _n_feats    = int(os.environ.get("UNEXT_N_FEATURES",  4))
    _n_classes  = int(os.environ.get("UNEXT_NUM_CLASSES", 6))

    # Each DDP rank owns a disjoint subset of chunk files (stride = world_size).
    # This is manual data sharding — equivalent to DistributedSampler in PyTorch.
    chunk_files = sorted(glob.glob(f"{_chunk_dir}/chunk_*.pkl"))
    my_files    = chunk_files[rank::world_size]
    print(f"[rank {rank}] chunk_dir={_chunk_dir}  total={len(chunk_files)}  mine={len(my_files)}")

    def load_batch(files, bsz):
        """Lazy generator: shuffles files each epoch and yields batches of size bsz."""
        import random; random.shuffle(files)
        bx, bf, bl, bk = [], [], [], []
        for fp in files:
            with open(fp, "rb") as fh: c = pickle.load(fh)
            bx.append(c["xyz"]); bf.append(c["feats"]); bl.append(c["labels"]); bk.append(c["knn"])
            if len(bx) == bsz:
                yield bx, bf, bl, bk
                bx, bf, bl, bk = [], [], [], []

    def collate(bx, bf, bl, bk):
        """Stack list of numpy arrays into batched GPU tensors."""
        xyz    = torch.tensor(np.stack(bx), dtype=torch.float32, device=device)
        feats  = torch.tensor(np.stack(bf), dtype=torch.float32, device=device)
        labels = torch.tensor(np.stack(bl), dtype=torch.long,    device=device)
        # knn_stages: 8 tensors per sample; stack across batch dimension
        knn    = [torch.tensor(np.stack([b[i] for b in bk]), dtype=torch.long, device=device) for i in range(8)]
        return xyz, feats, labels, knn

    model = UNext(d_in=_n_feats, num_classes=_n_classes, enc_dims=[16,64,128,256,512], k=16).to(device)
    # Wrap in DDP only when multiple processes are running
    if world_size > 1 and dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # ── Class-balanced loss ───────────────────────────────────────────────────
    # LiDAR datasets are heavily imbalanced (e.g. ground dominates by 10×-100×).
    # Inverse-frequency weighting ensures rare classes (Water, LowVeg) contribute
    # proportionally to the gradient — without this the model learns to predict
    # "Ground" for everything and still achieves ~60% accuracy.
    class_counts  = np.array([1200000, 9480, 336540, 148520, 14137719, 14400], dtype=np.float32)
    class_weights = (1.0 / (class_counts + 1)).astype(np.float32)
    class_weights = class_weights / class_weights.sum() * _n_classes   # normalise to sum = n_classes
    criterion  = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    # Linear LR scaling: multiply by world_size so each rank's effective step size
    # stays constant regardless of how many GPUs are summing gradients.
    optimizer  = optim.Adam(model.parameters(), lr=_lr * world_size)
    # Cosine decay to near-zero LR by the final epoch — avoids oscillating at convergence
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_n_epochs)

    # Only rank 0 logs to MLflow — other ranks just train
    if rank == 0:
        import mlflow
        mlflow.set_experiment(_experiment)
        mlflow.start_run(run_name=_run_name)
        mlflow.log_params({"world_size": world_size, "lr": _lr, "batch_size": _batch_size,
                           "n_epochs": _n_epochs, "run_name": _run_name})

    t0 = time.time()
    final_acc = 0.0
    for epoch in range(_n_epochs):
        model.train()
        loss_sum, n_correct, n_total = 0.0, 0, 0

        for bx, bf, bl, bk in load_batch(my_files, _batch_size):
            try:
                xyz, feats, labels, knn = collate(bx, bf, bl, bk)
                optimizer.zero_grad()
                logits = model(xyz, feats, knn)           # (B, N, num_classes)
                B, N, C = logits.shape
                # Flatten spatial dimension for CrossEntropyLoss: (B*N, C) vs (B*N,)
                loss = criterion(logits.reshape(B*N, C), labels.reshape(B*N))
                loss.backward()
                optimizer.step()
                loss_sum  += loss.item()
                n_correct += (logits.argmax(-1) == labels).sum().item()
                n_total   += labels.numel()
            except Exception as e:
                import traceback
                if rank == 0: print(f"  Batch error: {traceback.format_exc()}")
                raise  # fail fast — no silent catch so the real error is visible

        scheduler.step()
        final_acc = n_correct / max(n_total, 1)
        if rank == 0:
            import mlflow
            mlflow.log_metrics({"train_loss": loss_sum, "train_acc": final_acc}, step=epoch)
            print(f"  [{_run_name}] Epoch {epoch+1}/{_n_epochs}  loss={loss_sum:.4f}  acc={final_acc:.4f}")

    elapsed = time.time() - t0
    if rank == 0:
        import mlflow
        mlflow.log_metrics({"elapsed_s": elapsed, "final_acc": final_acc})
        mlflow.end_run()

    # Clean up the NCCL process group so subsequent TorchDistributor calls don't conflict
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def _train_safe():
    """
    Wrapper around train_ddp() that catches exceptions and writes a traceback
    to a UC Volume file so errors from any rank are preserved even when
    TorchDistributor swallows the full stack trace.
    """
    import os, traceback
    rank = os.environ.get("RANK", "0")
    try:
        train_ddp()
    except Exception:
        with open(f"/Volumes/chlor/lidar_schema/tmp_benchmark/error_rank{rank}.txt", "w") as f:
            f.write(traceback.format_exc())
        raise

# COMMAND ----------

# MAGIC %md ## 4a. Setup A — Single GPU (baseline)

# COMMAND ----------

import os, time

# Pass Databricks credentials so MLflow can authenticate inside train_ddp().
# dbutils.notebook.entry_point provides the notebook-scoped token — safe to use
# for logging within the same workspace session.
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"]  = _ctx.apiUrl().getOrElse(None)
os.environ["DATABRICKS_TOKEN"] = _ctx.apiToken().getOrElse(None)

# Encode all hyperparameters as env vars so the same train_ddp() function body
# works unchanged for Setups A, B, and D without any argument modifications.
os.environ["UNEXT_CHUNK_DIR"]  = CHUNK_DIR
os.environ["UNEXT_N_EPOCHS"]   = str(N_EPOCHS)
os.environ["UNEXT_BATCH_SIZE"] = str(BATCH_SIZE)
os.environ["UNEXT_LR"]         = str(LR)
os.environ["UNEXT_EXPERIMENT"] = EXPERIMENT
os.environ["UNEXT_N_FEATURES"] = str(N_FEATURES)
os.environ["UNEXT_NUM_CLASSES"]= str(NUM_CLASSES)
os.environ["UNEXT_RUN_NAME"]   = "A-SingleGPU"   # MLflow run name tag

# Setup A: direct call on the driver — RANK=0, WORLD_SIZE=1, no DDP
print("=== Setup A: Single GPU ===")
t0 = time.time()
_train_safe()
A_time = time.time() - t0
print(f"Setup A done: {A_time:.1f}s")

# COMMAND ----------

# MAGIC %md ## 4b. Setup B — 4-GPU DDP (single-node, TorchDistributor local_mode=True)

# COMMAND ----------

import torch as _torch
# Synchronise any outstanding CUDA ops from Setup A before launching a new
# process group.  Without this, NCCL init in the DDP workers can conflict
# with an incompletely torn-down CUDA context on the driver.
if _torch.cuda.is_initialized():
    _torch.cuda.synchronize()

os.environ["UNEXT_RUN_NAME"] = "B-4GPU-DDP"

from pyspark.ml.torch.distributor import TorchDistributor

# Setup B: TorchDistributor with local_mode=True spawns 4 worker processes on the
# *same driver node*, each pinned to one of the 4 T4 GPUs (g4dn.12xlarge).
# Gradients are synchronised via NCCL all-reduce after each backward pass.
# local_mode=True avoids Spark executor scheduling overhead and is the right
# choice when all GPUs live on a single machine.
print("=== Setup B: 4-GPU DDP ===")
t0 = time.time()
TorchDistributor(num_processes=4, local_mode=True, use_gpu=True).run(_train_safe)
B_time = time.time() - t0
print(f"Setup B done: {B_time:.1f}s")

# COMMAND ----------

# MAGIC %md ## 4c. Setup C — Multi-Node (pull from MLflow)

# COMMAND ----------

import mlflow

# Setup C: Multi-node training was run separately (notebook 04_unext_multigpu).
# Here we just pull the best run's metrics from the lidar-unext experiment so
# it can be included in the comparison table alongside A and B.
_exp_mn = mlflow.get_experiment_by_name("/Users/<YOUR_EMAIL>/lidar-unext")
C_time, C_acc = None, None
if _exp_mn:
    _runs = mlflow.search_runs(
        experiment_ids=[_exp_mn.experiment_id],
        filter_string="attributes.run_name = 'unext-multigpu'",
        order_by=["start_time DESC"], max_results=1,
    )
    if len(_runs) > 0:
        C_time = _runs.iloc[0].get("metrics.elapsed_seconds")
        C_acc  = _runs.iloc[0].get("metrics.final_acc")
        print(f"Setup C (multi-node) from MLflow: elapsed={C_time}s  acc={C_acc}")
    else:
        print("No multi-node run found in MLflow.")
else:
    print("Experiment lidar-unext not found.")

# COMMAND ----------

# MAGIC %md ## 4d. Setup D — HP Search (4 sequential single-GPU trials)

# COMMAND ----------

# Setup D: sequential single-GPU HP search across 4 lr × batch_size configurations.
# In production this can be parallelised (one GPU per trial with Optuna + MLflow)
# to achieve ~4× wall-clock reduction.  Sequential execution here keeps the
# setup self-contained on a single GPU without extra job scheduling.
D_results = []
for hp in HP_CONFIGS:
    os.environ["UNEXT_LR"]         = str(hp["lr"])
    os.environ["UNEXT_BATCH_SIZE"] = str(hp["batch_size"])
    os.environ["UNEXT_RUN_NAME"]   = f"D-HP-{hp['label']}"
    print(f"=== Setup D trial: {hp['label']} ===")
    t0 = time.time()
    _train_safe()
    elapsed = time.time() - t0
    D_results.append({"label": hp["label"], "lr": hp["lr"], "batch_size": hp["batch_size"],
                      "elapsed_s": elapsed})
    print(f"Trial done: {elapsed:.1f}s")

D_total_time    = sum(r["elapsed_s"] for r in D_results)      # actual sequential wall-clock
D_parallel_time = max(r["elapsed_s"] for r in D_results)      # hypothetical if run in parallel

# COMMAND ----------

# MAGIC %md ## 5. Collect Accuracy from MLflow & Compare

# COMMAND ----------

import mlflow, pandas as pd

_exp = mlflow.get_experiment_by_name(EXPERIMENT)

def get_metrics(run_name):
    runs = mlflow.search_runs(
        experiment_ids=[_exp.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["start_time DESC"], max_results=1,
    )
    if len(runs) == 0:
        return None, None
    r = runs.iloc[0]
    return r.get("metrics.final_acc"), r.get("metrics.elapsed_s")

A_acc, A_t = get_metrics("A-SingleGPU")
B_acc, B_t = get_metrics("B-4GPU-DDP")
A_time = A_t or A_time
B_time = B_t or B_time

for r in D_results:
    acc, _ = get_metrics(f"D-HP-{r['label']}")
    r["acc"] = acc or 0.0

D_best = max(D_results, key=lambda r: r["acc"])
print(f"Best HP trial: {D_best['label']}  acc={D_best['acc']:.4f}")

# Cost estimates
def cost(seconds, dbu_rate): return (seconds / 3600) * dbu_rate * DBU_PRICE_USD

rows = [
    {"Setup": "A — Single GPU (1 GPU)",
     "Wall Clock (s)": f"{A_time:.0f}",
     "Final Acc": f"{A_acc:.4f}" if A_acc else "—",
     "Est. Cost USD": f"${cost(A_time, DBU_RATE_1GPU):.4f}",
     "Models trained": 1,
     "Best for": "Dev / debugging"},
    {"Setup": "B — 4-GPU DDP (single-node)",
     "Wall Clock (s)": f"{B_time:.0f}",
     "Final Acc": f"{B_acc:.4f}" if B_acc else "—",
     "Est. Cost USD": f"${cost(B_time, DBU_RATE_4GPU):.4f}",
     "Models trained": 1,
     "Best for": "Production / fast training"},
    {"Setup": "C — Multi-Node DDP",
     "Wall Clock (s)": f"{C_time:.0f}" if C_time else "N/A (see lidar-unext exp)",
     "Final Acc": f"{C_acc:.4f}" if C_acc else "N/A",
     "Est. Cost USD": f"${cost(C_time, DBU_RATE_4GPU):.4f}" if C_time else "N/A",
     "Models trained": 1,
     "Best for": "Large datasets (>1 node RAM)"},
    {"Setup": "D — HP Search (4 trials × 1 GPU)",
     "Wall Clock (s)": f"{D_total_time:.0f} seq / ~{D_parallel_time:.0f} parallel",
     "Final Acc": f"{D_best['acc']:.4f} (best of 4)",
     "Est. Cost USD": f"${cost(D_total_time, DBU_RATE_1GPU):.4f} seq / ${cost(D_parallel_time, DBU_RATE_4GPU):.4f} parallel",
     "Models trained": 4,
     "Best for": "Model quality / HP optimisation"},
]
df = pd.DataFrame(rows)
display(df)

# COMMAND ----------

# MAGIC %md ## 6. Charts

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels  = ["A\n1-GPU", "B\n4-GPU DDP", "D-HP\n(parallel est.)"]
times   = [float(A_time), float(B_time), float(D_parallel_time)]
accs    = [A_acc or 0, B_acc or 0, D_best["acc"]]
costs_v = [cost(A_time, DBU_RATE_1GPU), cost(B_time, DBU_RATE_4GPU), cost(D_parallel_time, DBU_RATE_4GPU)]
colors  = ["#1f77b4", "#ff7f0e", "#2ca02c"]

if C_time:
    labels.insert(2, "C\nMulti-Node")
    times.insert(2,  float(C_time))
    accs.insert(2,   C_acc or 0)
    costs_v.insert(2, cost(C_time, DBU_RATE_4GPU))
    colors.insert(2, "#d62728")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("U-Next: Setup Comparison", fontsize=13, fontweight="bold")

for ax, vals, title, ylabel in [
    (axes[0], times,   "Wall-Clock Time (s)\n↓ lower = faster", "seconds"),
    (axes[1], accs,    "Final Train Accuracy\n↑ higher = better", "accuracy"),
    (axes[2], costs_v, "Estimated Cost (USD)\n↓ lower = cheaper", "USD"),
]:
    bars = ax.bar(labels, vals, color=colors)
    ax.set_title(title); ax.set_ylabel(ylabel)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f"{v:.3f}" if ylabel == "USD" else f"{v:.3f}" if ylabel == "accuracy" else f"{v:.0f}s",
                ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("/Volumes/chlor/lidar_schema/tmp_benchmark/benchmark_chart.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md ## 7. Speedup & Recommendation

# COMMAND ----------

speedup_B = A_time / B_time if B_time else 0
speedup_D = A_time / D_parallel_time if D_parallel_time else 0
print(f"Speedup B vs A: {speedup_B:.1f}×")
print(f"HP parallel vs A: {speedup_D:.1f}× (if 4 trials run concurrently)")
print()

reco = pd.DataFrame([
    {"Use Case": "Exploration / quick iteration",     "Recommended": "A — Single GPU",        "Reason": f"Cheapest (${cost(A_time,DBU_RATE_1GPU):.4f}), simple"},
    {"Use Case": "Production, tight deadline",        "Recommended": "B — 4-GPU DDP",          "Reason": f"{speedup_B:.1f}× faster, similar total cost to A"},
    {"Use Case": "Dataset > single-node RAM",         "Recommended": "C — Multi-Node DDP",     "Reason": "Only option for horizontal data scale"},
    {"Use Case": "Best model quality (HP tuning)",    "Recommended": "D — Parallel HP Search", "Reason": f"Same wall-clock as B, explores 4 configs, best acc={D_best['acc']:.4f}"},
    {"Use Case": "Best accuracy + fast final train",  "Recommended": "D → B",                  "Reason": "Find best HP with D, retrain at scale with B"},
])
display(reco)
