# Databricks notebook source
# MAGIC %md
# MAGIC # LiDAR Point Cloud — Visualization
# MAGIC
# MAGIC Visual exploration of the synthetic LiDAR dataset and model predictions.
# MAGIC
# MAGIC **Sections:**
# MAGIC 1. Class distribution (bar chart)
# MAGIC 2. 3D point cloud scatter — colored by class
# MAGIC 3. Top-down 2D density map per class
# MAGIC 4. Height (Z) distribution per class
# MAGIC 5. Confusion matrix heatmap
# MAGIC 6. Feature importance bar chart
# MAGIC 7. Per-tile accuracy

# COMMAND ----------

# MAGIC %md ## 0. Setup

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.cm as cm

CATALOG = "chlor"
SCHEMA  = "lidar_schema"

CLASS_NAMES  = {2: "Ground", 3: "LowVeg", 4: "MedVeg", 5: "HighVeg", 6: "Building", 9: "Water"}
CLASS_COLORS = {2: "#C2A97A", 3: "#90EE90", 4: "#228B22", 5: "#006400", 6: "#E07B54", 9: "#4682B4"}

def class_palette(classes):
    return [CLASS_COLORS.get(c, "#888888") for c in classes]

# COMMAND ----------

# MAGIC %md ## 1. Load Data

# COMMAND ----------

# Full point cloud (sample for plotting)
pts = spark.table(f"{CATALOG}.{SCHEMA}.point_cloud").sample(0.02, seed=42).toPandas()
pts["class_name"] = pts["classification"].map(CLASS_NAMES)
print(f"Plotting sample: {len(pts):,} points")

# Predictions table (sample)
preds = spark.table(f"{CATALOG}.{SCHEMA}.point_cloud_predictions").sample(0.02, seed=42).toPandas()
preds["class_name"]     = preds["classification"].map(CLASS_NAMES)
preds["pred_name"]      = preds["predicted_class"].map(CLASS_NAMES)

# Full counts for stats
counts_df = (spark.table(f"{CATALOG}.{SCHEMA}.point_cloud")
             .groupBy("classification").count().toPandas()
             .sort_values("classification"))
counts_df["class_name"] = counts_df["classification"].map(CLASS_NAMES)

# Per-tile accuracy — cast BOOLEAN correct to INT before avg
from pyspark.sql import functions as F
tile_acc = (spark.table(f"{CATALOG}.{SCHEMA}.point_cloud_predictions")
            .groupBy("tile_name")
            .agg(F.avg(F.col("correct").cast("int")).alias("accuracy"))
            .toPandas()
            .sort_values("tile_name"))

# COMMAND ----------

# MAGIC %md ## 2. Class Distribution

# COMMAND ----------

fig, ax = plt.subplots(figsize=(9, 5))
colors = class_palette(counts_df["classification"])
bars = ax.bar(counts_df["class_name"], counts_df["count"] / 1e6, color=colors, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, labels=[f"{v/1e6:.2f}M" for v in counts_df["count"]], padding=4, fontsize=10)
ax.set_title("Point Cloud — Class Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Class")
ax.set_ylabel("Points (millions)")
ax.set_ylim(0, counts_df["count"].max() / 1e6 * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md ## 3. 3D Point Cloud Scatter — Single Tile

# COMMAND ----------

# Pick one tile for 3D view
tile = pts[pts["tile_name"] == "tile_000.las"].copy()
# downsample to 15k for matplotlib 3D
if len(tile) > 15000:
    tile = tile.sample(15000, random_state=42)

fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection="3d")

for cls_id, grp in tile.groupby("classification"):
    color = CLASS_COLORS.get(cls_id, "#888")
    label = CLASS_NAMES.get(cls_id, str(cls_id))
    ax.scatter(grp["x"], grp["y"], grp["z"],
               c=color, label=label, s=0.4, alpha=0.7, depthshade=True)

ax.set_xlabel("X (m)", labelpad=6)
ax.set_ylabel("Y (m)", labelpad=6)
ax.set_zlabel("Z (m)", labelpad=6)
ax.set_title("3D Point Cloud — tile_000 (colored by class)", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", markerscale=8, framealpha=0.8)
ax.view_init(elev=25, azim=-60)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md ## 4. Top-Down 2D Density Map (all sampled tiles)

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
class_ids = [2, 3, 4, 5, 6, 9]

for ax, cls_id in zip(axes.flat, class_ids):
    grp = pts[pts["classification"] == cls_id]
    if len(grp) == 0:
        ax.set_visible(False)
        continue
    h = ax.hist2d(grp["x"], grp["y"], bins=120,
                  cmap=mcolors.LinearSegmentedColormap.from_list(
                      "", ["white", CLASS_COLORS[cls_id]]),
                  norm=mcolors.LogNorm())
    plt.colorbar(h[3], ax=ax, label="log(count)")
    ax.set_title(f"{CLASS_NAMES[cls_id]} (class {cls_id})", fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

fig.suptitle("Top-Down Density Maps by Class", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md ## 5. Height (Z) Distribution per Class

# COMMAND ----------

fig, ax = plt.subplots(figsize=(11, 5))

for cls_id in sorted(CLASS_NAMES):
    grp = pts[pts["classification"] == cls_id]["z"]
    if len(grp) < 10:
        continue
    ax.hist(grp, bins=100, alpha=0.6, color=CLASS_COLORS[cls_id],
            label=CLASS_NAMES[cls_id], density=True, range=(-1, 30))

ax.set_xlabel("Z height (m)")
ax.set_ylabel("Density")
ax.set_title("Height Distribution per Class", fontsize=13, fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md ## 6. Confusion Matrix Heatmap

# COMMAND ----------

from sklearn.metrics import confusion_matrix

# Use the predictions sample
classes_present = sorted(preds["classification"].unique())
labels = [CLASS_NAMES[c] for c in classes_present]

cm = confusion_matrix(preds["classification"], preds["predicted_class"], labels=classes_present)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalize

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, data, title, fmt in [
    (axes[0], cm,      "Confusion Matrix (counts)",       "d"),
    (axes[1], cm_norm, "Confusion Matrix (row-normalized)", ".2f"),
]:
    im = ax.imshow(data, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")
    thresh = data.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = data[i, j]
            txt = f"{val:{fmt}}" if fmt == ".2f" else f"{int(val):,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if val > thresh else "black")

plt.suptitle("Model Prediction Quality", fontsize=14, fontweight="bold")
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md ## 7. Feature Importance

# COMMAND ----------

# Re-use importance values computed in training notebook (stored as a simple lookup)
feature_importance = {
    "z":                 0.0,   # filled by XGBoost — placeholder
    "z_above_cell_mean": 0.0,
    "z_cell_mean":       0.0,
    "z_cell_range":      0.0,
    "z_cell_std":        0.0,
    "intensity":         0.0,
    "x":                 0.0,
    "y":                 0.0,
}

# If predictions table has importance metadata, load it; otherwise train a quick proxy
try:
    fi_df = spark.table(f"{CATALOG}.{SCHEMA}.feature_importance").toPandas()
except Exception:
    # Quick proxy: train lightweight XGBoost on 5% sample
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    FEATURES = ["x", "y", "z", "intensity", "z_cell_mean", "z_cell_std", "z_cell_range", "z_above_cell_mean"]
    CELL = 2.0

    s = preds.copy()
    s["cell_x"] = (s["x"] // CELL).astype(int)
    s["cell_y"] = (s["y"] // CELL).astype(int)
    s["cell_id"] = s["cell_x"].astype(str) + "_" + s["cell_y"].astype(str) + "_" + s["tile_name"]
    cs = s.groupby("cell_id")["z"].agg(z_cell_mean="mean", z_cell_std="std",
                                         z_cell_min="min", z_cell_max="max").reset_index()
    cs["z_cell_range"] = cs["z_cell_max"] - cs["z_cell_min"]
    cs["z_cell_std"]   = cs["z_cell_std"].fillna(0)
    s = s.merge(cs[["cell_id","z_cell_mean","z_cell_std","z_cell_range"]], on="cell_id", how="left")
    s["z_above_cell_mean"] = s["z"] - s["z_cell_mean"]

    le = LabelEncoder()
    y  = le.fit_transform(s["classification"].values)
    X  = s[FEATURES].fillna(0).values.astype(np.float32)

    proxy = xgb.XGBClassifier(n_estimators=80, max_depth=6, use_label_encoder=False,
                               eval_metric="mlogloss", random_state=42)
    proxy.fit(X, y)
    fi_df = pd.DataFrame({"feature": FEATURES, "importance": proxy.feature_importances_})

fi_df = fi_df.sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(fi_df["feature"], fi_df["importance"], color="#5B9BD5", edgecolor="white")
ax.bar_label(bars, labels=[f"{v:.3f}" for v in fi_df["importance"]], padding=4, fontsize=9)
ax.set_xlabel("Importance Score")
ax.set_title("XGBoost Feature Importance", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md ## 8. Per-Tile Accuracy

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 5))
colors = ["#2ecc71" if a >= 0.9 else "#f39c12" if a >= 0.75 else "#e74c3c"
          for a in tile_acc["accuracy"]]
bars = ax.bar(tile_acc["tile_name"], tile_acc["accuracy"] * 100, color=colors, edgecolor="white")
ax.axhline(tile_acc["accuracy"].mean() * 100, color="#333", linestyle="--", linewidth=1.5,
           label=f"Mean: {tile_acc['accuracy'].mean()*100:.1f}%")
ax.set_ylim(0, 105)
ax.set_xlabel("Tile")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-Tile Classification Accuracy", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=45)
legend_patches = [Patch(color="#2ecc71", label="≥90%"),
                  Patch(color="#f39c12", label="75–90%"),
                  Patch(color="#e74c3c", label="<75%")]
ax.legend(handles=legend_patches + [plt.Line2D([0],[0], color="#333", linestyle="--", label=f"Mean: {tile_acc['accuracy'].mean()*100:.1f}%")])
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC | Chart | What it shows |
# MAGIC |---|---|
# MAGIC | Class distribution | Point counts per ASPRS class |
# MAGIC | 3D scatter | Spatial structure of a single tile |
# MAGIC | 2D density maps | Top-down footprint per class |
# MAGIC | Height histograms | Z separation between classes |
# MAGIC | Confusion matrix | Model precision/recall per class |
# MAGIC | Feature importance | Which features drive predictions |
# MAGIC | Per-tile accuracy | Consistency across 20 tiles |
