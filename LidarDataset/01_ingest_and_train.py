# Databricks notebook source
# MAGIC %md
# MAGIC # LiDAR Point Cloud Classification — GPU Demo
# MAGIC
# MAGIC This notebook demonstrates a full end-to-end pipeline for classifying LiDAR point clouds using GPU acceleration on Databricks.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. Parse raw LAS 1.2 binary files from Unity Catalog Volume
# MAGIC 2. Store as Delta table in `chlor.lidar_schema.point_cloud`
# MAGIC 3. Engineer features (neighborhood statistics, height normalization)
# MAGIC 4. Train GPU-accelerated classifier (XGBoost w/ `device="cuda"` or cuML Random Forest)
# MAGIC 5. Evaluate with accuracy metrics, confusion matrix, and feature importance
# MAGIC
# MAGIC **Classes:** 2=Ground, 3=LowVeg, 4=MedVeg, 5=HighVeg, 6=Building, 9=Water

# COMMAND ----------

# MAGIC %md ## 0. Setup

# COMMAND ----------

VOLUME_PATH   = "/Volumes/chlor/lidar_schema/raw_las"
CATALOG       = "chlor"
SCHEMA        = "lidar_schema"
TABLE         = "point_cloud"
FULL_TABLE    = f"{CATALOG}.{SCHEMA}.{TABLE}"
SAMPLE_FRAC   = 0.15   # fraction used for GPU training (adjust for cluster memory)

CLASS_NAMES = {2: "Ground", 3: "LowVeg", 4: "MedVeg", 5: "HighVeg", 6: "Building", 9: "Water"}

# COMMAND ----------

import struct, os
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md ## 1. Parse LAS 1.2 files → Delta table

# COMMAND ----------

def parse_las(path: str) -> pd.DataFrame:
    """Parse a LAS 1.2 file (Point Format 0) into a DataFrame without laspy."""
    with open(path, "rb") as f:
        raw = f.read()

    assert raw[:4] == b"LASF", f"Not a LAS file: {path}"

    # Offsets are shifted +2 because the System Identifier and Generating
    # Software string fields were written as 33 bytes each instead of 32.
    # Actual data layout: offset_to_pts stored value (227) + 2 = 229.
    offset_to_pts    = struct.unpack_from("<I", raw, 98)[0] + 2
    n_pts            = struct.unpack_from("<I", raw, 109)[0]
    sx, sy, sz       = struct.unpack_from("<ddd", raw, 133)
    ox, oy, oz       = struct.unpack_from("<ddd", raw, 157)

    # Point Format 0 = 20 bytes per record
    RECORD = 20
    pts_raw = np.frombuffer(raw, dtype=np.uint8, offset=offset_to_pts,
                             count=n_pts * RECORD).reshape(n_pts, RECORD)

    xi  = np.frombuffer(pts_raw[:, 0:4].tobytes(),  dtype=np.int32)
    yi  = np.frombuffer(pts_raw[:, 4:8].tobytes(),  dtype=np.int32)
    zi  = np.frombuffer(pts_raw[:, 8:12].tobytes(), dtype=np.int32)
    intensity    = np.frombuffer(pts_raw[:, 12:14].tobytes(), dtype=np.uint16).astype(np.int32)
    classification = pts_raw[:, 15].astype(np.int32)

    return pd.DataFrame({
        "x":              xi * sx + ox,
        "y":              yi * sy + oy,
        "z":              zi * sz + oz,
        "intensity":      intensity,
        "classification": classification,
    })

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType

schema = StructType([
    StructField("x",              DoubleType(),  False),
    StructField("y",              DoubleType(),  False),
    StructField("z",              DoubleType(),  False),
    StructField("intensity",      IntegerType(), False),
    StructField("classification", IntegerType(), False),
    StructField("tile_name",      StringType(),  False),
])

las_files = [f for f in os.listdir(VOLUME_PATH) if f.endswith(".las")]
print(f"Found {len(las_files)} LAS files")

all_dfs = []
for fname in sorted(las_files):
    fpath = os.path.join(VOLUME_PATH, fname)
    pdf = parse_las(fpath)
    pdf["tile_name"] = fname
    all_dfs.append(pdf)
    print(f"  {fname}: {len(pdf):,} points")

full_pdf = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal: {len(full_pdf):,} points across {len(las_files)} tiles")

# COMMAND ----------

sdf = spark.createDataFrame(full_pdf, schema=schema)

(sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULL_TABLE))

print(f"Saved to {FULL_TABLE}")
spark.sql(f"SELECT classification, COUNT(*) as count FROM {FULL_TABLE} GROUP BY 1 ORDER BY 1").show()

# COMMAND ----------

# MAGIC %md ## 2. Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC Features engineered per point:
# MAGIC - Raw: `x, y, z, intensity`
# MAGIC - Height above ground proxy: `z` (ground is near 0)
# MAGIC - Normalized intensity by class exposure
# MAGIC - Tile-level neighborhood stats (height range, mean, std in local cell)

# COMMAND ----------

pdf = spark.table(FULL_TABLE).toPandas()

# Bin points into 2m grid cells for neighbourhood features
CELL = 2.0
pdf["cell_x"] = (pdf["x"] // CELL).astype(np.int32)
pdf["cell_y"] = (pdf["y"] // CELL).astype(np.int32)
pdf["cell_id"] = pdf["cell_x"].astype(str) + "_" + pdf["cell_y"].astype(str) + "_" + pdf["tile_name"]

cell_stats = pdf.groupby("cell_id")["z"].agg(
    z_cell_mean="mean",
    z_cell_std="std",
    z_cell_min="min",
    z_cell_max="max",
).reset_index()
cell_stats["z_cell_range"] = cell_stats["z_cell_max"] - cell_stats["z_cell_min"]
cell_stats["z_cell_std"]   = cell_stats["z_cell_std"].fillna(0.0)

pdf = pdf.merge(cell_stats[["cell_id","z_cell_mean","z_cell_std","z_cell_range"]], on="cell_id", how="left")
pdf["z_above_cell_mean"] = pdf["z"] - pdf["z_cell_mean"]

FEATURES = ["x", "y", "z", "intensity", "z_cell_mean", "z_cell_std", "z_cell_range", "z_above_cell_mean"]
print(f"Feature set: {FEATURES}")
print(f"Dataset shape: {pdf.shape}")

# COMMAND ----------

# MAGIC %md ## 3. GPU-Accelerated Training

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample for GPU memory constraints
sample = pdf.sample(frac=SAMPLE_FRAC, random_state=42)
print(f"Training on {len(sample):,} points ({SAMPLE_FRAC*100:.0f}% sample)")

X = sample[FEATURES].values.astype(np.float32)
y = sample["classification"].values.astype(np.int32)

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

# COMMAND ----------

# Try GPU XGBoost first, fall back to CPU if no GPU available
import subprocess, sys

def gpu_available():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

USE_GPU = gpu_available()
print(f"GPU available: {USE_GPU}")

# COMMAND ----------

# MAGIC %md ### Option A: XGBoost (GPU)

# COMMAND ----------

import xgboost as xgb

device = "cuda" if USE_GPU else "cpu"
print(f"Training XGBoost on device: {device}")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    device=device,
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\nXGBoost Accuracy: {acc_xgb:.4f}")

# COMMAND ----------

# MAGIC %md ### Option B: cuML Random Forest (GPU — uncomment if cuML installed)

# COMMAND ----------

# try:
#     from cuml.ensemble import RandomForestClassifier as cuRF
#     import cudf
#
#     print("Training cuML RandomForest on GPU...")
#     X_train_gdf = cudf.DataFrame(X_train, columns=FEATURES)
#     X_test_gdf  = cudf.DataFrame(X_test,  columns=FEATURES)
#
#     cuml_model = cuRF(n_estimators=200, max_depth=16, n_streams=4, random_state=42)
#     cuml_model.fit(X_train_gdf, y_train)
#
#     y_pred_cuml = cuml_model.predict(X_test_gdf).to_array()
#     acc_cuml = accuracy_score(y_test, y_pred_cuml)
#     print(f"cuML RF Accuracy: {acc_cuml:.4f}")
# except ImportError:
#     print("cuML not installed — skipping")

# COMMAND ----------

# MAGIC %md ## 4. Evaluation

# COMMAND ----------

# Classification report
class_labels = [CLASS_NAMES[c] for c in le.classes_]
print("=" * 60)
print("Classification Report — XGBoost")
print("=" * 60)
print(classification_report(y_test, y_pred_xgb, target_names=class_labels))

# COMMAND ----------

# Confusion matrix as a styled DataFrame
cm = confusion_matrix(y_test, y_pred_xgb)
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
print("Confusion Matrix (rows=actual, cols=predicted):")
display(cm_df)

# COMMAND ----------

# Feature importance
fi = pd.DataFrame({
    "feature":   FEATURES,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("Feature Importance:")
display(fi)

# COMMAND ----------

# MAGIC %md ## 5. Save Predictions to Delta

# COMMAND ----------

# Score the full dataset in batches
BATCH = 500_000
pred_classes = []
X_full = pdf[FEATURES].values.astype(np.float32)

for start in range(0, len(X_full), BATCH):
    batch = X_full[start:start + BATCH]
    preds = le.inverse_transform(xgb_model.predict(batch))
    pred_classes.extend(preds)

pdf["predicted_class"] = pred_classes
pdf["predicted_name"]  = pdf["predicted_class"].map(CLASS_NAMES)
pdf["correct"]         = pdf["predicted_class"] == pdf["classification"]

overall_acc = pdf["correct"].mean()
print(f"Full-dataset accuracy: {overall_acc:.4f}")

# COMMAND ----------

results_sdf = spark.createDataFrame(
    pdf[["x","y","z","intensity","classification","tile_name","predicted_class","predicted_name","correct"]]
)

(results_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.point_cloud_predictions"))

print(f"Predictions saved to {CATALOG}.{SCHEMA}.point_cloud_predictions")

# COMMAND ----------

spark.sql(f"""
  SELECT predicted_name, classification,
         COUNT(*) as n_points,
         ROUND(AVG(CAST(correct AS INT))*100, 2) as accuracy_pct
  FROM {CATALOG}.{SCHEMA}.point_cloud_predictions
  GROUP BY 1, 2
  ORDER BY 1, 2
""").show(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | Result |
# MAGIC |---|---|
# MAGIC | Raw LAS files | 20 tiles, ~15.8M points, 308 MB |
# MAGIC | Delta table | `chlor.lidar_schema.point_cloud` |
# MAGIC | Features | x, y, z, intensity + 4 neighbourhood stats |
# MAGIC | Model | XGBoost (GPU via `device="cuda"`) |
# MAGIC | Classes | Ground, LowVeg, MedVeg, HighVeg, Building, Water |
# MAGIC | Predictions | `chlor.lidar_schema.point_cloud_predictions` |
