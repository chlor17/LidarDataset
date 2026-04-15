# Databricks notebook source
# MAGIC %md
# MAGIC # LiDAR Classification — Multi-Node GPU + Optuna
# MAGIC
# MAGIC Replaces deprecated Hyperopt with **Optuna** — same results, no deprecation warnings.
# MAGIC
# MAGIC - **Optuna + `ThreadPoolExecutor`** — runs `PARALLELISM` trials simultaneously on GPU workers
# MAGIC - **`XGBClassifier(device="cuda")`** — each worker trains on its own GPU with broadcast data
# MAGIC - **`SparkXGBClassifier`** — final retrain distributed across ALL GPU workers simultaneously
# MAGIC - **MLflow** — every trial logged; best model registered in Unity Catalog
# MAGIC
# MAGIC ```
# MAGIC  Driver                         Workers (GPU)
# MAGIC  ──────                         ─────────────
# MAGIC  Broadcast train/test data  →   Worker 1: XGBClassifier(device=cuda) — Trial N
# MAGIC  Optuna TPE search loop     →   Worker 2: XGBClassifier(device=cuda) — Trial N+1
# MAGIC                             →   Worker 3: XGBClassifier(device=cuda) — Trial N+2
# MAGIC                             →   Worker 4: XGBClassifier(device=cuda) — Trial N+3
# MAGIC  Best params ←──────────────    All results aggregated
# MAGIC  Final retrain: SparkXGBClassifier(num_workers=4) across ALL GPUs
# MAGIC ```
# MAGIC
# MAGIC **Prerequisite:** Run `01_ingest_and_train` first to populate `chlor.lidar_schema.point_cloud`.

# COMMAND ----------

# MAGIC %md ## 0. Config

# COMMAND ----------

CATALOG      = "chlor"
SCHEMA       = "lidar_schema"
SRC_TABLE    = f"{CATALOG}.{SCHEMA}.point_cloud"
RESULT_TABLE = f"{CATALOG}.{SCHEMA}.point_cloud_hyperopt_preds"
MODEL_NAME   = f"{CATALOG}.{SCHEMA}.lidar_xgb_classifier"

MAX_EVALS   = 5     # total Optuna trials (set to 20 for full run)
PARALLELISM = 4     # concurrent trials — one per GPU worker
SAMPLE_FRAC = 0.05  # fraction used for HPO (set to 0.30 for full run)

FEATURES = ["x", "y", "z", "intensity", "z_cell_mean", "z_cell_std", "z_cell_range", "z_above_cell_mean"]
CLASS_NAMES = {2: "Ground", 3: "LowVeg", 4: "MedVeg", 5: "HighVeg", 6: "Building", 9: "Water"}

# COMMAND ----------

# MAGIC %md ## 1. Load & Feature-Engineer in Spark

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

raw = spark.table(SRC_TABLE)

CELL = 2.0
featured = (
    raw
    .withColumn("cell_x", (F.col("x") / CELL).cast("int"))
    .withColumn("cell_y", (F.col("y") / CELL).cast("int"))
    .withColumn("cell_id", F.concat_ws("_", F.col("cell_x"), F.col("cell_y"), F.col("tile_name")))
)

cell_stats = (
    featured.groupBy("cell_id")
    .agg(
        F.mean("z").alias("z_cell_mean"),
        F.stddev("z").alias("z_cell_std"),
        F.max("z").alias("z_cell_max"),
        F.min("z").alias("z_cell_min"),
    )
    .withColumn("z_cell_range", F.col("z_cell_max") - F.col("z_cell_min"))
    .withColumn("z_cell_std",   F.coalesce(F.col("z_cell_std"), F.lit(0.0)))
    .drop("z_cell_max", "z_cell_min")
)

sdf = (
    featured.join(cell_stats, on="cell_id", how="left")
    .withColumn("z_above_cell_mean", F.col("z") - F.col("z_cell_mean"))
    .select(*(FEATURES + ["classification", "tile_name"]))
    .cache()
)

print(f"Total points: {sdf.count():,}")
sdf.groupBy("classification").count().orderBy("classification").show()

# COMMAND ----------

# MAGIC %md ## 2. Sample → Pandas → Broadcast to GPU Workers

# COMMAND ----------

sample_pdf = sdf.sample(SAMPLE_FRAC, seed=42).toPandas()
print(f"HPO sample: {len(sample_pdf):,} points")

X = sample_pdf[FEATURES].fillna(0).values.astype(np.float32)
le = LabelEncoder()
y = le.fit_transform(sample_pdf["classification"].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}   Classes: {le.classes_}")

# Broadcast so each Spark worker receives a local copy without re-serialising per trial
X_train_bc = sc.broadcast(X_train)
X_test_bc  = sc.broadcast(X_test)
y_train_bc = sc.broadcast(y_train)
y_test_bc  = sc.broadcast(y_test)

# COMMAND ----------

# MAGIC %md ## 3. Optuna — Parallel GPU Search
# MAGIC
# MAGIC Each trial runs `XGBClassifier(device="cuda")` on a GPU worker.
# MAGIC `n_jobs=PARALLELISM` in `study.optimize()` runs trials concurrently via threads,
# MAGIC each dispatching a Spark task to a separate GPU worker.

# COMMAND ----------

import mlflow
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor

optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_experiment("/Users/<YOUR_EMAIL>/lidar-optuna")

trial_results = []   # collected by all threads

def objective(trial):
    params = {
        "max_depth":        trial.suggest_categorical("max_depth",        [4, 6, 8, 10]),
        "learning_rate":    trial.suggest_float("learning_rate",          0.02, 0.3, log=True),
        "n_estimators":     trial.suggest_categorical("n_estimators",     [100, 200, 300, 500]),
        "subsample":        trial.suggest_float("subsample",              0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",       0.5, 1.0),
        "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5, 10]),
        "gamma":            trial.suggest_float("gamma",                  1e-4, 1.0, log=True),
    }

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)

        import warnings
        X_tr = X_train_bc.value
        X_te = X_test_bc.value
        y_tr = y_train_bc.value
        y_te = y_test_bc.value

        model = xgb.XGBClassifier(
            device="cuda",
            eval_metric="mlogloss",
            random_state=42,
            **params,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Falling back to prediction")
            preds = model.predict(X_te)
        accuracy = accuracy_score(y_te, preds)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", 1.0 - accuracy)

    trial_results.append({"trial": trial.number, "accuracy": accuracy, "loss": 1.0 - accuracy})
    return 1.0 - accuracy   # Optuna minimises

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)

with mlflow.start_run(run_name="optuna-lidar-xgb") as parent_run:
    study.optimize(objective, n_trials=MAX_EVALS, n_jobs=PARALLELISM, show_progress_bar=False)

best_params   = study.best_params
best_accuracy = 1.0 - study.best_value

print("\nBest hyperparameters:")
for k, v in best_params.items():
    print(f"  {k:22s}: {v}")
print(f"\nBest validation accuracy: {best_accuracy:.4f}")

# COMMAND ----------

# MAGIC %md ## 4. Trial Results — Loss Curve & Accuracy Distribution

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

trial_pdf = pd.DataFrame(sorted(trial_results, key=lambda r: r["trial"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(trial_pdf["trial"], trial_pdf["loss"], "o-", color="#5B9BD5", alpha=0.8, linewidth=1.5)
ax.axhline(trial_pdf["loss"].min(), color="#e74c3c", linestyle="--",
           label=f'Best: {trial_pdf["loss"].min():.4f}')
ax.plot(trial_pdf["trial"], trial_pdf["loss"].cummin(),
        color="#2ecc71", linewidth=2, label="Running best")
ax.set_xlabel("Trial"); ax.set_ylabel("Loss (1 − accuracy)")
ax.set_title("Optuna Search — Loss per Trial", fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
ax.hist(trial_pdf["accuracy"], bins=15, color="#5B9BD5", edgecolor="white", alpha=0.85)
ax.axvline(best_accuracy, color="#e74c3c", linestyle="--", linewidth=2,
           label=f"Best: {best_accuracy:.4f}")
ax.set_xlabel("Validation Accuracy"); ax.set_ylabel("Count")
ax.set_title("Accuracy Distribution Across Trials", fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False)

plt.suptitle(f"Optuna — {MAX_EVALS} trials, parallelism={PARALLELISM}", fontsize=13, fontweight="bold")
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md ## 5. Retrain Final Model — All GPUs via SparkXGBClassifier

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from xgboost.spark import SparkXGBClassifier

# SparkXGBClassifier requires labels in [0, num_class) — use StringIndexer to remap
# {2,3,4,5,6,9} → {0,1,2,3,4,5}
assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
label_indexer = StringIndexer(inputCol="classification", outputCol="label", stringOrderType="alphabetAsc")
label_indexer_model = label_indexer.fit(sdf)
# Store mapping for decoding predictions later: index → original class id
idx_to_class = {i: int(v) for i, v in enumerate(label_indexer_model.labels)}
print("Label mapping (index → class):", idx_to_class)

train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=42)
train_vec = assembler.transform(label_indexer_model.transform(train_sdf)).select("features", "label")
test_vec  = assembler.transform(label_indexer_model.transform(test_sdf)).select("features", "label")

print(f"Final retrain: SparkXGBClassifier across {PARALLELISM} GPU workers...")

with mlflow.start_run(run_name="lidar-final-model") as final_run:
    mlflow.log_params(best_params)

    final_clf = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        num_workers=PARALLELISM,
        use_gpu=True,
        device="cuda",
        eval_metric="mlogloss",
        **best_params,
    )
    final_model = final_clf.fit(train_vec)
    final_preds = final_model.transform(test_vec)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    final_acc = evaluator.evaluate(final_preds)
    mlflow.log_metric("test_accuracy", final_acc)

    per_class = (
        final_preds
        .withColumn("correct", (F.col("prediction") == F.col("label")).cast("int"))
        .groupBy("label")
        .agg(F.avg("correct").alias("accuracy"), F.count("*").alias("n_points"))
        .orderBy("label")
        .toPandas()
    )
    # Decode indexed label back to original ASPRS class id for display
    per_class["class_id"]   = per_class["label"].astype(int).map(idx_to_class)
    per_class["class_name"] = per_class["class_id"].map(CLASS_NAMES)
    for _, row in per_class.iterrows():
        mlflow.log_metric(f"acc_{row['class_name']}", row["accuracy"])

    from mlflow.models import infer_signature
    sample_input  = train_vec.limit(100).toPandas()
    sample_preds  = final_model.transform(train_vec.limit(100)).select("prediction").toPandas()
    signature = infer_signature(sample_input, sample_preds)
    mlflow.spark.log_model(
        final_model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        signature=signature,
    )

print(f"\nFinal test accuracy: {final_acc:.4f}")
print(f"Model registered: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md ## 6. Per-Class Accuracy

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2ecc71" if a >= 0.9 else "#f39c12" if a >= 0.75 else "#e74c3c"
          for a in per_class["accuracy"]]
bars = ax.bar(per_class["class_name"], per_class["accuracy"] * 100, color=colors, edgecolor="white")
ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in per_class["accuracy"]], padding=4, fontsize=11)
ax.axhline(final_acc * 100, color="#333", linestyle="--", linewidth=1.5,
           label=f"Overall: {final_acc*100:.1f}%")
ax.set_ylim(0, 110); ax.set_xlabel("Class"); ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-Class Accuracy — Final Model (Best Hyperparams)", fontsize=13, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md ## 7. Hyperparameter Importance

# COMMAND ----------

importances = optuna.importance.get_param_importances(study)
imp_df = pd.DataFrame(importances.items(), columns=["param", "importance"]).sort_values("importance")

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(imp_df["param"], imp_df["importance"], color="#5B9BD5", edgecolor="white")
ax.set_xlabel("Importance Score (FAnova)")
ax.set_title("Hyperparameter Importance — Optuna", fontsize=13, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md ## 8. Save Predictions to Delta

# COMMAND ----------

# Use indexed labels (StringIndexer output) for SparkXGBClassifier inference
all_vec   = assembler.transform(label_indexer_model.transform(sdf)).select("features", "label", "tile_name", "classification")
all_preds = final_model.transform(all_vec)

# Map predicted index back to original class id
idx_to_class_expr = F.create_map([F.lit(x) for pair in idx_to_class.items() for x in pair])
save_df = all_preds.select(
    F.col("classification"),
    idx_to_class_expr[F.col("prediction").cast("int")].alias("predicted_class"),
    "tile_name",
    (F.col("prediction") == F.col("label")).alias("correct"),
)

(save_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(RESULT_TABLE))

full_acc = (save_df
    .agg(F.avg(F.col("correct").cast("int")).alias("acc"))
    .collect()[0]["acc"])

print(f"Full-dataset accuracy: {full_acc:.4f}")
print(f"Predictions saved → {RESULT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | | Value |
# MAGIC |---|---|
# MAGIC | Tuning framework | Optuna (replaces deprecated Hyperopt) |
# MAGIC | Trials | 20 (TPE sampler) |
# MAGIC | Trial parallelism | 4 concurrent threads → GPU workers |
# MAGIC | Per-trial model | `XGBClassifier(device="cuda")` on broadcast data |
# MAGIC | Final model | `SparkXGBClassifier(num_workers=4, device="cuda")` |
# MAGIC | Registered model | `chlor.lidar_schema.lidar_xgb_classifier` |
# MAGIC | Predictions table | `chlor.lidar_schema.point_cloud_hyperopt_preds` |
# MAGIC | MLflow experiment | `/Users/<YOUR_EMAIL>/lidar-optuna` |
