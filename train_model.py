"""
Train/export the UrbanNest deep housing-price model.

Place the original cleaned dataset at:
    data/cleaned_real_estate_final_price_v2.csv

Then run:
    python train_model.py

This follows the preprocessing and deep-network architecture in the supplied
UrbanNest notebook, with the missing input_dim explicitly derived from the
processed training matrix.
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/cleaned_real_estate_final_price_v2.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Copy your original cleaned_real_estate_final_price_v2.csv into the data folder."
    )

df = pd.read_csv(DATA_PATH)
df = df.dropna().reset_index(drop=True)

X = df.drop(columns=["price"])
y = df["price"]

# Match notebook stratification logic
df["price_decile"] = pd.qcut(
    df["price"], q=10, labels=False, duplicates="drop"
)
df["strata"] = (
    df["state"].astype(str) + "_" + df["price_decile"].astype(str)
)

# The notebook repairs rare strata before splitting.
# For a clean exported training pipeline, combine any singleton strata into
# a common fallback group.
counts = df["strata"].value_counts()
rare = counts[counts < 2].index
df.loc[df["strata"].isin(rare), "strata"] = "Other"

X = df.drop(columns=["price", "price_decile", "strata"])
y = df["price"]

X_temp, X_test, y_temp, y_test, strata_temp, strata_test = train_test_split(
    X, y, df["strata"],
    test_size=0.20,
    random_state=42,
    stratify=df["strata"]
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.20,
    random_state=42,
    stratify=strata_temp
)

categorical_cols = ["status", "city", "state", "zip3"]
numerical_cols = [
    "bed",
    "bath",
    "acre_lot",
    "house_size",
    "has_prior_sale",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            categorical_cols,
        ),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

y_train = y_train.to_numpy(dtype="float32")
y_val = y_val.to_numpy(dtype="float32")
y_test = y_test.to_numpy(dtype="float32")

# This is the fix for the notebook's NameError.
input_dim = X_train_processed.shape[1]

deep_model = keras.Sequential([
    layers.Input(shape=(input_dim,)),

    layers.Dense(128),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dropout(0.20),

    layers.Dense(64),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dropout(0.20),

    layers.Dense(32),
    layers.LeakyReLU(negative_slope=0.2),

    layers.Dense(16),
    layers.LeakyReLU(negative_slope=0.2),

    layers.Dense(1),
])

deep_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss="mse",
    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

deep_model.fit(
    X_train_processed,
    y_train,
    validation_data=(X_val_processed, y_val),
    epochs=300,
    batch_size=8192,
    callbacks=[early_stopping],
    verbose=1,
)

deep_model.save(os.path.join(ARTIFACT_DIR, "housing_price_model.keras"))
joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))

print("\nArtifacts created successfully:")
print(" - artifacts/housing_price_model.keras")
print(" - artifacts/preprocessor.joblib")
print(f"Processed input dimension: {input_dim}")
