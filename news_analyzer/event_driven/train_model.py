# --- train_model.py ---
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    f1_score,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from datetime import datetime
from typing import List, Dict

from neural_network import RoBERTaLSTMCNN
import database
from SHARED_CONSTANTS import NUMERICAL_FEATURE_NAMES
from config import (
    LABEL_PROFIT_THRESHOLD,
    LABEL_LOSS_THRESHOLD,
    LABEL_MAX_VOLATILITY_RANGE,
    LABEL_MAX_CONFIDENCE_PERF,
)

# --- Configuration ---
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = (
    4  # Simulate a larger batch size (BATCH_SIZE * STEPS = 32)
)


class TradingDataset(Dataset):
    """Custom PyTorch Dataset for our trading data."""

    def __init__(
        self,
        texts: List[str],
        numerical_features: pd.DataFrame,
        direction_labels: List[int],
        confidence_labels: List[float],
        volatility_labels: List[float],
        tokenizer: RobertaTokenizer,
    ):
        self.texts = texts
        if isinstance(numerical_features, pd.DataFrame):
            self.numerical_features = torch.tensor(
                numerical_features.values, dtype=torch.float32
            )
        else:
            self.numerical_features = torch.tensor(
                numerical_features, dtype=torch.float32
            )

        self.direction_labels = torch.tensor(direction_labels, dtype=torch.long)
        self.confidence_labels = torch.tensor(confidence_labels, dtype=torch.float32)
        self.volatility_labels = torch.tensor(volatility_labels, dtype=torch.float32)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "numerical_features": self.numerical_features[idx],
            "direction_labels": self.direction_labels[idx],
            "confidence_labels": self.confidence_labels[idx],
            "volatility_labels": self.volatility_labels[idx],
        }


def create_multi_task_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates outcome labels for direction, confidence, and volatility
    using the BEST performance across multiple intraday checkpoints, making the model smarter.
    """
    perf_cols = [
        "perf_30_min_pct",
        "perf_60_min_pct",
        "perf_240_min_pct",
        "perf_eod_pct",
    ]

    # Drop rows where all performance data is missing, as they are unusable for labeling.
    df = df.dropna(subset=perf_cols, how="all").copy()

    # Ensure all performance columns are numeric, fill any remaining NaNs with 0
    for col in perf_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.fillna({col: 0 for col in perf_cols}, inplace=True)

    if df.empty:
        print(
            "Warning: No rows remaining after dropping trades with no performance data. Cannot create labels."
        )
        return df

    # --- Step 1: Determine the best outcome for labeling ---
    df["best_gain"] = df[perf_cols].max(axis=1)
    df["worst_loss"] = df[perf_cols].min(axis=1)

    # --- Step 2: Create Direction Label ---
    # Define thresholds for a "significant" move
    def get_direction_label(row) -> str:
        if row["best_gain"] >= LABEL_PROFIT_THRESHOLD:
            return "LONG"
        elif row["worst_loss"] <= LOSS_THRESHOLD:
            return "SHORT"
        else:
            return "NONE"

    df["direction_label"] = df.apply(get_direction_label, axis=1)

    # --- Step 3: Create Volatility Label ---
    # Volatility is the range of outcomes. A larger range means higher volatility.
    df["perf_range"] = df["best_gain"] - df["worst_loss"]
    # Normalize to 0-1, clipping at a max range to prevent outliers from dominating.
    df["volatility_label"] = np.clip(df["perf_range"] / LABEL_MAX_VOLATILITY_RANGE, 0, 1)

    # --- Step 4: Create Confidence Label ---
    # Confidence is the magnitude of the move in the labeled direction.
    def get_confidence_magnitude(row):
        if row["direction_label"] == "LONG":
            return row["best_gain"]
        elif row["direction_label"] == "SHORT":
            return abs(row["worst_loss"])
        else:
            # For 'NONE' trades, confidence is high if the move was small.
            # We'll use 1 minus the normalized max absolute deviation from zero.
            max_abs_move = max(abs(row["best_gain"]), abs(row["worst_loss"]))
            normalized_deviation = min(max_abs_move / LABEL_PROFIT_THRESHOLD, 1.0)
            return 1.0 - normalized_deviation

    df["confidence_magnitude"] = df.apply(get_confidence_magnitude, axis=1)

    # For LONG/SHORT, normalize against a max performance cap
    long_short_mask = df["direction_label"].isin(["LONG", "SHORT"])
    df.loc[long_short_mask, "confidence_label"] = np.clip(
        df.loc[long_short_mask, "confidence_magnitude"] / LABEL_MAX_CONFIDENCE_PERF, 0, 1
    )

    # For NONE, the confidence_magnitude is already scaled between 0 and 1.
    df.loc[~long_short_mask, "confidence_label"] = df.loc[
        ~long_short_mask, "confidence_magnitude"
    ]

    # Ensure no NaNs in the final label
    df["confidence_label"].fillna(0.5, inplace=True)

    # Drop intermediate columns
    df = df.drop(
        columns=["best_gain", "worst_loss", "perf_range", "confidence_magnitude"]
    )

    return df


def train_model():
    """Main function to load data, train the model, save it with a version, and log performance."""
    print("--- Starting Multi-Task Model Training Process ---")

    training_data = database.get_training_data()
    if not training_data:
        print(
            "Error: No training data found in the database. Run price_tracker.py first."
        )
        return

    df = pd.DataFrame(training_data)
    df = create_multi_task_labels(df)

    if df.empty:
        print("Error: No valid data available for training after labeling. Exiting.")
        return

    print("Labeled training data counts:\n", df["direction_label"].value_counts())

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    numerical_feature_columns = [f"feature_{name}" for name in NUMERICAL_FEATURE_NAMES]

    df_for_split = df[
        numerical_feature_columns
        + ["news_text", "direction_label", "confidence_label", "volatility_label"]
    ].copy()
    df_for_split[numerical_feature_columns] = df_for_split[
        numerical_feature_columns
    ].fillna(0)

    train_df, val_df = train_test_split(
        df_for_split,
        test_size=0.2,
        random_state=42,
        stratify=df_for_split["direction_label"],
    )

    print("Fitting feature scaler...")
    scaler = StandardScaler()
    train_numerical_scaled_array = scaler.fit_transform(
        train_df[numerical_feature_columns]
    )
    val_numerical_scaled_array = scaler.transform(val_df[numerical_feature_columns])

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    scaler_filename = f"feature_scaler_{timestamp_str}.pkl"
    scaler_and_columns = {"scaler": scaler, "columns": numerical_feature_columns}
    joblib.dump(scaler_and_columns, scaler_filename)
    print(f"Feature scaler and column order have been saved to '{scaler_filename}'.")

    train_numerical_scaled_df = pd.DataFrame(
        train_numerical_scaled_array,
        columns=numerical_feature_columns,
        index=train_df.index,
    )
    val_numerical_scaled_df = pd.DataFrame(
        val_numerical_scaled_array,
        columns=numerical_feature_columns,
        index=val_df.index,
    )

    label_map = {"LONG": 0, "SHORT": 1, "NONE": 2}
    train_dataset = TradingDataset(
        texts=train_df["news_text"].tolist(),
        numerical_features=train_numerical_scaled_df,
        direction_labels=train_df["direction_label"].map(label_map).tolist(),
        confidence_labels=train_df["confidence_label"].tolist(),
        volatility_labels=train_df["volatility_label"].tolist(),
        tokenizer=tokenizer,
    )
    val_dataset = TradingDataset(
        texts=val_df["news_text"].tolist(),
        numerical_features=val_numerical_scaled_df,
        direction_labels=val_df["direction_label"].map(label_map).tolist(),
        confidence_labels=val_df["confidence_label"].tolist(),
        volatility_labels=val_df["volatility_label"].tolist(),
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoBERTaLSTMCNN(
        num_numerical_features=len(numerical_feature_columns), num_classes=3
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- NEW: Add a learning rate scheduler ---
    num_training_steps = (len(train_loader) * NUM_EPOCHS) // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    weights = compute_class_weight(
        "balanced", classes=np.unique(df["direction_label"]), y=df["direction_label"]
    )
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(
        f"Calculated Class Weights: {dict(zip(np.unique(df['direction_label']), weights))}"
    )

    loss_dir_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss_reg_fn = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical_features = batch["numerical_features"].to(device)
            dir_labels = batch["direction_labels"].to(device)
            conf_labels = batch["confidence_labels"].to(device)
            vol_labels = batch["volatility_labels"].to(device)

            dir_logits, conf_preds, vol_preds, _ = model(
                input_ids, attention_mask, numerical_features
            )

            loss_dir = loss_dir_fn(dir_logits, dir_labels)
            loss_conf = loss_reg_fn(conf_preds, conf_labels)
            loss_vol = loss_reg_fn(vol_preds, vol_labels)

            # Combine and scale loss for accumulation
            loss = (
                loss_dir + (0.5 * loss_conf) + (0.5 * loss_vol)
            ) / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            total_loss += loss.item()

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss * GRADIENT_ACCUMULATION_STEPS / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

    print("\n--- Validating Model ---")
    model.eval()
    all_dir_preds, all_dir_labels = [], []
    all_conf_preds, all_conf_labels = [], []
    all_vol_preds, all_vol_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical_features = batch["numerical_features"].to(device)
            dir_labels, conf_labels, vol_labels = (
                batch["direction_labels"],
                batch["confidence_labels"],
                batch["volatility_labels"],
            )

            dir_logits, conf_preds, vol_preds, _ = model(
                input_ids, attention_mask, numerical_features
            )

            all_dir_preds.extend(torch.argmax(dir_logits, dim=1).cpu().numpy())
            all_dir_labels.extend(dir_labels.numpy())
            all_conf_preds.extend(conf_preds.cpu().numpy())
            all_conf_labels.extend(conf_labels.numpy())
            all_vol_preds.extend(vol_preds.cpu().numpy())
            all_vol_labels.extend(vol_labels.numpy())

    print("\n--- Direction Classification Report ---")
    idx_to_label = {v: k for k, v in label_map.items()}
    pred_labels = [idx_to_label[int(p)] for p in all_dir_preds]
    true_labels = [idx_to_label[int(l)] for l in all_dir_labels]
    report_dict = classification_report(
        true_labels, pred_labels, zero_division=0, output_dict=True
    )
    print(classification_report(true_labels, pred_labels, zero_division=0))

    print("\n--- Regression Performance ---")
    conf_mse = mean_squared_error(all_conf_labels, all_conf_preds)
    vol_mse = mean_squared_error(all_vol_labels, all_vol_preds)
    print(f"Confidence Prediction MSE: {conf_mse:.4f}")
    print(f"Volatility Prediction MSE: {vol_mse:.4f}")

    model_version_str = f"roberta_hybrid_{timestamp_str}.pth"
    training_log_data = {
        "training_timestamp": datetime.now().isoformat(),
        "model_version": model_version_str,
        "scaler_path": scaler_filename,
        "avg_training_loss": avg_train_loss,
        "val_accuracy": report_dict["accuracy"],
        "val_f1_macro": report_dict["macro avg"]["f1-score"],
        "val_confidence_mse": conf_mse,
        "val_volatility_mse": vol_mse,
        "notes": f"Trained on {len(df)} samples.",
    }
    database.log_training_run(training_log_data)
    print("\nTraining run performance has been logged to the database.")

    torch.save(model.state_dict(), model_version_str)
    print(f"\n--- Model training complete. Saved to '{model_version_str}' ---")


if __name__ == "__main__":
    train_model()
