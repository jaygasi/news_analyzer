# --- ml_models.py ---
import torch
from transformers import RobertaTokenizer
import asyncio
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from neural_network import RoBERTaLSTMCNN
import database
from SHARED_CONSTANTS import NUMERICAL_FEATURE_NAMES

# --- Constants ---
NUMERICAL_PROCESSOR_WEIGHT_KEY = "numerical_processor.0.weight"


class ProfitMaximizingModel:
    """
    Wrapper for the RoBERTa+LSTM+CNN multi-task model.
    It handles loading the model, preprocessing data, and interpreting the multi-task predictions.
    NEW: It can now explain its predictions by analyzing attention weights.
    """

    def __init__(self):
        """Lightweight, non-blocking constructor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing model on device: {self.device}")
        self.tokenizer = None
        self.model = None
        self.scaler = None
        self.scaler_columns = None
        self.is_trained = False
        self.idx_to_label = {0: "LONG", 1: "SHORT", 2: "NONE"}
        self.numerical_feature_names = NUMERICAL_FEATURE_NAMES

    async def initialize(self):
        """Asynchronously initializes the model, tokenizer, and scaler."""
        print(f"ProfitMaximizingModel using device: {self.device}")
        loop = asyncio.get_running_loop()
        # Run the entire blocking initialization in an executor to not block the event loop
        await loop.run_in_executor(None, self._initialize_sync)

    def _initialize_sync(self):
        """Contains the synchronous initialization logic."""
        print("Searching for latest model version...")
        artifacts = database.get_latest_model_path()
        model_path = artifacts.get("model_version")
        scaler_path = artifacts.get("scaler_path")

        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Initialize model structure on the CPU directly.
        self.model = RoBERTaLSTMCNN(
            num_numerical_features=len(self.numerical_feature_names), num_classes=3
        )

        # Handle model loading
        if model_path:
            try:
                # Load state dict onto the CPU
                state_dict = torch.load(model_path, map_location="cpu")

                # If the saved model has different feature dimensions, reshape the weights
                if NUMERICAL_PROCESSOR_WEIGHT_KEY in state_dict:
                    old_weight = state_dict[NUMERICAL_PROCESSOR_WEIGHT_KEY]
                    if old_weight.shape[1] != len(self.numerical_feature_names):
                        print(
                            f"Adjusting numerical feature dimension from {old_weight.shape[1]} to {len(self.numerical_feature_names)}"
                        )
                        # Initialize new weight matrix with zeros
                        new_weight = torch.zeros(
                            (old_weight.shape[0], len(self.numerical_feature_names))
                        )
                        # Copy over the weights for features we're keeping
                        min_features = min(
                            old_weight.shape[1], len(self.numerical_feature_names)
                        )
                        new_weight[:, :min_features] = old_weight[:, :min_features]
                        state_dict[NUMERICAL_PROCESSOR_WEIGHT_KEY] = new_weight

                # Load the weights. Use assign=True to load a state_dict into a model
                # that was initialized on the 'meta' device. This is the correct way
                # to handle this pattern in modern PyTorch/Transformers.
                self.model.load_state_dict(state_dict, strict=False, assign=True)
                print("Model weights loaded and adjusted successfully")
                self.is_trained = True
            except Exception as e:
                print(f"Error loading model weights: {e}")
                self.is_trained = False
        else:
            print("No model path found. Using untrained model.")
            self.is_trained = False

        # Move the fully loaded model to the target device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load scaler
        try:
            if scaler_path:
                scaler_obj = joblib.load(scaler_path)
                if (
                    isinstance(scaler_obj, dict)
                    and "scaler" in scaler_obj
                    and "columns" in scaler_obj
                ):
                    self.scaler = scaler_obj.get("scaler")
                    self.scaler_columns = scaler_obj.get("columns")
                else:  # Backwards compatibility with old scaler files
                    self.scaler = scaler_obj
                    self.scaler_columns = None
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None
            self.scaler_columns = None

    def _prepare_inputs(
        self, news_text: str, numerical_features_dict: dict
    ) -> Dict[str, torch.Tensor]:
        """Prepares model inputs and ensures they're on the correct device."""
        # Tokenize text
        inputs = self.tokenizer(
            news_text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move input tensors to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Process numerical features
        numerical_features = self._process_numerical_features(numerical_features_dict)
        numerical_tensor = torch.tensor(numerical_features, dtype=torch.float32).to(
            self.device
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "numerical_features": numerical_tensor,
        }

    def _process_numerical_features(self, numerical_features_dict: dict) -> np.ndarray:
        """Process and scale numerical features."""
        if self.scaler and self.scaler_columns:
            # Create DataFrame with expected columns
            df = pd.DataFrame([numerical_features_dict])
            for col in self.scaler_columns:
                if col not in df.columns:
                    df[col] = 0.0
            df_ordered = df[self.scaler_columns]
            return self.scaler.transform(df_ordered)
        else:
            # Fallback to unscaled features
            features = []
            for name in self.numerical_feature_names:
                features.append(numerical_features_dict.get(name, 0.0))
            return np.array([features])

    def _get_explanation_from_attentions(
        self, attentions: tuple, input_ids: torch.Tensor, top_n: int = 5
    ) -> list[str]:
        """
        Processes attention scores to find the most influential words in the input text.
        """
        # --- FIX: Add a guard clause to handle cases where attentions are not returned ---
        if not attentions or not isinstance(attentions, tuple) or len(attentions) == 0:
            logging.warning("No attention weights found or returned, cannot generate explanation.")
            return []

        # 1. Select attentions from the last layer and average across all heads.
        last_layer_attentions = attentions[-1].squeeze(
            0
        )  # Shape: [num_heads, seq_len, seq_len]
        avg_attentions = last_layer_attentions.mean(dim=0)  # Shape: [seq_len, seq_len]

        # 2. Get a single importance score for each token by summing the attention it received.
        # We are interested in how much attention each token *receives* from all other tokens (including itself).
        # So, sum over the second dimension (dim=1) if avg_attentions[i, j] means attention from token j to token i.
        # Or sum over dim=0 if it means attention from token i to token j.
        # For RoBERTa, output_attentions is a tuple of (batch_size, num_heads, seq_len, seq_len)
        # After squeeze(0) and mean(dim=0), we have (seq_len, seq_len).
        # Let's assume attentions[i, j] is attention from token j to token i.
        # We want to sum the attention *paid to* each token.
        token_importance_scores = avg_attentions.sum(
            dim=0
        )  # Summing columns: attention *to* token j from all others.

        # 3. Get tokens from input_ids, ignoring special tokens.
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0).tolist()
        )  # Ensure input_ids is on CPU and list

        # 4. Pair tokens with scores and sort to find the most important ones.
        scored_tokens = []
        for token, score in zip(tokens, token_importance_scores):
            if token not in [
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            ]:
                scored_tokens.append((token, score.item()))

        scored_tokens.sort(key=lambda x: x[1], reverse=True)

        # 5. Merge sub-tokens (e.g., 'Ġanal', 'yst') back into whole words and correctly average their scores.
        merged_tokens = []
        for token, score in scored_tokens:
            if token.startswith("Ġ"):  # 'Ġ' indicates a new word/space
                merged_tokens.append({"text": token[1:], "score_sum": score, "token_count": 1})
            elif merged_tokens:  # Check if merged_tokens is not empty
                # Append to the last word if it's a sub-token
                merged_tokens[-1]["text"] += token
                merged_tokens[-1]["score_sum"] += score
                merged_tokens[-1]["token_count"] += 1
            # If a token doesn't start with 'Ġ' and merged_tokens is empty, it's a subword
            # at the beginning of the text (e.g., from truncation). We'll ignore it for simplicity
            # as we can't reconstruct the full word.

        # Calculate the true average score for each merged token
        for token_data in merged_tokens:
            token_data["score"] = token_data["score_sum"] / token_data["token_count"]

        # 6. Return the top N most influential words/phrases.
        merged_tokens.sort(key=lambda x: x["score"], reverse=True)
        return [token["text"] for token in merged_tokens[:top_n]]

    def predict(self, news_text: str, numerical_features: dict) -> dict:
        """Makes a prediction with proper error handling and device management."""
        if not self.is_trained:
            return {
                "direction": "NONE",
                "predicted_confidence": 0.0,
                "predicted_volatility": 0.0,
                "explanation": [],
            }

        try:
            self.model.eval()  # Ensure model is in evaluation mode
            with torch.no_grad():
                # Prepare inputs (moves to correct device)
                model_inputs = self._prepare_inputs(news_text, numerical_features)

                # Get predictions
                direction_logits, confidence_pred, volatility_pred, attentions = (
                    self.model(**model_inputs)
                )

                # Process results
                probabilities = torch.softmax(direction_logits, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1)
                predicted_label = self.idx_to_label[predicted_idx.item()]

                # Get explanation using attention weights
                explanation = self._get_explanation_from_attentions(
                    attentions, model_inputs["input_ids"]
                )

                return {
                    "direction": predicted_label,
                    "predicted_confidence": float(
                        confidence_pred.cpu().item()
                    ),  # Ensure we can serialize
                    "predicted_volatility": float(
                        volatility_pred.cpu().item()
                    ),  # Ensure we can serialize
                    "explanation": explanation,
                }

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return {
                "direction": "NONE",
                "predicted_confidence": 0.0,
                "predicted_volatility": 0.0,
                "explanation": [],
                "error": str(e),
            }
