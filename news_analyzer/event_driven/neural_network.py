import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from SHARED_CONSTANTS import NUMERICAL_FEATURE_NAMES


class RoBERTaLSTMCNN(nn.Module):
    """
    A multi-task hybrid neural network that combines RoBERTa, LSTM, and CNN.
    It predicts three outputs simultaneously: direction, confidence, and volatility.
    """

    def __init__(
        self,
        num_numerical_features: int = len(
            NUMERICAL_FEATURE_NAMES
        ),  # Dynamically set from constants
        num_classes: int = 3,
        dropout_rate: float = 0.4,
    ):
        super(RoBERTaLSTMCNN, self).__init__()

        # 1. RoBERTa Text Processing Branch (Shared Base)
        roberta_config = RobertaConfig.from_pretrained(
            "roberta-base",
            output_attentions=True,  # Configure to output attentions at initialization
            attn_implementation="eager",  # Use eager attention to support output_attentions
            torch_dtype=torch.float32,  # Explicitly set dtype
        )
        # Initialize RoBERTa
        self.roberta = RobertaModel.from_pretrained(
            "roberta-base",
            config=roberta_config,
            torch_dtype=torch.float32,
        )

        roberta_output_dim = self.roberta.config.hidden_size  # 768

        # 2. LSTM Branch (Shared Base)
        self.lstm = nn.LSTM(
            roberta_output_dim,
            256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate,
        )

        # 3. CNN Branch (Shared Base)
        # Using 1D convolutions for sequence data
        self.conv1 = nn.Conv1d(roberta_output_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(roberta_output_dim, 128, kernel_size=5, padding=2)
        # Additional CNN layer for more complex feature extraction
        self.conv3 = nn.Conv1d(roberta_output_dim, 128, kernel_size=7, padding=3)

        # 4. Numerical Features Branch (Shared Base)
        self.numerical_processor = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 5. Combined Feature Dimension
        # Updated combined_dim to account for the new conv3 output
        combined_dim = (256 * 2) + (128 * 3) + 64  # LSTM (bidirectional) + CNNs (3 layers) + Numerical

        # --- Multi-Task Output Heads ---

        # Head 1: Direction Classifier
        self.direction_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

        # Head 2: Confidence Predictor (predicts a value between 0 and 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Head 3: Volatility Predictor (predicts a value between 0 and 1)
        self.volatility_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, numerical_features, token_type_ids=None):
        # Ensure all inputs are on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        numerical_features = numerical_features.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # RoBERTa forward pass
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # Pass token_type_ids to the underlying model. It will be None if not provided,
            # which is the correct behavior for RoBERTa and prevents potential errors.
            token_type_ids=token_type_ids,
        )
        roberta_output = roberta_outputs.last_hidden_state
        attentions = roberta_outputs.attentions

        # Process through LSTM
        # lstm_out contains all hidden states, while hidden contains the final states.
        _, (hidden, _) = self.lstm(roberta_output)

        # For a bidirectional LSTM, a robust representation is to concatenate the
        # final hidden states from the last layer of both directions.
        # hidden shape: (num_layers * 2, batch_size, hidden_size)
        lstm_features = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # CNN processing
        cnn_input = roberta_output.permute(0, 2, 1) # Permute for Conv1d: (batch_size, features, sequence_length)
        cnn_out1 = torch.relu(self.conv1(cnn_input))
        cnn_out2 = torch.relu(self.conv2(cnn_input))
        cnn_out3 = torch.relu(self.conv3(cnn_input)) # New CNN layer

        # Max pooling over time for each CNN output
        cnn_features = torch.cat(
            (torch.max(cnn_out1, dim=2).values,
             torch.max(cnn_out2, dim=2).values,
             torch.max(cnn_out3, dim=2).values), # Concatenate new CNN features
            dim=1,
        )

        # Process numerical features
        numerical_out = self.numerical_processor(numerical_features)

        # Combine all features
        combined_features = torch.cat(
            (lstm_features, cnn_features, numerical_out), dim=1
        )

        # Get predictions from each head
        direction_logits = self.direction_head(combined_features)
        confidence_pred = self.confidence_head(combined_features)
        volatility_pred = self.volatility_head(combined_features)

        # Return the attentions as the fourth item
        return (
            direction_logits,
            confidence_pred.squeeze(-1),
            volatility_pred.squeeze(-1),
            attentions,
        )
