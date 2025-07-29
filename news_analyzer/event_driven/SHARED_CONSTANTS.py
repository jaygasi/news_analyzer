# --- SHARED_CONSTANTS.py ---

# This list defines the numerical features the model uses.
# The order MUST be consistent between training and prediction.
NUMERICAL_FEATURE_NAMES = [
    "finbert_score",
    "finbert_tone_score",
    "keyword_score",
    "alpha_vantage_relevance",
    "alpha_vantage_sentiment",
    # --- Features from FMP Profile & Quote Data ---
    "market_cap",
    "beta", # Beta is available in the FMP profile
    "volume",
    "change_percent",
]