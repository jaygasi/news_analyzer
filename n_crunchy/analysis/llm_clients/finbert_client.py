import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple
from .base_llm_client import BaseLLMClient
from utils.utils import get_logger
from utils.exceptions import LLMAPIError

logger = get_logger(__name__)

class FinBERTClient(BaseLLMClient):
    """
    Client for local FinBERT model for sentiment analysis.
    Maps sentiment to directional impact.
    """
    def __init__(self):
        # FinBERT doesn't use an API key in the traditional sense, but requires a model.
        # We'll override BaseLLMClient's __init__ for this.
        self.is_configured = False
        try:
            logger.info("Loading FinBERT tokenizer and model (ProsusAI/finbert)... This may take a moment.")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.is_configured = True
            logger.info("FinBERT model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.is_configured = False
            # We don't raise here, just log, so other LLMs can still run.
            # If FinBERT is crucial, consider raising MissingConfigError or similar.

    def analyze_text(self, text: str, ticker: str, title: str) -> Tuple[float, str]:
        if not self.is_configured:
            return 0.0, "FinBERT model not loaded."

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT typically outputs scores for 'positive', 'negative', 'neutral'
            # Map these to directional impact
            # Model's label mapping: 0: negative, 1: neutral, 2: positive (check model card)
            sentiment_labels = ['negative', 'neutral', 'positive']
            sentiment_scores = predictions[0].tolist() # Convert tensor to list
            
            # Get the sentiment with the highest probability
            max_score_idx = sentiment_scores.index(max(sentiment_scores))
            predicted_sentiment = sentiment_labels[max_score_idx]
            confidence = sentiment_scores[max_score_idx]

            directional_score = 0.0
            reason = f"FinBERT predicted: {predicted_sentiment.capitalize()} (Confidence: {confidence:.2f})."

            if predicted_sentiment == 'positive':
                directional_score = 0.6 * confidence # Scale by confidence
            elif predicted_sentiment == 'negative':
                directional_score = -0.6 * confidence # Scale by confidence
            # Neutral remains 0.0

            return directional_score, reason

        except Exception as e:
            logger.error(f"Error during FinBERT analysis for {ticker}: {e}")
            raise LLMAPIError(f"FinBERT analysis failed: {e}") from e