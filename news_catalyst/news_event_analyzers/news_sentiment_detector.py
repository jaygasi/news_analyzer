from langdetect import detect, LangDetectException
import re
import os
from typing import Any, Optional, Dict

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Finbert config
FINBERT_MAX_TOKENS = 512


class NewsSentimentDetector:
    """Singleton instance for News Sentiment Detection"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        # If the instance does not exist, create it
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None, device: Optional[str] = None):
        # Initialize the instance if it's the first time
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.labels = ["positive", "negative", "neutral"]
            self.model = model
            self.tokenizer = tokenizer
            self.device = device or "cpu"
            self.is_available = model is not None and tokenizer is not None

    def detect_english(self, text: str) -> bool:
        """Detect if text is in English"""
        try:
            return detect(text) == 'en'
        except (LangDetectException, Exception):
            return False

    def clean_article_text(self, text: str) -> str:
        """Clean article text by removing extra whitespace"""
        # Remove line breaks, tabs, and extra whitespace
        return re.sub(r'\s+', ' ', text)

    def detect_sentiment_for_article(self, title: str, text: str) -> float:
        """Detect sentiment for an article using FinBERT if available"""
        if not self.is_available:
            return 0.0
            
        news_text = f"{title} {text}"
        if not news_text:
            return 0.0
            
        try:
            # Truncate the news_text to the max length the model can handle
            tokens = self.tokenizer(
                news_text, 
                return_tensors="pt", 
                max_length=FINBERT_MAX_TOKENS, 
                truncation=True, 
                padding="max_length"
            ).to(self.device)

            # Forward pass to get model output
            with torch.no_grad():
                outputs = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])

            # Softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_idx = torch.argmax(probabilities, dim=1).item()
            probability = probabilities[0][sentiment_idx].item()

            # Assign positive or negative sign based on sentiment label
            if self.labels[sentiment_idx] == "positive":
                return probability  # Positive sentiment
            elif self.labels[sentiment_idx] == "negative":
                return -probability  # Negative sentiment
            else:
                return 0.0  # Neutral sentiment
        except Exception as e:
            from utils.log_utils import logw
            logw(f"Error in sentiment detection: {str(e)}")
            return 0.0

    def process_article(self, article: Dict[str, str]) -> float:
        """Process a single article dictionary with 'title' and 'text' keys"""
        # Ensure article has 'title' and 'text' keys and check language
        if not all(key in article for key in ("title", "text")):
            raise ValueError("Article dictionary must contain 'title' and 'text' keys")

        if not self.detect_english(article["text"]):
            return 0.0

        # Clean the article text and detect sentiment
        title = self.clean_article_text(article["title"])
        text = self.clean_article_text(article["text"])
        sentiment_score = self.detect_sentiment_for_article(title, text)

        return sentiment_score