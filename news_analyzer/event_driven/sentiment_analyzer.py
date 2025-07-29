import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import asyncio
import functools
import logging


class SentimentAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"SentimentAnalyzer: Device set to use {self.device}")
        self.finbert_pipeline = None
        self.secondary_model_pipeline = None

    async def initialize(self):
        """Asynchronously initializes the sentiment models in a non-blocking way."""
        loop = asyncio.get_running_loop()
        device_num = 0 if self.device == "cuda" else -1

        logging.info("SentimentAnalyzer: Starting model initialization...")
        load_finbert_task = loop.run_in_executor(
            None, functools.partial(self._load_model, "ProsusAI/finbert", device_num)
        )
        load_finbert_tone_task = loop.run_in_executor(
            None,
            functools.partial(self._load_model, "yiyanghkust/finbert-tone", device_num),
        )

        self.finbert_pipeline, self.secondary_model_pipeline = await asyncio.gather(
            load_finbert_task, load_finbert_tone_task
        )

        if self.finbert_pipeline and self.secondary_model_pipeline:
            logging.info("SentimentAnalyzer: FinBERT and FinBERT-Tone models loaded successfully.")
        else:
            logging.error(
                "SentimentAnalyzer: One or both sentiment models failed to load. Sentiment analysis may be limited."
            )

    def _load_model(self, model_name: str, device: int):
        """
        Helper function to load a pipeline with explicit device placement.
        """
        try:
            logging.info(f"SentimentAnalyzer: Loading model: {model_name} on device: {device}")

            # Determine the target device string for torch
            target_device = "cuda" if device == 0 and torch.cuda.is_available() else "cpu"

            # Manually load model and tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Explicitly move the model to the target device to avoid 'meta' tensor issues.
            model.to(target_device)

            # Create pipeline with the pre-loaded components
            pipe = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=device, # This tells the pipeline where to run inference
                batch_size=8,
            )
            logging.info(f"SentimentAnalyzer: Successfully loaded and moved {model_name} to {target_device}.")
            return pipe
        except ImportError as ie:
            logging.error(f"SentimentAnalyzer: Missing library for {model_name}: {ie}. Please ensure transformers and torch are installed.")
            return None
        except Exception as e:
            logging.error(f"SentimentAnalyzer: Error loading pipeline for model {model_name}: {e}", exc_info=True)
            return None

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of a single text.
        NOTE: For performance, use `analyze_sentiments_batch` for multiple texts.
        """
        if not text:
            return {
                "finbert_sentiment": {"label": "neutral", "score": 0.5},
                "finbert_tone_sentiment": {"label": "neutral", "score": 0.5},
            }
        results = self.analyze_sentiments_batch([text])
        return results[0]

    @torch.no_grad()
    def analyze_sentiments_batch(self, texts: list[str]) -> list[dict]:
        """
        Analyzes a batch of texts for sentiment using two financial models.
        """
        if not texts:
            return []

        default_sentiment = {"label": "neutral", "score": 0.5}

        # --- FinBERT ---
        try:
            if self.finbert_pipeline:
                finbert_results = self.finbert_pipeline(texts, truncation=True, max_length=512)
            else:
                logging.warning("SentimentAnalyzer: FinBERT pipeline not initialized. Using default sentiment.")
                finbert_results = [default_sentiment] * len(texts)
        except Exception as e:
            logging.error(f"SentimentAnalyzer: Error during batch analysis with FinBERT: {e}", exc_info=True)
            finbert_results = [default_sentiment] * len(texts)

        # --- FinBERT-Tone ---
        try:
            if self.secondary_model_pipeline:
                finbert_tone_results = self.secondary_model_pipeline(texts, truncation=True, max_length=512)
            else:
                logging.warning("SentimentAnalyzer: FinBERT-Tone pipeline not initialized. Using default sentiment.")
                finbert_tone_results = [default_sentiment] * len(texts)
        except Exception as e:
            logging.error(f"SentimentAnalyzer: Error during batch analysis with FinBERT-Tone: {e}", exc_info=True)
            finbert_tone_results = [default_sentiment] * len(texts)

        # Combine the results
        return [
            {
                "finbert_sentiment": (
                    finbert_results[i]
                    if i < len(finbert_results)
                    else default_sentiment
                ),
                "finbert_tone_sentiment": (
                    finbert_tone_results[i]
                    if i < len(finbert_tone_results)
                    else default_sentiment
                ),
            }
            for i in range(len(texts))
        ]
