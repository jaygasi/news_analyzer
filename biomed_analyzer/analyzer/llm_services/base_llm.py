# biomed_analyzer/analyzer/llm_services/base_llm.py
from abc import ABC, abstractmethod
from typing import Optional
# Ensure correct relative imports if models or utils are in the same package level or sub-package
from ..models import LLMAnalysisResult 
from ..utils import clean_text_for_llm 

class BaseLLM(ABC):
    @abstractmethod
    def analyze_news_sentiment(
        self,
        headline: str,
        summary: Optional[str] = None,
        article_text: Optional[str] = None, 
        ticker: Optional[str] = None
    ) -> Optional[LLMAnalysisResult]:
        pass

    def _construct_prompt(self, headline: str, summary: Optional[str], article_text: Optional[str], ticker: Optional[str]) -> str:
        cleaned_headline = clean_text_for_llm(headline)
        cleaned_summary = clean_text_for_llm(summary if summary else "")
        # cleaned_article_text = clean_text_for_llm(article_text if article_text else "") # If using article text

        prompt = f"""
        You are a specialized financial analyst for the biomedical and pharmaceutical sector.
        Your task is to analyze news for its potential short-term impact on the stock price of the involved company.
        Focus on news related to clinical trials, research findings, study results, FDA announcements, drug development milestones, or similar pivotal events.

        Company Ticker: {ticker if ticker else "Not specified, infer from context"}
        News Headline: "{cleaned_headline}"
        News Summary (if available): {cleaned_summary if cleaned_summary else "No summary provided."}
        
        Based ONLY on the provided headline and summary:
        1. Predict the likely short-term stock price movement: UP, DOWN, or NEUTRAL.
        2. Provide a confidence score for this prediction: An integer from 1 (low) to 10 (high).
        3. Briefly state your reasoning, highlighting key factors from the news.

        Output your response in the following exact format, with each field on a new line:
        Predicted Movement: [UP, DOWN, or NEUTRAL]
        Confidence: [Integer from 1 to 10]
        Reasoning: [Your concise reasoning (1-2 sentences)]

        Example Reasoning: "Positive Phase 2 trial results for their main drug candidate are likely to boost investor confidence." OR "FDA rejection of their new drug application will negatively impact the stock." OR "The study results were inconclusive and match expectations, likely leading to neutral stock movement."
        """
        return prompt