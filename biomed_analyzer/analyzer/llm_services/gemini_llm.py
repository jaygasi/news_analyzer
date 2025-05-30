# biomed_analyzer/analyzer/llm_services/gemini_llm.py
import google.generativeai as genai
from typing import Optional
import re

from .base_llm import BaseLLM
from ..models import LLMAnalysisResult
from ..config import config
from ..utils import logger 

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: Optional[str]): 
        if not api_key:
            logger.error("Gemini API key not provided to constructor.")
            raise ValueError("Gemini API key is required.")
        try:
            genai.configure(api_key=api_key)
            # Use the basic model configuration for better compatibility
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("Gemini LLM initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini LLM: {e}")
            raise

    def _parse_llm_response(self, raw_text: str, ticker: Optional[str]) -> Optional[LLMAnalysisResult]:
        try:
            data = {}
            movement_match = re.search(r"Predicted Movement:\s*(UP|DOWN|NEUTRAL)", raw_text, re.IGNORECASE)
            confidence_match = re.search(r"Confidence:\s*(\d{1,2})", raw_text, re.IGNORECASE) 
            reasoning_match = re.search(r"Reasoning:\s*(.+)", raw_text, re.IGNORECASE | re.DOTALL)

            if movement_match:
                data["predicted_movement"] = movement_match.group(1).upper()
            else:
                logger.warning(f"Gemini: Could not parse 'Predicted Movement' for {ticker or 'Unknown Ticker'}. Raw: '{raw_text[:150]}...'")
                return None
            
            if confidence_match:
                confidence_val = int(confidence_match.group(1))
                data["confidence"] = max(1, min(10, confidence_val)) # Clamped in parser
            else:
                logger.warning(f"Gemini: Could not parse 'Confidence' for {ticker or 'Unknown Ticker'}. Raw: '{raw_text[:150]}...'")
                return None

            if reasoning_match:
                data["reasoning"] = reasoning_match.group(1).strip()
            else:
                logger.warning(f"Gemini: Could not parse 'Reasoning' for {ticker or 'Unknown Ticker'}. Raw: '{raw_text[:150]}...'")
                return None
            
            return LLMAnalysisResult(**data) # Pydantic model handles validation

        except ValueError as ve: 
            logger.error(f"Gemini: Error converting parsed confidence to int for {ticker or 'Unknown Ticker'}: {ve}. Raw: '{raw_text[:150]}...'")
            return None
        except Exception as e: 
            logger.error(f"Gemini: General error parsing LLM response for {ticker or 'Unknown Ticker'}: {e}. Raw: '{raw_text[:150]}...'")
            return None

    def analyze_news_sentiment(
        self,
        headline: str,
        summary: Optional[str] = None,
        article_text: Optional[str] = None, 
        ticker: Optional[str] = None
    ) -> Optional[LLMAnalysisResult]:
        
        prompt = self._construct_prompt(
            headline=headline, summary=summary,
            article_text=article_text, ticker=ticker
        )

        try:
            # Use simple generate_content call without request_options
            # This is compatible with all versions of google-generativeai
            response = self.model.generate_content(prompt)
            
            # Check if response has text
            if hasattr(response, 'text') and response.text:
                raw_text_response = response.text.strip()
                parsed_analysis = self._parse_llm_response(raw_text_response, ticker)
                return parsed_analysis
            else:
                logger.error(f"Gemini API returned empty response for {ticker or 'N/A'}")
                return None

        except Exception as e: 
            logger.error(f"Error during Gemini API call for {ticker or 'N/A'}: {e}")
            if "API key not valid" in str(e): 
                logger.critical("GEMINI API KEY IS INVALID or not authorized. Check .env and Google AI Studio.")
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                logger.error("Gemini API quota/rate limit exceeded. Consider upgrading or waiting.")
            elif "request_options" in str(e).lower():
                logger.error("Incompatible API version - using fallback method")
            return None