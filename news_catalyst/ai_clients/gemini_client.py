"""
Google Gemini AI Client for news analysis
"""
import os
from typing import Optional, Dict, Any

from utils.log_utils import *
from config import GEMINI_MODEL

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


class GeminiClient:
    """Google Gemini AI client for news analysis"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            logw("Google Gemini AI library not available. Install with: pip install google-generativeai")
            self.client = None
            self.model = None
            return
            
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name or GEMINI_MODEL
        
        if not self.api_key:
            logw("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
            self.client = None
            self.model = None
            return
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.client = genai
            logi(f"✅ Gemini client initialized successfully with model: {self.model_name}")
        except Exception as e:
            loge(f"Failed to initialize Gemini client: {str(e)}")
            self.client = None
            self.model = None

    def is_available(self) -> bool:
        """Check if Gemini client is available"""
        return self.model is not None

    def analyze_news_sentiment(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze news sentiment using Gemini"""
        if not self.is_available():
            return None
            
        try:
            prompt = f"""
            Analyze the sentiment of this news article regarding its potential to increase stock prices.
            Provide a score from 0.0 (very negative) to 1.0 (very positive), and include your confidence 
            in the analysis with a score from 0.0 to 1.0.
            
            Respond only with a JSON object in this exact format:
            {{"sentiment": 0.7, "confidence": 0.8}}
            
            News text: {text[:2000]}
            """

            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                return None
                
            # Extract JSON from response
            import json
            try:
                # Clean the response text
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                
                return {
                    'gemini_sentiment': float(result.get('sentiment', 0.5)),
                    'gemini_confidence': float(result.get('confidence', 0.5))
                }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logw(f"Failed to parse Gemini response: {str(e)}")
                return None
                
        except Exception as e:
            loge(f"Error in Gemini sentiment analysis: {str(e)}")
            return None

    def analyze_phase_results(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze phase results from news content using Gemini"""
        if not self.is_available():
            return None
            
        try:
            prompt = f"""
            Extract the following metrics from this clinical trial news content if available.
            If not available, return 0 for the metric.
            
            Metrics to extract:
            - milestone_achieved: Phase 3 clinical trial success (1=success, 0=fail)
            - orr_score: Overall Response Rate as percentage (0-1)
            - omi_score: Overall Magnitude of Improvement (0-1) 
            - sva_score: Severe Adverse Effects percentage (0-1)
            
            Respond only with JSON in this exact format:
            {{"milestone_achieved": 1, "orr_score": 0.65, "omi_score": 0.8, "sva_score": 0.1}}
            
            News content: {text[:2000]}
            """

            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                return None
                
            import json
            try:
                # Clean the response text
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                
                # Calculate phase score
                milestone_achieved = float(result.get('milestone_achieved', 0))
                orr_score = float(result.get('orr_score', 0))
                omi_score = float(result.get('omi_score', 0))
                sva_score = float(result.get('sva_score', 0))
                
                phase_score = (
                    milestone_achieved * 0.4 +
                    orr_score * 0.2 +
                    omi_score * 0.2 -
                    sva_score * 0.2
                )
                
                return {
                    'milestone_achieved': milestone_achieved,
                    'orr_score': orr_score,
                    'omi_score': omi_score,
                    'sva_score': sva_score,
                    'phase_score': phase_score
                }
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logw(f"Failed to parse Gemini phase analysis response: {str(e)}")
                return None
                
        except Exception as e:
            loge(f"Error in Gemini phase analysis: {str(e)}")
            return None

    def analyze_news_topic(self, title: str, content: str) -> Optional[Dict[str, Any]]:
        """Analyze news topic and categorize using Gemini"""
        if not self.is_available():
            return None
            
        try:
            prompt = f"""
            Analyze this news article and categorize it into one of these topics:
            - FDA_APPROVAL
            - FDA_REJECTION  
            - POSITIVE_PHASE2_RESULT
            - NEGATIVE_PHASE2_RESULT
            - POSITIVE_PHASE3_RESULT
            - NEGATIVE_PHASE3_RESULT
            - POSITIVE_FINANCIAL_PERFORMANCE
            - NEGATIVE_FINANCIAL_PERFORMANCE
            - MARKET_GROWTH
            - INNOVATION
            - STOCK_BUYBACK
            - STOCK_SPLIT
            - LAWSUIT_WIN
            - UNKNOWN
            
            Also provide a confidence score from 0.0 to 1.0 and count relevant keywords.
            
            Respond only with JSON in this exact format:
            {{"topic": "POSITIVE_PHASE3_RESULT", "confidence": 0.9, "keyword_count": 5}}
            
            Title: {title[:200]}
            Content: {content[:1800]}
            """

            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                return None
                
            import json
            try:
                # Clean the response text
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                
                return {
                    'gemini_topic': result.get('topic', 'UNKNOWN'),
                    'gemini_confidence': float(result.get('confidence', 0.0)),
                    'gemini_keyword_count': int(result.get('keyword_count', 0))
                }
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logw(f"Failed to parse Gemini topic analysis response: {str(e)}")
                return None
                
        except Exception as e:
            loge(f"Error in Gemini topic analysis: {str(e)}")
            return None