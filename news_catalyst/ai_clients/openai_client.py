from typing import Optional, Dict, Any
import json
import re

from utils.log_utils import *

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAIClient:
    """Singleton class for OpenAI APIs"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, openai_api_key: Optional[str] = None):
        # Check if the instance already exists
        if not hasattr(self, 'initialized'):
            self.initialized = True
            
            if not OPENAI_AVAILABLE:
                logw("OpenAI library not available. Install with: pip install openai")
                self.client = None
                return
                
            try:
                self.client = OpenAI(api_key=openai_api_key)
                self.openai_model = "gpt-3.5-turbo"
                logi("✅ OpenAI client initialized successfully")
            except Exception as e:
                loge(f"Failed to initialize OpenAI client: {str(e)}")
                self.client = None

    def is_available(self) -> bool:
        """Check if OpenAI client is available"""
        return self.client is not None

    def calculate_phase_result_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate a phase result score based on milestone achievement and other metrics"""
        milestone_achieved = results.get('milestone_achieved')
        if milestone_achieved is None:
            results['phase_score'] = None
            return results

        # Normalize scores and ensure they are valid
        orr_score = results.get('orr_score', 0)
        omi_score = results.get('omi_score', 0)
        sva_score = results.get('sva_score', 0)

        # Calculate weighted score
        results['phase_score'] = (
            milestone_achieved * 0.4 +  # Milestone weight
            orr_score * 0.2 +  # ORR weight
            omi_score * 0.2 -  # OMI weight
            sva_score * 0.2    # Negative SVA weight
        )
        return results

    def analyze_news_sentiment(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze news sentiment using OpenAI"""
        if not self.is_available():
            return None
            
        try:
            prompt = (
                "Analyze the sentiment of this news article regarding its potential to increase stock prices. "
                "Provide a score from 0.0 (negative) to 1.0 (positive), and include your confidence in the analysis "
                "with a score from 0.0 to 1.0. Respond only with a JSON object in this format: "
                '{"openai_sentiment": "0.0 to 1.0", "openai_confidence": "0.0 to 1.0"}'
            )

            messages = [
                {"role": "system", "content": "You are a financial news analyst. Only return the JSON object."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": text[:2000]}  # Limit text length
            ]

            response = self.client.chat.completions.create(
                model=self.openai_model,
                temperature=0,
                messages=messages
            )

            # Extract the response content
            response_content = response.choices[0].message.content.strip()

            # Parse the JSON content
            try:
                results_json = json.loads(response_content.replace("'", "\""))  # Handle single quotes
            except json.JSONDecodeError:
                loge(f"Failed to parse JSON from response: {response_content}")
                return None

            # Extract and normalize metrics
            results = {
                'openai_sentiment': float(results_json.get('openai_sentiment', 0)),
                'openai_confidence': float(results_json.get('openai_confidence', 0))
            }

            return results
        except Exception as ex:
            loge(f"Exception during OpenAI sentiment analysis: {ex}")
            return None

    def analyze_phase_results(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze phase results from news content using OpenAI and extract metrics"""
        if not self.is_available():
            return None
            
        try:
            prompt = (
                "Extract the following metrics from the news content if available. "
                "If not available, return N/A for the metric."
                "Milestone achieved (e.g. Phase 3 clinical trial, FDA approval) (score: 1=success, 0=fail). "
                "Overall Response Rate (orr_score): The percentage of patients who achieved a level of improvement (score: 0 - 1). "
                "Overall Magnitude of Improvement (omi_score): The degree to which the drug improves symptoms or outcomes (score: 0 - 1). "
                "Percentage of patients with Severe Adverse Effects (sva_score) (score: 0 - 1). "
                "Format the response as a JSON object, structured as follows: "
                "{'milestone_achieved': '<0 or 1>', 'orr_score': '<score>', 'omi_score': '<score>', 'sva_score': '<score>'}."
            )

            messages = [
                {"role": "system", "content": "You are a news analyst with a medical background. Only return the JSON object."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": text[:2000]}  # Limit text length
            ]

            response = self.client.chat.completions.create(
                model=self.openai_model,
                temperature=0,
                messages=messages
            )

            # Extract the response content
            response_content = response.choices[0].message.content.strip()

            # Parse the JSON content
            try:
                results_json = json.loads(response_content.replace("'", "\""))  # Handle single quotes
            except json.JSONDecodeError:
                loge(f"Failed to parse JSON from response: {response_content}")
                return None

            # Extract and normalize metrics
            results = {
                'milestone_achieved': float(results_json.get('milestone_achieved', 0)),
                'orr_score': float(results_json.get('orr_score', 0)),
                'omi_score': float(results_json.get('omi_score', 0)),
                'sva_score': float(results_json.get('sva_score', 0))
            }

            # Calculate the overall milestone score for phase
            results = self.calculate_phase_result_score(results)

            return results
        except Exception as ex:
            loge(f"Exception during OpenAI phase analysis: {ex}")
            return None