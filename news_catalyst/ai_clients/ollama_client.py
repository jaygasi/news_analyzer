import ollama
from utils.log_utils import *
from typing import Optional, Dict, Any
import json


class OllamaClient:
    def __init__(self, model: str = 'llama3.2'):
        self.model = model
        self.is_connected = self._test_connection()

    def _test_connection(self) -> bool:
        """Test connection to Ollama"""
        try:
            # Try to list models to test connection
            ollama.list()
            logi(f"✅ Ollama client connected successfully with model {self.model}")
            return True
        except Exception as e:
            logw(f"⚠️  Ollama connection failed: {str(e)}")
            return False

    def is_available(self) -> bool:
        """Check if Ollama client is available"""
        return self.is_connected

    def query(self, prompt: str, role: str = 'You are an experienced financial analyst') -> Optional[str]:
        """Send a query to Ollama"""
        if not self.is_available():
            return None
            
        try:
            # Ensure input is properly formatted as a dictionary
            messages = [
                {'role': 'system', 'content': role},
                {'role': 'user', 'content': prompt}
            ]

            # Send the prompt to the model and get the response
            response = ollama.chat(
                model=self.model,
                messages=messages
            )

            # Extract and return the specific response content
            content = response.get('message', {}).get('content', '')
            return content
        except Exception as ex:
            loge(f"Exception in OllamaClient: {str(ex)}")
            return None

    def analyze_news_sentiment(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze news sentiment using Ollama"""
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

            response = self.query(prompt)
            
            if not response:
                return None
                
            # Extract JSON from response
            try:
                # Clean the response text
                response_text = response.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                result = json.loads(response_text)
                
                return {
                    'ollama_sentiment': float(result.get('sentiment', 0.5)),
                    'ollama_confidence': float(result.get('confidence', 0.5))
                }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logw(f"Failed to parse Ollama response: {str(e)}")
                return None
                
        except Exception as e:
            loge(f"Error in Ollama sentiment analysis: {str(e)}")
            return None

    def analyze_phase_results(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze phase results using Ollama"""
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

            response = self.query(prompt)
            
            if not response:
                return None
                
            try:
                # Clean the response text
                response_text = response.strip()
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
                logw(f"Failed to parse Ollama phase analysis response: {str(e)}")
                return None
                
        except Exception as e:
            loge(f"Error in Ollama phase analysis: {str(e)}")
            return None