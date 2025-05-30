"""
Enhanced LLM Interface with Rate Limiting and Better Error Handling
"""

import json
import logging
import traceback
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional
import google.generativeai as genai
from config.config import Config
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def analyze_sentiment(self, title: str, content: str, ticker: str) -> Dict:
        """Analyze sentiment and generate trading signals"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured"""
        pass

    def _create_analysis_prompt(self, title: str, content: str, ticker: str) -> str:
        """Create standardized analysis prompt for any LLM"""
        return f"""
        You are an expert financial analyst. Analyze this earnings-related news article for {ticker}.

        Article Title: {title}
        Article Content: {content}

        Provide analysis as JSON with exactly these fields:

        1. sentiment_score: Number between -1 and 1 where:
           -1 = Very negative (major disappointments, significant misses)
           -0.5 = Negative (minor disappointments, slight misses)
           0 = Neutral (mixed signals, unclear impact)
           0.5 = Positive (minor beats, positive developments)  
           1 = Very positive (major beats, exceptional results)

        2. confidence_score: Your confidence (0 to 1) where:
           0.9-1.0 = Very high confidence (clear signals)
           0.7-0.9 = High confidence (strong signals)
           0.5-0.7 = Medium confidence (mixed signals)
           0.3-0.5 = Low confidence (unclear information)
           0.0-0.3 = Very low confidence (insufficient data)

        3. trade_signal: One of "LONG", "SHORT", or "NEUTRAL"

        4. reasoning: 2-3 sentences explaining your analysis

        5. key_factors: Array of key factors influencing your decision

        Consider:
        - Earnings beats/misses and magnitude
        - Revenue growth trends  
        - Guidance changes
        - Management commentary
        - Market conditions
        - Competitive position

        Respond ONLY with valid JSON - no additional text.
        """
    
    def _validate_response(self, result: Dict) -> Dict:
        """Validate and clean LLM response"""
        # Ensure required keys exist
        required_keys = ['sentiment_score', 'confidence_score', 'trade_signal', 'reasoning']
        for key in required_keys:
            if key not in result:
                if 'score' in key:
                    result[key] = 0.0
                elif key == 'trade_signal':
                    result[key] = 'NEUTRAL'
                else:
                    result[key] = 'Analysis incomplete'
        
        # Validate ranges
        result['sentiment_score'] = max(-1.0, min(1.0, float(result.get('sentiment_score', 0))))
        result['confidence_score'] = max(0.0, min(1.0, float(result.get('confidence_score', 0))))
        
        # Validate signal
        if result.get('trade_signal') not in ['LONG', 'SHORT', 'NEUTRAL']:
            result['trade_signal'] = 'NEUTRAL'
        
        # Ensure key_factors exists
        if 'key_factors' not in result:
            result['key_factors'] = ['standard_analysis']
        
        return result
    
    def _error_response(self, error_msg: str) -> Dict:
        """Return error response in standard format"""
        return {
            "sentiment_score": 0.0,
            "confidence_score": 0.0,
            "trade_signal": "NEUTRAL",
            "reasoning": f"Analysis failed: {error_msg}",
            "key_factors": ["analysis_error"]
        }

class GeminiProvider(BaseLLMProvider):
    """Enhanced Google Gemini provider with rate limiting"""
    
    def __init__(self):
        """Initialize Gemini provider with rate limiting"""
        try:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
            self._available = bool(Config.GOOGLE_API_KEY)
            
            # Rate limiting configuration
            self.requests_per_minute = 15  # Free tier limit
            self.request_times = []  # Track request timestamps
            self.last_request_time = 0
            self.min_delay_seconds = 4  # Minimum delay between requests
            
            logger.info(f"✅ Gemini provider initialized with rate limiting (max {self.requests_per_minute}/min)")
            
        except ImportError:
            logger.warning("Google AI library not installed. Run: pip install google-generativeai")
            self._available = False
        except Exception as e:
            logger.error(f"Gemini initialization error: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        """Check if Gemini is properly configured"""
        return self._available
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting to avoid quota exceeded errors"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # If we're at the rate limit, wait
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0]) + 1
            if sleep_time > 0:
                logger.info(f"⏳ Rate limit reached, waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        
        # Ensure minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay_seconds:
            sleep_time = self.min_delay_seconds - time_since_last
            logger.debug(f"⏳ Waiting {sleep_time:.1f}s for minimum delay...")
            time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(time.time())
        self.last_request_time = time.time()
    
    def analyze_sentiment(self, title: str, content: str, ticker: str) -> Dict:
        """Analyze using Google Gemini with enhanced error handling and rate limiting"""
        if not self._available:
            return self._error_response("Gemini provider not available")
        
        prompt = self._create_analysis_prompt(title, content, ticker)
        cleaned_text = ""
        
        try:
            # Implement rate limiting
            self._wait_for_rate_limit()
            
            logger.debug(f"🔍 Analyzing {ticker} article: {title[:50]}...")
            
            # Get response from Gemini with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    break
                except google_exceptions.ResourceExhausted as e:
                    logger.warning(f"⏳ Rate limit hit for {ticker}, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        # Extract retry delay from error if available
                        retry_delay = getattr(e, 'retry_delay', None)
                        if retry_delay and hasattr(retry_delay, 'seconds'):
                            delay = retry_delay.seconds + 5  # Add buffer
                        else:
                            delay = 60  # Default delay
                        logger.info(f"⏳ Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                    else:
                        raise e
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Attempt {attempt + 1} failed for {ticker}: {e}")
                        time.sleep(5)  # Short delay before retry
                    else:
                        raise e
            
            if not response or not hasattr(response, 'text'):
                logger.error(f"❌ Empty response from Gemini for {ticker}")
                return self._error_response("Empty response from Gemini")
            
            # Clean and parse response
            cleaned_text = response.text.strip()
            logger.debug(f"📦 Gemini response for {ticker} (length: {len(cleaned_text)})")
            
            if not cleaned_text:
                logger.error(f"❌ Empty text in Gemini response for {ticker}")
                return self._error_response("Empty text in response")
            
            # Parse JSON response
            try:
                result = json.loads(cleaned_text)
            except json.JSONDecodeError:
                # Try to extract JSON if direct parsing fails
                start_idx = cleaned_text.find('{')
                end_idx = cleaned_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = cleaned_text[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    logger.error(f"❌ No valid JSON found in Gemini response for {ticker}")
                    logger.debug(f"Raw response: {cleaned_text[:500]}...")
                    return self._error_response("Invalid JSON response")
            
            # Log success
            logger.info(f"✅ {ticker} Gemini analysis complete: "
                       f"Sentiment: {result.get('sentiment_score', 0):.2f}, "
                       f"Confidence: {result.get('confidence_score', 0):.2f}, "
                       f"Signal: {result.get('trade_signal', 'NEUTRAL')}")
            
            return self._validate_response(result)
            
        except google_exceptions.ResourceExhausted as e:
            logger.error(f"❌ Gemini rate limit exceeded for {ticker}: {str(e)}")
            return self._error_response("Rate limit exceeded - using fallback analysis")
            
        except google_exceptions.GoogleAPICallError as e:
            logger.error(f"❌ Gemini API error for {ticker}: {str(e)}")
            return self._error_response(f"API error: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ Gemini analysis error for {ticker}: {str(e)}")
            logger.debug(f"📄 Full error details:\n{traceback.format_exc()}")
            if cleaned_text:
                logger.debug(f"❌ Raw response:\n{cleaned_text[:500]}...")
            return self._error_response(str(e))

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            self._available = bool(Config.ANTHROPIC_API_KEY)
        except ImportError:
            logger.warning("Anthropic library not installed. Run: pip install anthropic")
            self._available = False
        except Exception as e:
            logger.error(f"Claude initialization error: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def analyze_sentiment(self, title: str, content: str, ticker: str) -> Dict:
        """Analyze using Claude"""
        prompt = self._create_analysis_prompt(title, content, ticker)
        
        try:
            response = self.client.messages.create(
                model=Config.CLAUDE_MODEL,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = json.loads(response.content[0].text)
            return self._validate_response(result)
            
        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
            return self._error_response(str(e))

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self):
        try:
            import openai
            self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            self._available = bool(Config.OPENAI_API_KEY)
        except ImportError:
            logger.warning("OpenAI library not installed. Run: pip install openai")
            self._available = False
        except Exception as e:
            logger.error(f"OpenAI initialization error: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def analyze_sentiment(self, title: str, content: str, ticker: str) -> Dict:
        """Analyze using OpenAI GPT"""
        prompt = self._create_analysis_prompt(title, content, ticker)
        
        try:
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            return self._validate_response(result)
            
        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")
            return self._error_response(str(e))

class GrokProvider(BaseLLMProvider):
    """Grok (X AI) provider"""
    
    def __init__(self):
        try:
            import openai  # Grok uses OpenAI-compatible API
            self.client = openai.OpenAI(
                api_key=Config.GROK_API_KEY,
                base_url="https://api.x.ai/v1"
            )
            self._available = bool(Config.GROK_API_KEY)
        except ImportError:
            logger.warning("OpenAI library not installed for Grok. Run: pip install openai")
            self._available = False
        except Exception as e:
            logger.error(f"Grok initialization error: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def analyze_sentiment(self, title: str, content: str, ticker: str) -> Dict:
        """Analyze using Grok"""
        prompt = self._create_analysis_prompt(title, content, ticker)
        
        try:
            response = self.client.chat.completions.create(
                model=Config.GROK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            return self._validate_response(result)
            
        except Exception as e:
            logger.error(f"Grok analysis error: {e}")
            return self._error_response(str(e))

class LLMManager:
    """Manager for multiple LLM providers with fallback support"""
    
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers"""
        
        # Initialize providers based on configuration
        if Config.USE_CLAUDE and Config.ANTHROPIC_API_KEY:
            self.providers['claude'] = ClaudeProvider()
            
        if Config.USE_OPENAI and Config.OPENAI_API_KEY:
            self.providers['openai'] = OpenAIProvider()
            
        if Config.USE_GEMINI and Config.GOOGLE_API_KEY:
            self.providers['gemini'] = GeminiProvider()
            
        if Config.USE_GROK and Config.GROK_API_KEY:
            self.providers['grok'] = GrokProvider()
        
        # Filter to only available providers
        self.providers = {name: provider for name, provider in self.providers.items() 
                         if provider.is_available()}
        
        if not self.providers:
            logger.warning("No LLM providers available! Check your configuration.")
        else:
            logger.info(f"Initialized LLM providers: {list(self.providers.keys())}")
    
    def get_primary_provider(self) -> Optional[BaseLLMProvider]:
        """Get the primary LLM provider based on preference order"""
        
        # Priority order (can be configured)
        priority_order = Config.LLM_PRIORITY_ORDER
        
        for provider_name in priority_order:
            if provider_name in self.providers:
                return self.providers[provider_name]
        
        # Fallback to any available provider
        if self.providers:
            return next(iter(self.providers.values()))
        
        return None
    
    def analyze_with_fallback(self, title: str, content: str, ticker: str) -> Dict:
        """Analyze with primary provider and fallback support"""
        
        # Try primary provider first
        primary = self.get_primary_provider()
        if primary:
            try:
                result = primary.analyze_sentiment(title, content, ticker)
                if result['confidence_score'] > 0:  # Success
                    return result
            except Exception as e:
                logger.warning(f"Primary provider failed: {e}")
        
        # Try fallback providers
        for name, provider in self.providers.items():
            if provider != primary:  # Don't retry primary
                try:
                    logger.info(f"Trying fallback provider: {name}")
                    result = provider.analyze_sentiment(title, content, ticker)
                    if result['confidence_score'] > 0:
                        return result
                except Exception as e:
                    logger.warning(f"Fallback provider {name} failed: {e}")
        
        # All providers failed
        logger.error("All LLM providers failed")
        return {
            "sentiment_score": 0.0,
            "confidence_score": 0.0,
            "trade_signal": "NEUTRAL", 
            "reasoning": "All LLM analysis providers failed",
            "key_factors": ["provider_error"]
        }
    
    def get_available_providers(self) -> list:
        """Get list of available provider names"""
        return list(self.providers.keys())
    
    def switch_primary_provider(self, provider_name: str) -> bool:
        """Switch to a different primary provider"""
        if provider_name in self.providers:
            # Update config priority
            Config.LLM_PRIORITY_ORDER = [provider_name] + [p for p in Config.LLM_PRIORITY_ORDER if p != provider_name]
            return True
        return False
