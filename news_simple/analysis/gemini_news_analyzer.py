"""
Fixed Gemini LLM-powered news analyzer with resolved import issues
"""
import json
import re
import time
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning, log_debug

# Fix for Google Generative AI import issues
try:
    import google.generativeai as genai
    # Try to import the specific modules that Pylance expects
    try:
        from google.generativeai.client import configure
        from google.generativeai.generative_models import GenerativeModel
        SPECIFIC_IMPORTS_AVAILABLE = True
    except ImportError:
        # If specific imports fail, we'll use a different approach
        SPECIFIC_IMPORTS_AVAILABLE = False
        configure = None
        GenerativeModel = None
    
    GENAI_AVAILABLE = True
except ImportError:
    log_error("google-generativeai not installed. Run: pip install google-generativeai")
    GENAI_AVAILABLE = False
    SPECIFIC_IMPORTS_AVAILABLE = False
    genai = None
    configure = None
    GenerativeModel = None


@dataclass
class GeminiAnalysis:
    """Gemini analysis result with structured output"""
    sentiment_score: float      # -1 to 1 (normalized from 1-10 scale)
    market_impact: float        # 0 to 1 (normalized from 1-10 scale)
    confidence: float           # 0 to 1 (normalized from 1-10 scale)
    reasoning: str              # AI explanation
    expected_move: str          # "up", "down", "sideways"
    move_magnitude: str         # "small", "medium", "large"
    catalyst_type: str          # "earnings", "fda", "analyst", "corporate", "general"
    raw_response: str           # Full AI response for debugging


class GeminiNewsAnalyzer:
    """Fixed news analyzer using Google Gemini LLM with proper imports"""
    
    def __init__(self) -> None:
        """Initialize Gemini analyzer with fixed import handling"""
        if not GENAI_AVAILABLE:
            log_warning("Google Generative AI library not available - Gemini analysis disabled")
            self.enabled = False
            return
        
        self.api_key = CONFIG.get_api_key('gemini')
        if not self.api_key:
            log_warning("GEMINI_API_KEY not found - Gemini analysis disabled")
            self.enabled = False
            return
        
        self.model_name = CONFIG.get_gemini_model()
        
        # Conservative rate limiting for free tier
        self.max_requests_per_minute = 10
        self.max_requests_per_day = 1000
        self.request_timestamps = []
        self.daily_request_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Store references to avoid repeated attribute access
        self.model: Optional[Any] = None
        
        self.enabled = self._initialize_model()
        
        if self.enabled:
            self._setup_prompts()
    
    def _initialize_model(self) -> bool:
        """Initialize Gemini model with multiple fallback methods."""
        initialization_methods = [
            self._init_with_specific_imports,
            self._init_with_reflection,
            self._init_fallback_flash
        ]
        
        for method in initialization_methods:
            try:
                method()
                log_info(f"Gemini analyzer initialized successfully with model: {self.model_name}")
                log_info(f"Rate limits: {self.max_requests_per_minute} RPM, {self.max_requests_per_day} RPD")
                return True
            except Exception as e:
                log_debug(f"Initialization method failed: {e}")
                continue
        
        log_error("All Gemini initialization methods failed")
        return False
    
    def _init_with_specific_imports(self) -> None:
        """Primary initialization method using specific imports."""
        if not SPECIFIC_IMPORTS_AVAILABLE or configure is None or GenerativeModel is None:
            raise ImportError("Specific imports not available")
        
        configure(api_key=self.api_key)
        self.model = GenerativeModel(self.model_name)
    
    def _init_with_reflection(self) -> None:
        """Alternative initialization using reflection to avoid direct attribute access."""
        if genai is None:
            raise ImportError("genai not available")
        
        # Use getattr to avoid Pylance warnings about private imports
        configure_func = getattr(genai, 'configure', None)
        model_class = getattr(genai, 'GenerativeModel', None)
        
        if configure_func is None or model_class is None:
            raise ImportError("Required functions not available via reflection")
        
        configure_func(api_key=self.api_key)
        self.model = model_class(self.model_name)
    
    def _init_fallback_flash(self) -> None:
        """Fallback to flash model using reflection."""
        if genai is None:
            raise ImportError("genai not available")
        
        configure_func = getattr(genai, 'configure', None)
        model_class = getattr(genai, 'GenerativeModel', None)
        
        if configure_func is None or model_class is None:
            raise ImportError("Required functions not available for fallback")
        
        configure_func(api_key=self.api_key)
        self.model = model_class('gemini-1.5-flash')
        self.model_name = 'gemini-1.5-flash'
        log_info("Using fallback flash model")
    
    def _check_rate_limits(self) -> bool:
        """Enhanced rate limiting check with daily reset"""
        now = datetime.now()
        current_date = now.date()
        
        # Reset daily counter if new day
        if current_date != self.last_reset_date:
            self.daily_request_count = 0
            self.last_reset_date = current_date
            log_debug("Daily Gemini rate limit counter reset")
        
        # Check daily limit
        if self.daily_request_count >= self.max_requests_per_day:
            log_warning(f"Gemini daily rate limit reached: {self.daily_request_count}/{self.max_requests_per_day}")
            return False
        
        # Clean old timestamps
        minute_ago = now - timedelta(minutes=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        # Check per-minute limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            log_debug(f"Gemini per-minute rate limit reached: {len(self.request_timestamps)}/{self.max_requests_per_minute}")
            return False
        
        return True
    
    def _record_request(self) -> None:
        """Record a request for rate limiting"""
        self.request_timestamps.append(datetime.now())
        self.daily_request_count += 1
        
        if self.daily_request_count % 10 == 0:
            log_debug(f"Gemini usage: {self.daily_request_count}/{self.max_requests_per_day} daily, "
                     f"{len(self.request_timestamps)}/{self.max_requests_per_minute} per minute")
    
    def _wait_for_rate_limit(self) -> bool:
        """Wait for rate limit reset if close to limit"""
        if not self.request_timestamps or len(self.request_timestamps) < self.max_requests_per_minute:
            return True
        
        oldest_request = min(self.request_timestamps)
        wait_until = oldest_request + timedelta(minutes=1, seconds=5)
        now = datetime.now()
        
        if now < wait_until:
            sleep_time = (wait_until - now).total_seconds()
            if 0 < sleep_time < 60:
                log_info(f"Waiting {sleep_time:.1f}s for Gemini rate limit reset...")
                time.sleep(sleep_time)
                return True
            else:
                log_warning("Gemini rate limit wait time too long, skipping")
                return False
        
        return True
    
    def _setup_prompts(self) -> None:
        """Setup specialized prompts optimized for Flash model"""
        self.base_prompt = """
You are a financial analyst evaluating news for stock trading.

Analyze this news and respond with ONLY a JSON object:

COMPANY: {symbol}
PRICE: ${current_price}
TITLE: {title}
CONTENT: {content}

JSON Response:
{{
    "sentiment": <1-10 where 1=very bearish, 5=neutral, 10=very bullish>,
    "market_impact": <1-10 where 1=no impact, 10=major catalyst>,
    "confidence": <1-10 in your assessment>,
    "reasoning": "<brief 1-2 sentence explanation>",
    "expected_move": "<up/down/sideways>",
    "move_magnitude": "<small/medium/large>",
    "catalyst_type": "<earnings/fda/analyst/corporate/general>"
}}
"""
        
        self.topic_prompts = {
            'earnings': "You are analyzing EARNINGS news. Focus on beats/misses, guidance, and outlook.\n\n{base_analysis}",
            'biotech': "You are analyzing BIOTECH/PHARMA news. Focus on trials, FDA actions, and drug approvals.\n\n{base_analysis}",
            'analyst': "You are analyzing ANALYST coverage. Focus on rating changes, price targets, and credibility.\n\n{base_analysis}"
        }
    
    def analyze_news(self, symbol: str, title: str, content: str, 
                    current_price: float, topic: str = "general") -> Optional[GeminiAnalysis]:
        """Analyze news with improved rate limiting and error handling"""
        if not self.enabled or self.model is None:
            return None
        
        # Check and wait for rate limits
        if not self._check_rate_limits() or not self._wait_for_rate_limit():
            log_debug(f"Skipping Gemini analysis for {symbol} due to rate limits")
            return None
        
        try:
            self._record_request()
            
            # Get appropriate prompt
            prompt = self._get_prompt_for_topic(topic).format(
                symbol=symbol,
                current_price=current_price,
                title=title,
                content=content[:1200],
                base_analysis=self.base_prompt.format(
                    symbol=symbol,
                    current_price=current_price,
                    title=title,
                    content=content[:1200]
                )
            )
            
            # Generate response with error handling
            response = self._generate_response(prompt)
            
            if not response or not hasattr(response, 'text') or not response.text:
                log_warning(f"Empty response from Gemini for {symbol}")
                return None
            
            analysis = self._parse_gemini_response(response.text, symbol)
            
            if analysis:
                log_debug(f"Gemini SUCCESS: {symbol} sentiment={analysis.sentiment_score:.2f}, "
                         f"impact={analysis.market_impact:.2f}, conf={analysis.confidence:.2f}")
            
            return analysis
            
        except Exception as e:
            self._handle_api_error(e, symbol)
            return None
    
    def _generate_response(self, prompt: str) -> Any:
        """Generate response with multiple API version compatibility."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            # Try to use generation config if available
            try:
                from google.generativeai.types import GenerationConfig
                return self.model.generate_content(
                    prompt,
                    generation_config=GenerationConfig(
                        max_output_tokens=300,
                        temperature=0.1
                    )
                )
            except ImportError:
                # Fallback if types not available
                pass
            
            # Simple generation without config
            return self.model.generate_content(prompt)
            
        except Exception as e:
            # Final fallback
            log_debug(f"Generation error: {e}, trying basic method")
            return self.model.generate_content(prompt)
    
    def _handle_api_error(self, error: Exception, symbol: str) -> None:
        """Handle specific API errors."""
        error_str = str(error)
        if any(keyword in error_str.lower() for keyword in ["quota", "rate", "limit"]):
            log_warning(f"Gemini rate limit hit for {symbol}: {error}")
            self.daily_request_count = self.max_requests_per_day
        else:
            log_error(f"Gemini API error for {symbol}: {error}")
    
    def _get_prompt_for_topic(self, topic: str) -> str:
        """Get specialized prompt based on news topic"""
        return self.topic_prompts.get(topic, self.base_prompt)
    
    def _parse_gemini_response(self, response_text: str, symbol: str) -> Optional[GeminiAnalysis]:
        """Enhanced response parsing with better error handling"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                log_warning(f"No JSON found in Gemini response for {symbol}")
                return self._fallback_parse(response_text, symbol)
            
            data = json.loads(json_match.group(0))
            
            # Validate required fields
            required_fields = ['sentiment', 'market_impact', 'confidence', 'reasoning']
            if not all(field in data for field in required_fields):
                log_warning(f"Missing required fields in Gemini response for {symbol}")
                return self._fallback_parse(response_text, symbol)
            
            # Normalize scores
            sentiment_normalized = self._normalize_score(data['sentiment'], -1.0, 1.0, 1.0, 10.0, 5.5)
            market_impact_normalized = self._normalize_score(data['market_impact'], 0.0, 1.0, 1.0, 10.0)
            confidence_normalized = self._normalize_score(data['confidence'], 0.0, 1.0, 1.0, 10.0)
            
            return GeminiAnalysis(
                sentiment_score=sentiment_normalized,
                market_impact=market_impact_normalized,
                confidence=confidence_normalized,
                reasoning=str(data['reasoning'])[:150],
                expected_move=str(data.get('expected_move', 'sideways')),
                move_magnitude=str(data.get('move_magnitude', 'small')),
                catalyst_type=str(data.get('catalyst_type', 'general')),
                raw_response=response_text
            )
            
        except json.JSONDecodeError as e:
            log_warning(f"JSON parsing error for {symbol}: {e}")
            return self._fallback_parse(response_text, symbol)
        except Exception as e:
            log_error(f"Error parsing Gemini response for {symbol}: {e}")
            return None
    
    @staticmethod
    def _normalize_score(value: float, min_out: float, max_out: float, 
                        min_in: float = 1.0, max_in: float = 10.0, center: Optional[float] = None) -> float:
        """Normalize score from input range to output range."""
        try:
            if center is not None:
                # For sentiment: center around neutral
                if value > center:
                    normalized = (float(value) - center) / (max_in - center)
                else:
                    normalized = (float(value) - center) / (center - min_in)
                normalized = max(min_out, min(max_out, normalized))
            else:
                # Standard normalization
                normalized = (float(value) - min_in) / (max_in - min_in)
                normalized = min_out + normalized * (max_out - min_out)
                normalized = max(min_out, min(max_out, normalized))
            
            return normalized
        except (ValueError, TypeError, ZeroDivisionError):
            # Return neutral value if normalization fails
            return (min_out + max_out) / 2.0
    
    def _fallback_parse(self, response_text: str, symbol: str) -> Optional[GeminiAnalysis]:
        """Improved fallback parsing when JSON extraction fails"""
        try:
            text_lower = response_text.lower()
            
            # Sentiment keywords mapping
            sentiment_keywords = {
                0.8: ['very bullish', 'strong buy', 'very positive'],
                0.4: ['bullish', 'positive', 'buy'],
                -0.4: ['bearish', 'negative', 'sell'],
                -0.8: ['very bearish', 'strong sell', 'very negative']
            }
            
            sentiment = 0.0
            for score, keywords in sentiment_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    sentiment = score
                    break
            
            # Market impact keywords
            impact_keywords = {
                0.8: ['major catalyst', 'significant', 'large impact'],
                0.3: ['minor', 'small impact', 'limited']
            }
            
            market_impact = 0.5  # Default
            for score, keywords in impact_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    market_impact = score
                    break
            
            # Extract reasoning
            sentences = response_text.split('.')[:2]
            reasoning = '. '.join(sentences)[:150]
            
            return GeminiAnalysis(
                sentiment_score=sentiment,
                market_impact=market_impact,
                confidence=0.5,  # Lower confidence for fallback
                reasoning=reasoning,
                expected_move='sideways',
                move_magnitude='medium',
                catalyst_type='general',
                raw_response=response_text
            )
            
        except Exception as e:
            log_error(f"Fallback parsing failed for {symbol}: {e}")
            return None
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get current usage statistics"""
        return {
            'daily_requests': self.daily_request_count,
            'daily_limit': self.max_requests_per_day,
            'minute_requests': len(self.request_timestamps),
            'minute_limit': self.max_requests_per_minute,
            'remaining_daily': self.max_requests_per_day - self.daily_request_count,
            'remaining_minute': self.max_requests_per_minute - len(self.request_timestamps)
        }
    
    def batch_analyze(self, news_items: List[Dict]) -> List[Optional[GeminiAnalysis]]:
        """Analyze multiple news items with intelligent rate limiting"""
        if not self.enabled:
            return [None] * len(news_items)
        
        results = []
        for item in news_items:
            if not self._check_rate_limits():
                log_info(f"Stopping batch analysis due to rate limits after {len(results)} items")
                results.extend([None] * (len(news_items) - len(results)))
                break
            
            try:
                analysis = self.analyze_news(
                    symbol=item.get('symbol', ''),
                    title=item.get('title', ''),
                    content=item.get('content', ''),
                    current_price=item.get('current_price', 0.0),
                    topic=item.get('topic', 'general')
                )
                results.append(analysis)
                
                # Smart batching delay
                if len(results) % 5 == 0:
                    time.sleep(1)
                    
            except Exception as e:
                log_error(f"Batch analysis error: {e}")
                results.append(None)
        
        return results
    
    @lru_cache(maxsize=10)
    def get_market_context_analysis(self, market_regime: str, vix_level: float) -> str:
        """Get market context for better analysis"""
        if not self.enabled or self.model is None:
            return "neutral"
        
        try:
            prompt = f"""
Current market conditions:
- Market Regime: {market_regime}
- VIX Level: {vix_level}

In 1-2 sentences, describe how these conditions should affect:
1. Risk appetite for individual stock moves
2. Whether news-driven trades are likely to work
3. Expected holding periods

Response:
"""
            response = self.model.generate_content(prompt)
            return response.text if response and hasattr(response, 'text') and response.text else "neutral conditions"
            
        except Exception as e:
            log_debug(f"Market context analysis error: {e}")
            return "neutral conditions"