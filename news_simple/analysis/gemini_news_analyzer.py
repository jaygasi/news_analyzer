"""
Fixed Gemini LLM-powered news analyzer with resolved rate limiting and model issues
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

# Store the imported module in a variable to help type checkers
_gemini_lib_module: Optional[Any] = None
try:
    import google.generativeai
    _gemini_lib_module = google.generativeai
    GENAI_AVAILABLE = True
except ImportError:
    log_error("google-generativeai not installed. Run: pip install google-generativeai")
    GENAI_AVAILABLE = False


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


class OptimizedGeminiNewsAnalyzer:
    """Optimized news analyzer using Google Gemini LLM with fixed rate limiting"""
    
    def __init__(self) -> None:
        """Initialize Gemini analyzer with optimized rate limiting"""
        if not GENAI_AVAILABLE:
            log_warning("Google Generative AI library not available - Gemini analysis disabled")
            self.enabled = False
            return
        
        self.api_key = CONFIG.get_api_key('gemini')
        if not self.api_key:
            log_warning("GEMINI_API_KEY not found - Gemini analysis disabled")
            self.enabled = False
            return
        
        # Get and validate model name
        self.model_name = CONFIG.get_gemini_model()
        
        # More conservative rate limiting for stability
        self.max_requests_per_minute = 5  # Reduced from 8
        self.max_requests_per_day = 400   # Reduced from 800
        self.request_timestamps: List[datetime] = []
        self.daily_request_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Request spacing to avoid bursts
        self.min_request_spacing = 12.0  # Increased from 8 seconds
        self.last_request_time = 0.0
        
        # Rate limit tracking
        self.rate_limit_hit = False
        self.last_rate_limit_warning = 0.0
        self.warning_cooldown = 300  # 5 minutes between warnings
        
        self.model: Optional[Any] = None
        self.enabled = self._initialize_model()
        
        if self.enabled:
            self._setup_optimized_prompts()
    
    def _initialize_model(self) -> bool:
        """Initialize Gemini model with robust error handling."""
        # This method is called from __init__ only if GENAI_AVAILABLE is True.
        # Therefore, _gemini_lib_module should be the imported module.
        if not _gemini_lib_module:
            log_error("Gemini library module is None, cannot initialize model. This indicates an issue with GENAI_AVAILABLE logic.")
            return False
        try:
            # Configure with API key
            _gemini_lib_module.configure(api_key=self.api_key)
            
            # Validate model name and create model
            self.model = _gemini_lib_module.GenerativeModel(self.model_name)
            
            log_info(f"Gemini analyzer initialized successfully with model: {self.model_name}")
            log_info(f"Rate limits: {self.max_requests_per_minute} RPM, {self.max_requests_per_day} RPD")
            return True
            
        except Exception as e:
            log_error(f"Failed to initialize Gemini model '{self.model_name}': {e}")
            
            # Try fallback model
            try:
                self.model_name = 'gemini-1.5-flash'
                self.model = _gemini_lib_module.GenerativeModel(self.model_name)
                log_info(f"Using fallback model: {self.model_name}")
                return True
            except Exception as fallback_error:
                log_error(f"Fallback model also failed: {fallback_error}")
                return False
    
    def _check_enhanced_rate_limits(self) -> bool:
        """Enhanced rate limiting with daily reset and request spacing."""
        now = datetime.now()
        current_date = now.date()
        
        # Reset daily counter if new day
        if current_date != self.last_reset_date:
            self.daily_request_count = 0
            self.last_reset_date = current_date
            self.request_timestamps.clear()
            self.rate_limit_hit = False
            log_debug("Daily Gemini rate limit counter reset")
        
        # If we've hit rate limits, don't spam warnings
        if self.rate_limit_hit:
            return False
        
        # Check daily limit
        if self.daily_request_count >= self.max_requests_per_day:
            self._log_rate_limit_warning("daily", self.daily_request_count, self.max_requests_per_day)
            self.rate_limit_hit = True
            return False
        
        # Check request spacing
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_spacing:
            return False
        
        # Clean old timestamps (keep only last minute)
        minute_ago = now - timedelta(minutes=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        # Check per-minute limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            self._log_rate_limit_warning("minute", len(self.request_timestamps), self.max_requests_per_minute)
            return False
        
        return True
    
    def _log_rate_limit_warning(self, limit_type: str, current: int, maximum: int) -> None:
        """Log rate limit warning with cooldown to prevent spam."""
        current_time = time.time()
        if current_time - self.last_rate_limit_warning > self.warning_cooldown:
            log_warning(f"Gemini {limit_type} rate limit reached: {current}/{maximum}")
            self.last_rate_limit_warning = current_time
    
    def _wait_for_rate_limit_reset(self) -> bool:
        """Wait for rate limit reset with timeout."""
        if not self.request_timestamps:
            return True
        
        # Wait for request spacing
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_spacing:
            wait_time = self.min_request_spacing - time_since_last
            if wait_time > 0 and wait_time <= 20:  # Max 20 second wait
                log_debug(f"Waiting {wait_time:.1f}s for request spacing...")
                time.sleep(wait_time)
                return True
            return False
        
        # Wait for per-minute limit reset
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            oldest_request = min(self.request_timestamps)
            wait_until = oldest_request + timedelta(minutes=1, seconds=10)
            now = datetime.now()
            
            if now < wait_until:
                sleep_time = (wait_until - now).total_seconds()
                if 0 < sleep_time <= 30:  # Max 30 second wait
                    log_debug(f"Waiting {sleep_time:.1f}s for rate limit reset...")
                    time.sleep(sleep_time)
                    return True
                else:
                    return False
        
        return True
    
    def _record_request(self) -> None:
        """Record a request for rate limiting tracking."""
        now = datetime.now()
        self.request_timestamps.append(now)
        self.daily_request_count += 1
        self.last_request_time = time.time()
        
        # Only log every 50 requests to reduce log noise
        if self.daily_request_count % 50 == 0:
            log_debug(f"Gemini usage: {self.daily_request_count}/{self.max_requests_per_day} daily, "
                     f"{len(self.request_timestamps)}/{self.max_requests_per_minute} per minute")
    
    def _setup_optimized_prompts(self) -> None:
        """Setup optimized prompts for better results."""
        self.base_prompt = """
Analyze this financial news and return ONLY a JSON object.

COMPANY: {symbol} (Price: ${current_price})
TITLE: {title}
CONTENT: {content}

Return exactly this JSON format:
{{
    "sentiment": <number 1-10 where 1=very bearish, 5=neutral, 10=very bullish>,
    "market_impact": <number 1-10 where 1=no impact, 10=major catalyst>,
    "confidence": <number 1-10 in your assessment>,
    "reasoning": "<brief 1-2 sentence explanation>",
    "expected_move": "<up/down/sideways>",
    "move_magnitude": "<small/medium/large>",
    "catalyst_type": "<earnings/fda/analyst/corporate/general>"
}}
"""
        
        self.topic_prompts = {
            'earnings': "Focus on earnings beats/misses, guidance, and revenue trends.\n\n{base_analysis}",
            'biotech': "Focus on FDA approvals, clinical trials, and regulatory news.\n\n{base_analysis}",
            'analyst': "Focus on rating changes, price targets, and analyst credibility.\n\n{base_analysis}"
        }
    
    def analyze_news(self, symbol: str, title: str, content: str, 
                    current_price: float, topic: str = "general") -> Optional[GeminiAnalysis]:
        """Analyze news with enhanced rate limiting and error handling."""
        if not self.enabled or self.model is None:
            return None
        
        # Enhanced rate limit checking
        if not self._check_enhanced_rate_limits():
            return None
        
        if not self._wait_for_rate_limit_reset():
            return None
        
        try:
            self._record_request()
            
            # Get appropriate prompt
            prompt = self._get_optimized_prompt(symbol, title, content, current_price, topic)
            
            # Generate response with timeout
            response = self._generate_with_timeout(prompt)
            
            if not response or not hasattr(response, 'text') or not response.text:
                log_debug(f"Empty response from Gemini for {symbol}")
                return None
            
            analysis = self._parse_gemini_response(response.text, symbol)
            
            if analysis:
                log_debug(f"Gemini SUCCESS: {symbol} sentiment={analysis.sentiment_score:.2f}, "
                         f"impact={analysis.market_impact:.2f}, conf={analysis.confidence:.2f}")
            
            return analysis
            
        except Exception as e:
            self._handle_api_error(e, symbol)
            return None
    
    def _get_optimized_prompt(self, symbol: str, title: str, content: str, 
                             current_price: float, topic: str) -> str:
        """Get optimized prompt based on topic."""
        base_analysis = self.base_prompt.format(
            symbol=symbol,
            current_price=current_price,
            title=title[:200],  # Truncate for efficiency
            content=content[:800]  # Truncate for efficiency
        )
        
        if topic in self.topic_prompts:
            return self.topic_prompts[topic].format(base_analysis=base_analysis)
        
        return base_analysis
    
    def _generate_with_timeout(self, prompt: str) -> Any:
        """Generate response with timeout and optimized configuration."""
        if self.model is None:
            # This should ideally not be reached due to guards in calling methods (e.g., analyze_news).
            log_error("Gemini model is None in _generate_with_timeout. Aborting generation.")
            return None
        try:
            # Try with generation config for better control
            generation_config = {
                'max_output_tokens': 200,
                'temperature': 0.1,
                'top_p': 0.8
            }
            
            return self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
        except Exception:
            # Fallback to simple generation
            return self.model.generate_content(prompt)
    
    def _handle_api_error(self, error: Exception, symbol: str) -> None:
        """Handle specific API errors with appropriate responses."""
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ["quota", "rate", "limit"]):
            self.rate_limit_hit = True
            self._log_rate_limit_warning("API", self.daily_request_count, self.max_requests_per_day)
        elif "model" in error_str and "format" in error_str:
            log_error(f"Gemini model format error for {symbol}: {error}")
            # Disable temporarily to prevent repeated errors
            self.enabled = False
        else:
            log_debug(f"Gemini API error for {symbol}: {error}")
    
    def _parse_gemini_response(self, response_text: str, symbol: str) -> Optional[GeminiAnalysis]:
        """Enhanced response parsing with better error handling."""
        try:
            # Clean and extract JSON
            json_text = self._extract_json_from_response(response_text)
            if not json_text:
                return self._fallback_parse(response_text, symbol)
            
            data = json.loads(json_text)
            
            # Validate required fields with defaults
            required_fields = {
                'sentiment': 5, 'market_impact': 5, 'confidence': 5, 
                'reasoning': 'Analysis completed'
            }
            
            for field, default in required_fields.items():
                if field not in data:
                    data[field] = default
            
            # Normalize scores with safe conversion
            sentiment_normalized = self._safe_normalize_score(
                data['sentiment'], -1.0, 1.0, 1.0, 10.0, 5.5
            )
            market_impact_normalized = self._safe_normalize_score(
                data['market_impact'], 0.0, 1.0, 1.0, 10.0
            )
            confidence_normalized = self._safe_normalize_score(
                data['confidence'], 0.0, 1.0, 1.0, 10.0
            )
            
            return GeminiAnalysis(
                sentiment_score=sentiment_normalized,
                market_impact=market_impact_normalized,
                confidence=confidence_normalized,
                reasoning=str(data['reasoning'])[:150],
                expected_move=str(data.get('expected_move', 'sideways')),
                move_magnitude=str(data.get('move_magnitude', 'small')),
                catalyst_type=str(data.get('catalyst_type', 'general')),
                raw_response=response_text[:500]  # Truncate for memory
            )
            
        except json.JSONDecodeError:
            return self._fallback_parse(response_text, symbol)
        except Exception as e:
            log_error(f"Error parsing Gemini response for {symbol}: {e}")
            return None
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extract JSON from response text with multiple patterns."""
        patterns = [
            r'\{[^{}]*\}',  # Simple JSON object
            r'\{.*?\}',     # Non-greedy JSON
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                # Return the JSON part (group 1 if it exists, otherwise group 0)
                return match.group(1) if match.groups() else match.group(0)
        
        return None
    
    @staticmethod
    def _safe_normalize_score(value: Any, min_out: float, max_out: float, 
                             min_in: float = 1.0, max_in: float = 10.0, 
                             center: Optional[float] = None) -> float:
        """Safely normalize score with error handling."""
        try:
            num_value = float(value)
            
            if center is not None:
                # For sentiment: center around neutral
                if num_value > center:
                    normalized = (num_value - center) / (max_in - center)
                else:
                    normalized = (num_value - center) / (center - min_in)
                return max(min_out, min(max_out, normalized))
            else:
                # Standard normalization
                normalized = (num_value - min_in) / (max_in - min_in)
                normalized = min_out + normalized * (max_out - min_out)
                return max(min_out, min(max_out, normalized))
                
        except (ValueError, TypeError, ZeroDivisionError):
            # Return neutral value
            return (min_out + max_out) / 2.0
    
    def _fallback_parse(self, response_text: str, symbol: str) -> Optional[GeminiAnalysis]:
        """Improved fallback parsing with keyword detection."""
        try:
            text_lower = response_text.lower()
            
            # Enhanced sentiment detection
            sentiment_score = self._detect_sentiment_from_text(text_lower)
            market_impact = self._detect_impact_from_text(text_lower)
            
            # Extract reasoning (first sentence or two)
            sentences = response_text.split('.')[:2]
            reasoning = '. '.join(sentences)[:150]
            
            return GeminiAnalysis(
                sentiment_score=sentiment_score,
                market_impact=market_impact,
                confidence=0.4,  # Lower confidence for fallback
                reasoning=reasoning,
                expected_move='sideways',
                move_magnitude='medium',
                catalyst_type='general',
                raw_response=response_text[:500]
            )
            
        except Exception as e:
            log_error(f"Fallback parsing failed for {symbol}: {e}")
            return None
    
    def _detect_sentiment_from_text(self, text_lower: str) -> float:
        """Detect sentiment from text using keyword analysis."""
        positive_keywords = [
            'bullish', 'positive', 'buy', 'strong', 'good', 'excellent',
            'outperform', 'beat', 'growth', 'revenue increase'
        ]
        negative_keywords = [
            'bearish', 'negative', 'sell', 'weak', 'poor', 'terrible',
            'underperform', 'miss', 'decline', 'revenue decrease'
        ]
        
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count) * 0.6
    
    def _detect_impact_from_text(self, text_lower: str) -> float:
        """Detect market impact from text."""
        high_impact_keywords = [
            'major', 'significant', 'large', 'huge', 'massive', 'breakthrough'
        ]
        low_impact_keywords = [
            'minor', 'small', 'limited', 'slight', 'minimal'
        ]
        
        if any(word in text_lower for word in high_impact_keywords):
            return 0.8
        elif any(word in text_lower for word in low_impact_keywords):
            return 0.3
        
        return 0.5
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get current usage statistics."""
        return {
            'daily_requests': self.daily_request_count,
            'daily_limit': self.max_requests_per_day,
            'minute_requests': len(self.request_timestamps),
            'minute_limit': self.max_requests_per_minute,
            'remaining_daily': max(0, self.max_requests_per_day - self.daily_request_count),
            'remaining_minute': max(0, self.max_requests_per_minute - len(self.request_timestamps)),
            'rate_limit_hit': self.rate_limit_hit
        }
    
    @lru_cache(maxsize=5)
    def get_market_context_analysis(self, market_regime: str, vix_level: float) -> str:
        """Get cached market context for better analysis."""
        if not self.enabled or self.model is None or self.rate_limit_hit:
            return "neutral"
        
        # Only analyze if we have spare capacity
        if not self._check_enhanced_rate_limits():
            return "neutral conditions"
        
        try:
            prompt = f"""
Current market: {market_regime}, VIX: {vix_level}

In 1 sentence, how should traders adjust risk for individual stock moves?

Response:
"""
            response = self.model.generate_content(prompt)
            return response.text[:200] if response and hasattr(response, 'text') else "neutral conditions"
            
        except Exception:
            return "neutral conditions"


# Create alias for backwards compatibility
GeminiNewsAnalyzer = OptimizedGeminiNewsAnalyzer