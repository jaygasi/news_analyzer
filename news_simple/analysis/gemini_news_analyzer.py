"""
Fixed Gemini LLM-powered news analyzer with resolved import issues
"""
import json
import re
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning, log_debug

# Fix for Google Generative AI import issues
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    log_error("google-generativeai not installed. Run: pip install google-generativeai")
    GENAI_AVAILABLE = False
    genai = None


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
        
        # Get model configuration from environment with better default
        self.model_name = CONFIG.get_gemini_model()
        
        # Rate limiting for Gemini 1.5 Flash free tier
        self.max_requests_per_minute = 10  # Conservative (Flash allows 15)
        self.max_requests_per_day = 1000   # Conservative (Flash allows 1500)
        self.request_timestamps = []
        self.daily_request_count = 0
        self.last_reset_date = datetime.now().date()
        
        try:
            # Fix for configure import issue - try different approaches
            if hasattr(genai, 'configure'):
                genai.configure(api_key=self.api_key)
            else:
                # Alternative configuration method for different versions
                from google.generativeai import configure
                configure(api_key=self.api_key)
            
            self.model = genai.GenerativeModel(self.model_name)
            self.enabled = True
            log_info(f"Gemini analyzer initialized successfully with model: {self.model_name}")
            log_info(f"Rate limits: {self.max_requests_per_minute} RPM, {self.max_requests_per_day} RPD")
            
        except ImportError as e:
            log_error(f"Import error with Gemini library: {e}")
            log_info("Try: pip install --upgrade google-generativeai")
            self.enabled = False
            
        except AttributeError as e:
            log_error(f"Gemini API method not found: {e}")
            log_info("Trying alternative initialization...")
            try:
                # Alternative initialization for different library versions
                self._alternative_init()
            except Exception as alt_error:
                log_error(f"Alternative Gemini init failed: {alt_error}")
                self.enabled = False
                
        except Exception as e:
            log_error(f"Failed to initialize Gemini: {e}")
            log_info("Falling back to gemini-1.5-flash model...")
            try:
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.model_name = 'gemini-1.5-flash'
                self.enabled = True
                log_info("Gemini fallback successful with gemini-1.5-flash")
            except Exception as fallback_error:
                log_error(f"Gemini fallback also failed: {fallback_error}")
                self.enabled = False
        
        # Initialize prompts
        if self.enabled:
            self._setup_prompts()
    
    def _alternative_init(self) -> None:
        """Alternative initialization method for different library versions"""
        try:
            # Method 1: Direct import and configure
            from google.generativeai import configure, GenerativeModel
            configure(api_key=self.api_key)
            self.model = GenerativeModel(self.model_name)
            self.enabled = True
            log_info(f"Alternative Gemini init successful with {self.model_name}")
            
        except Exception as e:
            # Method 2: Try older import structure  
            log_debug(f"Method 1 failed: {e}, trying method 2...")
            import google.generativeai as genai_alt
            genai_alt.configure(api_key=self.api_key)
            self.model = genai_alt.GenerativeModel(self.model_name)
            self.enabled = True
            log_info(f"Alternative Gemini init method 2 successful")
    
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
        
        # Clean old timestamps (older than 1 minute)
        minute_ago = now - timedelta(minutes=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        # Check per-minute limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            log_debug(f"Gemini per-minute rate limit reached: {len(self.request_timestamps)}/{self.max_requests_per_minute}")
            return False
        
        return True
    
    def _record_request(self) -> None:
        """Record a request for rate limiting"""
        now = datetime.now()
        self.request_timestamps.append(now)
        self.daily_request_count += 1
        
        if self.daily_request_count % 10 == 0:  # Log every 10 requests
            log_debug(f"Gemini usage: {self.daily_request_count}/{self.max_requests_per_day} daily, "
                     f"{len(self.request_timestamps)}/{self.max_requests_per_minute} per minute")
    
    def _wait_for_rate_limit(self) -> bool:
        """Wait for rate limit reset if close to limit"""
        if not self.request_timestamps:
            return True
        
        # If we're at the limit, wait for the oldest request to age out
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            oldest_request = min(self.request_timestamps)
            wait_until = oldest_request + timedelta(minutes=1, seconds=5)  # Add 5s buffer
            now = datetime.now()
            
            if now < wait_until:
                sleep_time = (wait_until - now).total_seconds()
                if sleep_time > 0 and sleep_time < 60:  # Don't wait more than 1 minute
                    log_info(f"Waiting {sleep_time:.1f}s for Gemini rate limit reset...")
                    time.sleep(sleep_time)
                    return True
                else:
                    log_warning("Gemini rate limit wait time too long, skipping")
                    return False
        
        return True
    
    def _setup_prompts(self) -> None:
        """Setup specialized prompts optimized for Flash model"""
        
        # Optimized for Flash model (faster, more efficient)
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
        
        # Specialized prompts for different topics
        self.earnings_prompt = """
You are analyzing EARNINGS news. Focus on beats/misses, guidance, and outlook.

{base_analysis}
"""
        
        self.biotech_prompt = """
You are analyzing BIOTECH/PHARMA news. Focus on trials, FDA actions, and drug approvals.

{base_analysis}
"""
        
        self.analyst_prompt = """
You are analyzing ANALYST coverage. Focus on rating changes, price targets, and credibility.

{base_analysis}
"""
    
    def analyze_news(self, symbol: str, title: str, content: str, 
                    current_price: float, topic: str = "general") -> Optional[GeminiAnalysis]:
        """Analyze news with improved rate limiting and error handling"""
        if not self.enabled:
            return None
        
        # Check rate limits
        if not self._check_rate_limits():
            log_debug(f"Skipping Gemini analysis for {symbol} due to rate limits")
            return None
        
        # Optionally wait if close to rate limit
        if not self._wait_for_rate_limit():
            return None
        
        try:
            # Record the request
            self._record_request()
            
            # Select appropriate prompt based on topic
            prompt = self._get_prompt_for_topic(topic)
            
            # Format the prompt
            formatted_prompt = prompt.format(
                symbol=symbol,
                current_price=current_price,
                title=title,
                content=content[:1200],  # Shorter content for Flash efficiency
                base_analysis=self.base_prompt.format(
                    symbol=symbol,
                    current_price=current_price,
                    title=title,
                    content=content[:1200]
                )
            )
            
            # Get Gemini response with timeout - handle different API versions
            try:
                # Try newer API with generation config
                if hasattr(genai, 'types') and hasattr(genai.types, 'GenerationConfig'):
                    response = self.model.generate_content(
                        formatted_prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=300,  # Shorter responses for rate limit efficiency
                            temperature=0.1         # More deterministic
                        )
                    )
                else:
                    # Fallback for older API versions
                    response = self.model.generate_content(formatted_prompt)
            except Exception as api_error:
                log_debug(f"Primary API call failed: {api_error}, trying fallback...")
                # Simple fallback
                response = self.model.generate_content(formatted_prompt)
            
            if not response or not response.text:
                log_warning(f"Empty response from Gemini for {symbol}")
                return None
            
            # Parse structured response
            analysis = self._parse_gemini_response(response.text, symbol)
            
            if analysis:
                log_debug(f"Gemini SUCCESS: {symbol} sentiment={analysis.sentiment_score:.2f}, "
                         f"impact={analysis.market_impact:.2f}, conf={analysis.confidence:.2f}")
            
            return analysis
            
        except Exception as e:
            # Handle specific API errors
            error_str = str(e)
            if "quota" in error_str.lower() or "rate" in error_str.lower():
                log_warning(f"Gemini rate limit hit for {symbol}: {e}")
                # Temporarily disable to avoid spam
                self.daily_request_count = self.max_requests_per_day
            else:
                log_error(f"Gemini API error for {symbol}: {e}")
            return None
    
    def _get_prompt_for_topic(self, topic: str) -> str:
        """Get specialized prompt based on news topic"""
        if topic == "earnings":
            return self.earnings_prompt
        elif topic == "biotech":
            return self.biotech_prompt
        elif topic == "analyst":
            return self.analyst_prompt
        else:
            return self.base_prompt
    
    def _parse_gemini_response(self, response_text: str, symbol: str) -> Optional[GeminiAnalysis]:
        """Enhanced response parsing with better error handling"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                log_warning(f"No JSON found in Gemini response for {symbol}")
                return self._fallback_parse(response_text, symbol)
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['sentiment', 'market_impact', 'confidence', 'reasoning']
            if not all(field in data for field in required_fields):
                log_warning(f"Missing required fields in Gemini response for {symbol}")
                return self._fallback_parse(response_text, symbol)
            
            # Normalize scores from 1-10 scale to appropriate ranges
            sentiment_normalized = (float(data['sentiment']) - 5.5) / 4.5  # 1-10 to -1 to 1
            market_impact_normalized = (float(data['market_impact']) - 1) / 9  # 1-10 to 0-1
            confidence_normalized = (float(data['confidence']) - 1) / 9  # 1-10 to 0-1
            
            # Clamp values to valid ranges
            sentiment_normalized = max(-1, min(1, sentiment_normalized))
            market_impact_normalized = max(0, min(1, market_impact_normalized))
            confidence_normalized = max(0, min(1, confidence_normalized))
            
            return GeminiAnalysis(
                sentiment_score=sentiment_normalized,
                market_impact=market_impact_normalized,
                confidence=confidence_normalized,
                reasoning=str(data['reasoning'])[:150],  # Limit length
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
    
    def _fallback_parse(self, response_text: str, symbol: str) -> Optional[GeminiAnalysis]:
        """Improved fallback parsing when JSON extraction fails"""
        try:
            # Simple keyword-based parsing as fallback
            text_lower = response_text.lower()
            
            # Extract sentiment (look for bullish/bearish indicators)
            sentiment = 0.0
            if any(word in text_lower for word in ['very bullish', 'strong buy', 'very positive']):
                sentiment = 0.8
            elif any(word in text_lower for word in ['bullish', 'positive', 'buy']):
                sentiment = 0.4
            elif any(word in text_lower for word in ['bearish', 'negative', 'sell']):
                sentiment = -0.4
            elif any(word in text_lower for word in ['very bearish', 'strong sell', 'very negative']):
                sentiment = -0.8
            
            # Extract market impact
            market_impact = 0.5  # Default moderate
            if any(word in text_lower for word in ['major catalyst', 'significant', 'large impact']):
                market_impact = 0.8
            elif any(word in text_lower for word in ['minor', 'small impact', 'limited']):
                market_impact = 0.3
            
            # Extract reasoning (first few sentences)
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
            # Check if we should continue based on rate limits
            if not self._check_rate_limits():
                log_info(f"Stopping batch analysis due to rate limits after {len(results)} items")
                # Fill remaining with None
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
                
                # Smart batching delay to avoid rate limits
                if len(results) % 5 == 0:
                    time.sleep(1)  # Brief pause every 5 requests
                    
            except Exception as e:
                log_error(f"Batch analysis error: {e}")
                results.append(None)
        
        return results
    
    def get_market_context_analysis(self, market_regime: str, vix_level: float) -> str:
        """Get market context for better analysis"""
        if not self.enabled:
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
            return response.text if response and response.text else "neutral conditions"
            
        except Exception as e:
            log_debug(f"Market context analysis error: {e}")
            return "neutral conditions"