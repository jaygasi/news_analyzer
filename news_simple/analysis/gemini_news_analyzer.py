"""
Gemini LLM-powered news analyzer for advanced sentiment and market impact analysis
"""
import google.generativeai as genai
import json
import re
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning, log_debug


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
    """Advanced news analyzer using Google Gemini LLM"""
    
    def __init__(self) -> None:
        """Initialize Gemini analyzer with API key and model configuration"""
        self.api_key = CONFIG.get_api_key('gemini')
        if not self.api_key:
            log_warning("GEMINI_API_KEY not found - Gemini analysis disabled")
            self.enabled = False
            return
        
        # Get model configuration from environment
        self.model_name = CONFIG.get_gemini_model()
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.enabled = True
            log_info(f"Gemini analyzer initialized successfully with model: {self.model_name}")
        except Exception as e:
            log_error(f"Failed to initialize Gemini with model {self.model_name}: {e}")
            log_info("Falling back to gemini-pro model...")
            try:
                self.model = genai.GenerativeModel('gemini-pro')
                self.model_name = 'gemini-pro'
                self.enabled = True
                log_info("Gemini fallback successful with gemini-pro")
            except Exception as fallback_error:
                log_error(f"Gemini fallback also failed: {fallback_error}")
                self.enabled = False
        
        # Initialize prompts
        self._setup_prompts()
    
    def _setup_prompts(self) -> None:
        """Setup specialized prompts for different news types with model optimization"""
        
        # Optimize prompts based on model capabilities
        if "1.5" in self.model_name:
            # Enhanced prompts for Gemini 1.5 models (better reasoning)
            self.base_prompt = """
You are an expert financial analyst with 15+ years of experience evaluating news for short-term trading impact. You understand market psychology, institutional behavior, and catalyst significance.

Analyze this news article and provide a structured assessment:

COMPANY: {symbol}
CURRENT PRICE: ${current_price}
TITLE: {title}
CONTENT: {content}

Provide your analysis in this EXACT JSON format:
{{
    "sentiment": <number 1-10 where 1=very bearish, 5=neutral, 10=very bullish>,
    "market_impact": <number 1-10 where 1=irrelevant, 5=moderate catalyst, 10=major market-moving event>,
    "confidence": <number 1-10 in your assessment quality>,
    "reasoning": "<2-3 sentence explanation with specific rationale>",
    "expected_move": "<up/down/sideways>",
    "move_magnitude": "<small/medium/large>",
    "catalyst_type": "<earnings/fda/analyst/corporate/general>"
}}

Consider these factors:
- How material is this news vs market expectations?
- What's the likely institutional response?
- Are there any competitive/sector implications?
- What's the typical market reaction to this type of catalyst?
- How does this fit the current market regime?

Response:
"""
        else:
            # Standard prompts for basic Gemini models
            self.base_prompt = """
You are an expert financial analyst evaluating news for short-term trading impact.

Analyze this news article and provide a structured assessment:

COMPANY: {symbol}
CURRENT PRICE: ${current_price}
TITLE: {title}
CONTENT: {content}

Provide your analysis in this EXACT JSON format:
{{
    "sentiment": <number 1-10 where 1=very bearish, 5=neutral, 10=very bullish>,
    "market_impact": <number 1-10 where 1=irrelevant, 5=moderate, 10=major catalyst>,
    "confidence": <number 1-10 in your assessment>,
    "reasoning": "<2-3 sentence explanation of your analysis>",
    "expected_move": "<up/down/sideways>",
    "move_magnitude": "<small/medium/large>",
    "catalyst_type": "<earnings/fda/analyst/corporate/general>"
}}

Consider:
- Is this material news that would move the stock?
- How does this compare to market expectations?
- What's the likely magnitude and duration of impact?

Response:
"""
        
        # Enhanced specialized prompts for 1.5 models
        if "1.5" in self.model_name:
            self.earnings_prompt = """
You are analyzing EARNINGS news with deep expertise in financial statement analysis and earnings quality assessment.

Focus on these critical factors:
- Beat/miss magnitude vs consensus estimates (revenue, EPS, guidance)
- Quality of earnings (recurring vs one-time items, cash flow alignment)
- Management commentary tone and forward guidance changes
- Sector positioning and competitive dynamics
- Historical earnings reaction patterns for this stock

{base_analysis}
"""
            
            self.biotech_prompt = """
You are analyzing BIOTECH/PHARMA news with deep expertise in drug development and regulatory processes.

Focus on these critical factors:
- Trial phase significance and statistical power (p-values, patient numbers)
- Competitive landscape analysis and differentiation
- Regulatory pathway complexity and timeline to approval
- Commercial potential and market size assessment
- Risk factors and probability of success

{base_analysis}
"""
            
            self.analyst_prompt = """
You are analyzing ANALYST coverage with expertise in sell-side research quality and market impact.

Focus on these critical factors:
- Analyst/firm reputation and track record
- Price target change magnitude and justification quality
- Upgrade/downgrade reasoning depth and data backing
- Contrarian vs consensus positioning
- Historical accuracy of this analyst on this stock

{base_analysis}
"""
        else:
            # Standard specialized prompts
            self.earnings_prompt = """
You are analyzing EARNINGS news. Focus on:
- Beat/miss vs expectations (revenue, EPS, guidance)
- Quality of earnings (one-time items, recurring revenue)
- Management commentary and outlook
- Sector implications

{base_analysis}
"""
            
            self.biotech_prompt = """
You are analyzing BIOTECH/PHARMA news. Focus on:
- Trial phase and statistical significance
- Competitive landscape and market size
- Regulatory pathway and timeline
- Commercial potential and revenue impact

{base_analysis}
"""
            
            self.analyst_prompt = """
You are analyzing ANALYST coverage. Focus on:
- Credibility of the analyst/firm
- Price target change magnitude
- Reasoning quality and data backing
- Contrarian vs consensus view

{base_analysis}
"""
    
    def analyze_news(self, symbol: str, title: str, content: str, 
                    current_price: float, topic: str = "general") -> Optional[GeminiAnalysis]:
        """Analyze news with Gemini LLM"""
        if not self.enabled:
            return None
        
        try:
            # Select appropriate prompt based on topic
            prompt = self._get_prompt_for_topic(topic)
            
            # Format the prompt
            formatted_prompt = prompt.format(
                symbol=symbol,
                current_price=current_price,
                title=title,
                content=content[:1500],  # Limit content length
                base_analysis=self.base_prompt.format(
                    symbol=symbol,
                    current_price=current_price,
                    title=title,
                    content=content[:1500]
                )
            )
            
            # Get Gemini response
            response = self.model.generate_content(formatted_prompt)
            
            if not response or not response.text:
                log_warning(f"Empty response from Gemini for {symbol}")
                return None
            
            # Parse structured response
            analysis = self._parse_gemini_response(response.text, symbol)
            
            if analysis:
                log_debug(f"Gemini analysis for {symbol}: "
                         f"sentiment={analysis.sentiment_score:.2f}, "
                         f"impact={analysis.market_impact:.2f}, "
                         f"conf={analysis.confidence:.2f}")
            
            return analysis
            
        except Exception as e:
            log_error(f"Error in Gemini analysis for {symbol}: {e}")
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
        """Parse Gemini's structured response"""
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
                reasoning=str(data['reasoning'])[:200],  # Limit length
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
        """Fallback parsing when JSON extraction fails"""
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
            sentences = response_text.split('.')[:3]
            reasoning = '. '.join(sentences)[:200]
            
            return GeminiAnalysis(
                sentiment_score=sentiment,
                market_impact=market_impact,
                confidence=0.6,  # Lower confidence for fallback
                reasoning=reasoning,
                expected_move='sideways',
                move_magnitude='medium',
                catalyst_type='general',
                raw_response=response_text
            )
            
        except Exception as e:
            log_error(f"Fallback parsing failed for {symbol}: {e}")
            return None
    
    def batch_analyze(self, news_items: List[Dict]) -> List[Optional[GeminiAnalysis]]:
        """Analyze multiple news items efficiently"""
        if not self.enabled:
            return [None] * len(news_items)
        
        results = []
        for item in news_items:
            try:
                analysis = self.analyze_news(
                    symbol=item.get('symbol', ''),
                    title=item.get('title', ''),
                    content=item.get('content', ''),
                    current_price=item.get('current_price', 0.0),
                    topic=item.get('topic', 'general')
                )
                results.append(analysis)
                
                # Rate limiting - Gemini has limits
                if len(results) % 5 == 0:
                    import time
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