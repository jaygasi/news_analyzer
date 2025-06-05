"""
Multi-LLM analyzer with fallback chain for directional prediction - Enhanced with toggles
Python 3.13.3 compatible
"""
import time
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Core imports that should always work
from config import Config
from utils.simple_logger import log_info, log_error, log_debug, log_warning

# Optional imports with error handling
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    log_warning("PyTorch/Transformers not available - FinBERT will be disabled")
    TORCH_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    log_warning("Google Generative AI not available - Gemini will be disabled")
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    log_warning("OpenAI not available - GPT models will be disabled")
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    log_warning("Anthropic not available - Claude will be disabled")
    ANTHROPIC_AVAILABLE = False

# Import emergency services
from analysis.emergency_data_services import EmergencyDataServices, DirectionalPrediction as EDSDirectionalPrediction


@dataclass
class DirectionalPrediction:
    """Result of directional analysis"""
    direction: str  # 'BUY', 'SELL', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    reasoning: str
    source: str  # Which service provided the prediction
    raw_score: float = 0.0
    
    # Multi-source specific data
    individual_predictions: List['DirectionalPrediction'] = field(default_factory=list)
    source_weights: Dict[str, float] = field(default_factory=dict)
    weighted_scores: Dict[str, float] = field(default_factory=dict)


class MultiLLMAnalyzer:
    """Enhanced analyzer with service toggles and multi-source support"""

    def __init__(self) -> None:
        """Initialize only enabled AI services"""
        self.services = {}
        self.quota_exhausted = set()

        # Enhanced keyword lists for robust fallback analysis
        self._init_enhanced_keywords()

        # Initialize emergency data services
        self.emergency_services = EmergencyDataServices()

        # Initialize only enabled services
        self._init_enabled_services()

        # Define service weights for combining predictions
        self._init_service_weights()

        enabled_count = len([s for s in self.services.values() if s.get('available')])
        log_info(f"Initialized {enabled_count} enabled AI services for analysis")

    def _init_enabled_services(self) -> None:
        """Initialize only enabled services based on config toggles"""
        
        # Primary services
        if Config.ENABLE_FINBERT and TORCH_AVAILABLE:
            self._init_finbert()
        else:
            log_info("FinBERT disabled by config or missing dependencies")
            
        if Config.ENABLE_GEMINI and GEMINI_AVAILABLE:
            self._init_gemini()
        else:
            log_info("Gemini disabled by config or missing dependencies")
            
        if Config.ENABLE_OPENAI and OPENAI_AVAILABLE:
            self._init_openai()
        else:
            log_info("OpenAI disabled by config or missing dependencies")
            
        if Config.ENABLE_CLAUDE and ANTHROPIC_AVAILABLE:
            self._init_anthropic()
        else:
            log_info("Claude disabled by config or missing dependencies")

        # Emergency services
        if Config.ENABLE_ALPHA_VANTAGE:
            self._init_alpha_vantage()
        else:
            log_info("Alpha Vantage disabled by config")
            
        if Config.ENABLE_POLYGON:
            self._init_polygon()
        else:
            log_info("Polygon disabled by config")
            
        if Config.ENABLE_TIINGO:
            self._init_tiingo()
        else:
            log_info("Tiingo disabled by config")

    def _init_service_weights(self) -> None:
        """Initialize weights for combining different service predictions"""
        # Get enabled services
        enabled_services = [name for name, service in self.services.items() if service.get('available')]
        
        if not enabled_services:
            log_warning("No services enabled!")
            return
        
        # Base weights (will be normalized)
        base_weights = {
            'finbert': 0.25,      # Specialized financial model
            'gemini': 0.20,       # Google's LLM
            'openai': 0.20,       # OpenAI's models
            'claude': 0.15,       # Anthropic's Claude
            'alpha_vantage': 0.08,  # News sentiment from Alpha Vantage
            'polygon': 0.06,        # Polygon news analysis
            'tiingo': 0.04,         # Tiingo news analysis
            'enhanced_keyword_analysis': 0.02
        }
        
        # Filter to only enabled services
        self.service_weights = {}
        total_weight = 0.0
        
        for service in enabled_services:
            if service in base_weights:
                self.service_weights[service] = base_weights[service]
                total_weight += base_weights[service]
        
        # Add keyword analysis if enabled
        if Config.ENABLE_KEYWORD_ANALYSIS:
            self.service_weights['enhanced_keyword_analysis'] = base_weights['enhanced_keyword_analysis']
            total_weight += base_weights['enhanced_keyword_analysis']
        
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            self.service_weights = {k: v / total_weight for k, v in self.service_weights.items()}
        
        log_info(f"Service weights (enabled only): {self.service_weights}")

    def _init_enhanced_keywords(self) -> None:
        """Initialize comprehensive keyword lists for robust analysis"""
        self.positive_keywords = [
            # Earnings & Financial Performance
            'beat', 'exceed', 'outperform', 'surpass', 'stronger', 'robust', 'solid',
            'growth', 'increase', 'rise', 'surge', 'jump', 'soar', 'climb', 'gain',
            'revenue growth', 'profit margin', 'earnings beat', 'guidance raised',
            'positive outlook', 'strong results', 'record revenue', 'improved margins',

            # Business Development
            'acquisition', 'merger', 'partnership', 'joint venture', 'collaboration',
            'expansion', 'launch', 'breakthrough', 'innovation', 'patent',
            'contract', 'deal', 'agreement', 'signed', 'secured', 'won',
            'new product', 'market expansion', 'strategic alliance',

            # Regulatory & Approvals
            'approval', 'cleared', 'authorized', 'granted', 'licensed',
            'fda approval', 'regulatory approval', 'certification', 'patent granted',
            'breakthrough therapy', 'fast track', 'orphan drug designation',

            # Market Position
            'market leader', 'competitive advantage', 'market share',
            'first mover', 'exclusive', 'monopoly', 'dominant', 'leadership',
            'outpacing competitors', 'gaining share', 'market dominance',

            # Investment & Funding
            'investment', 'funding', 'capital', 'ipo', 'dividend', 'dividend increase',
            'buyback', 'share repurchase', 'upgraded', 'buy rating', 'price target raised',
            'institutional buying', 'analyst upgrade', 'overweight rating',

            # Performance Indicators
            'success', 'achievement', 'milestone', 'record', 'all-time high',
            'outperformed', 'momentum', 'accelerated', 'improved', 'stellar',
            'exceptional', 'outstanding', 'impressive', 'strong demand'
        ]

        self.negative_keywords = [
            # Earnings & Financial Performance
            'miss', 'missed', 'below', 'decline', 'decrease', 'fall', 'drop',
            'plunge', 'crash', 'slump', 'weak', 'disappointing', 'poor',
            'loss', 'losses', 'deficit', 'shortfall', 'guidance lowered',
            'revenue decline', 'margin compression', 'weak outlook',

            # Business Challenges
            'bankruptcy', 'insolvent', 'restructuring', 'layoffs', 'cuts',
            'closure', 'shutdown', 'suspended', 'terminated', 'cancelled',
            'delayed', 'postponed', 'failed', 'unsuccessful', 'struggling',
            'cash crunch', 'debt burden', 'financial distress',

            # Legal & Regulatory Issues
            'lawsuit', 'litigation', 'investigation', 'probe', 'audit',
            'violation', 'fine', 'penalty', 'sanctions', 'banned',
            'rejected', 'denied', 'warning', 'recall', 'subpoena',
            'regulatory action', 'compliance issues', 'sec investigation',

            # Market Position
            'competition', 'losing share', 'market pressure', 'disrupted',
            'downgraded', 'sell rating', 'underperform', 'price target cut',
            'analyst downgrade', 'competitive threat', 'market share loss',

            # Operational Issues
            'supply chain', 'shortage', 'disruption', 'cyber attack',
            'data breach', 'fraud', 'scandal', 'controversy', 'crisis',
            'operational challenges', 'production issues', 'quality problems',

            # Performance Indicators
            'failure', 'setback', 'disappointed', 'concerns', 'risks',
            'uncertainty', 'volatility', 'pressure', 'challenges', 'headwinds',
            'deteriorating', 'weakening', 'struggling', 'disappointing results'
        ]

        # High-impact keywords that carry more weight
        self.high_impact_positive = [
            'fda approval', 'merger', 'acquisition', 'breakthrough therapy',
            'earnings beat', 'guidance raised', 'analyst upgrade', 'record revenue'
        ]

        self.high_impact_negative = [
            'bankruptcy', 'fda rejection', 'lawsuit', 'investigation',
            'earnings miss', 'guidance lowered', 'analyst downgrade', 'recall'
        ]

    def _init_finbert(self) -> None:
        """Initialize FinBERT model (local)"""
        if not TORCH_AVAILABLE:
            self.services['finbert'] = {'available': False}
            return
            
        try:
            model_name = "ProsusAI/finbert"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(device)
            model.eval()

            self.services['finbert'] = {
                'tokenizer': tokenizer,
                'model': model,
                'device': device,
                'available': True
            }

            log_info(f"FinBERT initialized on {device}")

        except Exception as e:
            log_error(f"Failed to initialize FinBERT: {e}")
            self.services['finbert'] = {'available': False}

    def _init_gemini(self) -> None:
        """Initialize Gemini API"""
        if not GEMINI_AVAILABLE:
            self.services['gemini'] = {'available': False}
            return
            
        try:
            if Config.GEMINI_API_KEY:
                genai.configure(api_key=Config.GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')

                self.services['gemini'] = {
                    'model': model,
                    'available': True,
                    'requests_today': 0,
                    'daily_limit': 1000
                }

                log_info("Gemini API initialized")
            else:
                self.services['gemini'] = {'available': False}

        except Exception as e:
            log_error(f"Failed to initialize Gemini: {e}")
            self.services['gemini'] = {'available': False}

    def _init_openai(self) -> None:
        """Initialize OpenAI API"""
        if not OPENAI_AVAILABLE:
            self.services['openai'] = {'available': False}
            return
            
        try:
            if Config.OPENAI_API_KEY:
                self.services['openai'] = {
                    'client': openai.OpenAI(api_key=Config.OPENAI_API_KEY),
                    'available': True,
                    'requests_today': 0,
                    'daily_limit': 500
                }

                log_info("OpenAI API initialized")
            else:
                self.services['openai'] = {'available': False}

        except Exception as e:
            log_error(f"Failed to initialize OpenAI: {e}")
            self.services['openai'] = {'available': False}

    def _init_anthropic(self) -> None:
        """Initialize Anthropic Claude API"""
        if not ANTHROPIC_AVAILABLE:
            self.services['claude'] = {'available': False}
            return
            
        try:
            if Config.ANTHROPIC_API_KEY:
                client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

                self.services['claude'] = {
                    'client': client,
                    'available': True,
                    'requests_today': 0,
                    'daily_limit': 300
                }

                log_info("Anthropic Claude API initialized")
            else:
                self.services['claude'] = {'available': False}

        except Exception as e:
            log_error(f"Failed to initialize Anthropic: {e}")
            self.services['claude'] = {'available': False}

    def _init_alpha_vantage(self) -> None:
        """Initialize Alpha Vantage emergency service"""
        if Config.ALPHA_VANTAGE_API_KEY:
            self.services['alpha_vantage'] = {
                'available': True,
                'requests_today': 0,
                'daily_limit': 500
            }
            log_info("Alpha Vantage emergency service initialized")

    def _init_polygon(self) -> None:
        """Initialize Polygon emergency service"""
        if Config.POLYGON_API_KEY:
            self.services['polygon'] = {
                'available': True,
                'requests_today': 0,
                'daily_limit': 500
            }
            log_info("Polygon emergency service initialized")

    def _init_tiingo(self) -> None:
        """Initialize Tiingo emergency service"""
        if Config.TIINGO_API_KEY:
            self.services['tiingo'] = {
                'available': True,
                'requests_today': 0,
                'daily_limit': 1000
            }
            log_info("Tiingo emergency service initialized")

    def analyze_news_direction(self, ticker: str, articles: List[Dict[str, Any]]) -> Optional[DirectionalPrediction]:
        """
        Analyze news articles using enabled services and combine their predictions.
        Enhanced to call multiple services and combine results.
        """
        if not articles:
            return None

        combined_text = self._combine_articles(articles)

        if len(combined_text) < 50:  # Too little content
            return None

        # Collect predictions from all available enabled sources
        predictions = []
        sources_used = []

        # Try enabled LLM services
        enabled_llm_services = Config.get_enabled_llm_services()
        for service_name in enabled_llm_services:
            if self._is_service_available(service_name):
                try:
                    prediction = self._analyze_with_service(service_name, ticker, combined_text)
                    if prediction:
                        predictions.append(prediction)
                        sources_used.append(service_name)
                        log_debug(f"Got prediction from {service_name}: {prediction.direction} ({prediction.confidence:.3f})")
                except Exception as e:
                    log_warning(f"{service_name} analysis failed: {e}")
                    self._mark_service_exhausted(service_name)

        # Try enabled emergency services
        enabled_emergency_services = Config.get_enabled_emergency_services()
        for service_name in enabled_emergency_services:
            if self._is_service_available(service_name):
                try:
                    prediction = self._analyze_with_emergency_service(service_name, ticker, combined_text)
                    if prediction:
                        predictions.append(prediction)
                        sources_used.append(service_name)
                        log_debug(f"Got prediction from {service_name}: {prediction.direction} ({prediction.confidence:.3f})")
                except Exception as e:
                    log_warning(f"Emergency service {service_name} failed: {e}")
                    self._mark_service_exhausted(service_name)

        # Add keyword analysis if enabled
        if Config.ENABLE_KEYWORD_ANALYSIS:
            try:
                keyword_prediction = self._enhanced_keyword_analysis(combined_text)
                if keyword_prediction:
                    predictions.append(keyword_prediction)
                    sources_used.append('enhanced_keyword_analysis')
                    log_debug(f"Got keyword prediction: {keyword_prediction.direction} ({keyword_prediction.confidence:.3f})")
            except Exception as e:
                log_warning(f"Keyword analysis failed: {e}")

        # Combine all predictions
        if predictions:
            combined_prediction = self._combine_predictions(predictions, sources_used)
            log_info(f"Combined {len(predictions)} predictions for {ticker}: {combined_prediction.direction} ({combined_prediction.confidence:.3f}) from {', '.join(sources_used)}")
            return combined_prediction
        else:
            log_warning(f"No predictions available for {ticker}")
            return None

    def _combine_predictions(self, predictions: List[DirectionalPrediction], sources_used: List[str]) -> DirectionalPrediction:
        """Combine multiple predictions using weighted averaging"""
        if not predictions:
            return None

        # Calculate weighted scores
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        source_weights_used = {}
        weighted_scores = {}

        reasoning_parts = []

        for prediction, source in zip(predictions, sources_used):
            # Get weight for this source
            weight = self.service_weights.get(source, 0.01)  # Default small weight for unknown sources
            source_weights_used[source] = weight

            # Convert prediction to numeric score
            if prediction.direction == 'BUY':
                score = prediction.confidence
            elif prediction.direction == 'SELL':
                score = -prediction.confidence
            else:  # NEUTRAL
                score = 0.0

            # Apply weighting
            weighted_score += score * weight
            weighted_confidence += prediction.confidence * weight
            total_weight += weight

            weighted_scores[source] = score * weight

            # Add to reasoning
            reasoning_parts.append(f"{source}: {prediction.direction}({prediction.confidence:.2f})")

        # Normalize by total weight
        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = 0.0
            final_confidence = 0.0

        # Determine final direction
        if final_score > 0.2:  # Lowered threshold
            direction = 'BUY'
            confidence = min(abs(final_score), 1.0)
        elif final_score < -0.2:  # Lowered threshold
            direction = 'SELL'
            confidence = min(abs(final_score), 1.0)
        else:
            direction = 'NEUTRAL'
            confidence = 1.0 - abs(final_score)

        # Boost confidence based on number of agreeing sources
        agreement_boost = min(0.1 * len(predictions), 0.3)  # Up to 30% boost for agreement
        confidence = min(confidence + agreement_boost, 1.0)

        reasoning = f"Multi-source analysis ({len(predictions)} sources): {'; '.join(reasoning_parts[:5])}"

        # Create enhanced prediction with multi-source data
        combined_prediction = DirectionalPrediction(
            direction=direction,
            confidence=confidence,
            reasoning=reasoning[:300],
            source="multi_source",
            raw_score=final_score
        )
        
        # Add multi-source data
        combined_prediction.individual_predictions = predictions
        combined_prediction.source_weights = source_weights_used
        combined_prediction.weighted_scores = weighted_scores
        
        return combined_prediction

    def _combine_articles(self, articles: List[Dict[str, Any]]) -> str:
        """Combine multiple articles into single text for analysis"""
        combined_parts = []

        for article in articles[:5]:  # Limit to 5 articles to avoid token limits
            title = str(article.get('title', '')).strip()
            text = str(article.get('text', '')).strip()

            if title:
                combined_parts.append(f"HEADLINE: {title}")
            if text:
                combined_parts.append(f"CONTENT: {text[:500]}")  # Limit content length

        return "\n\n".join(combined_parts)

    def _is_service_available(self, service_name: str) -> bool:
        """Check if service is available and not quota exhausted"""
        if service_name in self.quota_exhausted:
            return False

        service = self.services.get(service_name, {})
        if not service.get('available', False):
            return False

        # Check daily limits
        requests_today = service.get('requests_today', 0)
        daily_limit = service.get('daily_limit', 0)

        if daily_limit > 0 and requests_today >= daily_limit:
            self._mark_service_exhausted(service_name)
            return False

        return True

    def _mark_service_exhausted(self, service_name: str) -> None:
        """Mark service as quota exhausted"""
        self.quota_exhausted.add(service_name)
        log_warning(f"Service {service_name} marked as quota exhausted")

    def _analyze_with_service(self, service_name: str, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze with specific LLM service"""
        if service_name == 'finbert':
            return self._analyze_with_finbert(ticker, text)
        elif service_name == 'gemini':
            return self._analyze_with_gemini(ticker, text)
        elif service_name == 'openai':
            return self._analyze_with_openai(ticker, text)
        elif service_name == 'claude':
            return self._analyze_with_claude(ticker, text)

        return None

    def _analyze_with_emergency_service(self, service_name: str, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using real emergency fallback services and convert to local DirectionalPrediction."""
        try:
            eds_prediction: Optional[EDSDirectionalPrediction] = None
            if service_name == 'alpha_vantage':
                eds_prediction = self.emergency_services.analyze_with_alpha_vantage(ticker, text)
            elif service_name == 'polygon':
                eds_prediction = self.emergency_services.analyze_with_polygon(ticker, text)
            elif service_name == 'tiingo':
                eds_prediction = self.emergency_services.analyze_with_tiingo(ticker, text)
            else:
                return None

            # Convert EDSDirectionalPrediction to local DirectionalPrediction
            if eds_prediction:
                # Update request count
                if service_name in self.services:
                    self.services[service_name]['requests_today'] += 1

                return DirectionalPrediction(
                    direction=eds_prediction.direction,
                    confidence=eds_prediction.confidence,
                    reasoning=eds_prediction.reasoning,
                    source=eds_prediction.source,
                    raw_score=eds_prediction.raw_score
                )
            return None

        except Exception as e:
            log_error(f"Emergency service {service_name} error: {e}")
            return None

    def _analyze_with_finbert(self, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using FinBERT"""
        if not TORCH_AVAILABLE:
            return None
            
        try:
            service = self.services['finbert']
            tokenizer = service['tokenizer']
            model = service['model']
            device = service['device']

            # Tokenize
            inputs = tokenizer(
                text[:512],  # Limit to model's max length
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().numpy()[0]

            # FinBERT outputs: [positive, negative, neutral]
            positive_score = float(predictions[0])
            negative_score = float(predictions[1])
            neutral_score = float(predictions[2])

            # Determine direction
            if positive_score > negative_score and positive_score > neutral_score:
                direction = 'BUY'
                confidence = positive_score
            elif negative_score > positive_score and negative_score > neutral_score:
                direction = 'SELL'
                confidence = negative_score
            else:
                direction = 'NEUTRAL'
                confidence = neutral_score

            return DirectionalPrediction(
                direction=direction,
                confidence=confidence,
                reasoning=f"FinBERT scores - Positive: {positive_score:.3f}, Negative: {negative_score:.3f}, Neutral: {neutral_score:.3f}",
                source="finbert",
                raw_score=positive_score - negative_score
            )

        except Exception as e:
            log_error(f"FinBERT analysis error: {e}")
            return None

    def _analyze_with_gemini(self, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using Gemini"""
        if not GEMINI_AVAILABLE:
            return None
            
        try:
            service = self.services['gemini']
            model = service['model']

            prompt = self._create_llm_prompt(ticker, text)

            time.sleep(Config.LLM_REQUEST_DELAY)  # Rate limiting
            response = model.generate_content(prompt)

            service['requests_today'] += 1

            if response and response.text:
                return self._parse_llm_response(response.text, "gemini")
            return None

        except Exception as e:
            log_error(f"Gemini analysis error: {e}")
            return None

    def _analyze_with_openai(self, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using OpenAI"""
        if not OPENAI_AVAILABLE:
            return None
            
        try:
            service = self.services['openai']
            client = service['client']

            prompt = self._create_llm_prompt(ticker, text)

            time.sleep(Config.LLM_REQUEST_DELAY)  # Rate limiting
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )

            service['requests_today'] += 1

            if response.choices and response.choices[0].message.content:
                return self._parse_llm_response(response.choices[0].message.content, "openai")
            return None

        except Exception as e:
            log_error(f"OpenAI analysis error: {e}")
            return None

    def _analyze_with_claude(self, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using Anthropic Claude"""
        if not ANTHROPIC_AVAILABLE:
            return None
            
        try:
            service = self.services['claude']
            client = service['client']

            prompt = self._create_llm_prompt(ticker, text)

            time.sleep(Config.LLM_REQUEST_DELAY)  # Rate limiting
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            service['requests_today'] += 1

            if response.content and response.content[0].text:
                return self._parse_llm_response(response.content[0].text, "claude")
            return None

        except Exception as e:
            log_error(f"Claude analysis error: {e}")
            return None

    def _create_llm_prompt(self, ticker: str, text: str) -> str:
        """Create prompt for LLM analysis"""
        return f"""
Analyze the following financial news about {ticker} and predict if the stock price will go UP, DOWN, or stay NEUTRAL.

News content:
{text[:1500]}

Respond with ONLY a JSON object in this format:
{{
    "direction": "BUY|SELL|NEUTRAL",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Focus on:
- Earnings beats/misses
- Revenue changes
- Guidance updates
- FDA approvals/rejections
- Mergers/acquisitions
- Management changes
- Product launches
- Regulatory changes

Consider the actual financial impact on the company's value.
"""

    def _parse_llm_response(self, response_text: str, source: str) -> Optional[DirectionalPrediction]:
        """Parse LLM response into DirectionalPrediction"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                direction = data.get('direction', 'NEUTRAL').upper()
                if direction not in ['BUY', 'SELL', 'NEUTRAL']:
                    direction = 'NEUTRAL'

                confidence = float(data.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))

                reasoning = str(data.get('reasoning', 'No reasoning provided'))[:200]

                return DirectionalPrediction(
                    direction=direction,
                    confidence=confidence,
                    reasoning=reasoning,
                    source=source
                )

            # Fallback: simple text parsing
            response_lower = response_text.lower()
            if 'buy' in response_lower or 'bullish' in response_lower or 'positive' in response_lower:
                direction = 'BUY'
                confidence = 0.6
            elif 'sell' in response_lower or 'bearish' in response_lower or 'negative' in response_lower:
                direction = 'SELL'
                confidence = 0.6
            else:
                direction = 'NEUTRAL'
                confidence = 0.5

            return DirectionalPrediction(
                direction=direction,
                confidence=confidence,
                reasoning="Parsed from text response",
                source=source
            )

        except Exception as e:
            log_error(f"Error parsing {source} response: {e}")
            return None

    def _enhanced_keyword_analysis(self, text: str) -> Optional[DirectionalPrediction]:
        """Enhanced keyword-based analysis as additional input"""
        text_lower = text.lower()

        positive_score = 0.0
        negative_score = 0.0
        reasoning_parts = []

        # Check for high-impact keywords first (weighted more heavily)
        for keyword in self.high_impact_positive:
            if keyword in text_lower:
                positive_score += 2.0
                reasoning_parts.append(f"High-impact positive: {keyword}")

        for keyword in self.high_impact_negative:
            if keyword in text_lower:
                negative_score += 2.0
                reasoning_parts.append(f"High-impact negative: {keyword}")

        # Standard positive keywords
        for keyword in self.positive_keywords:
            if keyword in text_lower:
                positive_score += 1.0

        # Standard negative keywords
        for keyword in self.negative_keywords:
            if keyword in text_lower:
                negative_score += 1.0

        # Determine direction and confidence
        net_score = positive_score - negative_score
        total_score = positive_score + negative_score

        if total_score == 0:
            return DirectionalPrediction(
                direction='NEUTRAL',
                confidence=0.3,
                reasoning="No significant keywords found",
                source="enhanced_keyword_analysis"
            )

        # Calculate confidence based on score strength and total signals
        confidence = min(0.8, 0.4 + (abs(net_score) / max(total_score, 1)) * 0.4)

        if net_score > 1.0:
            direction = 'BUY'
        elif net_score < -1.0:
            direction = 'SELL'
        else:
            direction = 'NEUTRAL'
            confidence = 0.4

        reasoning = f"Enhanced keyword analysis: +{positive_score:.1f}/-{negative_score:.1f}. " + "; ".join(reasoning_parts[:3])

        return DirectionalPrediction(
            direction=direction,
            confidence=confidence,
            reasoning=reasoning[:200],
            source="enhanced_keyword_analysis",
            raw_score=net_score
        )

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {}

        for service_name, service_info in self.services.items():
            status[service_name] = {
                'available': service_info.get('available', False),
                'quota_exhausted': service_name in self.quota_exhausted,
                'requests_today': service_info.get('requests_today', 0),
                'daily_limit': service_info.get('daily_limit', 0),
                'weight': self.service_weights.get(service_name, 0.0)
            }

        return status