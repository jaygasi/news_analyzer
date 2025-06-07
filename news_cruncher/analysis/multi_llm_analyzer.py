"""
Multi-LLM analyzer with fallback chain for directional prediction - Enhanced with RoBERTa+LSTM/CNN hybrid
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

# Import enhanced neural analyzer
try:
    from analysis.enhanced_neural_analyzer import EnhancedNeuralAnalyzer, EnhancedPrediction
    ENHANCED_NEURAL_AVAILABLE = True
except ImportError:
    log_warning("Enhanced Neural Analyzer not available")
    ENHANCED_NEURAL_AVAILABLE = False


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
    """Enhanced analyzer with service toggles, multi-source support, and enhanced neural analysis"""

    def __init__(self) -> None:
        """Initialize only enabled AI services including enhanced neural analyzer"""
        self.services = {}
        self.quota_exhausted = set()

        # Enhanced keyword lists for robust fallback analysis
        self._init_enhanced_keywords()

        # Initialize emergency data services
        self.emergency_services = EmergencyDataServices()

        # Initialize enhanced neural analyzer (NEW)
        self.enhanced_neural = None
        if Config.ENABLE_ENHANCED_NEURAL and ENHANCED_NEURAL_AVAILABLE:
            try:
                self.enhanced_neural = EnhancedNeuralAnalyzer()
                if self.enhanced_neural.is_available:
                    log_info("✅ Enhanced Neural Analyzer (RoBERTa+LSTM+CNN) initialized successfully")
                else:
                    log_warning("❌ Enhanced Neural Analyzer failed to initialize")
                    self.enhanced_neural = None
            except Exception as e:
                log_error(f"Failed to initialize Enhanced Neural Analyzer: {e}")
                self.enhanced_neural = None

        # Initialize only enabled services
        self._init_enabled_services()

        # Define service weights for combining predictions
        self._init_service_weights()

        enabled_count = len([s for s in self.services.values() if s.get('available')])
        total_services = enabled_count + (1 if self.enhanced_neural else 0)
        log_info(f"Initialized {total_services} enabled AI services for analysis (including {enabled_count} traditional + enhanced neural)")

    def _init_enabled_services(self) -> None:
        """Initialize only enabled services based on config toggles"""
        
        # Primary services
        if Config.ENABLE_FINBERT and TORCH_AVAILABLE:
            self._init_finbert()
        
        if Config.ENABLE_GEMINI and GEMINI_AVAILABLE:
            self._init_gemini()
        
        if Config.ENABLE_OPENAI and OPENAI_AVAILABLE:
            self._init_openai()
        
        if Config.ENABLE_CLAUDE and ANTHROPIC_AVAILABLE:
            self._init_claude()
        
        # Emergency services
        if Config.ENABLE_ALPHA_VANTAGE:
            self._init_alpha_vantage()
        
        if Config.ENABLE_POLYGON:
            self._init_polygon()
        
        if Config.ENABLE_TIINGO:
            self._init_tiingo()
        
        # Keyword analysis always available
        if Config.ENABLE_KEYWORD_SENTIMENT:
            self._init_keyword_analysis()

    def _init_service_weights(self) -> None:
        """Define weights for combining predictions from different services"""
        self.service_weights = {
            # Enhanced neural analyzer gets highest weight due to superior accuracy
            'enhanced_neural': 0.45,  # NEW: Highest weight for 94-96% accuracy model
            
            # Traditional services (rebalanced)
            'finbert': 0.25,          # Reduced from 0.41 to accommodate enhanced neural
            'gemini': 0.20,           # Reduced from 0.33
            'openai': 0.12,           # Reduced from 0.20
            'claude': 0.10,           # Reduced from 0.15
            
            # Emergency services (unchanged)
            'alpha_vantage': 0.08,
            'polygon': 0.06,
            'tiingo': 0.03,
            
            # Keyword fallback (unchanged)
            'keyword': 0.02
        }

    def _init_enhanced_keywords(self) -> None:
        """Initialize enhanced financial keyword lists for robust fallback analysis"""
        
        # Comprehensive positive sentiment keywords
        self.positive_keywords = [
            # Performance & Results
            'beat', 'beats', 'exceeded', 'exceeds', 'outperformed', 'strong', 'robust',
            'solid', 'impressive', 'record', 'milestone', 'achievement', 'success',
            
            # Growth & Expansion  
            'growth', 'grew', 'growing', 'expansion', 'expanding', 'increase', 'increased',
            'rising', 'uptick', 'momentum', 'acceleration', 'scaling', 'breakthrough',
            
            # Financial Strength
            'profitable', 'profitability', 'margins', 'cash flow', 'revenue growth',
            'earnings growth', 'return on investment', 'shareholder value', 'dividend',
            
            # Market Position
            'market leader', 'competitive advantage', 'market share', 'innovation',
            'breakthrough', 'patent', 'partnership', 'acquisition', 'merger',
            
            # Future Outlook
            'optimistic', 'confident', 'positive outlook', 'raised guidance',
            'upgraded', 'buy rating', 'target price', 'analyst upgrade', 'bullish',
            
            # Operational Excellence
            'efficient', 'streamlined', 'cost savings', 'productivity', 'quality',
            'customer satisfaction', 'brand strength', 'operational excellence'
        ]

        # Comprehensive negative sentiment keywords  
        self.negative_keywords = [
            # Poor Performance
            'missed', 'miss', 'disappointing', 'weak', 'poor', 'declined', 'fell',
            'dropped', 'decrease', 'reduced', 'lower', 'worst', 'failure', 'setback',
            
            # Financial Troubles
            'loss', 'losses', 'debt', 'bankruptcy', 'restructuring', 'writedown',
            'impairment', 'margin compression', 'cash burn', 'liquidity concerns',
            
            # Legal & Regulatory
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
        """Initialize FinBERT with enhanced error handling"""
        if not TORCH_AVAILABLE:
            self.services['finbert'] = {'available': False}
            return
            
        try:
            # Suppress the specific warning about untrained weights
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")
                
                # Load with specific configuration to reduce warnings
                self.finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                    'ProsusAI/finbert',
                    local_files_only=False,
                    trust_remote_code=False
                )
            
            # Check if CUDA is available and move model to GPU
            if torch.cuda.is_available():
                self.finbert_model = self.finbert_model.cuda()
                device_info = "CUDA GPU"
            else:
                device_info = "CPU"
            
            self.services['finbert'] = {
                'available': True,
                'model': self.finbert_model,
                'tokenizer': self.finbert_tokenizer,
                'requests_today': 0,
                'quota_limit': float('inf'),  # Local model - no quota
                'device': device_info
            }
            
            log_info(f"✅ FinBERT initialized successfully on {device_info}")

        except Exception as e:
            log_error(f"Failed to initialize FinBERT: {e}")
            self.services['finbert'] = {'available': False}

    def _init_gemini(self) -> None:
        """Initialize Google Gemini"""
        if not GEMINI_AVAILABLE or not Config.GEMINI_API_KEY:
            self.services['gemini'] = {'available': False}
            return
            
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-pro')

            self.services['gemini'] = {
                'available': True,
                'model': model,
                'requests_today': 0,
                'quota_limit': 1500
            }
            
            log_info("✅ Gemini initialized successfully")

        except Exception as e:
            log_error(f"Failed to initialize Gemini: {e}")
            self.services['gemini'] = {'available': False}

    def _init_openai(self) -> None:
        """Initialize OpenAI"""
        if not OPENAI_AVAILABLE or not Config.OPENAI_API_KEY:
            self.services['openai'] = {'available': False}
            return
            
        try:
            client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

            self.services['openai'] = {
                'available': True,
                'client': client,
                'requests_today': 0,
                'quota_limit': 1000
            }
            
            log_info("✅ OpenAI initialized successfully")

        except Exception as e:
            log_error(f"Failed to initialize OpenAI: {e}")
            self.services['openai'] = {'available': False}

    def _init_claude(self) -> None:
        """Initialize Anthropic Claude"""
        if not ANTHROPIC_AVAILABLE or not Config.ANTHROPIC_API_KEY:
            self.services['claude'] = {'available': False}
            return
            
        try:
            client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

            self.services['claude'] = {
                'available': True,
                'client': client,
                'requests_today': 0,
                'quota_limit': 300
            }
            
            log_info("✅ Claude initialized successfully")

        except Exception as e:
            log_error(f"Failed to initialize Claude: {e}")
            self.services['claude'] = {'available': False}

    def _init_alpha_vantage(self) -> None:
        """Initialize Alpha Vantage emergency service"""
        self.services['alpha_vantage'] = {
            'available': bool(Config.ALPHA_VANTAGE_API_KEY),
            'requests_today': 0,
            'quota_limit': 500
        }

    def _init_polygon(self) -> None:
        """Initialize Polygon emergency service"""
        self.services['polygon'] = {
            'available': bool(Config.POLYGON_API_KEY),
            'requests_today': 0,
            'quota_limit': 500
        }

    def _init_tiingo(self) -> None:
        """Initialize Tiingo emergency service"""
        self.services['tiingo'] = {
            'available': bool(Config.TIINGO_API_KEY),
            'requests_today': 0,
            'quota_limit': 1000
        }

    def _init_keyword_analysis(self) -> None:
        """Initialize keyword-based analysis (always available)"""
        self.services['keyword'] = {
            'available': True,
            'requests_today': 0,
            'quota_limit': float('inf')
        }

    def analyze_ticker_multi_source(self, ticker: str, articles: List[Dict[str, Any]]) -> Optional[DirectionalPrediction]:
        """
        Analyze ticker using multi-source approach - this is an alias for analyze_news_direction
        This method exists to maintain compatibility with calling code
        """
        return self.analyze_news_direction(ticker, articles)
    
    def analyze_news_direction(self, ticker: str, articles: List[Dict[str, Any]]) -> Optional[DirectionalPrediction]:
        """
        Analyze news direction using all available services including enhanced neural analyzer
        
        Returns multi-source consensus prediction with enhanced neural analysis priority
        """
        if not articles:
            return None

        # Combine all article text
        combined_text = ' '.join([
            f"{article.get('title', '')} {article.get('text', '')}"
            for article in articles
        ])

        if not combined_text.strip():
            return None

        log_debug(f"Analyzing {len(articles)} articles for {ticker} using all available services")

        predictions = []
        successful_services = []

        # 1. PRIORITY: Enhanced Neural Analysis (NEW)
        if self.enhanced_neural:
            try:
                enhanced_pred = self.enhanced_neural.analyze_text(ticker, combined_text)
                if enhanced_pred:
                    # Convert EnhancedPrediction to DirectionalPrediction
                    neural_prediction = DirectionalPrediction(
                        direction=enhanced_pred.direction,
                        confidence=enhanced_pred.confidence,
                        reasoning=enhanced_pred.reasoning,
                        source=enhanced_pred.source,
                        raw_score=enhanced_pred.raw_score
                    )
                    predictions.append(neural_prediction)
                    successful_services.append('enhanced_neural')
                    log_debug(f"✅ Enhanced Neural: {ticker} - {enhanced_pred.direction} ({enhanced_pred.confidence:.3f})")
            except Exception as e:
                log_error(f"Enhanced neural analysis failed for {ticker}: {e}")

        # 2. Traditional FinBERT Analysis
        if 'finbert' in self.services and self.services['finbert']['available']:
            prediction = self._analyze_with_finbert(ticker, combined_text)
            if prediction:
                predictions.append(prediction)
                successful_services.append('finbert')

        # 3. LLM Services
        for service_name in ['gemini', 'openai', 'claude']:
            if service_name in self.services and self.services[service_name]['available']:
                if service_name not in self.quota_exhausted:
                    prediction = self._analyze_with_llm(service_name, ticker, combined_text)
                    if prediction:
                        predictions.append(prediction)
                        successful_services.append(service_name)

        # 4. Emergency Services
        for service_name in ['alpha_vantage', 'polygon', 'tiingo']:
            if service_name in self.services and self.services[service_name]['available']:
                if service_name not in self.quota_exhausted:
                    prediction = self._analyze_with_emergency_service(service_name, ticker, combined_text)
                    if prediction:
                        predictions.append(prediction)
                        successful_services.append(service_name)

        # 5. Keyword Analysis (Fallback)
        if Config.ENABLE_KEYWORD_SENTIMENT:
            prediction = self._analyze_with_keywords(ticker, combined_text)
            if prediction:
                predictions.append(prediction)
                successful_services.append('keyword')

        # Combine predictions
        if predictions:
            combined_prediction = self._combine_predictions(predictions, successful_services)
            log_debug(f"Combined prediction for {ticker}: {combined_prediction.direction} ({combined_prediction.confidence:.3f}) from {len(predictions)} services")
            return combined_prediction
        else:
            log_warning(f"No successful predictions for {ticker}")
            return None

    def _combine_predictions(self, predictions: List[DirectionalPrediction], services_used: List[str]) -> DirectionalPrediction:
        """Combine multiple predictions using weighted voting with enhanced neural priority"""
        
        if len(predictions) == 1:
            single_pred = predictions[0]
            single_pred.individual_predictions = [single_pred]
            single_pred.source_weights = {single_pred.source: 1.0}
            single_pred.weighted_scores = {single_pred.source: single_pred.raw_score}
            return single_pred

        # Calculate weighted scores
        total_weight = 0
        weighted_buy_score = 0
        weighted_sell_score = 0
        
        source_weights_used = {}
        weighted_scores = {}
        
        for prediction, service in zip(predictions, services_used):
            weight = self.service_weights.get(service, 0.1)
            total_weight += weight
            source_weights_used[service] = weight
            
            if prediction.direction == 'BUY':
                weighted_buy_score += prediction.confidence * weight
                weighted_scores[service] = prediction.confidence * weight
            elif prediction.direction == 'SELL':
                weighted_sell_score += prediction.confidence * weight
                weighted_scores[service] = -prediction.confidence * weight
            else:  # NEUTRAL
                weighted_scores[service] = 0
        
        # Normalize weights
        if total_weight > 0:
            for service in source_weights_used:
                source_weights_used[service] /= total_weight
                
            weighted_buy_score /= total_weight
            weighted_sell_score /= total_weight

        # Determine final direction
        if weighted_buy_score > weighted_sell_score and weighted_buy_score > 0.3:
            direction = 'BUY'
            confidence = weighted_buy_score
            raw_score = weighted_buy_score - weighted_sell_score
        elif weighted_sell_score > weighted_buy_score and weighted_sell_score > 0.3:
            direction = 'SELL' 
            confidence = weighted_sell_score
            raw_score = weighted_sell_score - weighted_buy_score
        else:
            direction = 'NEUTRAL'
            confidence = max(weighted_buy_score, weighted_sell_score, 0.5)
            raw_score = 0

        # Create reasoning
        service_summaries = []
        for pred, service in zip(predictions, services_used):
            weight = source_weights_used.get(service, 0)
            service_summaries.append(f"{service}: {pred.direction} ({pred.confidence:.2f}, weight: {weight:.2f})")
        
        reasoning = f"Multi-source consensus from {len(predictions)} services: {' | '.join(service_summaries)}"

        return DirectionalPrediction(
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            source='multi_source',
            raw_score=raw_score,
            individual_predictions=predictions,
            source_weights=source_weights_used,
            weighted_scores=weighted_scores
        )

    def _analyze_with_emergency_service(self, service_name: str, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using emergency data services"""
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
        """Analyze using FinBERT model"""
        if not self.services.get('finbert', {}).get('available', False):
            return None
            
        try:
            model = self.services['finbert']['model']
            tokenizer = self.services['finbert']['tokenizer']
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # FIXED: Check device properly and move inputs to same device as model
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Convert to numpy for easier handling
            scores = predictions.cpu().numpy()[0]
            
            # FinBERT outputs: [negative, neutral, positive]
            negative_score = float(scores[0])
            neutral_score = float(scores[1])
            positive_score = float(scores[2])
            
            # Determine direction based on highest score
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

    def _analyze_with_llm(self, service_name: str, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using LLM services (Gemini, OpenAI, Claude)"""
        if service_name == 'gemini':
            return self._analyze_with_gemini(ticker, text)
        elif service_name == 'openai':
            return self._analyze_with_openai(ticker, text)
        elif service_name == 'claude':
            return self._analyze_with_claude(ticker, text)
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

            if response.choices and response.choices[0].message:
                return self._parse_llm_response(response.choices[0].message.content, "openai")
            return None

        except Exception as e:
            log_error(f"OpenAI analysis error: {e}")
            return None

    def _analyze_with_claude(self, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using Claude"""
        if not ANTHROPIC_AVAILABLE:
            return None
            
        try:
            service = self.services['claude']
            client = service['client']

            prompt = self._create_llm_prompt(ticker, text)

            time.sleep(Config.LLM_REQUEST_DELAY)  # Rate limiting
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            service['requests_today'] += 1

            if message.content and len(message.content) > 0:
                return self._parse_llm_response(message.content[0].text, "claude")
            return None

        except Exception as e:
            log_error(f"Claude analysis error: {e}")
            return None

    def _analyze_with_keywords(self, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using enhanced keyword sentiment"""
        try:
            text_lower = text.lower()
            
            positive_score = 0
            negative_score = 0
            
            # Count regular keywords
            for keyword in self.positive_keywords:
                if keyword in text_lower:
                    positive_score += 1
            
            for keyword in self.negative_keywords:
                if keyword in text_lower:
                    negative_score += 1
            
            # Add weight for high-impact keywords
            for keyword in self.high_impact_positive:
                if keyword in text_lower:
                    positive_score += 2
            
            for keyword in self.high_impact_negative:
                if keyword in text_lower:
                    negative_score += 2
            
            total_score = positive_score + negative_score
            
            if total_score == 0:
                return DirectionalPrediction(
                    direction='NEUTRAL',
                    confidence=0.3,
                    reasoning="No significant sentiment keywords found",
                    source="keyword",
                    raw_score=0
                )
            
            # Calculate direction and confidence
            if positive_score > negative_score:
                direction = 'BUY'
                confidence = min(0.8, positive_score / (total_score + 2))
                raw_score = (positive_score - negative_score) / total_score
            elif negative_score > positive_score:
                direction = 'SELL'
                confidence = min(0.8, negative_score / (total_score + 2))
                raw_score = (negative_score - positive_score) / total_score
            else:
                direction = 'NEUTRAL'
                confidence = 0.5
                raw_score = 0
            
            reasoning = f"Keyword analysis: {positive_score} positive, {negative_score} negative keywords"
            
            return DirectionalPrediction(
                direction=direction,
                confidence=confidence,
                reasoning=reasoning,
                source="keyword",
                raw_score=raw_score
            )

        except Exception as e:
            log_error(f"Keyword analysis error for {ticker}: {e}")
            return None

    def _create_llm_prompt(self, ticker: str, text: str) -> str:
        """Create standardized prompt for LLM services"""
        return f"""
        Analyze the following financial news about {ticker} and determine if it suggests a BUY, SELL, or NEUTRAL stance.
        
        Consider:
        - Financial performance indicators
        - Future outlook and guidance
        - Market position and competitive factors
        - Risk factors and challenges
        
        News text: {text[:1000]}
        
        Respond in JSON format:
        {{
            "direction": "BUY|SELL|NEUTRAL",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}
        """

    def _parse_llm_response(self, response_text: str, source: str) -> Optional[DirectionalPrediction]:
        """Parse LLM response into DirectionalPrediction"""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                direction = data.get('direction', '').upper()
                if direction not in ['BUY', 'SELL', 'NEUTRAL']:
                    direction = 'NEUTRAL'
                
                confidence = float(data.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))
                
                reasoning = data.get('reasoning', f'{source} analysis')
                
                # Calculate raw score
                if direction == 'BUY':
                    raw_score = confidence
                elif direction == 'SELL':
                    raw_score = -confidence
                else:
                    raw_score = 0
                
                return DirectionalPrediction(
                    direction=direction,
                    confidence=confidence,
                    reasoning=f"{source}: {reasoning}",
                    source=source,
                    raw_score=raw_score
                )
        
        except Exception as e:
            log_debug(f"Failed to parse {source} JSON response: {e}")
        
        # Fallback parsing
        response_lower = response_text.lower()
        
        if 'buy' in response_lower and 'sell' not in response_lower:
            direction = 'BUY'
            confidence = 0.6
        elif 'sell' in response_lower and 'buy' not in response_lower:
            direction = 'SELL'
            confidence = 0.6
        else:
            direction = 'NEUTRAL'
            confidence = 0.5
        
        raw_score = confidence if direction == 'BUY' else (-confidence if direction == 'SELL' else 0)
        
        return DirectionalPrediction(
            direction=direction,
            confidence=confidence,
            reasoning=f"{source}: {response_text[:100]}",
            source=source,
            raw_score=raw_score
        )

    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services including enhanced neural analyzer"""
        status = {}
        
        # Add enhanced neural analyzer status
        if self.enhanced_neural:
            neural_info = self.enhanced_neural.get_model_info()
            status['enhanced_neural'] = {
                'available': neural_info.get('available', False),
                'type': 'Enhanced Neural (RoBERTa+LSTM+CNN)',
                'accuracy': neural_info.get('expected_accuracy', 'N/A'),
                'device': neural_info.get('device', 'N/A'),
                'parameters': neural_info.get('total_parameters', 'N/A')
            }
        else:
            status['enhanced_neural'] = {
                'available': False,
                'type': 'Enhanced Neural (RoBERTa+LSTM+CNN)',
                'accuracy': 'N/A',
                'device': 'N/A',
                'parameters': 'N/A'
            }
        
        # Add traditional services
        for service_name, service_info in self.services.items():
            status[service_name] = {
                'available': service_info.get('available', False),
                'requests_today': service_info.get('requests_today', 0),
                'quota_limit': service_info.get('quota_limit', 'N/A'),
                'quota_exhausted': service_name in self.quota_exhausted
            }
        
        return status
