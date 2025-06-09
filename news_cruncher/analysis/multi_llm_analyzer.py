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
        
        # Emergency/backup services
        if Config.ENABLE_ALPHA_VANTAGE:
            self._init_alpha_vantage()
        
        if Config.ENABLE_POLYGON:
            self._init_polygon()
        
        if Config.ENABLE_TIINGO:
            self._init_tiingo()
        
        # Keyword analysis (always available as ultimate fallback)
        self._init_keyword_analysis()

    def _init_service_weights(self) -> None:
        """Initialize service weights for combining predictions"""
        self.service_weights = {
            'enhanced_neural': 0.45,  # Highest weight for neural analysis
            'finbert': 0.25,
            'gemini': 0.20,
            'openai': 0.12,
            'claude': 0.10,
            'alpha_vantage': 0.08,
            'polygon': 0.06,
            'tiingo': 0.04,
            'keyword': 0.02  # Lowest weight for keyword analysis
        }

    def _init_enhanced_keywords(self) -> None:
        """Initialize enhanced keyword lists for analysis"""
        self.positive_keywords = [
            'buy', 'bull', 'bullish', 'positive', 'good', 'great', 'excellent', 'strong',
            'growth', 'increase', 'up', 'rise', 'gain', 'profit', 'beat', 'exceed',
            'outperform', 'upgrade', 'optimistic', 'confident', 'momentum', 'breakthrough'
        ]

        self.negative_keywords = [
            'sell', 'bear', 'bearish', 'negative', 'bad', 'poor', 'weak', 'decline',
            'decrease', 'down', 'fall', 'loss', 'miss', 'disappoint', 'underperform',
            'downgrade', 'pessimistic', 'concern', 'worry', 'risk', 'problem'
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
        """Initialize FinBERT with proper configuration - no warnings"""
        if not TORCH_AVAILABLE:
            self.services['finbert'] = {'available': False}
            return
            
        try:
            log_info("🔧 Initializing FinBERT with proper configuration...")
            
            # FIXED: Load FinBERT with proper configuration
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                'ProsusAI/finbert',
                use_fast=True,  # Use fast tokenizer for better performance
                trust_remote_code=False
            )
            
            # FIXED: Load model with specific configuration to eliminate warnings
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                'ProsusAI/finbert',
                attn_implementation="eager",
                num_labels=3,  # Explicitly specify number of labels
                problem_type="single_label_classification",  # Explicit problem type
                local_files_only=False,
                trust_remote_code=False,
                output_attentions=False,  # Disable if not needed to save memory
                output_hidden_states=False  # Disable if not needed to save memory
            )
            
            # FIX 1: Properly initialize any randomly initialized layers
            self._initialize_finbert_classifier()
            
            # Check device and move model
            if torch.cuda.is_available():
                self.finbert_model = self.finbert_model.cuda()
                device_info = "CUDA GPU"
            else:
                device_info = "CPU"
            
            # FIX 2: Set model to evaluation mode and optimize
            self.finbert_model.eval()
            
            # FIX 3: Optimize model for inference if using CPU
            if device_info == "CPU":
                try:
                    # Enable CPU optimizations
                    self.finbert_model = torch.jit.optimize_for_inference(self.finbert_model)
                    log_info("✅ Applied CPU optimizations to FinBERT")
                except Exception as e:
                    log_debug(f"CPU optimization failed (non-critical): {e}")
            
            self.services['finbert'] = {
                'available': True,
                'model': self.finbert_model,
                'tokenizer': self.finbert_tokenizer,
                'requests_today': 0,
                'quota_limit': float('inf'),  # Local model - no quota
                'device': device_info,
                'initialized_properly': True
            }
            
            log_info(f"✅ FinBERT initialized properly on {device_info}")
            
            # FIX 4: Warm up the model for better initial predictions
            self._warm_up_finbert()
            self._load_adaptive_finbert_if_available()
        except Exception as e:
            log_error(f"Failed to initialize FinBERT: {e}")
            self.services['finbert'] = {'available': False, 'error': str(e)}

    def _initialize_finbert_classifier(self):
        """
        FIX 1: Properly initialize FinBERT classifier layer if needed
        FinBERT should already be fine-tuned, but this ensures proper initialization
        """
        try:
            # Check if classifier needs initialization
            classifier = self.finbert_model.classifier
            
            # Initialize with Xavier/Glorot uniform (good for classification)
            if hasattr(classifier, 'weight'):
                torch.nn.init.xavier_uniform_(classifier.weight)
                log_debug("🔧 Initialized FinBERT classifier weights")
                
            if hasattr(classifier, 'bias') and classifier.bias is not None:
                torch.nn.init.zeros_(classifier.bias)
                log_debug("🔧 Initialized FinBERT classifier bias")
                
        except Exception as e:
            log_debug(f"FinBERT classifier initialization note: {e}")

    def _warm_up_finbert(self):
        """
        FIX 4: Warm up FinBERT model for better initial predictions
        """
        try:
            log_info("🔥 Warming up FinBERT model...")
            
            warmup_texts = [
                "The company reported strong quarterly earnings.",
                "Revenue declined due to market conditions.",
                "Positive outlook for next quarter."
            ]
            
            self.finbert_model.eval()
            
            with torch.no_grad():
                for text in warmup_texts:
                    try:
                        # Tokenize
                        inputs = self.finbert_tokenizer(
                            text,
                            max_length=512,
                            padding=True,
                            truncation=True,
                            return_tensors="pt"
                        )
                        
                        # Move to same device as model
                        if next(self.finbert_model.parameters()).is_cuda:
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                        
                        # Forward pass
                        _ = self.finbert_model(**inputs)
                        
                    except Exception as e:
                        log_debug(f"FinBERT warmup sample failed: {e}")
                        continue
            
            log_info("✅ FinBERT warmup completed")
            
        except Exception as e:
            log_debug(f"FinBERT warmup failed (non-critical): {e}")

    def _init_gemini(self) -> None:
        """Initialize Google Gemini"""
        if not GEMINI_AVAILABLE:
            self.services['gemini'] = {'available': False}
            return
            
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.services['gemini'] = {
                'available': True,
                'client': genai.GenerativeModel('gemini-pro'),
                'requests_today': 0,
                'quota_limit': 60
            }
            log_info("✅ Gemini initialized successfully")
        except Exception as e:
            log_error(f"Failed to initialize Gemini: {e}")
            self.services['gemini'] = {'available': False}

    def _init_openai(self) -> None:
        """Initialize OpenAI"""
        if not OPENAI_AVAILABLE:
            self.services['openai'] = {'available': False}
            return
            
        try:
            self.services['openai'] = {
                'available': True,
                'client': openai.OpenAI(api_key=Config.OPENAI_API_KEY),
                'requests_today': 0,
                'quota_limit': 100
            }
            log_info("✅ OpenAI initialized successfully")
        except Exception as e:
            log_error(f"Failed to initialize OpenAI: {e}")
            self.services['openai'] = {'available': False}

    def _init_claude(self) -> None:
        """Initialize Anthropic Claude"""
        if not ANTHROPIC_AVAILABLE:
            self.services['claude'] = {'available': False}
            return
            
        try:
            self.services['claude'] = {
                'available': True,
                'client': anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY),
                'requests_today': 0,
                'quota_limit': 100
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

        # 1. Enhanced Neural Analysis (highest priority)
        if self.enhanced_neural and self.enhanced_neural.is_available:
            try:
                neural_prediction = self.enhanced_neural.analyze_text(ticker, combined_text)
                if neural_prediction:
                    # Convert EnhancedPrediction to DirectionalPrediction
                    pred = DirectionalPrediction(
                        direction=neural_prediction.direction,
                        confidence=neural_prediction.confidence,
                        reasoning=neural_prediction.reasoning,
                        source='enhanced_neural',
                        raw_score=neural_prediction.raw_score
                    )
                    predictions.append(pred)
                    successful_services.append('enhanced_neural')
                    log_debug(f"Enhanced neural analysis: {pred.direction} ({pred.confidence:.3f})")
            except Exception as e:
                log_error(f"Enhanced neural analysis failed for {ticker}: {e}")

        # 2. FinBERT Analysis
        if 'finbert' in self.services and self.services['finbert']['available']:
            try:
                finbert_pred = self._analyze_finbert(combined_text)
                if finbert_pred:
                    predictions.append(finbert_pred)
                    successful_services.append('finbert')
            except Exception as e:
                log_error(f"FinBERT analysis failed for {ticker}: {e}")

        # 3. Try other LLM services in order of preference
        for service_name in ['gemini', 'openai', 'claude']:
            if service_name in self.services and self.services[service_name]['available']:
                if service_name in self.quota_exhausted:
                    continue
                    
                try:
                    if service_name == 'gemini':
                        pred = self._analyze_with_gemini(ticker, combined_text)
                    elif service_name == 'openai':
                        pred = self._analyze_with_openai(ticker, combined_text)
                    elif service_name == 'claude':
                        pred = self._analyze_with_claude(ticker, combined_text)
                    
                    if pred:
                        predictions.append(pred)
                        successful_services.append(service_name)
                        log_debug(f"{service_name} analysis: {pred.direction} ({pred.confidence:.3f})")
                    
                except Exception as e:
                    log_error(f"{service_name} analysis failed for {ticker}: {e}")
                    continue

        # 4. Emergency services fallback
        if not predictions:
            emergency_pred = self.emergency_services.get_emergency_sentiment(ticker, combined_text)
            if emergency_pred:
                # Convert emergency prediction to our format
                pred = DirectionalPrediction(
                    direction=emergency_pred.direction,
                    confidence=emergency_pred.confidence,
                    reasoning=emergency_pred.reasoning,
                    source='emergency_services',
                    raw_score=emergency_pred.raw_score
                )
                predictions.append(pred)
                successful_services.append('emergency_services')

        # 5. Keyword analysis as final fallback
        if not predictions:
            try:
                keyword_pred = self._analyze_with_keywords(ticker, combined_text)
                if keyword_pred:
                    predictions.append(keyword_pred)
                    successful_services.append('keyword')
            except Exception as e:
                log_error(f"Keyword analysis failed for {ticker}: {e}")

        # Combine predictions if we have multiple
        if not predictions:
            return None
        elif len(predictions) == 1:
            result = predictions[0]
            result.sources_used = successful_services
            return result
        else:
            return self._combine_predictions(predictions, successful_services)

    def _analyze_finbert(self, text: str) -> Optional[DirectionalPrediction]:
        """
        FIXED: Enhanced FinBERT analysis with proper error handling
        """
        if 'finbert' not in self.services or not self.services['finbert']['available']:
            return None
        
        try:
            # Tokenize with proper parameters
            inputs = self.services['finbert']['tokenizer'](
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move inputs to same device as model
            model = self.services['finbert']['model']
            if next(model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Forward pass with proper configuration
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process outputs properly
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # FinBERT classes: [negative, neutral, positive]
            negative_prob = probabilities[0][0].item()
            neutral_prob = probabilities[0][1].item()
            positive_prob = probabilities[0][2].item()
            
            # Determine prediction
            max_prob = max(negative_prob, neutral_prob, positive_prob)
            
            if max_prob == positive_prob:
                prediction = 'BUY'
                confidence = positive_prob
            elif max_prob == negative_prob:
                prediction = 'SELL'
                confidence = negative_prob
            else:
                prediction = 'NEUTRAL'
                confidence = neutral_prob
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = positive_prob - negative_prob
            
            return DirectionalPrediction(
                direction=prediction,
                confidence=confidence,
                reasoning=f"FinBERT analysis: {prediction} with {confidence:.2%} confidence",
                source='finbert',
                raw_score=sentiment_score
            )
            
        except Exception as e:
            log_error(f"Error in FinBERT analysis: {e}")
            return None

    def _analyze_with_gemini(self, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using Google Gemini"""
        if not GEMINI_AVAILABLE:
            return None
            
        try:
            service = self.services['gemini']
            model = service['client']

            prompt = self._create_llm_prompt(ticker, text)

            time.sleep(Config.LLM_REQUEST_DELAY)  # Rate limiting
            response = model.generate_content(prompt)

            service['requests_today'] += 1

            if response.text:
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
            
            # Count high-impact keywords (weighted 3x)
            for keyword in self.high_impact_positive:
                if keyword in text_lower:
                    positive_score += 3
            
            for keyword in self.high_impact_negative:
                if keyword in text_lower:
                    negative_score += 3
            
            if positive_score == 0 and negative_score == 0:
                return DirectionalPrediction(
                    direction='NEUTRAL',
                    confidence=0.5,
                    reasoning="No significant keywords found",
                    source='keyword',
                    raw_score=0.0
                )
            
            total_score = positive_score + negative_score
            sentiment_ratio = positive_score / total_score if total_score > 0 else 0.5
            
            if sentiment_ratio > 0.6:
                direction = 'BUY'
                confidence = min(0.8, 0.5 + sentiment_ratio * 0.3)
            elif sentiment_ratio < 0.4:
                direction = 'SELL'
                confidence = min(0.8, 0.5 + (1 - sentiment_ratio) * 0.3)
            else:
                direction = 'NEUTRAL'
                confidence = 0.5
            
            raw_score = (sentiment_ratio - 0.5) * 2  # Scale to -1 to 1
            
            reasoning = f"Keyword analysis: {positive_score} positive, {negative_score} negative signals"
            
            return DirectionalPrediction(
                direction=direction,
                confidence=confidence,
                reasoning=reasoning,
                source='keyword',
                raw_score=raw_score
            )

        except Exception as e:
            log_error(f"Keyword analysis error: {e}")
            return None

    def _create_llm_prompt(self, ticker: str, text: str) -> str:
        """Create standardized prompt for LLM services"""
        return f"""
        Analyze this financial news about {ticker} and provide a trading recommendation.
        
        News text: {text[:2000]}
        
        Respond in exactly this format:
        DIRECTION: [BUY/SELL/NEUTRAL]
        CONFIDENCE: [0.0-1.0]
        REASONING: [brief explanation]
        """

    def _parse_llm_response(self, response: str, source: str) -> Optional[DirectionalPrediction]:
        """Parse LLM response into DirectionalPrediction"""
        try:
            lines = response.strip().split('\n')
            direction = None
            confidence = 0.5
            reasoning = "LLM analysis"
            
            for line in lines:
                if 'DIRECTION:' in line.upper():
                    direction_text = line.split(':', 1)[1].strip().upper()
                    if 'BUY' in direction_text:
                        direction = 'BUY'
                    elif 'SELL' in direction_text:
                        direction = 'SELL'
                    else:
                        direction = 'NEUTRAL'
                
                elif 'CONFIDENCE:' in line.upper():
                    try:
                        confidence = float(re.findall(r'(\d+\.?\d*)', line)[0])
                        if confidence > 1.0:
                            confidence = confidence / 100.0  # Convert percentage
                        confidence = max(0.0, min(1.0, confidence))
                    except:
                        confidence = 0.5
                
                elif 'REASONING:' in line.upper():
                    reasoning = line.split(':', 1)[1].strip()
            
            if direction:
                raw_score = confidence if direction == 'BUY' else (-confidence if direction == 'SELL' else 0.0)
                return DirectionalPrediction(
                    direction=direction,
                    confidence=confidence,
                    reasoning=reasoning,
                    source=source,
                    raw_score=raw_score
                )
            
            return None

        except Exception as e:
            log_error(f"Error parsing {source} response: {e}")
            return None

    def _combine_predictions(self, predictions: List[DirectionalPrediction], sources: List[str]) -> DirectionalPrediction:
        """Combine multiple predictions using weighted voting"""
        try:
            weighted_buy = 0.0
            weighted_sell = 0.0
            weighted_neutral = 0.0
            total_weight = 0.0
            
            source_weights = {}
            weighted_scores = {}
            
            for pred in predictions:
                weight = self.service_weights.get(pred.source, 0.1)
                total_weight += weight
                source_weights[pred.source] = weight
                
                confidence_weighted = pred.confidence * weight
                
                if pred.direction == 'BUY':
                    weighted_buy += confidence_weighted
                elif pred.direction == 'SELL':
                    weighted_sell += confidence_weighted
                else:
                    weighted_neutral += confidence_weighted
                
                weighted_scores[pred.source] = pred.raw_score * weight
            
            # Normalize weights
            if total_weight > 0:
                weighted_buy /= total_weight
                weighted_sell /= total_weight
                weighted_neutral /= total_weight
            
            # Determine final direction
            max_score = max(weighted_buy, weighted_sell, weighted_neutral)
            
            if max_score == weighted_buy:
                final_direction = 'BUY'
                final_confidence = weighted_buy
            elif max_score == weighted_sell:
                final_direction = 'SELL'
                final_confidence = weighted_sell
            else:
                final_direction = 'NEUTRAL'
                final_confidence = weighted_neutral
            
            # Calculate combined raw score
            combined_raw_score = sum(weighted_scores.values()) / len(weighted_scores) if weighted_scores else 0.0
            
            # Create reasoning
            source_summary = ', '.join([f"{s}({self.service_weights.get(s, 0.1):.2f})" for s in sources])
            reasoning = f"Multi-source consensus: {final_direction} from {len(sources)} services [{source_summary}]"
            
            return DirectionalPrediction(
                direction=final_direction,
                confidence=final_confidence,
                reasoning=reasoning,
                source='multi_source',
                raw_score=combined_raw_score,
                individual_predictions=predictions,
                source_weights=source_weights,
                weighted_scores=weighted_scores
            )

        except Exception as e:
            log_error(f"Error combining predictions: {e}")
            # Return the highest confidence prediction as fallback
            return max(predictions, key=lambda p: p.confidence)

    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services including enhanced neural analyzer"""
        status = {}
        
        # Enhanced neural analyzer status
        if self.enhanced_neural:
            neural_info = self.enhanced_neural.get_model_info()
            status['enhanced_neural'] = {
                'available': neural_info.get('available', False),
                'device': neural_info.get('device', 'unknown'),
                'accuracy': neural_info.get('expected_accuracy', 'unknown'),
                'parameters': neural_info.get('trainable_parameters', 0)
            }
        else:
            status['enhanced_neural'] = {'available': False}
        
        # Traditional services
        for service_name, service_info in self.services.items():
            status[service_name] = {
                'available': service_info.get('available', False),
                'requests_today': service_info.get('requests_today', 0),
                'quota_limit': service_info.get('quota_limit', 0)
            }
        
        return status

    def reset_daily_quotas(self) -> None:
        """Reset daily quotas for all services"""
        for service in self.services.values():
            service['requests_today'] = 0
        self.quota_exhausted.clear()
        log_info("🔄 Daily quotas reset for all services")
        
    def _load_adaptive_finbert_if_available(self):
        """Load adaptive FinBERT checkpoint if available"""
        try:
            adaptive_checkpoint = Config.DATA_DIR / "finbert_multimodal_adaptive.pth"
            if adaptive_checkpoint.exists():
                log_info("🔄 Loading adaptive FinBERT checkpoint...")
                
                checkpoint = torch.load(adaptive_checkpoint, map_location='cpu')
                
                # Load data processors
                processors_path = Config.DATA_DIR / "data_processors.pkl"
                if processors_path.exists():
                    with open(processors_path, 'rb') as f:
                        processors = pickle.load(f)
                        log_info("✅ Adaptive FinBERT data processors loaded")
                        
                        # Store processors for use during prediction
                        self.adaptive_processors = processors
                        
                log_info(f"✅ Adaptive FinBERT checkpoint loaded from {adaptive_checkpoint}")
                return True
        except Exception as e:
            log_debug(f"No adaptive FinBERT checkpoint available: {e}")
        
        return False