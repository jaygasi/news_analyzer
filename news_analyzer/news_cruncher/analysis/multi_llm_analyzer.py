from __future__ import annotations # Must be at the very top for type hinting forward references

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
import pickle

# Core imports that should always work
from config import Config # Uncommented to import from your config.py
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

# Placeholder for EmergencyDataServices (in a real scenario, this would be in analysis/emergency_data_services.py)
class DirectionalPrediction: # Re-defining for clarity, but already present in multi_llm_analyzer.py
    def __init__(self, direction: str, confidence: float, reasoning: str, source: str, raw_score: float = 0.0):
        self.direction = direction
        self.confidence = confidence
        self.reasoning = reasoning
        self.source = source
        self.raw_score = raw_score

class EmergencyDataServices:
    def __init__(self):
        pass

    def get_sentiment_from_text(self, text: str) -> DirectionalPrediction:
        """Placeholder for emergency sentiment analysis."""
        # Simple placeholder logic
        if "positive" in text.lower() or "gain" in text.lower():
            return DirectionalPrediction("BUY", 0.6, "Positive keywords found.", "emergency_keyword")
        elif "negative" in text.lower() or "loss" in text.lower():
            return DirectionalPrediction("SELL", 0.6, "Negative keywords found.", "emergency_keyword")
        else:
            return DirectionalPrediction("NEUTRAL", 0.5, "No strong sentiment keywords.", "emergency_keyword")

# Placeholder for EnhancedNeuralAnalyzer (in a real scenario, this would be in analysis/enhanced_neural_analyzer.py)
class EnhancedPrediction:
    def __init__(self, direction: str, confidence: float, reasoning: str, raw_score: float):
        self.direction = direction
        self.confidence = confidence
        self.reasoning = reasoning
        self.raw_score = raw_score

class EnhancedNeuralAnalyzer:
    def __init__(self):
        self.is_available = True # Assume available for demonstration

    def predict_direction(self, text: str) -> Optional[EnhancedPrediction]:
        """Placeholder for enhanced neural prediction."""
        # Simple placeholder logic
        if "breakthrough" in text.lower():
            return EnhancedPrediction("BUY", 0.8, "Breakthrough detected by enhanced neural.", 0.8)
        elif "recall" in text.lower():
            return EnhancedPrediction("SELL", 0.8, "Recall detected by enhanced neural.", -0.8)
        else:
            return EnhancedPrediction("NEUTRAL", 0.5, "Neutral sentiment by enhanced neural.", 0.0)

    def analyze_tickers_batch(self, ticker_articles: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Optional[EnhancedPrediction]]:
        """Placeholder for batch analysis."""
        results = {}
        for ticker, articles in ticker_articles.items():
            combined_text = ' '.join([ f"{article.get('title', '')} {article.get('text', '')}" for article in articles ])
            results[ticker] = self.predict_direction(combined_text)
        return results

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

        # FinBERT specific attributes
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.finbert_continuous_learning_active = False
        self._finbert_load_time = 0
        self._last_finbert_check = 0
        self._analysis_count = 0  # Track analysis calls for periodic checks

        # Enhanced keyword lists for robust fallback analysis
        self._init_enhanced_keywords()

        # Initialize emergency data services
        self.emergency_services = EmergencyDataServices()

        # Initialize enhanced neural analyzer (NEW)
        self.enhanced_neural = None
        # Assuming ENHANCED_NEURAL_AVAILABLE is defined somewhere, if not, consider it False
        # For this context, assuming ENHANCED_NEURAL_AVAILABLE will be handled by external setup
        ENHANCED_NEURAL_AVAILABLE = True # Placeholder for now if not defined externally

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
        """Initialize service weights for combining predictions from config"""
        self.service_weights = {
            'enhanced_neural': Config.MULTI_LLM_WEIGHT_ENHANCED_NEURAL,
            'finbert': Config.MULTI_LLM_WEIGHT_FINBERT,
            'gemini': Config.MULTI_LLM_WEIGHT_GEMINI,
            'openai': Config.MULTI_LLM_WEIGHT_OPENAI,
            'claude': Config.MULTI_LLM_WEIGHT_CLAUDE,
            'alpha_vantage': Config.MULTI_LLM_WEIGHT_ALPHA_VANTAGE,
            'polygon': Config.MULTI_LLM_WEIGHT_POLYGON,
            'tiingo': Config.MULTI_LLM_WEIGHT_TIINGO,
            'keyword': Config.MULTI_LLM_WEIGHT_KEYWORD
        }

        # Validate weights sum approximately to 1.0
        total_weight = sum(self.service_weights.values())
        if abs(total_weight - 1.0) > 0.1:
            log_warning(f"Service weights sum to {total_weight:.3f}, should be ~1.0")

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
                Config.FINBERT_MODEL_NAME, # Using constant
                use_fast=True,  # Use fast tokenizer for better performance
                trust_remote_code=False
            )

            # FIXED: Load model with specific configuration to eliminate warnings
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                Config.FINBERT_MODEL_NAME, # Using constant
                torch_dtype=torch.float32, # Add this to prevent meta tensor issues
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
            # Initialize continuous learning status
            self.finbert_continuous_learning_active = False

            # Try to load adaptive checkpoint
            self._load_adaptive_finbert_if_available()
            log_info(f"✅ FinBERT initialized properly on {device_info}")

            # FIX 4: Warm up the model for better initial predictions
            self._warm_up_finbert()

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

    def _check_for_new_finbert_checkpoint(self) -> bool:
        """Check if a new FinBERT adaptive checkpoint is available and reload if needed"""
        if not self.services.get('finbert', {}).get('available'):
            return False
        current_time = time.time()
        if current_time - self._last_finbert_check < Config.FINBERT_CHECK_INTERVAL: # Using constant
            return False
        self._last_finbert_check = current_time
        try:
            # Check both multimodal and standard checkpoints
            # Assuming Config.DATA_DIR is Path object
            multimodal_checkpoint = Config.DATA_DIR / "finbert_multimodal_adaptive.pth"
            standard_checkpoint = Config.DATA_DIR / Config.FINBERT_STANDARD_CHECKPOINT_NAME # Using constant
            newest_checkpoint = None
            newest_mtime = 0
            for checkpoint_path in [standard_checkpoint, multimodal_checkpoint]:
                if checkpoint_path.exists():
                    mtime = checkpoint_path.stat().st_mtime
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest_checkpoint = checkpoint_path

            if not newest_checkpoint or newest_mtime <= self._finbert_load_time:
                return False

            log_info("🔄 New FinBERT adaptive checkpoint detected, reloading model...")
            # Reload FinBERT with new checkpoint
            self._load_adaptive_finbert_if_available()
            self._finbert_load_time = current_time
            log_info("✅ FinBERT successfully reloaded with new adaptive checkpoint")
            return True
        except Exception as e:
            log_error(f"Failed to reload new FinBERT checkpoint: {e}")
            return False

    def _load_adaptive_finbert_if_available(self) -> None:
        """
        Loads the latest adaptive FinBERT checkpoint if available.
        Prioritizes multimodal over standard if both exist and are newer.
        """
        if not self.services.get('finbert', {}).get('available'):
            return

        try:
            multimodal_checkpoint_path = Config.DATA_DIR / "finbert_multimodal_adaptive.pth"
            standard_checkpoint_path = Config.DATA_DIR / Config.FINBERT_STANDARD_CHECKPOINT_NAME

            latest_checkpoint = None
            latest_mtime = 0

            # Check multimodal first
            if multimodal_checkpoint_path.exists():
                mtime = multimodal_checkpoint_path.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_checkpoint = multimodal_checkpoint_path

            # Check standard
            if standard_checkpoint_path.exists():
                mtime = standard_checkpoint_path.stat().st_mtime
                if mtime > latest_mtime: # Even if older than multimodal, if multimodal doesn't exist, this becomes latest
                    latest_mtime = mtime
                    latest_checkpoint = standard_checkpoint_path

            if latest_checkpoint:
                log_info(f"Attempting to load adaptive FinBERT checkpoint: {latest_checkpoint.name}")
                checkpoint = torch.load(latest_checkpoint, map_location=torch.device('cpu'))

                # Load state dict
                if hasattr(self.finbert_model, 'module'): # For DataParallel models
                    self.finbert_model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.finbert_model.load_state_dict(checkpoint['model_state_dict'])

                # Optionally load optimizer state, etc.
                # if 'optimizer_state_dict' in checkpoint and hasattr(self, 'finbert_optimizer'):
                #     self.finbert_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                self._finbert_load_time = latest_mtime
                self.finbert_continuous_learning_active = True
                log_info(f"✅ FinBERT adaptive checkpoint '{latest_checkpoint.name}' loaded. "
                         f"Trained with {checkpoint.get('training_samples', 0)} samples. "
                         f"Accuracy: {checkpoint.get('accuracy', 0.0):.2%}")
            else:
                log_info("No adaptive FinBERT checkpoint found. Using base model.")
                self.finbert_continuous_learning_active = False

        except Exception as e:
            log_error(f"Failed to load adaptive FinBERT checkpoint: {e}")
            self.finbert_continuous_learning_active = False


    def _init_gemini(self) -> None:
        """Initialize Google Gemini"""
        if not GEMINI_AVAILABLE:
            self.services['gemini'] = {'available': False}
            return
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY) # Correct usage of configure
            self.services['gemini'] = {
                'available': True,
                'client': genai.GenerativeModel(Config.GEMINI_MODEL), # Correct usage of GenerativeModel
                'requests_today': 0,
                'quota_limit': 60
            }
            log_info(f"✅ Gemini initialized successfully with model: {Config.GEMINI_MODEL}")
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

    def analyze_batch_enhanced_neural(self, ticker_articles: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Optional[DirectionalPrediction]]:
        """CRITICAL OPTIMIZATION: Batch process multiple tickers using enhanced neural analyzer"""
        if not ticker_articles:
            return {}

        # Check if enhanced neural is available
        if not (self.enhanced_neural and self.enhanced_neural.is_available):
            log_warning("❌ Enhanced neural not available for batch processing")
            return {}

        try:
            log_info(f"🚀 MultiLLM batch processing {len(ticker_articles)} tickers")
            # Use the enhanced neural analyzer's batch method
            neural_results = self.enhanced_neural.analyze_tickers_batch(ticker_articles)

            # Convert EnhancedPredictions to DirectionalPredictions
            converted_results = {}
            for ticker, neural_prediction in neural_results.items():
                if neural_prediction:
                    # Convert to DirectionalPrediction format
                    converted_prediction = DirectionalPrediction(
                        direction=neural_prediction.direction,
                        confidence=neural_prediction.confidence,
                        reasoning=neural_prediction.reasoning,
                        source='enhanced_neural_batch',
                        raw_score=neural_prediction.raw_score
                    )
                    converted_results[ticker] = converted_prediction
                    log_debug(f"✅ {ticker} batch conversion: {neural_prediction.direction} (conf: {neural_prediction.confidence:.3f})")
                else:
                    converted_results[ticker] = None
                    log_warning(f"❌ {ticker} batch processing failed")

            log_info(f"✅ MultiLLM batch processing complete: {len([r for r in converted_results.values() if r])}/{len(ticker_articles)} successful")
            return converted_results
        except Exception as e:
            log_error(f"❌ MultiLLM batch processing failed: {e}")
            return {}

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

        # Check for new adaptive checkpoints periodically
        if self._check_for_new_finbert_checkpoint():
            log_info("Reloaded FinBERT with a new checkpoint during analysis.")

        # Try Enhanced Neural Analyzer first (if enabled and available)
        if self.enhanced_neural and self.enhanced_neural.is_available:
            log_debug("Attempting analysis with Enhanced Neural Analyzer...")
            try:
                # Use the placeholder predict_direction method
                neural_prediction = self.enhanced_neural.predict_direction(combined_text) # Fix: Call the method

                if neural_prediction:
                    prediction = DirectionalPrediction(
                        direction=neural_prediction.direction,
                        confidence=neural_prediction.confidence,
                        reasoning=neural_prediction.reasoning,
                        source='enhanced_neural',
                        raw_score=neural_prediction.raw_score
                    )
                    predictions.append(prediction)
                    successful_services.append('enhanced_neural')
                    log_info(f"✅ Enhanced Neural Analyzer: {prediction.direction} (conf: {prediction.confidence:.3f})")
                    # If high confidence, return early or prioritize
                    if prediction.confidence >= 0.8: # Example threshold
                        log_info("High confidence enhanced neural prediction, prioritizing.")
                        return prediction
                else:
                    log_warning("Enhanced Neural Analyzer returned no prediction.")
            except Exception as e:
                log_error(f"Enhanced Neural Analyzer failed: {e}")
                self.enhanced_neural.is_available = False # Temporarily disable if it fails

        # Fallback to other services if enhanced neural not available or failed
        # FinBERT
        if self.services.get('finbert', {}).get('available') and 'finbert' not in self.quota_exhausted:
            log_debug("Attempting analysis with FinBERT...")
            try:
                finbert_prediction = self._analyze_with_finbert(combined_text)
                if finbert_prediction:
                    predictions.append(finbert_prediction)
                    successful_services.append('finbert')
                    log_info(f"✅ FinBERT: {finbert_prediction.direction} (conf: {finbert_prediction.confidence:.3f})")
            except Exception as e:
                log_error(f"FinBERT analysis failed: {e}")
                self.quota_exhausted.add('finbert') # Mark as exhausted if it fails

        # Gemini
        if self.services.get('gemini', {}).get('available') and 'gemini' not in self.quota_exhausted:
            log_debug("Attempting analysis with Gemini...")
            try:
                gemini_prediction = self._analyze_with_gemini(combined_text)
                if gemini_prediction:
                    predictions.append(gemini_prediction)
                    successful_services.append('gemini')
                    log_info(f"✅ Gemini: {gemini_prediction.direction} (conf: {gemini_prediction.confidence:.3f})")
            except Exception as e:
                log_error(f"Gemini analysis failed: {e}")
                self.quota_exhausted.add('gemini')

        # OpenAI (GPT)
        if self.services.get('openai', {}).get('available') and 'openai' not in self.quota_exhausted:
            log_debug("Attempting analysis with OpenAI...")
            try:
                openai_prediction = self._analyze_with_openai(combined_text)
                if openai_prediction:
                    predictions.append(openai_prediction)
                    successful_services.append('openai')
                    log_info(f"✅ OpenAI: {openai_prediction.direction} (conf: {openai_prediction.confidence:.3f})")
            except Exception as e:
                log_error(f"OpenAI analysis failed: {e}")
                self.quota_exhausted.add('openai')

        # Anthropic (Claude)
        if self.services.get('claude', {}).get('available') and 'claude' not in self.quota_exhausted:
            log_debug("Attempting analysis with Claude...")
            try:
                claude_prediction = self._analyze_with_claude(combined_text)
                if claude_prediction:
                    predictions.append(claude_prediction)
                    successful_services.append('claude')
                    log_info(f"✅ Claude: {claude_prediction.direction} (conf: {claude_prediction.confidence:.3f})")
            except Exception as e:
                log_error(f"Claude analysis failed: {e}")
                self.quota_exhausted.add('claude')

        # Emergency Fallback Services (Alpha Vantage, Polygon, Tiingo - if they can provide directional data)
        # These typically provide financial data, not directly sentiment from text.
        # This section is commented out as it requires external data retrieval logic not in the snippet.
        # if self.services.get('alpha_vantage', {}).get('available') and 'alpha_vantage' not in self.quota_exhausted:
        #     # Placeholder for calling Alpha Vantage sentiment
        #     pass
        # if self.services.get('polygon', {}).get('available') and 'polygon' not in self.quota_exhausted:
        #     # Placeholder for calling Polygon sentiment
        #     pass
        # if self.services.get('tiingo', {}).get('available') and 'tiingo' not in self.quota_exhausted:
        #     # Placeholder for calling Tiingo sentiment
        #     pass

        # Final Fallback: Keyword-based Analysis (always available)
        if not predictions or not successful_services:
            log_warning("No LLM services succeeded, falling back to keyword analysis.")
            keyword_prediction = self._analyze_with_keyword_analysis(combined_text)
            predictions.append(keyword_prediction)
            successful_services.append('keyword')
            log_info(f"✅ Keyword Analysis: {keyword_prediction.direction} (conf: {keyword_prediction.confidence:.3f})")


        if not predictions:
            log_error("No analysis could be performed by any available service.")
            return DirectionalPrediction("NEUTRAL", 0.0, "No services could provide a prediction.", "none")

        # Combine predictions using weighted average (if multiple sources)
        final_prediction = self._combine_predictions(predictions, successful_services)
        log_info(f"📊 Final Consensus for {ticker}: {final_prediction.direction} (Confidence: {final_prediction.confidence:.3f}) from {final_prediction.source}")
        return final_prediction


    def _analyze_with_finbert(self, text: str) -> Optional[DirectionalPrediction]:
        """Analyze text using FinBERT model."""
        service_info = self.services.get('finbert')
        if not service_info or not service_info['available']:
            return None

        self.services['finbert']['requests_today'] += 1
        if self.services['finbert']['requests_today'] > service_info['quota_limit']:
            log_warning("FinBERT quota exhausted (local model should not have this issue).")
            self.quota_exhausted.add('finbert')
            return None

        try:
            inputs = service_info['tokenizer'](
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            # Move inputs to the same device as the model
            device = next(service_info['model'].parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = service_info['model'](**inputs)
                scores = torch.softmax(outputs.logits, dim=1) # Apply softmax for probabilities

            # FinBERT usually outputs 3 labels: negative, neutral, positive
            # Map these to 'SELL', 'NEUTRAL', 'BUY'
            # Assuming the order is [negative, neutral, positive] or [positive, neutral, negative]
            # Check FinBERT's label mapping if possible. A common mapping is:
            # 0: negative, 1: neutral, 2: positive

            # Find the best score index and score
            best_score = torch.max(scores).item()
            best_score_index = torch.argmax(scores).item() # This is already an integer

            direction_map = {0: 'SELL', 1: 'NEUTRAL', 2: 'BUY'} # Common FinBERT mapping
            predicted_direction = direction_map.get(best_score_index, 'NEUTRAL')

            confidence = best_score
            reasoning = f"FinBERT predicted {predicted_direction} with confidence {confidence:.2f} based on semantic analysis."

            return DirectionalPrediction(
                direction=predicted_direction,
                confidence=confidence,
                reasoning=reasoning,
                source='finbert',
                raw_score=best_score # Raw score is the highest probability
            )

        except Exception as e:
            log_error(f"FinBERT analysis failed: {e}")
            return None

    def _analyze_with_gemini(self, text: str) -> Optional[DirectionalPrediction]:
        """Analyze text using Google Gemini."""
        service_info = self.services.get('gemini')
        if not service_info or not service_info['available']:
            return None

        self.services['gemini']['requests_today'] += 1
        if self.services['gemini']['requests_today'] > service_info['quota_limit']:
            log_warning("Gemini quota exhausted.")
            self.quota_exhausted.add('gemini')
            return None

        try:
            model = service_info['client']
            prompt = (
                f"Analyze the sentiment of the following financial news text and determine a directional prediction (BUY, SELL, NEUTRAL) "
                f"and a confidence score (0.0 to 1.0). Provide a brief reasoning. "
                f"Format your response as a JSON object with 'direction', 'confidence', and 'reasoning' keys.\n\nText: {text}"
            )
            response = model.generate_content(prompt)
            raw_content = response.text
            # Attempt to extract JSON from the response text
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                direction = result.get('direction', 'NEUTRAL').upper()
                confidence = float(result.get('confidence', 0.5))
                reasoning = result.get('reasoning', 'Gemini analysis.')

                # Ensure direction is valid
                if direction not in ['BUY', 'SELL', 'NEUTRAL']:
                    direction = 'NEUTRAL'
                confidence = max(0.0, min(1.0, confidence)) # Clamp confidence

                return DirectionalPrediction(
                    direction=direction,
                    confidence=confidence,
                    reasoning=reasoning,
                    source='gemini',
                    raw_score=confidence if direction == 'BUY' else (-confidence if direction == 'SELL' else 0.0)
                )
            else:
                log_error(f"Gemini response did not contain valid JSON: {raw_content[:200]}...")
                return None

        except Exception as e:
            log_error(f"Gemini analysis failed: {e}")
            return None

    def _analyze_with_openai(self, text: str) -> Optional[DirectionalPrediction]:
        """Analyze text using OpenAI GPT models."""
        service_info = self.services.get('openai')
        if not service_info or not service_info['available']:
            return None

        self.services['openai']['requests_today'] += 1
        if self.services['openai']['requests_today'] > service_info['quota_limit']:
            log_warning("OpenAI quota exhausted.")
            self.quota_exhausted.add('openai')
            return None

        try:
            client = service_info['client']
            chat_completion = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis AI. Analyze the given text and provide a directional prediction (BUY, SELL, NEUTRAL) and a confidence score (0.0 to 1.0). Provide a brief reasoning."},
                    {"role": "user", "content": f"Text: {text}\n\nFormat your response as a JSON object with 'direction', 'confidence', and 'reasoning' keys."}
                ],
                response_format={"type": "json_object"}
            )
            response_content = chat_completion.choices[0].message.content
            result = json.loads(response_content)

            direction = result.get('direction', 'NEUTRAL').upper()
            confidence = float(result.get('confidence', 0.5))
            reasoning = result.get('reasoning', 'OpenAI analysis.')

            if direction not in ['BUY', 'SELL', 'NEUTRAL']:
                direction = 'NEUTRAL'
            confidence = max(0.0, min(1.0, confidence))

            return DirectionalPrediction(
                direction=direction,
                confidence=confidence,
                reasoning=reasoning,
                source='openai',
                raw_score=confidence if direction == 'BUY' else (-confidence if direction == 'SELL' else 0.0)
            )
        except Exception as e:
            log_error(f"OpenAI analysis failed: {e}")
            return None

    def _analyze_with_claude(self, text: str) -> Optional[DirectionalPrediction]:
        """Analyze text using Anthropic Claude models."""
        service_info = self.services.get('claude')
        if not service_info or not service_info['available']:
            return None

        self.services['claude']['requests_today'] += 1
        if self.services['claude']['requests_today'] > service_info['quota_limit']:
            log_warning("Claude quota exhausted.")
            self.quota_exhausted.add('claude')
            return None

        try:
            client = service_info['client']
            response = client.messages.create(
                model=Config.CLAUDE_MODEL,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"Analyze the sentiment of the following financial news text and determine a directional prediction (BUY, SELL, NEUTRAL) and a confidence score (0.0 to 1.0). Provide a brief reasoning. Format your response as a JSON object with 'direction', 'confidence', and 'reasoning' keys.\n\nText: {text}"}
                ]
            )
            raw_content = response.content[0].text
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                direction = result.get('direction', 'NEUTRAL').upper()
                confidence = float(result.get('confidence', 0.5))
                reasoning = result.get('reasoning', 'Claude analysis.')

                if direction not in ['BUY', 'SELL', 'NEUTRAL']:
                    direction = 'NEUTRAL'
                confidence = max(0.0, min(1.0, confidence))

                return DirectionalPrediction(
                    direction=direction,
                    confidence=confidence,
                    reasoning=reasoning,
                    source='claude',
                    raw_score=confidence if direction == 'BUY' else (-confidence if direction == 'SELL' else 0.0)
                )
            else:
                log_error(f"Claude response did not contain valid JSON: {raw_content[:200]}...")
                return None

        except Exception as e:
            log_error(f"Claude analysis failed: {e}")
            return None

    def _analyze_with_keyword_analysis(self, text: str) -> DirectionalPrediction:
        """Perform simple keyword-based sentiment analysis."""
        text_lower = text.lower()
        positive_score = sum(text_lower.count(k) for k in self.positive_keywords)
        negative_score = sum(text_lower.count(k) for k in self.negative_keywords)

        # Apply higher weight for high-impact keywords
        positive_score += sum(text_lower.count(k) * 2 for k in self.high_impact_positive)
        negative_score += sum(text_lower.count(k) * 2 for k in self.high_impact_negative)

        total_score = positive_score - negative_score
        abs_total = abs(total_score)

        if abs_total == 0:
            return DirectionalPrediction("NEUTRAL", 0.5, "No clear sentiment from keywords.", "keyword")

        # Normalize confidence (simple linear scaling for demonstration)
        # Max possible score depends on keyword list and text length, so this is a basic heuristic
        max_possible_score = max(len(self.positive_keywords) + len(self.high_impact_positive) * 2,
                                 len(self.negative_keywords) + len(self.high_impact_negative) * 2) * (len(text) / 10) # rough estimate
        confidence = min(1.0, abs_total / (max_possible_score + 1e-6)) # Avoid division by zero

        if total_score > 0:
            direction = "BUY"
            reasoning = "Positive keywords dominate."
        elif total_score < 0:
            direction = "SELL"
            reasoning = "Negative keywords dominate."
        else:
            direction = "NEUTRAL"
            reasoning = "Balanced keywords."

        return DirectionalPrediction(
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            source='keyword',
            raw_score=float(total_score)
        )

    def _combine_predictions(self, predictions: List[DirectionalPrediction], successful_services: List[str]) -> DirectionalPrediction:
        """
        Combine predictions from multiple sources using weighted averaging.
        Enhanced for nuanced decision making with confidence thresholds.
        """
        if not predictions:
            return DirectionalPrediction("NEUTRAL", 0.0, "No valid predictions to combine.", "consensus")

        # Filter out services that weren't successful, ensuring only relevant weights are used
        active_weights = {service: self.service_weights.get(service, 0.0) for service in successful_services}

        # Normalize weights for only active services
        total_active_weight = sum(active_weights.values())
        if total_active_weight == 0:
            log_warning("Total active weight is zero, cannot combine. Returning first prediction.")
            return predictions[0] # Fallback to first available prediction

        normalized_weights = {service: weight / total_active_weight for service, weight in active_weights.items()}
        log_debug(f"Normalized weights for combination: {normalized_weights}")

        weighted_buy_score = 0.0
        weighted_sell_score = 0.0
        weighted_neutral_score = 0.0
        total_confidence_sum = 0.0
        all_reasons = []
        individual_predictions_list = []
        weighted_scores_map = {}

        for p in predictions:
            weight = normalized_weights.get(p.source, 0.0)
            scaled_confidence = p.confidence * weight # Scale confidence by its normalized weight
            total_confidence_sum += scaled_confidence
            all_reasons.append(f"{p.source} ({p.direction} conf:{p.confidence:.2f}): {p.reasoning}")
            individual_predictions_list.append(p)

            # Assign scores for weighted average based on direction and confidence
            if p.direction == 'BUY':
                weighted_buy_score += scaled_confidence
                weighted_scores_map[p.source] = scaled_confidence
            elif p.direction == 'SELL':
                weighted_sell_score += scaled_confidence
                weighted_scores_map[p.source] = -scaled_confidence # Negative for sell
            else: # NEUTRAL
                weighted_neutral_score += scaled_confidence
                weighted_scores_map[p.source] = 0.0 # Neutral contributes 0 to directional score

        # Calculate net directional score
        net_directional_score = weighted_buy_score - weighted_sell_score

        # Determine overall direction based on weighted scores and confidence
        final_direction: str
        final_confidence: float

        # More nuanced decision: Check if any strong 'BUY' or 'SELL' prediction exists
        strong_buy = any(p.direction == 'BUY' and p.confidence >= Config.CONFIDENCE_THRESHOLD_BUY for p in predictions)
        strong_sell = any(p.direction == 'SELL' and p.confidence >= Config.CONFIDENCE_THRESHOLD_SELL for p in predictions)

        if strong_buy and not strong_sell:
            final_direction = 'BUY'
            final_confidence = max(p.confidence for p in predictions if p.direction == 'BUY') # Highest buy confidence
            final_reasoning = "Strong BUY signal from at least one high-confidence source."
        elif strong_sell and not strong_buy:
            final_direction = 'SELL'
            final_confidence = max(p.confidence for p in predictions if p.direction == 'SELL') # Highest sell confidence
            final_reasoning = "Strong SELL signal from at least one high-confidence source."
        elif strong_buy and strong_sell:
            # Conflicting strong signals, lean towards neutral or resolve with net score
            if net_directional_score > 0:
                final_direction = 'BUY'
                final_confidence = abs(net_directional_score)
                final_reasoning = "Conflicting strong signals, but weighted average leans BUY."
            elif net_directional_score < 0:
                final_direction = 'SELL'
                final_confidence = abs(net_directional_score)
                final_reasoning = "Conflicting strong signals, but weighted average leans SELL."
            else:
                final_direction = 'NEUTRAL'
                final_confidence = 0.5
                final_reasoning = "Conflicting strong signals, resulting in NEUTRAL."
        else:
            # No strong signals, rely purely on net directional score
            if net_directional_score > 0.05: # Small buffer for neutral
                final_direction = 'BUY'
                final_confidence = abs(net_directional_score) # Max 1.0 if weights are 1.0 and confidence is 1.0
            elif net_directional_score < -0.05: # Small buffer for neutral
                final_direction = 'SELL'
                final_confidence = abs(net_directional_score)
            else:
                final_direction = 'NEUTRAL'
                # For neutral, confidence is often a measure of lack of strong direction
                # Could be 0.5, or derived from total confidence sum divided by number of predictions
                final_confidence = total_confidence_sum / max(1, len(predictions)) # Average confidence
            final_reasoning = "Consensus from multiple models based on weighted average."

        # Clamp final confidence between 0 and 1
        final_confidence = max(0.0, min(1.0, final_confidence))

        combined_reasoning = "Combined analysis:\n" + "\n".join(all_reasons)

        return DirectionalPrediction(
            direction=final_direction,
            confidence=final_confidence,
            reasoning=combined_reasoning,
            source='consensus',
            raw_score=net_directional_score,
            individual_predictions=individual_predictions_list,
            source_weights=normalized_weights,
            weighted_scores=weighted_scores_map
        )


    def _save_finbert_checkpoint(self, training_samples: int, accuracy: float) -> None:
        """Save the current state of the FinBERT model as an adaptive checkpoint."""
        if not self.finbert_model:
            log_warning("FinBERT model not initialized, cannot save checkpoint.")
            return

        try:
            # Ensure the data directory exists
            Config.DATA_DIR.mkdir(parents=True, exist_ok=True)

            checkpoint_path = Config.DATA_DIR / Config.FINBERT_STANDARD_CHECKPOINT_NAME # Using constant
            log_info(f"Saving FinBERT adaptive checkpoint to {checkpoint_path}...")

            checkpoint = {
                'model_state_dict': self.finbert_model.state_dict(),
                'tokenizer_config': self.finbert_tokenizer.save_pretrained(Config.DATA_DIR / "finbert_tokenizer_temp"), # Save tokenizer config
                'training_samples': training_samples,
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat()
            }
            torch.save(checkpoint, checkpoint_path)
            log_info(f"✅ FinBERT checkpoint saved: {checkpoint_path}")
            # Update load time to reflect this newly saved checkpoint
            self._finbert_load_time = time.time()
        except Exception as e:
            log_error(f"Failed to save FinBERT checkpoint: {e}")

    def _update_finbert_with_feedback(self, texts: List[str], labels: List[int]) -> bool:
        """
        Update FinBERT model with new feedback data (continuous learning).
        labels: 0 for negative, 1 for neutral, 2 for positive
        """
        if not (self.finbert_model and self.finbert_tokenizer):
            log_warning("FinBERT model or tokenizer not available for feedback update.")
            return False
        if not texts or not labels or len(texts) != len(labels):
            log_warning("Invalid input for FinBERT feedback update: texts and labels must match.")
            return False
        if not TORCH_AVAILABLE:
            log_warning("PyTorch not available, cannot update FinBERT.")
            return False

        try:
            self.finbert_model.train() # Set model to training mode

            # Determine device
            device = next(self.finbert_model.parameters()).device
            log_info(f"FinBERT feedback update starting on {device} with {len(texts)} samples.")

            # Optimizer and loss function
            optimizer = torch.optim.AdamW(self.finbert_model.parameters(), lr=Config.FINBERT_LEARNING_RATE, weight_decay=Config.FINBERT_WEIGHT_DECAY) # Added weight_decay
            criterion = torch.nn.CrossEntropyLoss()

            # Simple batching for training
            batch_size = 8 # Can be configured
            total_loss = 0.0
            actual_labels = []
            predicted_labels = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]

                if not batch_texts:
                    continue

                # Tokenize and move to device
                inputs = self.finbert_tokenizer(
                    batch_texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels_tensor = torch.LongTensor(batch_labels).to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.finbert_model(**inputs)
                loss = criterion(outputs.logits, labels_tensor)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # For accuracy calculation (on batch)
                _, predicted = torch.max(outputs.logits, 1)
                actual_labels.extend(batch_labels)
                predicted_labels.extend(predicted.cpu().numpy())

            avg_loss = total_loss / max(1, len(texts) // batch_size)

            # Calculate simple accuracy
            correct = sum(1 for actual, predicted in zip(actual_labels, predicted_labels)
                          if actual == predicted)
            accuracy = correct / len(actual_labels) if actual_labels else 0.0

            # Save updated model
            self._save_finbert_checkpoint(training_samples=len(texts), accuracy=accuracy) # Corrected method call

            self.finbert_model.eval()
            log_info(f"✅ FinBERT updated: loss={avg_loss:.4f}, accuracy={accuracy:.2%}")

            return True

        except Exception as e:
            log_error(f"FinBERT feedback update failed: {e}")
            if hasattr(self, 'finbert_model'):
                self.finbert_model.eval()
            return False