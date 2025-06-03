"""
Enhanced news analyzer with integrated social sentiment and analyst data processing
"""
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timezone
from functools import lru_cache
import numpy as np
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning, log_debug
from .technical_analyzer import TechnicalAnalyzer, TechnicalAnalysis
from .gemini_news_analyzer import OptimizedGeminiNewsAnalyzer, GeminiAnalysis
from .technical_strategies import TechnicalStrategies, StrategySignal
from .sentiment.enhanced_sentiment_scorer import EnhancedSentimentScorer, SentimentScore


@dataclass
class EnhancedNewsAnalysis:
    """Comprehensive news analysis result with all metrics and new data sources."""
    symbol: str
    title: str
    content: str
    sentiment_score: float
    confidence: float
    topic: str
    timestamp: pd.Timestamp
    finbert_score: float = 0.0
    keyword_score: float = 0.0
    gemini_score: float = 0.0
    gemini_confidence: float = 0.0
    gemini_reasoning: str = ""
    technical_analysis: Optional[TechnicalAnalysis] = None
    strategy_signals: Optional[List[StrategySignal]] = None
    best_strategy: Optional[StrategySignal] = None
    combined_confidence: float = 0.0
    
    # New fields for enhanced data sources
    social_sentiment_score: float = 0.0
    social_sentiment_label: str = ""
    analyst_action: str = ""
    analyst_grade_new: str = ""
    analyst_grade_previous: str = ""
    price_target: float = 0.0
    price_target_change: float = 0.0
    event_type: str = ""
    source_type: str = ""


class EnhancedNewsAnalyzer:
    """High-performance news analyzer with integrated social and analyst data."""
    
    # Class-level constants for performance
    BATCH_SIZE = 15
    MAX_CACHE_SIZE = 150
    FINBERT_MODEL_NAME = "ProsusAI/finbert"  # Renamed to avoid conflict
    
    # Enhanced topic patterns with new categories
    TOPIC_PATTERNS = {
        'earnings': re.compile(
            r'\b(?:earnings|revenue|profit|eps|quarterly|guidance|beat|miss|results|q[1-4])\b', 
            re.IGNORECASE
        ),
        'biotech': re.compile(
            r'\b(?:fda|approval|phase|trial|clinical|drug|therapy|efficacy|breakthrough|orphan|pdufa)\b', 
            re.IGNORECASE
        ),
        'ma': re.compile(
            r'\b(?:merger|acquisition|deal|buyout|takeover|acquire|merge|purchase|combine)\b', 
            re.IGNORECASE
        ),
        'analyst': re.compile(
            r'\b(?:upgrade|downgrade|target|analyst|rating|price target|initiat|recommendation)\b', 
            re.IGNORECASE
        ),
        'corporate_action': re.compile(
            r'\b(?:dividend|buyback|split|spinoff|distribution|repurchase|special dividend)\b', 
            re.IGNORECASE
        ),
        'social_sentiment': re.compile(
            r'\b(?:social|sentiment|bullish|bearish|trending|reddit|twitter|social media)\b',
            re.IGNORECASE
        ),
        'market_event': re.compile(
            r'\b(?:ipo|economic|federal|fed|interest rate|inflation|gdp|unemployment)\b',
            re.IGNORECASE
        )
    }
    
    # Analyst action confidence mapping
    ANALYST_CONFIDENCE_MAP = {
        'upgrade': 0.85,
        'downgrade': 0.85,
        'initiated': 0.75,
        'reiterated': 0.60,
        'maintained': 0.50
    }
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized configuration and safe error handling."""
        self.fmp_loader = fmp_loader
        self.technical_analyzer = TechnicalAnalyzer(fmp_loader)
        self.technical_strategies = TechnicalStrategies(fmp_loader)
        self.gemini_analyzer = OptimizedGeminiNewsAnalyzer()
        self.sentiment_scorer = EnhancedSentimentScorer()
        
        # Optimized caching
        self.recent_analyses_cache = []
        self.analysis_count = 0
        self.cache_hits = 0
        
        # Enhanced caching for new data sources
        self.social_sentiment_cache = {}
        self.analyst_data_cache = {}
        self.price_target_cache = {}
        
        # Device setup for FinBERT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finbert_model_instance = None  # Renamed to avoid conflict
        self.finbert_tokenizer = None
        
        # Performance tracking
        self._setup_performance_tracking()
        
        # Initialize components
        self._initialize_finbert_safe()
        
        # Set universe cache in FMP loader for targeted news fetching
        try:
            if hasattr(self.fmp_loader, 'set_universe_cache'):
                # This will be set by the main system after universe initialization
                pass
        except Exception as e:
            log_debug(f"Could not set universe cache: {e}")
    
    def _setup_performance_tracking(self) -> None:
        """Setup performance monitoring."""
        self.processing_times = {
            'finbert': [],
            'sentiment': [],
            'technical': [],
            'strategy': [],
            'social': [],
            'analyst': [],
            'total': []
        }
    
    def _initialize_finbert_safe(self) -> None:
        """Initialize FinBERT with safe error handling and no torch.compile."""
        try:
            log_info("Loading FinBERT model with safe configuration...")
            
            # Load with optimization flags
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                self.FINBERT_MODEL_NAME,
                use_fast=True,
                return_tensors="pt"
            )
            
            self.finbert_model_instance = AutoModelForSequenceClassification.from_pretrained(
                self.FINBERT_MODEL_NAME,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            self.finbert_model_instance.to(self.device)
            self.finbert_model_instance.eval()
            
            # Set inference context safely
            self.inference_context = torch.inference_mode if hasattr(torch, 'inference_mode') else torch.no_grad
            
            log_info(f"FinBERT initialized successfully on {self.device}")
            
        except Exception as e:
            log_error(f"Failed to initialize FinBERT: {e}")
            self.finbert_model_instance = None
            self.finbert_tokenizer = None
    
    def set_universe(self, universe: List[str]) -> None:
        """Set universe for targeted data fetching."""
        try:
            if hasattr(self.fmp_loader, 'set_universe_cache'):
                self.fmp_loader.set_universe_cache(universe)
                log_debug(f"Set universe cache with {len(universe)} symbols")
        except Exception as e:
            log_debug(f"Could not set universe cache: {e}")
    
    def analyze_news_with_technical(self, news_df: pd.DataFrame, 
                                  current_prices: pd.DataFrame) -> List[EnhancedNewsAnalysis]:
        """Enhanced news analysis with social sentiment and analyst data integration."""
        if news_df is None or news_df.empty:
            return []
        
        log_info(f"Analyzing {len(news_df)} articles with enhanced pipeline")
        
        # Create price lookup for efficiency
        price_lookup = self._create_price_lookup_safe(current_prices)
        
        # Process in optimized batches
        results = []
        for i in range(0, len(news_df), self.BATCH_SIZE):
            batch_df = news_df.iloc[i:i + self.BATCH_SIZE]
            batch_results = self._process_batch_enhanced(batch_df, price_lookup)
            results.extend([r for r in batch_results if r is not None])
        
        # Filter and log results
        high_confidence_results = [
            analysis for analysis in results 
            if analysis.combined_confidence >= CONFIG.min_confidence_score
        ]
        
        if high_confidence_results:
            log_info(f"Generated {len(high_confidence_results)} high-confidence analyses")
            self._log_top_results_enhanced(high_confidence_results[:3])
        
        self.analysis_count += len(results)
        return results
    
    def _create_price_lookup_safe(self, current_prices: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Create price lookup with safe error handling and correct return type."""
        try:
            if current_prices is None or current_prices.empty:
                return {}
            
            # Ensure we return the correct type
            price_dict = current_prices.set_index('symbol').to_dict('index')
            return {str(k): dict(v) for k, v in price_dict.items()}
        except Exception as e:
            log_debug(f"Error creating price lookup: {e}")
            return {}
    
    def _process_batch_enhanced(self, batch_df: pd.DataFrame, 
                               price_lookup: Dict[str, Dict[str, Any]]) -> List[Optional[EnhancedNewsAnalysis]]:
        """Process batch with enhanced data sources."""
        # Validate and prepare batch data
        valid_data = []
        for _, row in batch_df.iterrows():
            try:
                symbol = str(row.get('symbol', '')).strip()
                title = str(row.get('title', '')).strip()
                content = str(row.get('content', '')).strip()
                
                if all([symbol, title, content]):
                    valid_data.append({
                        'symbol': symbol,
                        'title': title,
                        'content': content,
                        'source': str(row.get('source', '')).strip(),
                        'full_text': f"{title} {content}",
                        'row_data': row  # Keep full row for enhanced data extraction
                    })
            except Exception as e:
                log_debug(f"Error preparing batch data: {e}")
                continue
        
        if not valid_data:
            return [None] * len(batch_df)
        
        # Batch FinBERT processing
        finbert_results = self._process_finbert_batch_safe([item['full_text'] for item in valid_data])
        
        # Process individual articles with enhanced data
        results = []
        for i, data in enumerate(valid_data):
            try:
                finbert_sentiment, finbert_conf = finbert_results[i] if i < len(finbert_results) else (0.0, 0.0)
                analysis = self._process_single_article_enhanced(
                    data, price_lookup, finbert_sentiment, finbert_conf
                )
                results.append(analysis)
            except Exception as e:
                log_error(f"Error processing {data.get('symbol', 'unknown')}: {e}")
                results.append(None)
        
        # Pad results to match batch size
        while len(results) < len(batch_df):
            results.append(None)
        
        return results
    
    def _process_finbert_batch_safe(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Safe FinBERT batch processing with comprehensive error handling."""
        if not self.finbert_model_instance or not texts:
            return [(0.0, 0.0)] * len(texts)
        
        try:
            # Prepare texts (truncate for efficiency)
            processed_texts = [text[:512] for text in texts if text.strip()]
            
            if not processed_texts:
                return [(0.0, 0.0)] * len(texts)
            
            # Batch tokenization
            inputs = self.finbert_tokenizer(
                processed_texts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
                add_special_tokens=True
            ).to(self.device)
            
            # Inference with safe context - fixed the optional call issue
            with self.inference_context():
                outputs = self.finbert_model_instance(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions_cpu = predictions.cpu().float().numpy()
            
            # Process results safely
            results = []
            for probs in predictions_cpu:
                try:
                    pos_prob, neg_prob, _ = probs
                    sentiment = float(pos_prob - neg_prob)
                    confidence = float(max(probs))
                    results.append((sentiment, confidence))
                except Exception as e:
                    log_debug(f"Error processing FinBERT result: {e}")
                    results.append((0.0, 0.0))
            
            # Pad if necessary
            while len(results) < len(texts):
                results.append((0.0, 0.0))
            
            return results
            
        except Exception as e:
            log_error(f"FinBERT batch processing error: {e}")
            return [(0.0, 0.0)] * len(texts)
    
    def _process_single_article_enhanced(self, data: Dict, price_lookup: Dict[str, Dict[str, Any]],
                                        finbert_sentiment: float, finbert_conf: float) -> Optional[EnhancedNewsAnalysis]:
        """Process single article with enhanced data sources."""
        try:
            symbol = data['symbol']
            title = data['title']
            content = data['content']
            source = data['source']
            full_text = data['full_text']
            row_data = data['row_data']
            
            # Get price and topic efficiently
            price_data = price_lookup.get(symbol)
            current_price = float(price_data.get('lastSalePrice', 0)) if price_data else 0.0
            topic = self._detect_topic_enhanced(full_text, source)
            
            # Enhanced sentiment analysis
            enhanced_score = self._get_enhanced_sentiment_safe(
                title, content, symbol, current_price, topic, source
            )
            
            # Extract enhanced data from source
            enhanced_data = self._extract_enhanced_data(row_data, current_price)
            
            # Social sentiment integration
            social_sentiment = self._get_social_sentiment_safe(symbol, enhanced_data)
            
            # Analyst data integration
            analyst_data = self._get_analyst_data_safe(symbol, enhanced_data)
            
            # Gemini analysis (with built-in rate limiting)
            gemini_sentiment, gemini_conf, gemini_reasoning = self._get_gemini_analysis_safe(
                symbol, title, content, current_price, topic
            )
            
            # Enhanced ensemble sentiment calculation
            final_sentiment, news_confidence = self._calculate_enhanced_ensemble_sentiment(
                finbert_sentiment, finbert_conf,
                enhanced_score.final_sentiment if enhanced_score else 0.0,
                enhanced_score.quality_confidence if enhanced_score else 0.0,
                gemini_sentiment, gemini_conf,
                social_sentiment, analyst_data
            )
            
            # Technical analysis
            technical_analysis = self._get_technical_analysis_safe(symbol, price_data)
            
            # Strategy analysis
            strategy_signals, best_strategy = self._get_strategy_analysis_safe(
                symbol, current_price, technical_analysis, final_sentiment
            )
            
            # Enhanced combined confidence calculation
            combined_confidence = self._calculate_enhanced_combined_confidence(
                news_confidence, technical_analysis, final_sentiment, 
                strategy_signals, social_sentiment, analyst_data
            )
            
            # Update cache safely
            self._update_analysis_cache_safe(symbol, final_sentiment, topic, 
                                           enhanced_score.quality_confidence if enhanced_score else 0.0)
            
            return EnhancedNewsAnalysis(
                symbol=symbol,
                title=title,
                content=content,
                sentiment_score=final_sentiment,
                confidence=news_confidence,
                topic=topic,
                timestamp=pd.Timestamp.now(tz=timezone.utc),
                finbert_score=finbert_sentiment,
                keyword_score=enhanced_score.final_sentiment if enhanced_score else 0.0,
                gemini_score=gemini_sentiment,
                gemini_confidence=gemini_conf,
                gemini_reasoning=gemini_reasoning,
                technical_analysis=technical_analysis,
                strategy_signals=strategy_signals or [],
                best_strategy=best_strategy,
                combined_confidence=combined_confidence,
                
                # Enhanced fields
                social_sentiment_score=social_sentiment.get('score', 0.0),
                social_sentiment_label=social_sentiment.get('label', ''),
                analyst_action=analyst_data.get('action', ''),
                analyst_grade_new=analyst_data.get('grade_new', ''),
                analyst_grade_previous=analyst_data.get('grade_previous', ''),
                price_target=analyst_data.get('price_target', 0.0),
                price_target_change=analyst_data.get('price_target_change', 0.0),
                event_type=enhanced_data.get('event_type', ''),
                source_type=source
            )
            
        except Exception as e:
            log_error(f"Error processing enhanced single article: {e}")
            return None
    
    @lru_cache(maxsize=1000)
    def _detect_topic_enhanced(self, text: str, source: str) -> str:
        """Enhanced topic detection with source awareness."""
        try:
            # Source-based topic detection first
            if source in ['analyst_changes', 'price_targets']:
                return 'analyst'
            elif source in ['social_sentiment']:
                return 'social_sentiment'
            elif source in ['market_events']:
                return 'market_event'
            elif source in ['insider_trading', 'senate_trading']:
                return 'insider'
            
            # Pattern-based detection
            text_lower = text.lower()
            topic_scores = {}
            
            for topic, pattern in self.TOPIC_PATTERNS.items():
                try:
                    matches = len(pattern.findall(text_lower))
                    if matches > 0:
                        topic_scores[topic] = matches
                except Exception as e:
                    log_debug(f"Error matching pattern for topic {topic}: {e}")
                    continue
            
            return max(topic_scores.items(), key=lambda x: x[1])[0] if topic_scores else 'general'
        except Exception as e:
            log_debug(f"Error in enhanced topic detection: {e}")
            return 'general'
    
    def _extract_enhanced_data(self, row_data: pd.Series, current_price: float) -> Dict[str, Any]:
        """Extract enhanced data from row - removed unused symbol parameter."""
        enhanced_data = {}
        
        try:
            # Extract analyst data if present
            for field in ['analyst_action', 'analyst_grade_new', 'analyst_grade_previous', 
                         'analyst_company', 'price_target', 'target_high', 'target_low']:
                if field in row_data:
                    enhanced_data[field] = row_data[field]
            
            # Extract social sentiment data if present
            for field in ['social_sentiment_score', 'social_sentiment_label']:
                if field in row_data:
                    enhanced_data[field] = row_data[field]
            
            # Extract event data if present
            for field in ['event_type']:
                if field in row_data:
                    enhanced_data[field] = row_data[field]
            
            # Calculate price target change if we have price target
            if 'price_target' in enhanced_data and enhanced_data['price_target'] and current_price:
                try:
                    target = float(enhanced_data['price_target'])
                    enhanced_data['price_target_change'] = (target - current_price) / current_price
                except (ValueError, ZeroDivisionError):
                    enhanced_data['price_target_change'] = 0.0
            
        except Exception as e:
            log_debug(f"Error extracting enhanced data: {e}")
        
        return enhanced_data
    
    def _get_social_sentiment_safe(self, symbol: str, enhanced_data: Dict) -> Dict[str, Any]:
        """Get social sentiment with caching and API fallback."""
        try:
            # Check if we already have social sentiment in the row data
            if 'social_sentiment_score' in enhanced_data:
                return {
                    'score': enhanced_data['social_sentiment_score'],
                    'label': enhanced_data.get('social_sentiment_label', ''),
                    'source': 'embedded'
                }
            
            # Check cache first
            if symbol in self.social_sentiment_cache:
                cache_time, cached_data = self.social_sentiment_cache[symbol]
                if (datetime.now() - cache_time).seconds < 1800:  # 30 minutes cache
                    return cached_data
            
            # Fetch from API if available
            try:
                sentiment_data = self.fmp_loader.get_social_sentiment_for_symbol(symbol)
                if sentiment_data:
                    result = {
                        'score': sentiment_data.get('sentiment', 0.0),
                        'label': sentiment_data.get('label', ''),
                        'source': 'api'
                    }
                    self.social_sentiment_cache[symbol] = (datetime.now(), result)
                    return result
            except Exception as e:
                log_debug(f"Could not fetch social sentiment for {symbol}: {e}")
            
            return {'score': 0.0, 'label': '', 'source': 'none'}
            
        except Exception as e:
            log_debug(f"Error getting social sentiment for {symbol}: {e}")
            return {'score': 0.0, 'label': '', 'source': 'error'}
    
    def _get_analyst_data_safe(self, symbol: str, enhanced_data: Dict) -> Dict[str, Any]:
        """Get analyst data with caching and API fallback."""
        try:
            # Check if we already have analyst data in the row
            if 'analyst_action' in enhanced_data:
                return {
                    'action': enhanced_data.get('analyst_action', ''),
                    'grade_new': enhanced_data.get('analyst_grade_new', ''),
                    'grade_previous': enhanced_data.get('analyst_grade_previous', ''),
                    'price_target': enhanced_data.get('price_target', 0.0),
                    'price_target_change': enhanced_data.get('price_target_change', 0.0),
                    'source': 'embedded'
                }
            
            # Check cache first
            if symbol in self.analyst_data_cache:
                cache_time, cached_data = self.analyst_data_cache[symbol]
                if (datetime.now() - cache_time).seconds < 3600:  # 1 hour cache
                    return cached_data
            
            # Fetch from API if available
            try:
                # Try to get price target
                price_target_data = self.fmp_loader.get_price_target_for_symbol(symbol)
                result = {'action': '', 'grade_new': '', 'grade_previous': '', 
                         'price_target': 0.0, 'price_target_change': 0.0, 'source': 'api'}
                
                if price_target_data:
                    result['price_target'] = price_target_data.get('priceTarget', 0.0)
                    # Calculate change if we have current price
                    # This would need current price passed in, for now set to 0
                    result['price_target_change'] = 0.0
                
                self.analyst_data_cache[symbol] = (datetime.now(), result)
                return result
                
            except Exception as e:
                log_debug(f"Could not fetch analyst data for {symbol}: {e}")
            
            return {'action': '', 'grade_new': '', 'grade_previous': '', 
                   'price_target': 0.0, 'price_target_change': 0.0, 'source': 'none'}
            
        except Exception as e:
            log_debug(f"Error getting analyst data for {symbol}: {e}")
            return {'action': '', 'grade_new': '', 'grade_previous': '', 
                   'price_target': 0.0, 'price_target_change': 0.0, 'source': 'error'}
    
    def _get_enhanced_sentiment_safe(self, title: str, content: str, symbol: str,
                                   current_price: float, topic: str, source: str) -> Optional[SentimentScore]:
        """Get enhanced sentiment with error handling."""
        try:
            return self.sentiment_scorer.calculate_enhanced_sentiment(
                title=title, 
                content=content, 
                symbol=symbol,
                current_price=current_price, 
                topic=topic,
                source=source, 
                recent_analyses=self.recent_analyses_cache[-20:]
            )
        except Exception as e:
            log_debug(f"Error in enhanced sentiment analysis: {e}")
            return None
    
    def _get_gemini_analysis_safe(self, symbol: str, title: str, content: str,
                                current_price: float, topic: str) -> Tuple[float, float, str]:
        """Get Gemini analysis with safe error handling."""
        if not self.gemini_analyzer.enabled or current_price <= 0:
            return 0.0, 0.0, ""
        
        try:
            result = self.gemini_analyzer.analyze_news(
                symbol=symbol, 
                title=title[:200], 
                content=content[:800],
                current_price=current_price, 
                topic=topic
            )
            
            if result:
                return result.sentiment_score, result.confidence, result.reasoning[:150]
                
        except Exception as e:
            log_debug(f"Gemini analysis failed for {symbol}: {e}")
        
        return 0.0, 0.0, ""
    
    def _calculate_enhanced_ensemble_sentiment(self, finbert_sentiment: float, finbert_conf: float,
                                             enhanced_sentiment: float, enhanced_conf: float,
                                             gemini_sentiment: float, gemini_conf: float,
                                             social_sentiment: Dict, analyst_data: Dict) -> Tuple[float, float]:
        """Enhanced ensemble sentiment calculation - refactored to reduce complexity."""
        try:
            # Base weights (adjusted to accommodate new sources)
            weights = {
                'finbert': CONFIG.finbert_weight * 0.8,
                'enhanced': CONFIG.keyword_weight * 0.8,
                'gemini': CONFIG.gemini_weight * 0.8,
                'social': 0.15,
                'analyst': 0.25
            }
            
            # Collect base sources
            sources = self._collect_base_sources(finbert_sentiment, finbert_conf, enhanced_sentiment, 
                                               enhanced_conf, gemini_sentiment, gemini_conf, weights)
            
            # Add social and analyst sources
            sources = self._add_enhanced_sources(sources, social_sentiment, analyst_data, weights)
            
            # Calculate ensemble
            return self._calculate_ensemble_result(sources)
            
        except Exception as e:
            log_debug(f"Error in enhanced ensemble sentiment calculation: {e}")
            return 0.0, 0.0
    
    def _collect_base_sources(self, finbert_sentiment: float, finbert_conf: float,
                            enhanced_sentiment: float, enhanced_conf: float,
                            gemini_sentiment: float, gemini_conf: float,
                            weights: Dict) -> List[Tuple[float, float, float]]:
        """Collect base sentiment sources."""
        return [
            (finbert_sentiment, finbert_conf, weights['finbert']),
            (enhanced_sentiment, enhanced_conf, weights['enhanced']),
            (gemini_sentiment, gemini_conf, weights['gemini'])
        ]
    
    def _add_enhanced_sources(self, sources: List[Tuple[float, float, float]], 
                            social_sentiment: Dict, analyst_data: Dict, 
                            weights: Dict) -> List[Tuple[float, float, float]]:
        """Add enhanced sources to the collection."""
        # Add social sentiment if available
        social_score = social_sentiment.get('score', 0.0)
        if abs(social_score) > 0.1:
            social_conf = 0.6
            sources.append((social_score, social_conf, weights['social']))
        
        # Add analyst sentiment if available
        analyst_sentiment, analyst_conf = self._calculate_analyst_sentiment(analyst_data)
        if analyst_conf > 0:
            sources.append((analyst_sentiment, analyst_conf, weights['analyst']))
        
        return sources
    
    def _calculate_ensemble_result(self, sources: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        """Calculate final ensemble result."""
        valid_sources = [(s, c, w) for s, c, w in sources if c > 0]
        
        if not valid_sources:
            return 0.0, 0.0
        
        if len(valid_sources) == 1:
            return valid_sources[0][0], valid_sources[0][1]
        
        # Weighted ensemble calculation
        numerator = sum(s * w * c for s, c, w in valid_sources)
        denominator = sum(w * c for s, c, w in valid_sources)
        
        ensemble_sentiment = numerator / denominator if denominator > 0 else 0.0
        
        # Enhanced confidence calculation
        base_confidence = sum(c for s, c, w in valid_sources) / len(valid_sources)
        source_bonus = 1.0 + (len(valid_sources) - 1) * 0.1
        
        final_confidence = min(base_confidence * source_bonus, 1.0)
        
        return ensemble_sentiment, final_confidence
    
    def _calculate_analyst_sentiment(self, analyst_data: Dict) -> Tuple[float, float]:
        """Calculate sentiment and confidence from analyst data."""
        try:
            action = analyst_data.get('action', '').lower()
            price_target_change = analyst_data.get('price_target_change', 0.0)
            
            sentiment = 0.0
            confidence = 0.0
            
            # Action-based sentiment
            if 'upgrade' in action:
                sentiment = 0.7
                confidence = self.ANALYST_CONFIDENCE_MAP.get('upgrade', 0.85)
            elif 'downgrade' in action:
                sentiment = -0.7
                confidence = self.ANALYST_CONFIDENCE_MAP.get('downgrade', 0.85)
            elif 'initiated' in action or 'coverage' in action:
                # Determine sentiment from grade if available
                grade_new = analyst_data.get('grade_new', '').lower()
                if any(word in grade_new for word in ['buy', 'strong buy', 'outperform']):
                    sentiment = 0.5
                elif any(word in grade_new for word in ['sell', 'strong sell', 'underperform']):
                    sentiment = -0.5
                else:
                    sentiment = 0.1  # Neutral positive for initiation
                confidence = self.ANALYST_CONFIDENCE_MAP.get('initiated', 0.75)
            
            # Price target-based sentiment adjustment
            if abs(price_target_change) > 0.05:  # 5% or more change
                if price_target_change > 0:
                    sentiment = max(sentiment, 0.3)
                    confidence = max(confidence, 0.6)
                else:
                    sentiment = min(sentiment, -0.3)
                    confidence = max(confidence, 0.6)
            
            return sentiment, confidence
            
        except Exception as e:
            log_debug(f"Error calculating analyst sentiment: {e}")
            return 0.0, 0.0
    
    def _get_technical_analysis_safe(self, symbol: str, price_data) -> Optional[TechnicalAnalysis]:
        """Get technical analysis with safe error handling."""
        if price_data:
            try:
                return self.technical_analyzer.analyze_symbol(symbol, pd.Series(price_data))
            except Exception as e:
                log_debug(f"Technical analysis failed for {symbol}: {e}")
        return None
    
    def _get_strategy_analysis_safe(self, symbol: str, current_price: float,
                                  technical_analysis: Optional[TechnicalAnalysis],
                                  sentiment_score: float) -> Tuple[List[StrategySignal], Optional[StrategySignal]]:
        """Get strategy analysis with safe error handling."""
        if technical_analysis and current_price > 0:
            try:
                signals = self.technical_strategies.analyze_with_strategies(
                    symbol=symbol, 
                    current_price=current_price,
                    technical_analysis=technical_analysis, 
                    news_sentiment=sentiment_score
                )
                
                if signals:
                    best_signal = self.technical_strategies.get_best_strategy(signals)
                    return signals, best_signal
                    
            except Exception as e:
                log_debug(f"Strategy analysis failed for {symbol}: {e}")
        
        return [], None
    
    def _calculate_enhanced_combined_confidence(self, news_confidence: float, 
                                              technical_analysis: Optional[TechnicalAnalysis],
                                              sentiment_score: float,
                                              strategy_signals: Optional[List[StrategySignal]],
                                              social_sentiment: Dict,
                                              analyst_data: Dict) -> float:
        """Enhanced combined confidence calculation with new data sources."""
        try:
            if technical_analysis is None:
                return news_confidence * 0.7
            
            tech_confidence = technical_analysis.technical_confidence
            
            # Base combination
            base_combined = (
                news_confidence * 0.4 +
                tech_confidence * 0.3 +
                0.3  # Reserved for additional factors
            )
            
            # Social sentiment factor
            social_factor = self._calculate_social_factor(social_sentiment, sentiment_score)
            
            # Analyst factor
            analyst_factor = self._calculate_analyst_factor(analyst_data)
            
            # Strategy alignment
            strategy_factor = self._calculate_strategy_factor(strategy_signals, sentiment_score)
            
            # Apply all factors
            final_confidence = base_combined * social_factor * analyst_factor * strategy_factor
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            log_debug(f"Error calculating enhanced combined confidence: {e}")
            return news_confidence * 0.7 if news_confidence else 0.0
    
    def _calculate_social_factor(self, social_sentiment: Dict, sentiment_score: float) -> float:
        """Calculate social sentiment factor."""
        social_score = social_sentiment.get('score', 0.0)
        if abs(social_score) > 0.3:
            # Social sentiment aligns with main sentiment
            return 1.15 if (social_score > 0) == (sentiment_score > 0) else 0.9
        return 1.0
    
    def _calculate_analyst_factor(self, analyst_data: Dict) -> float:
        """Calculate analyst factor."""
        analyst_conf = self._calculate_analyst_sentiment(analyst_data)[1]
        if analyst_conf > 0.7:
            return 1.2
        elif analyst_conf > 0.5:
            return 1.1
        return 1.0
    
    def _calculate_strategy_factor(self, strategy_signals: Optional[List[StrategySignal]], 
                                 sentiment_score: float) -> float:
        """Calculate strategy factor."""
        if not strategy_signals:
            return 1.0
        
        best_signal = max(strategy_signals, key=lambda x: (x.strength.value, x.confidence))
        signal_direction = 1 if best_signal.signal_type == "long" else -1
        sentiment_direction = 1 if sentiment_score > 0 else -1
        
        if signal_direction == sentiment_direction:
            return 1.0 + (best_signal.confidence * 0.2)
        else:
            return 0.9
    
    def _update_analysis_cache_safe(self, symbol: str, sentiment_score: float, 
                                  topic: str, quality_confidence: float) -> None:
        """Safe cache management with error handling."""
        try:
            current_time = datetime.now(timezone.utc)
            
            self.recent_analyses_cache.append({
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'timestamp': current_time,
                'topic': topic,
                'quality_confidence': quality_confidence
            })
            
            # Efficient cache pruning
            if len(self.recent_analyses_cache) > self.MAX_CACHE_SIZE:
                keep_count = int(self.MAX_CACHE_SIZE * 0.8)
                self.recent_analyses_cache = self.recent_analyses_cache[-keep_count:]
                self.cache_hits += 1
        except Exception as e:
            log_debug(f"Error updating analysis cache: {e}")
    
    def _log_top_results_enhanced(self, analyses: List[EnhancedNewsAnalysis]) -> None:
        """Log top analysis results with enhanced information."""
        try:
            for i, analysis in enumerate(analyses, 1):
                try:
                    tech_info = ""
                    if analysis.technical_analysis:
                        ta = analysis.technical_analysis
                        tech_info = f"tech={ta.technical_confidence:.2f} mom={ta.momentum_score:.2f}"
                    
                    strategy_info = ""
                    if analysis.best_strategy:
                        bs = analysis.best_strategy
                        strategy_info = f"strategy={bs.strategy_name}[{bs.signal_type.upper()}]"
                    
                    enhanced_info = ""
                    if analysis.social_sentiment_score != 0:
                        enhanced_info += f"social={analysis.social_sentiment_score:.2f} "
                    if analysis.analyst_action:
                        enhanced_info += f"analyst={analysis.analyst_action} "
                    if analysis.price_target > 0:
                        enhanced_info += f"target=${analysis.price_target:.2f} "
                    
                    log_info(
                        f"Top #{i}: {analysis.symbol} sentiment={analysis.sentiment_score:.3f} "
                        f"conf={analysis.combined_confidence:.3f} {tech_info} {strategy_info} "
                        f"{enhanced_info}topic={analysis.topic} source={analysis.source_type}"
                    )
                except Exception as e:
                    log_debug(f"Error logging enhanced result #{i}: {e}")
        except Exception as e:
            log_debug(f"Error in enhanced top results logging: {e}")
    
    def filter_high_confidence(self, analyses: List[EnhancedNewsAnalysis], 
                             use_combined_confidence: bool = True) -> List[EnhancedNewsAnalysis]:
        """Safe high confidence filtering with error handling."""
        if not analyses:
            return []
        
        try:
            confidence_attr = 'combined_confidence' if use_combined_confidence else 'confidence'
            threshold = CONFIG.min_confidence_score
            
            # Filter and sort in one pass
            high_conf = [
                analysis for analysis in analyses
                if analysis and getattr(analysis, confidence_attr, 0) >= threshold
            ]
            
            if high_conf:
                high_conf.sort(key=lambda x: getattr(x, confidence_attr, 0), reverse=True)
                log_info(f"Filtered to {len(high_conf)} high-confidence signals (enhanced)")
            
            return high_conf
        except Exception as e:
            log_error(f"Error filtering high confidence analyses: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics."""
        try:
            return {
                'total_analyses': self.analysis_count,
                'cache_size': len(self.recent_analyses_cache),
                'cache_hits': self.cache_hits,
                'finbert_enabled': self.finbert_model_instance is not None,
                'gemini_enabled': self.gemini_analyzer.enabled,
                'device': self.device,
                'batch_size': self.BATCH_SIZE,
                'max_cache_size': self.MAX_CACHE_SIZE,
                'social_sentiment_cache_size': len(self.social_sentiment_cache),
                'analyst_data_cache_size': len(self.analyst_data_cache),
                'enhanced_sources_count': 4  # social, analyst, market events, company news
            }
        except Exception as e:
            log_error(f"Error getting enhanced performance stats: {e}")
            return {'error': str(e)}