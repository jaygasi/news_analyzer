"""
Enhanced news analyzer with integrated social sentiment and analyst data processing - Rate limit optimized
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
    
    # New fields for enhanced data sources - SIMPLIFIED to avoid rate limits
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
    """High-performance news analyzer with rate-limit optimized processing."""
    
    # Class-level constants for performance
    BATCH_SIZE = 15
    MAX_CACHE_SIZE = 150
    FINBERT_MODEL_NAME = "ProsusAI/finbert"
    
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
        """Initialize with optimized configuration and rate limit awareness."""
        self.fmp_loader = fmp_loader
        self.technical_analyzer = TechnicalAnalyzer(fmp_loader)
        self.technical_strategies = TechnicalStrategies(fmp_loader)
        self.gemini_analyzer = OptimizedGeminiNewsAnalyzer()
        self.sentiment_scorer = EnhancedSentimentScorer()
        
        # Optimized caching
        self.recent_analyses_cache = []
        self.analysis_count = 0
        self.cache_hits = 0
        
        # DISABLED enhanced data sources to avoid 429 errors
        self.enhanced_data_enabled = False  # Temporarily disabled
        
        # Device setup for FinBERT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finbert_model_instance = None
        self.finbert_tokenizer = None
        
        # Performance tracking
        self._setup_performance_tracking()
        
        # Initialize components
        self._initialize_finbert_safe()
    
    def _setup_performance_tracking(self) -> None:
        """Setup performance monitoring."""
        self.processing_times = {
            'finbert': [],
            'sentiment': [],
            'technical': [],
            'strategy': [],
            'total': []
        }
    
    def _initialize_finbert_safe(self) -> None:
        """Initialize FinBERT with safe error handling."""
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
        """Enhanced news analysis with rate-limit optimized processing."""
        if news_df is None or news_df.empty:
            return []
        
        log_info(f"Analyzing {len(news_df)} articles with rate-limit optimized pipeline")
        
        # Create price lookup for efficiency
        price_lookup = self._create_price_lookup_safe(current_prices)
        
        # Process in optimized batches
        results = []
        for i in range(0, len(news_df), self.BATCH_SIZE):
            batch_df = news_df.iloc[i:i + self.BATCH_SIZE]
            batch_results = self._process_batch_optimized(batch_df, price_lookup)
            results.extend([r for r in batch_results if r is not None])
        
        # Filter and log results
        high_confidence_results = [
            analysis for analysis in results 
            if analysis.combined_confidence >= CONFIG.min_confidence_score
        ]
        
        if high_confidence_results:
            log_info(f"Generated {len(high_confidence_results)} high-confidence analyses")
            self._log_top_results_optimized(high_confidence_results[:3])
        
        self.analysis_count += len(results)
        return results
    
    def _create_price_lookup_safe(self, current_prices: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Create price lookup with safe error handling."""
        try:
            if current_prices is None or current_prices.empty:
                return {}
            
            price_dict = current_prices.set_index('symbol').to_dict('index')
            return {str(k): dict(v) for k, v in price_dict.items()}
        except Exception as e:
            log_debug(f"Error creating price lookup: {e}")
            return {}
    
    def _process_batch_optimized(self, batch_df: pd.DataFrame, 
                                price_lookup: Dict[str, Dict[str, Any]]) -> List[Optional[EnhancedNewsAnalysis]]:
        """Process batch with rate-limit optimized processing."""
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
                        'row_data': row
                    })
            except Exception as e:
                log_debug(f"Error preparing batch data: {e}")
                continue
        
        if not valid_data:
            return [None] * len(batch_df)
        
        # Batch FinBERT processing
        finbert_results = self._process_finbert_batch_safe([item['full_text'] for item in valid_data])
        
        # Process individual articles with optimized processing
        results = []
        for i, data in enumerate(valid_data):
            try:
                finbert_sentiment, finbert_conf = finbert_results[i] if i < len(finbert_results) else (0.0, 0.0)
                analysis = self._process_single_article_optimized(
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
            
            # Inference with safe context
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
    
    def _process_single_article_optimized(self, data: Dict, price_lookup: Dict[str, Dict[str, Any]],
                                         finbert_sentiment: float, finbert_conf: float) -> Optional[EnhancedNewsAnalysis]:
        """Process single article with rate-limit optimized processing."""
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
            
            # SIMPLIFIED: Extract enhanced data from source only (no API calls)
            enhanced_data = self._extract_enhanced_data_simple(row_data)
            
            # SIMPLIFIED: Get social sentiment from embedded data only
            social_sentiment = self._get_social_sentiment_embedded(enhanced_data)
            
            # SIMPLIFIED: Get analyst data from embedded data only
            analyst_data = self._get_analyst_data_embedded(enhanced_data)
            
            # Gemini analysis (with built-in rate limiting)
            gemini_sentiment, gemini_conf, gemini_reasoning = self._get_gemini_analysis_safe(
                symbol, title, content, current_price, topic
            )
            
            # Simplified ensemble sentiment calculation (no external API calls)
            final_sentiment, news_confidence = self._calculate_simplified_ensemble_sentiment(
                finbert_sentiment, finbert_conf,
                enhanced_score.final_sentiment if enhanced_score else 0.0,
                enhanced_score.quality_confidence if enhanced_score else 0.0,
                gemini_sentiment, gemini_conf
            )
            
            # Technical analysis
            technical_analysis = self._get_technical_analysis_safe(symbol, price_data)
            
            # Strategy analysis
            strategy_signals, best_strategy = self._get_strategy_analysis_safe(
                symbol, current_price, technical_analysis, final_sentiment
            )
            
            # Simplified combined confidence calculation
            combined_confidence = self._calculate_simplified_combined_confidence(
                news_confidence, technical_analysis, final_sentiment, strategy_signals
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
                
                # Simplified enhanced fields (no API calls)
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
            log_error(f"Error processing optimized single article: {e}")
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
    
    def _extract_enhanced_data_simple(self, row_data: pd.Series) -> Dict[str, Any]:
        """Extract enhanced data from row only (no API calls)."""
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
            
        except Exception as e:
            log_debug(f"Error extracting enhanced data: {e}")
        
        return enhanced_data
    
    def _get_social_sentiment_embedded(self, enhanced_data: Dict) -> Dict[str, Any]:
        """Get social sentiment from embedded data only (no API calls)."""
        try:
            # Only use embedded data - no API calls to avoid 429 errors
            if 'social_sentiment_score' in enhanced_data:
                return {
                    'score': enhanced_data['social_sentiment_score'],
                    'label': enhanced_data.get('social_sentiment_label', ''),
                    'source': 'embedded'
                }
            
            return {'score': 0.0, 'label': '', 'source': 'none'}
            
        except Exception as e:
            log_debug(f"Error getting embedded social sentiment: {e}")
            return {'score': 0.0, 'label': '', 'source': 'error'}
    
    def _get_analyst_data_embedded(self, enhanced_data: Dict) -> Dict[str, Any]:
        """Get analyst data from embedded data only (no API calls)."""
        try:
            # Only use embedded data - no API calls to avoid 429 errors
            if 'analyst_action' in enhanced_data:
                return {
                    'action': enhanced_data.get('analyst_action', ''),
                    'grade_new': enhanced_data.get('analyst_grade_new', ''),
                    'grade_previous': enhanced_data.get('analyst_grade_previous', ''),
                    'price_target': enhanced_data.get('price_target', 0.0),
                    'price_target_change': enhanced_data.get('price_target_change', 0.0),
                    'source': 'embedded'
                }
            
            return {'action': '', 'grade_new': '', 'grade_previous': '', 
                   'price_target': 0.0, 'price_target_change': 0.0, 'source': 'none'}
            
        except Exception as e:
            log_debug(f"Error getting embedded analyst data: {e}")
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
    
    def _calculate_simplified_ensemble_sentiment(self, finbert_sentiment: float, finbert_conf: float,
                                                enhanced_sentiment: float, enhanced_conf: float,
                                                gemini_sentiment: float, gemini_conf: float) -> Tuple[float, float]:
        """Simplified ensemble sentiment calculation without external API calls."""
        try:
            # Simplified weights (no external sources)
            weights = {
                'finbert': CONFIG.finbert_weight,
                'enhanced': CONFIG.keyword_weight,
                'gemini': CONFIG.gemini_weight
            }
            
            # Collect sources
            sources = [
                (finbert_sentiment, finbert_conf, weights['finbert']),
                (enhanced_sentiment, enhanced_conf, weights['enhanced']),
                (gemini_sentiment, gemini_conf, weights['gemini'])
            ]
            
            # Calculate ensemble
            valid_sources = [(s, c, w) for s, c, w in sources if c > 0]
            
            if not valid_sources:
                return 0.0, 0.0
            
            if len(valid_sources) == 1:
                return valid_sources[0][0], valid_sources[0][1]
            
            # Weighted ensemble calculation
            numerator = sum(s * w * c for s, c, w in valid_sources)
            denominator = sum(w * c for s, c, w in valid_sources)
            
            ensemble_sentiment = numerator / denominator if denominator > 0 else 0.0
            
            # Simplified confidence calculation
            base_confidence = sum(c for s, c, w in valid_sources) / len(valid_sources)
            source_bonus = 1.0 + (len(valid_sources) - 1) * 0.1
            
            final_confidence = min(base_confidence * source_bonus, 1.0)
            
            return ensemble_sentiment, final_confidence
            
        except Exception as e:
            log_debug(f"Error in simplified ensemble sentiment calculation: {e}")
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
    
    def _calculate_simplified_combined_confidence(self, news_confidence: float, 
                                                technical_analysis: Optional[TechnicalAnalysis],
                                                sentiment_score: float,
                                                strategy_signals: Optional[List[StrategySignal]]) -> float:
        """Simplified combined confidence calculation."""
        try:
            if technical_analysis is None:
                return news_confidence * 0.7
            
            tech_confidence = technical_analysis.technical_confidence
            
            # Base combination
            base_combined = (
                news_confidence * 0.5 +
                tech_confidence * 0.4 +
                0.1  # Reserved for strategy factor
            )
            
            # Strategy alignment
            if strategy_signals:
                best_signal = max(strategy_signals, key=lambda x: (x.strength.value, x.confidence))
                signal_direction = 1 if best_signal.signal_type == "long" else -1
                sentiment_direction = 1 if sentiment_score > 0 else -1
                
                if signal_direction == sentiment_direction:
                    base_combined *= 1.1
                else:
                    base_combined *= 0.95
            
            return min(base_combined, 1.0)
            
        except Exception as e:
            log_debug(f"Error calculating simplified combined confidence: {e}")
            return news_confidence * 0.7 if news_confidence else 0.0
    
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
    
    def _log_top_results_optimized(self, analyses: List[EnhancedNewsAnalysis]) -> None:
        """Log top analysis results with simplified information."""
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
                    
                    log_info(
                        f"Top #{i}: {analysis.symbol} sentiment={analysis.sentiment_score:.3f} "
                        f"conf={analysis.combined_confidence:.3f} {tech_info} {strategy_info} "
                        f"topic={analysis.topic} source={analysis.source_type}"
                    )
                except Exception as e:
                    log_debug(f"Error logging optimized result #{i}: {e}")
        except Exception as e:
            log_debug(f"Error in optimized top results logging: {e}")
    
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
                log_info(f"Filtered to {len(high_conf)} high-confidence signals (optimized)")
            
            return high_conf
        except Exception as e:
            log_error(f"Error filtering high confidence analyses: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get optimized performance statistics."""
        try:
            return {
                'total_analyses': self.analysis_count,
                'cache_size': len(self.recent_analyses_cache),
                'cache_hits': self.cache_hits,
                'finbert_enabled': self.finbert_model_instance is not None,
                'gemini_enabled': self.gemini_analyzer.enabled,
                'enhanced_data_enabled': self.enhanced_data_enabled,
                'device': self.device,
                'batch_size': self.BATCH_SIZE,
                'max_cache_size': self.MAX_CACHE_SIZE,
                'rate_limit_optimized': True
            }
        except Exception as e:
            log_error(f"Error getting optimized performance stats: {e}")
            return {'error': str(e)}