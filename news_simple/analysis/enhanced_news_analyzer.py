"""
Enhanced news analyzer using modular sentiment components with performance optimizations
"""
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timezone
from functools import lru_cache
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning, log_debug
from .technical_analyzer import TechnicalAnalyzer, TechnicalAnalysis
from .gemini_news_analyzer import OptimizedGeminiNewsAnalyzer, GeminiAnalysis
from .technical_strategies import TechnicalStrategies, StrategySignal
from .sentiment.enhanced_sentiment_scorer import EnhancedSentimentScorer, SentimentScore


@dataclass
class EnhancedNewsAnalysis:
    """Enhanced news analysis with comprehensive data"""
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


class EnhancedNewsAnalyzer:
    """Enhanced news analyzer with modular components and performance optimizations"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with modular components and optimized caching."""
        self.fmp_loader = fmp_loader
        self.technical_analyzer = TechnicalAnalyzer(fmp_loader)
        self.technical_strategies = TechnicalStrategies(fmp_loader)
        self.gemini_analyzer = OptimizedGeminiNewsAnalyzer()
        self.sentiment_scorer = EnhancedSentimentScorer()
        
        # Enhanced caching
        self.recent_analyses_cache = []
        self.max_cache_size = 100  # Increased cache size
        self.batch_size = 10  # Process in batches for better performance
        
        # Device and FinBERT setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.finbert_labels = ["positive", "negative", "neutral"]
        
        # Performance tracking
        self.analysis_count = 0
        self.cache_hits = 0
        
        self._load_finbert()
        self._initialize_topic_patterns()
    
    def _load_finbert(self) -> None:
        """Load FinBERT model with optimizations for performance."""
        try:
            log_info("Loading FinBERT model...")
            model_name = "ProsusAI/finbert"
            cache_dir = "./cache"
            
            # Load tokenizer with optimizations
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                use_fast=True  # Use fast tokenizer for better performance
            )
            
            # Load model with optimizations
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.finbert_model.to(self.device)
            self.finbert_model.eval()
            
            # Optimize for inference
            if self.device == "cuda":
                try:
                    # Enable CUDA optimizations
                    torch.backends.cudnn.benchmark = True
                    if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
                        self.finbert_model = torch.compile(self.finbert_model, mode="reduce-overhead")
                except Exception as e:
                    log_warning(f"Could not apply CUDA optimizations: {e}")
            
            log_info(f"FinBERT loaded successfully on {self.device} with optimizations")
            
        except Exception as e:
            log_error(f"Failed to load FinBERT: {e}")
            self.finbert_model = None
            self.finbert_tokenizer = None
    
    def _initialize_topic_patterns(self) -> None:
        """Initialize topic detection patterns with optimized regex compilation."""
        self._topic_patterns = {
            'earnings': re.compile(
                r'\b(earnings|revenue|profit|eps|quarterly|guidance|beat|miss|results)\b', 
                re.IGNORECASE
            ),
            'biotech': re.compile(
                r'\b(fda|approval|phase|trial|clinical|drug|therapy|efficacy|breakthrough|orphan)\b', 
                re.IGNORECASE
            ),
            'ma': re.compile(
                r'\b(merger|acquisition|deal|buyout|takeover|acquire|merge|purchase)\b', 
                re.IGNORECASE
            ),
            'analyst': re.compile(
                r'\b(upgrade|downgrade|target|analyst|rating|price target|initiat)\b', 
                re.IGNORECASE
            ),
            'corporate_action': re.compile(
                r'\b(dividend|buyback|split|spinoff|distribution|repurchase)\b', 
                re.IGNORECASE
            ),
            'business_development': re.compile(
                r'\b(contract|partnership|agreement|collaboration|alliance|licensing)\b', 
                re.IGNORECASE
            )
        }
    
    def _get_finbert_sentiment_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Get FinBERT sentiment analysis for batch of texts (performance optimization)."""
        if not self.finbert_model or not self.finbert_tokenizer or not texts:
            return [(0.0, 0.0)] * len(texts)
        
        try:
            # Clean and truncate texts
            cleaned_texts = [text.strip()[:384] for text in texts if text.strip()]
            if not cleaned_texts:
                return [(0.0, 0.0)] * len(texts)
            
            # Batch tokenization
            inputs = self.finbert_tokenizer(
                cleaned_texts,
                return_tensors="pt",
                max_length=384,
                truncation=True,
                padding=True,
                add_special_tokens=True
            ).to(self.device)
            
            results = []
            with torch.no_grad():
                if hasattr(torch, 'inference_mode'):
                    with torch.inference_mode():
                        outputs = self.finbert_model(**inputs)
                else:
                    outputs = self.finbert_model(**inputs)
                
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions_np = predictions.cpu().float().numpy()
                
                for probs in predictions_np:
                    pos_prob, neg_prob, _ = probs
                    sentiment = float(pos_prob - neg_prob)
                    confidence = float(max(probs))
                    results.append((sentiment, confidence))
            
            # Pad results if needed
            while len(results) < len(texts):
                results.append((0.0, 0.0))
            
            return results
            
        except Exception as e:
            log_error(f"FinBERT batch analysis error: {e}")
            return [(0.0, 0.0)] * len(texts)
    
    def _get_finbert_sentiment(self, text: str) -> Tuple[float, float]:
        """Get FinBERT sentiment analysis for single text (fallback method)."""
        if not self.finbert_model or not self.finbert_tokenizer:
            return 0.0, 0.0
        
        results = self._get_finbert_sentiment_batch([text])
        return results[0] if results else (0.0, 0.0)
    
    @lru_cache(maxsize=1000)
    def _detect_topic(self, text: str) -> str:
        """Detect article topic using cached patterns with performance optimization."""
        text_lower = text.lower()
        
        # Score each topic based on number of matches
        topic_scores = {}
        for topic, pattern in self._topic_patterns.items():
            matches = len(pattern.findall(text_lower))
            if matches > 0:
                topic_scores[topic] = matches
        
        # Return topic with highest score, or 'general' if no matches
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _enhanced_ensemble_score(self, finbert_sentiment: float, finbert_conf: float,
                                enhanced_sentiment: float, enhanced_conf: float,
                                gemini_sentiment: float, gemini_conf: float) -> Tuple[float, float]:
        """Calculate ensemble score from multiple sentiment sources with optimized weights."""
        
        # Pre-computed weights for performance
        finbert_weight = CONFIG.finbert_weight
        enhanced_weight = CONFIG.keyword_weight * 1.15
        gemini_weight = CONFIG.gemini_weight
        
        # Fast path for single source
        sources = [(finbert_sentiment, finbert_conf, finbert_weight),
                  (enhanced_sentiment, enhanced_conf, enhanced_weight),
                  (gemini_sentiment, gemini_conf, gemini_weight)]
        
        valid_sources = [(s, c, w) for s, c, w in sources if c > 0]
        
        if not valid_sources:
            return 0.0, 0.0
        
        if len(valid_sources) == 1:
            return valid_sources[0][0], valid_sources[0][1]
        
        # Calculate weighted sentiment
        weighted_sentiment = sum(s * w * c for s, c, w in valid_sources)
        total_weight = sum(w * c for s, c, w in valid_sources)
        
        ensemble_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence with bonuses
        base_confidence = sum(c for s, c, w in valid_sources) / len(valid_sources)
        
        # Quality bonus (enhanced sentiment typically has better quality metrics)
        quality_bonus = 1.0 + (enhanced_conf * 0.25)
        
        # Agreement bonus (when multiple sources agree on direction)
        sentiments = [s for s, c, w in valid_sources]
        if len(sentiments) >= 2:
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            agreement_ratio = max(positive_count, negative_count) / len(sentiments)
            agreement_bonus = 1.0 + (agreement_ratio * 0.25)
        else:
            agreement_bonus = 1.0
        
        # Gemini bonus for high-confidence Gemini predictions
        gemini_bonus = 1.15 if gemini_conf > 0.65 else 1.0
        
        final_confidence = min(
            base_confidence * quality_bonus * agreement_bonus * gemini_bonus, 
            1.0
        )
        
        return ensemble_sentiment, final_confidence
    
    def _combine_news_technical_confidence(self, news_confidence: float, 
                                         technical_analysis: Optional[TechnicalAnalysis],
                                         sentiment_score: float,
                                         strategy_signals: Optional[List[StrategySignal]] = None) -> float:
        """Combine news and technical confidence scores with optimized calculations."""
        if technical_analysis is None:
            return news_confidence * 0.65
        
        tech_confidence = technical_analysis.technical_confidence
        
        # Vectorized momentum alignment calculation
        sentiment_direction = 1 if sentiment_score > 0 else -1
        momentum_direction = 1 if technical_analysis.momentum_score > 0 else -1
        
        # Momentum alignment scoring
        if abs(sentiment_score) > 0.25 and abs(technical_analysis.momentum_score) > 0.15:
            momentum_alignment = 1.2 if sentiment_direction == momentum_direction else 0.75
        elif abs(sentiment_score) > 0.3 and abs(technical_analysis.momentum_score) > 0.2:
            momentum_alignment = 0.75 if sentiment_direction != momentum_direction else 1.0
        else:
            momentum_alignment = 1.0
        
        # Volume and liquidity factors
        volume_bonus = 1.08 if technical_analysis.volume_score > 0.65 else 1.0
        liquidity_factor = max(technical_analysis.liquidity_score, 0.25)
        
        # Strategy alignment bonus
        strategy_bonus = 1.0
        if strategy_signals:
            best_signal = max(strategy_signals, key=lambda x: (x.strength.value, x.confidence))
            strategy_direction = 1 if best_signal.signal_type == "long" else -1
            
            if strategy_direction == sentiment_direction:
                base_bonus = 1.0 + (best_signal.confidence * 0.25)
                strength_bonus = 1.05 if best_signal.strength.value >= 4 else 1.0
                strategy_bonus = base_bonus * strength_bonus
            else:
                strategy_bonus = 0.85
        
        # Optimized weighted combination
        combined = (
            news_confidence * 0.48 +
            tech_confidence * 0.28 +
            0.24 * strategy_bonus
        ) * momentum_alignment * volume_bonus * liquidity_factor
        
        return min(combined, 1.0)
    
    def analyze_news_with_technical(self, news_df: pd.DataFrame, 
                                  current_prices: pd.DataFrame) -> List[EnhancedNewsAnalysis]:
        """Analyze news with technical analysis using optimized batch processing."""
        if news_df is None or news_df.empty:
            return []
        
        log_info(f"Analyzing {len(news_df)} news articles with Enhanced Sentiment + Technical Analysis")
        
        # Create optimized price lookup
        price_lookup = {}
        if current_prices is not None and not current_prices.empty:
            price_lookup = current_prices.set_index('symbol').to_dict('index')
        
        # Process in batches for better performance
        results: List[EnhancedNewsAnalysis] = []
        batch_size = min(self.batch_size, len(news_df))
        
        for i in range(0, len(news_df), batch_size):
            batch_df = news_df.iloc[i:i + batch_size]
            batch_results = self._process_article_batch(batch_df, price_lookup)
            
            # Filter out None results and add valid analyses
            valid_results = [analysis for analysis in batch_results if analysis is not None]
            results.extend(valid_results)
            
            # Log progress for large batches
            if len(news_df) > 20:
                log_debug(f"Processed batch {i//batch_size + 1}/{(len(news_df)-1)//batch_size + 1}")
        
        # Filter and log high confidence results
        high_confidence_results = [
            analysis for analysis in results 
            if analysis.combined_confidence >= CONFIG.min_confidence_score
        ]
        
        if high_confidence_results:
            log_info(f"Generated {len(high_confidence_results)} high-confidence analyses")
            for analysis in high_confidence_results[:3]:  # Log top 3
                self._log_high_confidence_result(analysis)
        
        self.analysis_count += len(results)
        return results
    
    def _process_article_batch(self, batch_df: pd.DataFrame, price_lookup: Dict) -> List[Optional[EnhancedNewsAnalysis]]:
        """Process a batch of articles with optimized operations."""
        # Extract data for batch processing
        batch_data = []
        valid_indices = []
        
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            symbol = str(row.get('symbol', '')).strip()
            title = str(row.get('title', '')).strip()
            content = str(row.get('content', '')).strip()
            
            if all([symbol, title, content]):
                batch_data.append({
                    'symbol': symbol,
                    'title': title,
                    'content': content,
                    'source': str(row.get('source', '')).strip(),
                    'full_text': f"{title} {content}"
                })
                valid_indices.append(idx)
        
        if not batch_data:
            return [None] * len(batch_df)
        
        # Batch FinBERT analysis
        full_texts = [item['full_text'] for item in batch_data]
        finbert_results = self._get_finbert_sentiment_batch(full_texts)
        
        # Process individual articles with pre-computed FinBERT results
        # Fix: Properly type the results list to handle Optional[EnhancedNewsAnalysis]
        results: List[Optional[EnhancedNewsAnalysis]] = [None] * len(batch_df)
        
        for i, (batch_idx, data) in enumerate(zip(valid_indices, batch_data)):
            try:
                finbert_sentiment, finbert_conf = finbert_results[i]
                analysis = self._process_single_article_with_finbert(
                    data, price_lookup, finbert_sentiment, finbert_conf
                )
                results[batch_idx] = analysis
            except Exception as e:
                log_error(f"Error processing article for {data['symbol']}: {e}")
                results[batch_idx] = None
        
        return results
    
    def _process_single_article_with_finbert(self, data: Dict, price_lookup: Dict,
                                           finbert_sentiment: float, finbert_conf: float) -> Optional[EnhancedNewsAnalysis]:
        """Process single article with pre-computed FinBERT results."""
        symbol = data['symbol']
        title = data['title']
        content = data['content']
        source = data['source']
        full_text = data['full_text']
        
        # Get price data and topic
        price_data = price_lookup.get(symbol)
        current_price = float(price_data.get('lastSalePrice', 0)) if price_data else 0.0
        topic = self._detect_topic(full_text)
        
        # Enhanced sentiment analysis using modular scorer
        enhanced_score = self.sentiment_scorer.calculate_enhanced_sentiment(
            title=title, content=content, symbol=symbol,
            current_price=current_price, topic=topic,
            source=source, recent_analyses=self.recent_analyses_cache[-20:]
        )
        
        # Gemini analysis (with rate limiting built-in)
        gemini_sentiment, gemini_conf, gemini_reasoning = self._perform_gemini_analysis(
            symbol, title, content, current_price, topic
        )
        
        # Ensemble scoring
        final_sentiment, news_confidence = self._enhanced_ensemble_score(
            finbert_sentiment, finbert_conf,
            enhanced_score.final_sentiment, enhanced_score.quality_confidence,
            gemini_sentiment, gemini_conf
        )
        
        # Technical analysis
        technical_analysis = self._perform_technical_analysis(symbol, price_data)
        
        # Strategy analysis
        strategy_signals, best_strategy = self._perform_strategy_analysis(
            symbol, current_price, technical_analysis, final_sentiment
        )
        
        # Combined confidence
        combined_confidence = self._combine_news_technical_confidence(
            news_confidence, technical_analysis, final_sentiment, strategy_signals
        )
        
        # Update cache efficiently
        self._update_cache(symbol, final_sentiment, topic, enhanced_score.quality_confidence)
        
        return EnhancedNewsAnalysis(
            symbol=symbol,
            title=title,
            content=content,
            sentiment_score=final_sentiment,
            confidence=news_confidence,
            topic=topic,
            timestamp=pd.Timestamp.now(tz=timezone.utc),
            finbert_score=finbert_sentiment,
            keyword_score=enhanced_score.final_sentiment,
            gemini_score=gemini_sentiment,
            gemini_confidence=gemini_conf,
            gemini_reasoning=gemini_reasoning,
            technical_analysis=technical_analysis,
            strategy_signals=strategy_signals or [],
            best_strategy=best_strategy,
            combined_confidence=combined_confidence
        )
    
    def _perform_gemini_analysis(self, symbol: str, title: str, content: str,
                               current_price: float, topic: str) -> Tuple[float, float, str]:
        """Perform Gemini analysis with built-in rate limiting and caching."""
        if self.gemini_analyzer.enabled and current_price > 0:
            try:
                gemini_analysis = self.gemini_analyzer.analyze_news(
                    symbol=symbol, title=title[:150], content=content[:600],
                    current_price=current_price, topic=topic
                )
                if gemini_analysis:
                    return (gemini_analysis.sentiment_score,
                           gemini_analysis.confidence,
                           gemini_analysis.reasoning[:100])
            except Exception as e:
                log_warning(f"Gemini analysis failed for {symbol}: {e}")
        
        return 0.0, 0.0, ""
    
    def _perform_technical_analysis(self, symbol: str, price_data) -> Optional[TechnicalAnalysis]:
        """Perform technical analysis with caching."""
        if price_data is not None:
            try:
                return self.technical_analyzer.analyze_symbol(symbol, pd.Series(price_data))
            except Exception as e:
                log_warning(f"Technical analysis failed for {symbol}: {e}")
        return None
    
    def _perform_strategy_analysis(self, symbol: str, current_price: float,
                                 technical_analysis: Optional[TechnicalAnalysis],
                                 sentiment_score: float) -> Tuple[List[StrategySignal], Optional[StrategySignal]]:
        """Perform strategy analysis with error handling."""
        if technical_analysis and current_price > 0:
            try:
                strategy_signals = self.technical_strategies.analyze_with_strategies(
                    symbol=symbol, current_price=current_price,
                    technical_analysis=technical_analysis, news_sentiment=sentiment_score
                )
                
                if strategy_signals:
                    best_strategy = self.technical_strategies.get_best_strategy(strategy_signals)
                    return strategy_signals, best_strategy
                    
            except Exception as e:
                log_warning(f"Strategy analysis failed for {symbol}: {e}")
        
        return [], None
    
    def _update_cache(self, symbol: str, sentiment_score: float, topic: str, quality_confidence: float) -> None:
        """Update the recent analyses cache with memory management."""
        current_time = datetime.now(timezone.utc)
        
        # Add to cache
        self.recent_analyses_cache.append({
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'timestamp': current_time,
            'topic': topic,
            'quality_confidence': quality_confidence
        })
        
        # Efficient cache management
        if len(self.recent_analyses_cache) > self.max_cache_size:
            # Remove oldest 25% of entries
            keep_count = int(self.max_cache_size * 0.75)
            self.recent_analyses_cache = self.recent_analyses_cache[-keep_count:]
            self.cache_hits += 1
    
    def _log_high_confidence_result(self, analysis: EnhancedNewsAnalysis) -> None:
        """Log high confidence analysis results with optimized formatting."""
        # Technical info
        tech_info = ""
        if analysis.technical_analysis:
            ta = analysis.technical_analysis
            tech_info = (f"tech_conf={ta.technical_confidence:.2f} "
                        f"momentum={ta.momentum_score:.2f} "
                        f"liquidity={ta.liquidity_score:.2f}")
        
        # Gemini info
        gemini_info = ""
        if analysis.gemini_confidence > 0:
            gemini_info = f"gemini={analysis.gemini_score:.2f}({analysis.gemini_confidence:.2f}) "
        
        # Strategy info
        strategy_info = ""
        if analysis.best_strategy:
            bs = analysis.best_strategy
            strategy_info = (f"strategy={bs.strategy_name}[{bs.signal_type.upper()}] "
                           f"strength={bs.strength.name} conf={bs.confidence:.2f} ")
        
        log_info(
            f"HIGH CONFIDENCE: {analysis.symbol} "
            f"sentiment={analysis.sentiment_score:.3f} "
            f"finbert={analysis.finbert_score:.2f} enhanced={analysis.keyword_score:.2f} "
            f"{gemini_info}"
            f"{strategy_info}"
            f"news_conf={analysis.confidence:.3f} "
            f"combined_conf={analysis.combined_confidence:.3f} "
            f"{tech_info} topic={analysis.topic}"
        )
        
        if analysis.gemini_reasoning:
            log_info(f"  Gemini: {analysis.gemini_reasoning[:120]}...")
        
        if analysis.best_strategy:
            strategy_summary = self.technical_strategies.format_strategy_summary(analysis.best_strategy)
            log_info(f"  Strategy: {strategy_summary}")
    
    def filter_high_confidence(self, analyses: List[EnhancedNewsAnalysis], 
                             use_combined_confidence: bool = True) -> List[EnhancedNewsAnalysis]:
        """Filter for high confidence analyses with optimized processing."""
        if not analyses:
            return []
        
        confidence_attr = 'combined_confidence' if use_combined_confidence else 'confidence'
        threshold = CONFIG.min_confidence_score
        
        # Vectorized filtering
        high_conf = [
            analysis for analysis in analyses
            if analysis and getattr(analysis, confidence_attr, 0) >= threshold
        ]
        
        if high_conf:
            # Sort by confidence for better trade selection
            high_conf.sort(key=lambda x: getattr(x, confidence_attr, 0), reverse=True)
            log_info(f"Filtered to {len(high_conf)} high-confidence signals")
        else:
            self._log_detailed_signal_analysis(analyses, confidence_attr, threshold)
        
        return high_conf
    
    def _log_detailed_signal_analysis(self, analyses: List[EnhancedNewsAnalysis], 
                                    confidence_attr: str, threshold: float) -> None:
        """Log detailed analysis for debugging with optimized output."""
        log_info("=" * 60)
        log_info("DEBUG: SIGNAL ANALYSIS")
        log_info("=" * 60)
        
        if not analyses:
            log_info("DEBUG: No analyses to filter")
            return
        
        # Get valid analyses only
        valid_analyses = [a for a in analyses if a is not None]
        if not valid_analyses:
            log_info("DEBUG: No valid analyses")
            return
        
        scores = [getattr(a, confidence_attr, 0) for a in valid_analyses]
        max_score = max(scores) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        
        log_info(f"THRESHOLD: {threshold:.3f} | HIGHEST {confidence_attr.upper()}: {max_score:.3f} | AVG: {avg_score:.3f}")
        
        # Log top 5 analyses
        top_analyses = sorted(valid_analyses, key=lambda x: getattr(x, confidence_attr, 0), reverse=True)[:5]
        
        for i, analysis in enumerate(top_analyses, 1):
            log_info(f"\nSIGNAL {i}: {analysis.symbol}")
            log_info(f"   Title: {analysis.title[:60]}...")
            log_info(f"   Topic: {analysis.topic}")
            log_info(f"   Sentiment: {analysis.sentiment_score:.3f}")
            log_info(f"   News Confidence: {analysis.confidence:.3f}")
            
            if analysis.technical_analysis:
                log_info(f"   Technical Confidence: {analysis.technical_analysis.technical_confidence:.3f}")
            else:
                log_info("   Technical Analysis: MISSING")
            
            combined_conf = getattr(analysis, confidence_attr, 0)
            log_info(f"   {confidence_attr.upper()}: {combined_conf:.3f}")
            
            # Diagnostic info
            if combined_conf < threshold:
                issues = []
                if analysis.confidence < 0.45:
                    issues.append("Low news confidence")
                if analysis.technical_analysis and analysis.technical_analysis.technical_confidence < 0.25:
                    issues.append("Low technical confidence")
                if abs(analysis.sentiment_score) < 0.18:
                    issues.append("Weak sentiment")
                
                if issues:
                    log_info(f"   ISSUES: {', '.join(issues)}")
        
        log_info("=" * 60)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get analyzer performance statistics."""
        return {
            'total_analyses': self.analysis_count,
            'cache_size': len(self.recent_analyses_cache),
            'cache_hits': self.cache_hits,
            'finbert_enabled': self.finbert_model is not None,
            'gemini_enabled': self.gemini_analyzer.enabled,
            'device': self.device
        }


# Create alias for backwards compatibility
OptimizedEnhancedNewsAnalyzer = EnhancedNewsAnalyzer