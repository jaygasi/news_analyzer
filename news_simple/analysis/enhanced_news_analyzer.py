"""
Enhanced news analyzer using modular sentiment components
"""
import pandas as pd
from typing import Dict, Optional, Tuple, List
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
    """Enhanced news analyzer with modular components"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with modular components"""
        self.fmp_loader = fmp_loader
        self.technical_analyzer = TechnicalAnalyzer(fmp_loader)
        self.technical_strategies = TechnicalStrategies(fmp_loader)
        self.gemini_analyzer = OptimizedGeminiNewsAnalyzer()
        self.sentiment_scorer = EnhancedSentimentScorer()
        self.recent_analyses_cache = []
        self.max_cache_size = 50
        
        # Device and FinBERT setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.finbert_labels = ["positive", "negative", "neutral"]
        
        self._load_finbert()
        self._initialize_topic_patterns()
    
    def _load_finbert(self) -> None:
        """Load FinBERT model with optimizations"""
        try:
            log_info("Loading FinBERT model...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                "ProsusAI/finbert", cache_dir="./cache"
            )
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert", cache_dir="./cache"
            )
            self.finbert_model.to(self.device)
            self.finbert_model.eval()
            
            # Optimize for inference
            if hasattr(torch, 'jit') and self.device == "cuda":
                self.finbert_model = torch.jit.optimize_for_inference(self.finbert_model)
            
            if self.device == "cuda":
                self.finbert_model = self.finbert_model.half()
            
            log_info(f"FinBERT loaded successfully on {self.device}")
            
        except Exception as e:
            log_error(f"Failed to load FinBERT: {e}")
            self.finbert_model = None
            self.finbert_tokenizer = None
    
    def _initialize_topic_patterns(self) -> None:
        """Initialize topic detection patterns"""
        self._topic_patterns = {
            'earnings': re.compile(r'\b(earnings|revenue|profit|eps|quarterly|guidance|beat|miss)\b', re.IGNORECASE),
            'biotech': re.compile(r'\b(fda|approval|phase|trial|clinical|drug|therapy|efficacy)\b', re.IGNORECASE),
            'ma': re.compile(r'\b(merger|acquisition|deal|buyout|takeover|acquire|merge)\b', re.IGNORECASE),
            'analyst': re.compile(r'\b(upgrade|downgrade|target|analyst|rating|price target)\b', re.IGNORECASE),
            'corporate_action': re.compile(r'\b(dividend|buyback|split|spinoff|distribution)\b', re.IGNORECASE),
            'business_development': re.compile(r'\b(contract|partnership|agreement|collaboration|alliance)\b', re.IGNORECASE)
        }
    
    def _get_finbert_sentiment(self, text: str) -> Tuple[float, float]:
        """Get FinBERT sentiment analysis"""
        if not self.finbert_model or not self.finbert_tokenizer:
            return 0.0, 0.0
        
        try:
            text = text.strip()[:384]
            
            if not text:
                return 0.0, 0.0
            
            inputs = self.finbert_tokenizer(
                text, return_tensors="pt", max_length=384,
                truncation=True, padding=False, add_special_tokens=True
            ).to(self.device)
            
            with torch.no_grad():
                if hasattr(torch, 'inference_mode'):
                    with torch.inference_mode():
                        outputs = self.finbert_model(**inputs)
                else:
                    outputs = self.finbert_model(**inputs)
                
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = predictions.cpu().float().numpy()[0]
            pos_prob, neg_prob, _ = probs
            
            sentiment = float(pos_prob - neg_prob)
            confidence = float(max(probs))
            
            return sentiment, confidence
            
        except Exception as e:
            log_error(f"FinBERT analysis error: {e}")
            return 0.0, 0.0
    
    @lru_cache(maxsize=500)
    def _detect_topic(self, text: str) -> str:
        """Detect article topic using cached patterns"""
        text_lower = text.lower()
        
        for topic, pattern in self._topic_patterns.items():
            if pattern.search(text_lower):
                return topic
        
        return 'general'
    
    def _enhanced_ensemble_score(self, finbert_sentiment: float, finbert_conf: float,
                                enhanced_sentiment: float, enhanced_conf: float,
                                gemini_sentiment: float, gemini_conf: float) -> Tuple[float, float]:
        """Calculate ensemble score from multiple sentiment sources"""
        
        # Pre-computed weights
        finbert_weight = CONFIG.finbert_weight
        enhanced_weight = CONFIG.keyword_weight * 1.15
        gemini_weight = CONFIG.gemini_weight
        
        # Calculate weighted sentiment
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        if finbert_conf > 0.0:
            weight = finbert_weight * finbert_conf
            weighted_sentiment += finbert_sentiment * weight
            total_weight += weight
        
        if enhanced_conf > 0.0:
            weight = enhanced_weight * enhanced_conf
            weighted_sentiment += enhanced_sentiment * weight
            total_weight += weight
        
        if gemini_conf > 0.0:
            weight = gemini_weight * gemini_conf
            weighted_sentiment += gemini_sentiment * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0, 0.0
        
        ensemble_sentiment = weighted_sentiment / total_weight
        
        # Calculate confidence
        components = [conf for conf in [finbert_conf, enhanced_conf, gemini_conf] if conf > 0]
        if not components:
            return ensemble_sentiment, 0.0
        
        base_confidence = sum(components) / len(components)
        quality_bonus = 1.0 + (enhanced_conf * 0.25)
        
        # Agreement bonus
        sentiments = [s for s, c in [(finbert_sentiment, finbert_conf), 
                                   (enhanced_sentiment, enhanced_conf),
                                   (gemini_sentiment, gemini_conf)] if c > 0]
        
        agreement_bonus = 1.25 if len(sentiments) >= 2 and all(
            (s > 0) == (sentiments[0] > 0) for s in sentiments
        ) else 1.0
        
        gemini_bonus = 1.15 if gemini_conf > 0.65 else 1.0
        
        final_confidence = min(base_confidence * quality_bonus * agreement_bonus * gemini_bonus, 1.0)
        
        return ensemble_sentiment, final_confidence
    
    def _combine_news_technical_confidence(self, news_confidence: float, 
                                         technical_analysis: Optional[TechnicalAnalysis],
                                         sentiment_score: float,
                                         strategy_signals: Optional[List[StrategySignal]] = None) -> float:
        """Combine news and technical confidence scores"""
        if technical_analysis is None:
            return news_confidence * 0.65
        
        tech_confidence = technical_analysis.technical_confidence
        
        # Momentum alignment
        momentum_alignment = 1.2 if (
            abs(sentiment_score) > 0.25 and abs(technical_analysis.momentum_score) > 0.15 and
            (sentiment_score > 0) == (technical_analysis.momentum_score > 0)
        ) else (0.75 if abs(sentiment_score) > 0.3 and abs(technical_analysis.momentum_score) > 0.2 and
                (sentiment_score > 0) != (technical_analysis.momentum_score > 0) else 1.0)
        
        volume_bonus = 1.08 if technical_analysis.volume_score > 0.65 else 1.0
        liquidity_factor = max(technical_analysis.liquidity_score, 0.25)
        
        # Strategy bonus
        strategy_bonus = 1.0
        if strategy_signals:
            best_signal = max(strategy_signals, key=lambda x: (x.strength.value, x.confidence))
            strategy_direction = 1 if best_signal.signal_type == "long" else -1
            sentiment_direction = 1 if sentiment_score > 0 else -1
            
            if strategy_direction == sentiment_direction:
                strategy_bonus = 1.0 + (best_signal.confidence * 0.25)
                if best_signal.strength.value >= 4:
                    strategy_bonus *= 1.05
            else:
                strategy_bonus = 0.85
        
        combined = (
            news_confidence * 0.48 +
            tech_confidence * 0.28 +
            0.24 * strategy_bonus
        ) * momentum_alignment * volume_bonus * liquidity_factor
        
        return min(combined, 1.0)
    
    def analyze_news_with_technical(self, news_df: pd.DataFrame, 
                                  current_prices: pd.DataFrame) -> List[EnhancedNewsAnalysis]:
        """Analyze news with technical analysis using modular components"""
        if news_df is None or news_df.empty:
            return []
        
        log_info(f"Analyzing {len(news_df)} news articles with Enhanced Sentiment + Technical Analysis")
        
        # Create price lookup
        price_lookup = (
            current_prices.set_index('symbol').to_dict('index') 
            if current_prices is not None and not current_prices.empty 
            else {}
        )
        
        results = []
        for _, row in news_df.iterrows():
            try:
                analysis = self._process_single_article(row, price_lookup)
                if analysis:
                    results.append(analysis)
                    
                    if analysis.combined_confidence >= CONFIG.min_confidence_score:
                        self._log_high_confidence_result(analysis)
                        
            except Exception as e:
                symbol = row.get('symbol', 'UNKNOWN')
                log_error(f"Error analyzing news for {symbol}: {e}")
                continue
        
        return results
    
    def _process_single_article(self, row: pd.Series, price_lookup: Dict) -> Optional[EnhancedNewsAnalysis]:
        """Process single article with modular components"""
        # Extract essential fields
        symbol = str(row.get('symbol', '')).strip()
        title = str(row.get('title', '')).strip()
        content = str(row.get('content', '')).strip()
        
        if not all([symbol, title, content]):
            return None
        
        # Get price data and topic
        price_data = price_lookup.get(symbol)
        current_price = float(price_data.get('lastSalePrice', 0)) if price_data else 0.0
        topic = self._detect_topic(f"{title} {content}")
        source = str(row.get('source', '')).strip()
        
        # Sentiment analysis
        full_text = f"{title} {content}"
        
        # FinBERT analysis
        finbert_sentiment, finbert_conf = self._get_finbert_sentiment(full_text)
        
        # Enhanced sentiment analysis using modular scorer
        enhanced_score = self.sentiment_scorer.calculate_enhanced_sentiment(
            title=title, content=content, symbol=symbol,
            current_price=current_price, topic=topic,
            source=source, recent_analyses=self.recent_analyses_cache[-10:]
        )
        
        # Gemini analysis
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
        
        # Update cache
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
        """Perform Gemini analysis with rate limiting"""
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
        """Perform technical analysis"""
        if price_data is not None:
            return self.technical_analyzer.analyze_symbol(symbol, pd.Series(price_data))
        return None
    
    def _perform_strategy_analysis(self, symbol: str, current_price: float,
                                 technical_analysis: Optional[TechnicalAnalysis],
                                 sentiment_score: float) -> Tuple[List[StrategySignal], Optional[StrategySignal]]:
        """Perform strategy analysis"""
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
        """Update the recent analyses cache"""
        current_time = datetime.now(timezone.utc)
        self.recent_analyses_cache.append({
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'timestamp': current_time,
            'topic': topic,
            'quality_confidence': quality_confidence
        })
        
        # Keep cache size under control
        if len(self.recent_analyses_cache) > self.max_cache_size:
            self.recent_analyses_cache = self.recent_analyses_cache[-self.max_cache_size//2:]
    
    def _log_high_confidence_result(self, analysis: EnhancedNewsAnalysis) -> None:
        """Log high confidence analysis results"""
        # Technical info
        tech_info = ""
        if analysis.technical_analysis:
            tech_info = (f"tech_conf={analysis.technical_analysis.technical_confidence:.2f} "
                        f"momentum={analysis.technical_analysis.momentum_score:.2f} "
                        f"liquidity={analysis.technical_analysis.liquidity_score:.2f}")
        
        # Gemini info
        gemini_info = ""
        if analysis.gemini_confidence > 0:
            gemini_info = f"gemini={analysis.gemini_score:.2f}({analysis.gemini_confidence:.2f}) "
        
        # Strategy info
        strategy_info = ""
        if analysis.best_strategy:
            strategy_info = (f"strategy={analysis.best_strategy.strategy_name}[{analysis.best_strategy.signal_type.upper()}] "
                           f"strength={analysis.best_strategy.strength.name} conf={analysis.best_strategy.confidence:.2f} ")
        
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
        """Filter for high confidence analyses"""
        if not analyses:
            return []
        
        confidence_attr = 'combined_confidence' if use_combined_confidence else 'confidence'
        threshold = CONFIG.min_confidence_score
        
        high_conf = [
            analysis for analysis in analyses
            if getattr(analysis, confidence_attr, 0) >= threshold
        ]
        
        if high_conf:
            log_info(f"Filtered to {len(high_conf)} high-confidence signals")
        else:
            self._log_detailed_signal_analysis(analyses, confidence_attr, threshold)
        
        return high_conf
    
    def _log_detailed_signal_analysis(self, analyses: List[EnhancedNewsAnalysis], 
                                    confidence_attr: str, threshold: float) -> None:
        """Log detailed analysis for debugging"""
        log_info("=" * 60)
        log_info("DEBUG: SIGNAL ANALYSIS")
        log_info("=" * 60)
        
        if not analyses:
            log_info("DEBUG: No analyses to filter")
            return
        
        scores = [getattr(a, confidence_attr, 0) for a in analyses]
        max_score = max(scores) if scores else 0
        log_info(f"THRESHOLD: {threshold:.3f} | HIGHEST {confidence_attr.upper()}: {max_score:.3f}")
        
        # Log top 5 analyses
        top_analyses = sorted(analyses, key=lambda x: getattr(x, confidence_attr, 0), reverse=True)[:5]
        
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
            
            log_info(f"   COMBINED CONFIDENCE: {analysis.combined_confidence:.3f}")
            
            if analysis.combined_confidence < threshold:
                if analysis.confidence < 0.45:
                    log_info("   ISSUE: Low news confidence")
                if analysis.technical_analysis and analysis.technical_analysis.technical_confidence < 0.25:
                    log_info("   ISSUE: Low technical confidence")
                if abs(analysis.sentiment_score) < 0.18:
                    log_info("   ISSUE: Weak sentiment")
        
        log_info("=" * 60)


# Create alias for backwards compatibility
OptimizedEnhancedNewsAnalyzer = EnhancedNewsAnalyzer