"""
Optimized news analyzer with Gemini LLM integration and professional technical strategies
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
from utils.simple_logger import log_info, log_error, log_warning
from analysis.technical_analyzer import TechnicalAnalyzer, TechnicalAnalysis
from analysis.gemini_news_analyzer import GeminiNewsAnalyzer, GeminiAnalysis
from analysis.technical_strategies import TechnicalStrategies, StrategySignal


@dataclass
class EnhancedNewsAnalysis:
    """Enhanced news analysis with comprehensive data including Gemini and strategies."""
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
    """Optimized news analyzer with Gemini LLM integration and professional strategies."""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized model loading, Gemini, and technical strategies."""
        self.fmp_loader = fmp_loader
        self.technical_analyzer = TechnicalAnalyzer(fmp_loader)
        self.technical_strategies = TechnicalStrategies(fmp_loader)
        
        # Initialize Gemini analyzer
        self.gemini_analyzer = GeminiNewsAnalyzer()
        
        # Device selection with fallback
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.finbert_labels = ["positive", "negative", "neutral"]
        
        # Initialize FinBERT with error handling
        self._load_finbert_safely()
        
        # Optimized keyword sets
        self._initialize_keywords()
    
    def _load_finbert_safely(self) -> None:
        """Load FinBERT model with comprehensive error handling."""
        try:
            log_info("Loading FinBERT model...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                "ProsusAI/finbert",
                cache_dir="./cache"
            )
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert",
                cache_dir="./cache"
            )
            self.finbert_model.to(self.device)
            self.finbert_model.eval()
            
            # Optimize for inference
            if hasattr(torch, 'jit') and self.device == "cuda":
                self.finbert_model = torch.jit.optimize_for_inference(self.finbert_model)
            
            log_info(f"FinBERT loaded successfully on {self.device}")
            
        except Exception as e:
            log_error(f"Failed to load FinBERT: {e}")
            log_warning("Falling back to keyword-only analysis")
            self.finbert_model = None
            self.finbert_tokenizer = None
    
    def _initialize_keywords(self) -> None:
        """Initialize optimized keyword sets for sentiment analysis."""
        # Use sets for O(1) lookup performance
        self.positive_keywords = {
            'beats', 'beat', 'exceeds', 'exceed', 'raises', 'upgrade', 'approval', 'approved',
            'growth', 'strong', 'positive', 'success', 'breakthrough', 'innovation',
            'revenue increase', 'profit increase', 'buyback', 'dividend', 'expansion',
            'outperform', 'surge', 'rally', 'boost', 'gain', 'rise', 'soar', 'bullish',
            'record', 'milestone', 'achievement', 'partnership', 'collaboration', 'deal',
            'cheap', 'undervalued', 'good buy', 'bargain', 'discount', 'value',
            'attractive price', 'buying opportunity', 'oversold', 'worth buying',
            'time to buy', 'attractive valuation', 'compelling value', 'good value',
            'reasonable price', 'fair value', 'target price increase', 'price target raised',
            'buy rating', 'accumulate', 'overweight', 'recommend buy'
        }
        
        self.negative_keywords = {
            'misses', 'miss', 'falls short', 'disappointing', 'decline', 'drop',
            'downgrade', 'concern', 'loss', 'cut', 'reduce', 'weak', 'struggle',
            'investigation', 'lawsuit', 'recall', 'bankruptcy', 'layoffs',
            'plunge', 'crash', 'fall', 'slump', 'tumble', 'sink', 'bearish',
            'warning', 'delay', 'setback', 'failure', 'reject', 'denied',
            'overvalued', 'expensive', 'overpriced', 'sell rating', 'avoid',
            'sell recommendation', 'underweight', 'price target cut',
            'target price lowered', 'poor value', 'too expensive', 'risky buy'
        }
        
        self.biotech_keywords = {
            'fda', 'approval', 'phase', 'trial', 'clinical', 'drug', 'therapy',
            'treatment', 'efficacy', 'safety', 'regulatory', 'orphan drug',
            'breakthrough therapy', 'fast track', 'priority review', 'biologics',
            'pipeline', 'indication', 'endpoint', 'biomarker'
        }
        
        # Compile regex patterns for better performance
        self._topic_patterns = {
            'earnings': re.compile(r'\b(earnings|revenue|profit|eps|quarterly|guidance|beat|miss)\b', re.IGNORECASE),
            'biotech': re.compile(r'\b(fda|approval|phase|trial|clinical|drug|therapy|efficacy)\b', re.IGNORECASE),
            'ma': re.compile(r'\b(merger|acquisition|deal|buyout|takeover|acquire|merge)\b', re.IGNORECASE),
            'analyst': re.compile(r'\b(upgrade|downgrade|target|analyst|rating|price target)\b', re.IGNORECASE),
            'corporate_action': re.compile(r'\b(dividend|buyback|split|spinoff|distribution)\b', re.IGNORECASE),
            'business_development': re.compile(r'\b(contract|partnership|agreement|collaboration|alliance)\b', re.IGNORECASE)
        }
    
    def _get_finbert_sentiment(self, text: str) -> Tuple[float, float]:
        """Optimized FinBERT sentiment analysis with batching support."""
        if not self.finbert_model or not self.finbert_tokenizer:
            return 0.0, 0.0
        
        try:
            text = text.strip()[:512]  # Truncate early to save processing
            
            if not text:
                return 0.0, 0.0
            
            # Tokenize with optimized parameters
            inputs = self.finbert_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=False,
                add_special_tokens=True
            ).to(self.device)
            
            # Inference with optimization
            with torch.no_grad():
                if hasattr(torch, 'inference_mode'):
                    with torch.inference_mode():
                        outputs = self.finbert_model(**inputs)
                else:
                    outputs = self.finbert_model(**inputs)
                
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = predictions.cpu().numpy()[0]
            
            # FinBERT label mapping: [positive, negative, neutral]
            pos_prob, neg_prob, neu_prob = probs
            
            sentiment = float(pos_prob - neg_prob)
            confidence = float(max(probs))
            
            return sentiment, confidence
            
        except Exception as e:
            log_error(f"FinBERT analysis error: {e}")
            return 0.0, 0.0
    
    def _calculate_keyword_score(self, text: str) -> Tuple[float, float]:
        """Optimized keyword-based sentiment analysis with set operations and phrase handling."""
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Use set intersections for efficient counting
        positive_matches = len(text_words & self.positive_keywords)
        negative_matches = len(text_words & self.negative_keywords)
        biotech_matches = len(text_words & self.biotech_keywords)
        
        # Enhanced phrase-based keyword matching
        positive_phrases = [
            'revenue increase', 'profit increase', 'breakthrough therapy', 'good buy',
            'buying opportunity', 'attractive price', 'worth buying', 'time to buy',
            'attractive valuation', 'compelling value', 'good value', 'reasonable price',
            'target price raised', 'price target increase', 'buy rating'
        ]
        
        negative_phrases = [
            'falls short', 'price target cut', 'target price lowered', 'sell rating',
            'sell recommendation', 'poor value', 'too expensive', 'risky buy'
        ]
        
        # Efficient phrase checking
        clean_text = re.sub(r'[^\w\s]', '', text_lower)
        
        positive_matches += sum(1 for phrase in positive_phrases if phrase in clean_text)
        negative_matches += sum(1 for phrase in negative_phrases if phrase in clean_text)
        
        # Special handling for valuation questions
        valuation_questions = [
            'is it a good buy', 'should you buy', 'worth buying', 'time to buy',
            'good investment', 'buy the dip', 'cheap stock'
        ]
        
        positive_matches += sum(1 for phrase in valuation_questions if phrase in clean_text)
        
        # Calculate sentiment
        total_keywords = positive_matches + negative_matches
        if total_keywords == 0:
            sentiment = 0.0
            confidence = 0.1
        else:
            sentiment = (positive_matches - negative_matches) / total_keywords
            confidence = min(total_keywords / 5.0, 1.0)
        
        # Biotech boost
        if biotech_matches > 0:
            confidence = min(confidence * 1.3, 1.0)
        
        return sentiment, confidence
    
    @lru_cache(maxsize=1000)
    def _detect_topic(self, text: str) -> str:
        """Optimized topic detection with cached regex patterns."""
        text_lower = text.lower()
        
        for topic, pattern in self._topic_patterns.items():
            if pattern.search(text_lower):
                return topic
        
        return 'general'
    
    def _ensemble_score(self, finbert_sentiment: float, finbert_conf: float,
                       keyword_sentiment: float, keyword_conf: float,
                       gemini_sentiment: float, gemini_conf: float) -> Tuple[float, float]:
        """Enhanced ensemble scoring with Gemini integration."""
        
        components = []
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        # Add components that are available
        if finbert_conf > 0.0:
            weight = CONFIG.finbert_weight * finbert_conf
            weighted_sentiment += finbert_sentiment * weight
            total_weight += weight
            components.append(finbert_conf)
        
        if keyword_conf > 0.0:
            weight = CONFIG.keyword_weight * keyword_conf
            weighted_sentiment += keyword_sentiment * weight
            total_weight += weight
            components.append(keyword_conf)
        
        if gemini_conf > 0.0:
            weight = CONFIG.gemini_weight * gemini_conf
            weighted_sentiment += gemini_sentiment * weight
            total_weight += weight
            components.append(gemini_conf)
        
        if total_weight == 0:
            return 0.0, 0.0
        
        ensemble_sentiment = weighted_sentiment / total_weight
        
        if not components:
            return ensemble_sentiment, 0.0
        
        # Enhanced confidence calculation
        base_confidence = sum(components) / len(components)
        
        # Agreement bonus calculation
        sentiments = []
        if finbert_conf > 0:
            sentiments.append(finbert_sentiment)
        if keyword_conf > 0:
            sentiments.append(keyword_sentiment)
        if gemini_conf > 0:
            sentiments.append(gemini_sentiment)
        
        agreement_bonus = 1.0
        if len(sentiments) >= 2:
            sentiment_std = pd.Series(sentiments).std()
            if sentiment_std < 0.3:
                agreement_bonus = 1.3
            elif sentiment_std > 0.7:
                agreement_bonus = 0.8
        
        # Gemini quality bonus
        gemini_bonus = 1.2 if gemini_conf > 0.7 else 1.0
        
        final_confidence = min(base_confidence * agreement_bonus * gemini_bonus, 1.0)
        
        return ensemble_sentiment, final_confidence
    
    def _combine_news_technical_confidence(self, news_confidence: float, 
                                         technical_analysis: Optional[TechnicalAnalysis],
                                         sentiment_score: float,
                                         strategy_signals: Optional[List[StrategySignal]] = None) -> float:
        """Combine news, technical, and strategy confidence with alignment checking."""
        if technical_analysis is None:
            return news_confidence * 0.7  # Penalty for missing technical data
        
        tech_confidence = technical_analysis.technical_confidence
        
        # Calculate alignment and bonuses
        momentum_alignment = self._calculate_momentum_alignment(sentiment_score, technical_analysis.momentum_score)
        volume_bonus = 1.1 if technical_analysis.volume_score > 0.7 else 1.0
        liquidity_factor = max(technical_analysis.liquidity_score, 0.3)
        
        # Strategy confirmation bonus
        strategy_bonus = self._calculate_strategy_bonus(strategy_signals, sentiment_score)
        
        # Combined calculation
        combined = (
            news_confidence * 0.5 +
            tech_confidence * 0.3 +
            0.2 * strategy_bonus
        ) * momentum_alignment * volume_bonus * liquidity_factor
        
        return min(combined, 1.0)
    
    def _calculate_momentum_alignment(self, sentiment_score: float, momentum_score: float) -> float:
        """Calculate momentum alignment bonus."""
        if abs(sentiment_score) > 0.3 and abs(momentum_score) > 0.2:
            if (sentiment_score > 0) == (momentum_score > 0):
                return 1.25  # Aligned signals
            else:
                return 0.7   # Contradictory signals
        return 1.0
    
    def _calculate_strategy_bonus(self, strategy_signals: Optional[List[StrategySignal]], sentiment_score: float) -> float:
        """Calculate strategy alignment bonus."""
        if not strategy_signals:
            return 1.0
        
        best_signal = max(strategy_signals, key=lambda x: (x.strength.value, x.confidence))
        
        strategy_direction = 1 if best_signal.signal_type == "long" else -1
        sentiment_direction = 1 if sentiment_score > 0 else -1
        
        if strategy_direction == sentiment_direction:
            strategy_multiplier = 1.0 + (best_signal.confidence * 0.3)
            
            if best_signal.strength.value >= 4:
                strategy_multiplier *= 1.1
            
            return strategy_multiplier
        else:
            return 0.8
    
    def analyze_news_with_technical(self, news_df: pd.DataFrame, 
                                  current_prices: pd.DataFrame) -> List[EnhancedNewsAnalysis]:
        """Optimized news analysis with batch processing and technical validation."""
        if news_df is None or news_df.empty:
            return []
        
        log_info(f"Analyzing {len(news_df)} news articles with FinBERT + Technical Analysis")
        
        # Pre-process current prices for efficient lookup
        price_lookup = (
            current_prices.set_index('symbol').to_dict('index') 
            if current_prices is not None and not current_prices.empty 
            else {}
        )
        
        # Process news articles
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
        """Process a single news article with comprehensive validation including Gemini and strategies."""
        # Extract and validate essential fields
        symbol = str(row.get('symbol', '')).strip()
        title = str(row.get('title', '')).strip()
        content = str(row.get('content', '')).strip()
        
        if not all([symbol, title, content]):
            return None
        
        # Get price data for technical analysis
        price_data = price_lookup.get(symbol)
        current_price = float(price_data.get('lastSalePrice', 0)) if price_data else 0.0
        
        # Topic detection
        topic = self._detect_topic(f"{title} {content}")
        
        # Multi-source sentiment analysis
        full_text = f"{title} {content}"
        
        # 1. FinBERT analysis
        finbert_sentiment, finbert_conf = self._get_finbert_sentiment(full_text)
        
        # 2. Keyword analysis
        keyword_sentiment, keyword_conf = self._calculate_keyword_score(full_text)
        
        # 3. Gemini analysis (if enabled)
        gemini_sentiment, gemini_conf, gemini_reasoning = 0.0, 0.0, ""
        if self.gemini_analyzer.enabled and current_price > 0:
            try:
                gemini_analysis = self.gemini_analyzer.analyze_news(
                    symbol=symbol,
                    title=title,
                    content=content,
                    current_price=current_price,
                    topic=topic
                )
                if gemini_analysis:
                    gemini_sentiment = gemini_analysis.sentiment_score
                    gemini_conf = gemini_analysis.confidence
                    gemini_reasoning = gemini_analysis.reasoning
            except Exception as e:
                log_warning(f"Gemini analysis failed for {symbol}: {e}")
        
        # Enhanced ensemble scoring
        final_sentiment, news_confidence = self._ensemble_score(
            finbert_sentiment, finbert_conf,
            keyword_sentiment, keyword_conf,
            gemini_sentiment, gemini_conf
        )
        
        # Technical analysis
        technical_analysis = None
        if price_data is not None:
            technical_analysis = self.technical_analyzer.analyze_symbol(symbol, pd.Series(price_data))
        
        # Technical strategy analysis
        strategy_signals = []
        best_strategy = None
        if technical_analysis and current_price > 0:
            try:
                strategy_signals = self.technical_strategies.analyze_with_strategies(
                    symbol=symbol,
                    current_price=current_price,
                    technical_analysis=technical_analysis,
                    news_sentiment=final_sentiment
                )
                
                if strategy_signals:
                    best_strategy = self.technical_strategies.get_best_strategy(strategy_signals)
                    
            except Exception as e:
                log_warning(f"Strategy analysis failed for {symbol}: {e}")
        
        # Enhanced combined confidence calculation
        combined_confidence = self._combine_news_technical_confidence(
            news_confidence, technical_analysis, final_sentiment, strategy_signals
        )
        
        return EnhancedNewsAnalysis(
            symbol=symbol,
            title=title,
            content=content,
            sentiment_score=final_sentiment,
            confidence=news_confidence,
            topic=topic,
            timestamp=pd.Timestamp.now(tz=timezone.utc),
            finbert_score=finbert_sentiment,
            keyword_score=keyword_sentiment,
            gemini_score=gemini_sentiment,
            gemini_confidence=gemini_conf,
            gemini_reasoning=gemini_reasoning,
            technical_analysis=technical_analysis,
            strategy_signals=strategy_signals or [],
            best_strategy=best_strategy,
            combined_confidence=combined_confidence
        )
    
    def _log_high_confidence_result(self, analysis: EnhancedNewsAnalysis) -> None:
        """Log high confidence analysis results with all component details including strategies."""
        tech_info = ""
        if analysis.technical_analysis:
            tech = analysis.technical_analysis
            tech_info = (
                f"tech_conf={tech.technical_confidence:.2f} "
                f"momentum={tech.momentum_score:.2f} "
                f"liquidity={tech.liquidity_score:.2f}"
            )
        
        gemini_info = ""
        if analysis.gemini_confidence > 0:
            gemini_info = f"gemini={analysis.gemini_score:.2f}({analysis.gemini_confidence:.2f}) "
        
        strategy_info = ""
        if analysis.best_strategy:
            strategy = analysis.best_strategy
            strategy_info = (
                f"strategy={strategy.strategy_name}[{strategy.signal_type.upper()}] "
                f"strength={strategy.strength.name} conf={strategy.confidence:.2f} "
            )
        
        log_info(
            f"HIGH CONFIDENCE: {analysis.symbol} "
            f"sentiment={analysis.sentiment_score:.3f} "
            f"finbert={analysis.finbert_score:.2f} keyword={analysis.keyword_score:.2f} "
            f"{gemini_info}"
            f"{strategy_info}"
            f"news_conf={analysis.confidence:.3f} "
            f"combined_conf={analysis.combined_confidence:.3f} "
            f"{tech_info} topic={analysis.topic}"
        )
        
        if analysis.gemini_reasoning:
            log_info(f"  Gemini Reasoning: {analysis.gemini_reasoning[:150]}...")
        
        if analysis.best_strategy:
            strategy_summary = self.technical_strategies.format_strategy_summary(analysis.best_strategy)
            log_info(f"  Strategy: {strategy_summary}")
            if analysis.best_strategy.reasoning:
                log_info(f"  Strategy Reasoning: {analysis.best_strategy.reasoning[:150]}...")
    
    def filter_high_confidence(self, analyses: List[EnhancedNewsAnalysis], 
                             use_combined_confidence: bool = True) -> List[EnhancedNewsAnalysis]:
        """Filter for high confidence analyses with detailed debugging."""
        if not analyses:
            return []
        
        confidence_attr = 'combined_confidence' if use_combined_confidence else 'confidence'
        threshold = CONFIG.min_confidence_score
        
        high_conf = [
            analysis for analysis in analyses
            if getattr(analysis, confidence_attr, 0) >= threshold
        ]
        
        if high_conf:
            log_info(f"Filtered to {len(high_conf)} high-confidence signals using {confidence_attr} "
                    f"(threshold: {threshold})")
        else:
            self._log_detailed_signal_analysis(analyses, confidence_attr, threshold)
        
        return high_conf
    
    def _log_detailed_signal_analysis(self, analyses: List[EnhancedNewsAnalysis], 
                                    confidence_attr: str, threshold: float) -> None:
        """Log detailed analysis of why signals were rejected."""
        log_info("=" * 80)
        log_info("DEBUG: DETAILED SIGNAL ANALYSIS")
        log_info("=" * 80)
        
        if not analyses:
            log_info("DEBUG: No analyses to filter")
            return
        
        scores = [getattr(a, confidence_attr, 0) for a in analyses]
        max_score = max(scores) if scores else 0
        log_info(f"THRESHOLD: {threshold:.3f} | HIGHEST {confidence_attr.upper()}: {max_score:.3f}")
        
        for i, analysis in enumerate(analyses, 1):
            self._log_single_signal_details(i, len(analyses), analysis, threshold)
        
        self._log_improvement_suggestions(threshold)
    
    def _log_single_signal_details(self, index: int, total: int, 
                                 analysis: EnhancedNewsAnalysis, threshold: float) -> None:
        """Log details for a single signal."""
        log_info(f"\nSIGNAL {index}/{total}: {analysis.symbol}")
        log_info(f"   Title: {analysis.title[:80]}...")
        log_info(f"   Topic: {analysis.topic}")
        log_info(f"   Sentiment Score: {analysis.sentiment_score:.3f}")
        log_info(f"   FinBERT: {analysis.finbert_score:.3f}")
        log_info(f"   Keyword: {analysis.keyword_score:.3f}")
        log_info(f"   News Confidence: {analysis.confidence:.3f}")
        
        if analysis.technical_analysis:
            tech = analysis.technical_analysis
            log_info(f"   Technical Confidence: {tech.technical_confidence:.3f}")
            log_info(f"   Liquidity Score: {tech.liquidity_score:.3f}")
            log_info(f"   Momentum Score: {tech.momentum_score:.3f}")
            log_info(f"   Volume Score: {tech.volume_score:.3f}")
            log_info(f"   RSI: {tech.rsi:.1f}")
            log_info(f"   Price Trend: {tech.price_trend}")
        else:
            log_info("   Technical Analysis: MISSING")
        
        log_info(f"   COMBINED CONFIDENCE: {analysis.combined_confidence:.3f}")
        
        if analysis.combined_confidence < threshold:
            reasons = self._get_rejection_reasons(analysis)
            log_info(f"   REJECTION REASONS: {', '.join(reasons) if reasons else 'Below threshold'}")
    
    def _get_rejection_reasons(self, analysis: EnhancedNewsAnalysis) -> List[str]:
        """Get reasons why a signal was rejected."""
        reasons = []
        
        if analysis.confidence < 0.5:
            reasons.append("Low news confidence")
        if analysis.technical_analysis and analysis.technical_analysis.technical_confidence < 0.3:
            reasons.append("Low technical confidence")
        if analysis.technical_analysis and analysis.technical_analysis.liquidity_score < 0.3:
            reasons.append("Poor liquidity")
        if abs(analysis.sentiment_score) < 0.25:
            reasons.append("Weak sentiment")
        
        return reasons
    
    def _log_improvement_suggestions(self, threshold: float) -> None:
        """Log suggestions for getting more trades."""
        log_info("=" * 80)
        log_info("SUGGESTIONS TO GET TRADES:")
        log_info(f"1. Lower confidence threshold from {threshold} to 0.5 in config.py")
        log_info("2. Wait for stronger news sentiment (earnings, FDA approvals)")
        log_info("3. Check if market is in favorable regime")
        log_info("4. Ensure sufficient volume and liquidity")
        log_info("=" * 80)