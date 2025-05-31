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
    gemini_score: float = 0.0           # NEW: Gemini sentiment score
    gemini_confidence: float = 0.0      # NEW: Gemini confidence
    gemini_reasoning: str = ""          # NEW: Gemini reasoning
    technical_analysis: Optional[TechnicalAnalysis] = None
    strategy_signals: Optional[List[StrategySignal]] = None  # NEW: Technical strategy signals
    best_strategy: Optional[StrategySignal] = None  # NEW: Best strategy recommendation
    combined_confidence: float = 0.0


class EnhancedNewsAnalyzer:
    """Optimized news analyzer with Gemini LLM integration and professional strategies."""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized model loading, Gemini, and technical strategies."""
        self.fmp_loader = fmp_loader
        self.technical_analyzer = TechnicalAnalyzer(fmp_loader)
        self.technical_strategies = TechnicalStrategies(fmp_loader)  # NEW: Professional strategies
        
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
        # Positive sentiment keywords - ENHANCED FOR STOCK ANALYSIS
        self.positive_keywords = {
            'beats', 'beat', 'exceeds', 'exceed', 'raises', 'upgrade', 'approval', 'approved',
            'growth', 'strong', 'positive', 'success', 'breakthrough', 'innovation',
            'revenue increase', 'profit increase', 'buyback', 'dividend', 'expansion',
            'outperform', 'surge', 'rally', 'boost', 'gain', 'rise', 'soar', 'bullish',
            'record', 'milestone', 'achievement', 'partnership', 'collaboration', 'deal',
            
            # ADDED: Stock valuation and buying keywords
            'cheap', 'undervalued', 'good buy', 'bargain', 'discount', 'value',
            'attractive price', 'buying opportunity', 'oversold', 'worth buying',
            'time to buy', 'attractive valuation', 'compelling value', 'good value',
            'reasonable price', 'fair value', 'target price increase', 'price target raised',
            'buy rating', 'accumulate', 'overweight', 'recommend buy'
        }
        
        # Negative sentiment keywords - ENHANCED
        self.negative_keywords = {
            'misses', 'miss', 'falls short', 'disappointing', 'decline', 'drop',
            'downgrade', 'concern', 'loss', 'cut', 'reduce', 'weak', 'struggle',
            'investigation', 'lawsuit', 'recall', 'bankruptcy', 'layoffs',
            'plunge', 'crash', 'fall', 'slump', 'tumble', 'sink', 'bearish',
            'warning', 'delay', 'setback', 'failure', 'reject', 'denied',
            
            # ADDED: Stock negative valuation keywords
            'overvalued', 'expensive', 'overpriced', 'sell rating', 'avoid',
            'sell recommendation', 'underweight', 'price target cut',
            'target price lowered', 'poor value', 'too expensive', 'risky buy'
        }
        
        # Biotech-specific keywords (high impact)
        self.biotech_keywords = {
            'fda', 'approval', 'phase', 'trial', 'clinical', 'drug', 'therapy',
            'treatment', 'efficacy', 'safety', 'regulatory', 'orphan drug',
            'breakthrough therapy', 'fast track', 'priority review', 'biologics',
            'pipeline', 'indication', 'endpoint', 'biomarker'
        }
    
    def _get_finbert_sentiment(self, text: str) -> Tuple[float, float]:
        """Optimized FinBERT sentiment analysis with batching support."""
        if not self.finbert_model or not self.finbert_tokenizer:
            return 0.0, 0.0
        
        try:
            # Optimize text preprocessing
            text = text.strip()[:512]  # Truncate early to save processing
            
            if not text:
                return 0.0, 0.0
            
            # Tokenize with optimized parameters
            inputs = self.finbert_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=False,  # No padding needed for single input
                add_special_tokens=True
            ).to(self.device)
            
            # Inference with optimization
            with torch.no_grad():
                if hasattr(torch, 'inference_mode'):
                    with torch.inference_mode():
                        outputs = self.finbert_model(**inputs)
                else:
                    outputs = self.finbert_model(**inputs)
                
                # Efficient softmax calculation
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = predictions.cpu().numpy()[0]
            
            # FinBERT label mapping: [positive, negative, neutral]
            pos_prob = float(probs[0])
            neg_prob = float(probs[1])
            neu_prob = float(probs[2])
            
            # Calculate sentiment score
            sentiment = pos_prob - neg_prob
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
        
        # Check for phrases (ignore punctuation)
        clean_text = text_lower.replace('?', '').replace('!', '').replace('.', '').replace(',', '')
        
        for phrase in positive_phrases:
            if phrase in clean_text:
                positive_matches += 1
        
        for phrase in negative_phrases:
            if phrase in clean_text:
                negative_matches += 1
        
        # Special handling for valuation questions (common in stock analysis)
        valuation_questions = [
            'is it a good buy', 'should you buy', 'worth buying', 'time to buy',
            'good investment', 'buy the dip', 'cheap stock'
        ]
        
        for phrase in valuation_questions:
            if phrase in clean_text:
                positive_matches += 1  # Treat valuation questions as positive interest
        
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
    
    def _detect_topic(self, text: str) -> str:
        """Optimized topic detection with regex patterns."""
        text_lower = text.lower()
        
        # Use compiled regex patterns for better performance
        topic_patterns = {
            'earnings': r'\b(earnings|revenue|profit|eps|quarterly|guidance|beat|miss)\b',
            'biotech': r'\b(fda|approval|phase|trial|clinical|drug|therapy|efficacy)\b',
            'ma': r'\b(merger|acquisition|deal|buyout|takeover|acquire|merge)\b',
            'analyst': r'\b(upgrade|downgrade|target|analyst|rating|price target)\b',
            'corporate_action': r'\b(dividend|buyback|split|spinoff|distribution)\b',
            'business_development': r'\b(contract|partnership|agreement|collaboration|alliance)\b'
        }
        
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, text_lower):
                return topic
        
        return 'general'
    
    def _ensemble_score(self, finbert_sentiment: float, finbert_conf: float,
                       keyword_sentiment: float, keyword_conf: float,
                       gemini_sentiment: float, gemini_conf: float) -> Tuple[float, float]:
        """Enhanced ensemble scoring with Gemini integration."""
        
        # Handle missing components gracefully
        total_weight = 0.0
        weighted_sentiment = 0.0
        confidence_components = []
        
        # FinBERT component
        if finbert_conf > 0.0:
            finbert_weight = CONFIG.finbert_weight * finbert_conf
            weighted_sentiment += finbert_sentiment * finbert_weight
            total_weight += finbert_weight
            confidence_components.append(finbert_conf)
        
        # Keyword component
        if keyword_conf > 0.0:
            keyword_weight = CONFIG.keyword_weight * keyword_conf
            weighted_sentiment += keyword_sentiment * keyword_weight
            total_weight += keyword_weight
            confidence_components.append(keyword_conf)
        
        # Gemini component (highest priority when available)
        if gemini_conf > 0.0:
            gemini_weight = CONFIG.gemini_weight * gemini_conf
            weighted_sentiment += gemini_sentiment * gemini_weight
            total_weight += gemini_weight
            confidence_components.append(gemini_conf)
        
        # Calculate ensemble sentiment
        if total_weight == 0:
            return 0.0, 0.0
        
        ensemble_sentiment = weighted_sentiment / total_weight
        
        # Enhanced confidence calculation
        if not confidence_components:
            return ensemble_sentiment, 0.0
        
        # Base confidence from components
        base_confidence = sum(confidence_components) / len(confidence_components)
        
        # Agreement bonus - check alignment between components
        agreement_bonus = 1.0
        sentiments = []
        if finbert_conf > 0:
            sentiments.append(finbert_sentiment)
        if keyword_conf > 0:
            sentiments.append(keyword_sentiment)
        if gemini_conf > 0:
            sentiments.append(gemini_sentiment)
        
        # Calculate sentiment agreement
        if len(sentiments) >= 2:
            sentiment_std = pd.Series(sentiments).std()
            if sentiment_std < 0.3:  # Good agreement
                agreement_bonus = 1.3
            elif sentiment_std > 0.7:  # Poor agreement
                agreement_bonus = 0.8
        
        # Gemini quality bonus (Gemini reasoning adds confidence)
        gemini_bonus = 1.0
        if gemini_conf > 0.7:  # High confidence Gemini analysis
            gemini_bonus = 1.2
        
        final_confidence = min(base_confidence * agreement_bonus * gemini_bonus, 1.0)
        
        return ensemble_sentiment, final_confidence
    
    def _combine_news_technical_confidence(self, news_confidence: float, 
                                         technical_analysis: Optional[TechnicalAnalysis],
                                         sentiment_score: float,
                                         strategy_signals: List[StrategySignal] = None) -> float:
        """Combine news, technical, and strategy confidence with alignment checking."""
        if technical_analysis is None:
            return news_confidence * 0.7  # Penalty for missing technical data
        
        tech_confidence = technical_analysis.technical_confidence
        
        # Momentum alignment scoring
        momentum_alignment = 1.0
        tech_momentum = technical_analysis.momentum_score
        
        if abs(sentiment_score) > 0.3 and abs(tech_momentum) > 0.2:
            if (sentiment_score > 0) == (tech_momentum > 0):
                momentum_alignment = 1.25  # Aligned signals
            else:
                momentum_alignment = 0.7   # Contradictory signals
        
        # Volume confirmation
        volume_bonus = 1.0
        if technical_analysis.volume_score > 0.7:
            volume_bonus = 1.1
        
        # Liquidity requirement (hard constraint)
        liquidity_factor = max(technical_analysis.liquidity_score, 0.3)
        
        # NEW: Strategy confirmation bonus
        strategy_bonus = 1.0
        if strategy_signals:
            # Get best strategy signal
            best_signal = max(strategy_signals, key=lambda x: (x.strength.value, x.confidence))
            
            # Check strategy-sentiment alignment
            strategy_direction = 1 if best_signal.signal_type == "long" else -1
            sentiment_direction = 1 if sentiment_score > 0 else -1
            
            if strategy_direction == sentiment_direction:
                # Strategy and sentiment align
                strategy_multiplier = 1.0 + (best_signal.confidence * 0.3)  # Up to 30% bonus
                
                # Additional bonus for strong strategies
                if best_signal.strength.value >= 4:  # Strong or Very Strong
                    strategy_multiplier *= 1.1
                
                strategy_bonus = strategy_multiplier
            else:
                # Strategy contradicts sentiment
                strategy_bonus = 0.8
        
        # Combined calculation with strategy integration
        combined = (
            news_confidence * 0.5 +      # News confidence (reduced to make room for strategy)
            tech_confidence * 0.3 +      # Technical confidence  
            (strategy_bonus - 1.0) * 0.2  # Strategy bonus (0.2 weight for strategy component)
        ) * momentum_alignment * volume_bonus * liquidity_factor
        
        # Ensure we add back the base confidence
        combined = news_confidence * 0.5 + tech_confidence * 0.3 + 0.2 + (strategy_bonus - 1.0) * 0.2
        combined *= momentum_alignment * volume_bonus * liquidity_factor
        
        return min(combined, 1.0)
    
    def analyze_news_with_technical(self, news_df: pd.DataFrame, 
                                  current_prices: pd.DataFrame) -> List[EnhancedNewsAnalysis]:
        """Optimized news analysis with batch processing and technical validation."""
        results = []
        
        if news_df is None or news_df.empty:
            return results
        
        log_info(f"Analyzing {len(news_df)} news articles with FinBERT + Technical Analysis")
        
        # Pre-process current prices for efficient lookup
        price_lookup = {}
        if current_prices is not None and not current_prices.empty:
            for _, row in current_prices.iterrows():
                price_lookup[row['symbol']] = row
        
        # Process news articles
        for idx, row in news_df.iterrows():
            try:
                analysis = self._process_single_article(row, price_lookup)
                if analysis:
                    results.append(analysis)
                    
                    # Log high confidence results
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
        current_price = 0.0
        if price_data is not None:
            current_price = float(price_data.get('lastSalePrice', 0))
        
        # Topic detection (needed for Gemini)
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
            technical_analysis = self.technical_analyzer.analyze_symbol(symbol, price_data)
        
        # NEW: Technical strategy analysis
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
        
        # Enhanced combined confidence calculation with strategy integration
        combined_confidence = self._combine_news_technical_confidence(
            news_confidence, technical_analysis, final_sentiment, strategy_signals
        )
        
        # Create enhanced analysis result with all components
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
            tech_info = (
                f"tech_conf={analysis.technical_analysis.technical_confidence:.2f} "
                f"momentum={analysis.technical_analysis.momentum_score:.2f} "
                f"liquidity={analysis.technical_analysis.liquidity_score:.2f}"
            )
        
        # Include Gemini information
        gemini_info = ""
        if analysis.gemini_confidence > 0:
            gemini_info = f"gemini={analysis.gemini_score:.2f}({analysis.gemini_confidence:.2f}) "
        
        # Include strategy information  
        strategy_info = ""
        if analysis.best_strategy:
            strategy = analysis.best_strategy
            strategy_info = (f"strategy={strategy.strategy_name}[{strategy.signal_type.upper()}] "
                           f"strength={strategy.strength.name} conf={strategy.confidence:.2f} ")
        
        log_info(f"HIGH CONFIDENCE: {analysis.symbol} "
                f"sentiment={analysis.sentiment_score:.3f} "
                f"finbert={analysis.finbert_score:.2f} keyword={analysis.keyword_score:.2f} "
                f"{gemini_info}"
                f"{strategy_info}"
                f"news_conf={analysis.confidence:.3f} "
                f"combined_conf={analysis.combined_confidence:.3f} "
                f"{tech_info} topic={analysis.topic}")
        
        # Log Gemini reasoning if available
        if analysis.gemini_reasoning:
            log_info(f"  Gemini Reasoning: {analysis.gemini_reasoning[:150]}...")
        
        # Log best strategy details if available
        if analysis.best_strategy:
            strategy = analysis.best_strategy
            log_info(f"  Strategy: {self.technical_strategies.format_strategy_summary(strategy)}")
            if strategy.reasoning:
                log_info(f"  Strategy Reasoning: {strategy.reasoning[:150]}...")
    
    def filter_high_confidence(self, analyses: List[EnhancedNewsAnalysis], 
                             use_combined_confidence: bool = True) -> List[EnhancedNewsAnalysis]:
        """Filter for high confidence analyses with detailed debugging."""
        if not analyses:
            return []
        
        confidence_attr = 'combined_confidence' if use_combined_confidence else 'confidence'
        threshold = CONFIG.min_confidence_score
        
        # Use list comprehension for efficiency
        high_conf = [
            analysis for analysis in analyses
            if getattr(analysis, confidence_attr, 0) >= threshold
        ]
        
        if high_conf:
            log_info(f"Filtered to {len(high_conf)} high-confidence signals using {confidence_attr} "
                    f"(threshold: {threshold})")
        else:
            # DETAILED DEBUG LOGGING FOR REJECTED SIGNALS
            log_info("=" * 80)
            log_info("DEBUG: DETAILED SIGNAL ANALYSIS")
            log_info("=" * 80)
            
            if analyses:
                scores = [getattr(a, confidence_attr, 0) for a in analyses]
                max_score = max(scores) if scores else 0
                log_info(f"THRESHOLD: {threshold:.3f} | HIGHEST {confidence_attr.upper()}: {max_score:.3f}")
                
                # Analyze each signal in detail
                for i, analysis in enumerate(analyses, 1):
                    log_info(f"\nSIGNAL {i}/{len(analyses)}: {analysis.symbol}")
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
                        log_info(f"   Bid-Ask Spread: {tech.bid_ask_spread:.3f}")
                    else:
                        log_info(f"   Technical Analysis: MISSING")
                    
                    log_info(f"   COMBINED CONFIDENCE: {analysis.combined_confidence:.3f}")
                    
                    # Explain why it failed
                    if analysis.combined_confidence < threshold:
                        reasons = []
                        if analysis.confidence < 0.5:
                            reasons.append("Low news confidence")
                        if analysis.technical_analysis and analysis.technical_analysis.technical_confidence < 0.3:
                            reasons.append("Low technical confidence")
                        if analysis.technical_analysis and analysis.technical_analysis.liquidity_score < 0.3:
                            reasons.append("Poor liquidity")
                        if abs(analysis.sentiment_score) < 0.25:
                            reasons.append("Weak sentiment")
                        
                        log_info(f"   REJECTION REASONS: {', '.join(reasons) if reasons else 'Below threshold'}")
                
                log_info("=" * 80)
                log_info("SUGGESTIONS TO GET TRADES:")
                log_info(f"1. Lower confidence threshold from {threshold} to 0.5 in config.py")
                log_info("2. Wait for stronger news sentiment (earnings, FDA approvals)")
                log_info("3. Check if market is in favorable regime")
                log_info("4. Ensure sufficient volume and liquidity")
                log_info("=" * 80)
            else:
                log_info("DEBUG: No analyses to filter")
        
        return high_conf