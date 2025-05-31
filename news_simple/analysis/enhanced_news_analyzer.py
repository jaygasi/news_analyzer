"""
Optimized news analyzer with improved processing and comprehensive debugging
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


@dataclass
class EnhancedNewsAnalysis:
    """Enhanced news analysis with comprehensive data and type hints."""
    symbol: str
    title: str
    content: str
    sentiment_score: float
    confidence: float
    topic: str
    timestamp: pd.Timestamp
    finbert_score: float = 0.0
    keyword_score: float = 0.0
    technical_analysis: Optional[TechnicalAnalysis] = None
    combined_confidence: float = 0.0


class EnhancedNewsAnalyzer:
    """Optimized news analyzer with improved performance and error handling."""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized model loading and keyword sets."""
        self.fmp_loader = fmp_loader
        self.technical_analyzer = TechnicalAnalyzer(fmp_loader)
        
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
                       keyword_sentiment: float, keyword_conf: float) -> Tuple[float, float]:
        """Optimized ensemble scoring with dynamic weighting."""
        # Handle missing FinBERT
        if finbert_conf == 0.0:
            return keyword_sentiment, keyword_conf
        
        # Dynamic weights based on confidence levels
        finbert_weight = finbert_conf * 0.7
        keyword_weight = keyword_conf * 0.3
        
        total_weight = finbert_weight + keyword_weight
        if total_weight == 0:
            return 0.0, 0.0
        
        # Weighted sentiment
        ensemble_sentiment = (
            finbert_sentiment * finbert_weight + 
            keyword_sentiment * keyword_weight
        ) / total_weight
        
        # Agreement bonus
        agreement_bonus = 1.0
        if abs(finbert_sentiment - keyword_sentiment) < 0.3:
            agreement_bonus = 1.2
        
        # Combined confidence with agreement bonus
        ensemble_confidence = min(
            (finbert_conf * keyword_conf) ** 0.5 * agreement_bonus, 
            1.0
        )
        
        return ensemble_sentiment, ensemble_confidence
    
    def _combine_news_technical_confidence(self, news_confidence: float, 
                                         technical_analysis: Optional[TechnicalAnalysis],
                                         sentiment_score: float) -> float:
        """Combine news and technical confidence with alignment checking."""
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
        
        # Combined calculation
        combined = (
            news_confidence * 0.6 +      # News confidence (primary)
            tech_confidence * 0.4        # Technical confidence (secondary)
        ) * momentum_alignment * volume_bonus * liquidity_factor
        
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
        """Process a single news article with comprehensive validation."""
        # Extract and validate essential fields
        symbol = str(row.get('symbol', '')).strip()
        title = str(row.get('title', '')).strip()
        content = str(row.get('content', '')).strip()
        
        if not all([symbol, title, content]):
            return None
        
        # Get price data for technical analysis
        price_data = price_lookup.get(symbol)
        
        # News sentiment analysis
        full_text = f"{title} {content}"
        finbert_sentiment, finbert_conf = self._get_finbert_sentiment(full_text)
        keyword_sentiment, keyword_conf = self._calculate_keyword_score(full_text)
        final_sentiment, news_confidence = self._ensemble_score(
            finbert_sentiment, finbert_conf, keyword_sentiment, keyword_conf
        )
        
        # Technical analysis
        technical_analysis = None
        if price_data is not None:
            technical_analysis = self.technical_analyzer.analyze_symbol(symbol, price_data)
        
        # Combined confidence calculation
        combined_confidence = self._combine_news_technical_confidence(
            news_confidence, technical_analysis, final_sentiment
        )
        
        # Topic detection
        topic = self._detect_topic(full_text)
        
        # Create analysis result with timezone-aware timestamp
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
            technical_analysis=technical_analysis,
            combined_confidence=combined_confidence
        )
    
    def _log_high_confidence_result(self, analysis: EnhancedNewsAnalysis) -> None:
        """Log high confidence analysis results with technical details."""
        tech_info = ""
        if analysis.technical_analysis:
            tech_info = (
                f"tech_conf={analysis.technical_analysis.technical_confidence:.2f} "
                f"momentum={analysis.technical_analysis.momentum_score:.2f} "
                f"liquidity={analysis.technical_analysis.liquidity_score:.2f}"
            )
        
        log_info(f"HIGH CONFIDENCE: {analysis.symbol} "
                f"sentiment={analysis.sentiment_score:.3f} "
                f"news_conf={analysis.confidence:.3f} "
                f"combined_conf={analysis.combined_confidence:.3f} "
                f"{tech_info} topic={analysis.topic}")
    
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