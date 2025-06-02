"""
Optimized news analyzer with improved performance and reduced memory usage
"""
import pandas as pd
from typing import Dict, Optional, Tuple, List, Set, Any
from dataclasses import dataclass
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from collections import defaultdict
import hashlib
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning, log_debug
from analysis.technical_analyzer import TechnicalAnalyzer, TechnicalAnalysis
from analysis.gemini_news_analyzer import OptimizedGeminiNewsAnalyzer, GeminiAnalysis
from analysis.technical_strategies import TechnicalStrategies, StrategySignal


# Optimized constants for repeated string literals
FDA_APPROVAL = 'fda approval'
FAST_TRACK = 'fast track'
PRIORITY_REVIEW = 'priority review'
WORTH_BUYING = 'worth buying'
TIME_TO_BUY = 'time to buy'


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


@dataclass
class OptimizedSentimentScore:
    """Optimized sentiment with quality metrics"""
    base_sentiment: float
    magnitude_score: float
    urgency_score: float
    credibility_score: float
    sector_relevance: float
    surprise_factor: float
    confirmation_score: float
    final_sentiment: float
    quality_confidence: float


class OptimizedKeywordAnalyzer:
    """Optimized keyword analysis with pre-compiled patterns and efficient lookups"""
    
    def __init__(self):
        self._initialize_optimized_keywords()
        self._initialize_magnitude_indicators()
        self._initialize_urgency_indicators()
        self._initialize_sector_keywords()
        self._initialize_credibility_indicators()
        self._compile_patterns()
        
    def _initialize_optimized_keywords(self):
        """Initialize optimized keywords with frozensets for faster lookups"""
        
        # Power keywords with highest predictive value
        self.power_positive_keywords = frozenset([
            'crushes', 'smashes', 'demolishes', 'obliterates', 'destroys estimates',
            'blowout', 'blockbuster', 'stellar', 'phenomenal', 'exceptional',
            'record-breaking', 'all-time high', 'massive beat', 'huge surprise',
            'explosive growth', 'rapid expansion', 'accelerating', 'momentum building',
            'scaling rapidly', 'breakthrough momentum', 'unprecedented growth',
            'cash flow surge', 'profit explosion', 'revenue acceleration',
            'margin expansion', 'cost savings realized', 'efficiency gains',
            'market domination', 'competitive advantage', 'market leader',
            'disrupting industry', 'game changer', 'revolutionary'
        ])
        
        self.power_negative_keywords = frozenset([
            'catastrophic', 'devastating', 'disastrous', 'nightmare', 'collapse',
            'free fall', 'plummeting', 'cratering', 'imploding', 'hemorrhaging',
            'cash crunch', 'liquidity crisis', 'burning cash', 'bleeding money',
            'massive losses', 'debt spiral', 'financial distress',
            'losing market share', 'competitive pressure', 'disrupted business',
            'obsolete technology', 'regulatory nightmare', 'investigation launched'
        ])
        
        # Optimized biotech/pharma keywords
        self.biotech_catalysts = {
            'positive': frozenset([
                FDA_APPROVAL, 'breakthrough designation', FAST_TRACK, PRIORITY_REVIEW,
                'orphan drug', 'accelerated approval', 'phase 3 success', 'meets endpoints',
                'statistically significant', 'pdufa date', 'nda accepted', 'bla accepted',
                'regulatory approval', 'cms approval', 'reimbursement approved'
            ]),
            'negative': frozenset([
                'fda rejection', 'complete response letter', 'crl', 'safety hold',
                'trial failure', 'missed endpoints', 'adverse events', 'safety concerns',
                'trial halted', 'regulatory delay', 'black box warning'
            ])
        }
        
        # Optimized tech keywords
        self.tech_catalysts = {
            'positive': frozenset([
                'ai breakthrough', 'patent approval', 'licensing deal', 'cloud growth',
                'user growth', 'subscription growth', 'platform adoption', 'digital transformation'
            ]),
            'negative': frozenset([
                'data breach', 'regulatory scrutiny', 'antitrust', 'user decline',
                'competition threat', 'platform issues', 'security vulnerability'
            ])
        }

    def _initialize_magnitude_indicators(self):
        """Optimized magnitude indicators with frozen dict"""
        self.magnitude_amplifiers = {
            'extreme': 2.0, 'massive': 1.8, 'huge': 1.6, 'significant': 1.4,
            'substantial': 1.3, 'major': 1.2, 'notable': 1.1,
            'slight': 0.7, 'minor': 0.6, 'small': 0.5, 'tiny': 0.3
        }
        
    def _initialize_urgency_indicators(self):
        """Optimized urgency indicators"""
        self.urgency_keywords = {
            'immediate': 1.0, 'urgent': 0.9, 'breaking': 0.9, 'just announced': 0.9,
            'developing': 0.8, 'emerging': 0.7, 'upcoming': 0.6, 'planned': 0.4,
            'potential': 0.3, 'possible': 0.2, 'rumored': 0.1
        }
        
        self.time_sensitivity = {
            'within hours': 1.0, 'today': 0.9, 'this week': 0.7,
            'this month': 0.5, 'this quarter': 0.3, 'next year': 0.1
        }

    def _initialize_sector_keywords(self):
        """Optimized sector-specific keywords"""
        self.sector_keywords = {
            'biotech': {
                'high_impact': frozenset(['fda', 'approval', 'trial', 'drug', 'therapy', 'treatment']),
                'catalysts': frozenset(['pdufa', 'breakthrough', 'orphan', FAST_TRACK, 'priority']),
                'risks': frozenset(['safety', 'adverse', 'crl', 'rejection', 'delay'])
            },
            'tech': {
                'high_impact': frozenset(['ai', 'cloud', 'subscription', 'platform', 'user growth']),
                'catalysts': frozenset(['breakthrough', 'patent', 'acquisition', 'partnership']),
                'risks': frozenset(['competition', 'regulation', 'data breach', 'antitrust'])
            },
            'energy': {
                'high_impact': frozenset(['oil', 'gas', 'renewable', 'production', 'reserves']),
                'catalysts': frozenset(['discovery', 'drilling', 'capacity', 'contract']),
                'risks': frozenset(['spill', 'accident', 'regulation', 'embargo'])
            }
        }

    def _initialize_credibility_indicators(self):
        """Optimized credibility indicators"""
        self.credible_sources = {
            'tier_1': frozenset(['reuters', 'bloomberg', 'wsj', 'ft', 'ap news']),
            'tier_2': frozenset(['cnbc', 'marketwatch', 'yahoo finance', 'seeking alpha']),
            'tier_3': frozenset(['motley fool', 'benzinga', 'zacks']),
            'company_direct': frozenset(['press release', 'sec filing', '8-k', '10-k'])
        }
        
        self.credibility_markers = {
            'high': frozenset(['sec filing', 'press release', 'earnings call', 'official statement',
                              'regulatory filing', 'management guidance', 'board approval']),
            'medium': frozenset(['analyst report', 'research note', 'expert opinion', 'industry study']),
            'low': frozenset(['rumor', 'speculation', 'unconfirmed', 'alleged', 'sources say'])
        }
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        self.percentage_patterns = [
            re.compile(r'(\d+(?:\.\d+)?)\s*%\s*(increase|growth|up|higher)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*%\s*(decrease|decline|down|lower)', re.IGNORECASE),
            re.compile(r'(doubled|tripled|quadrupled)', re.IGNORECASE),
            re.compile(r'(\d+)x\s*(growth|increase)', re.IGNORECASE)
        ]


class OptimizedMultiSourceValidator:
    """Optimized cross-reference validator with better memory management"""
    
    def __init__(self):
        self.news_cache = {}
        self.confirmation_window_hours = 4  # Reduced for efficiency
        self.max_cache_size = 200  # Reduced memory footprint
        
    def check_multi_source_confirmation(self, current_analysis: dict, 
                                      recent_analyses: Optional[List[dict]]) -> float:
        """Optimized multi-source confirmation checking"""
        if not recent_analyses or len(recent_analyses) < 2:
            return 0.5
        
        current_symbol = current_analysis['symbol']
        current_sentiment = current_analysis['sentiment_score']
        current_timestamp = current_analysis['timestamp']
        
        # Filter related articles efficiently
        cutoff_time = current_timestamp - timedelta(hours=self.confirmation_window_hours)
        related_articles = [
            art for art in recent_analyses[-20:]  # Only check last 20 for efficiency
            if (art['symbol'] == current_symbol and 
                art['timestamp'] > cutoff_time)
        ]
        
        if len(related_articles) < 2:
            return 0.5
        
        # Optimized sentiment agreement calculation
        sentiments = [art['sentiment_score'] for art in related_articles]
        sentiment_agreement = self._calculate_sentiment_agreement_optimized(current_sentiment, sentiments)
        
        # Simplified topic matching
        current_topic = current_analysis.get('topic', 'general')
        topic_matches = sum(1 for art in related_articles if art.get('topic') == current_topic)
        topic_score = topic_matches / len(related_articles)
        
        # Weighted combination
        confirmation_score = (sentiment_agreement * 0.7 + topic_score * 0.3)
        
        return confirmation_score
    
    def _calculate_sentiment_agreement_optimized(self, target_sentiment: float, other_sentiments: List[float]) -> float:
        """Optimized sentiment agreement calculation"""
        if not other_sentiments:
            return 0.5
        
        target_direction = 1 if target_sentiment > 0.1 else (-1 if target_sentiment < -0.1 else 0)
        
        agreements = sum(
            1 for sentiment in other_sentiments
            if (1 if sentiment > 0.1 else (-1 if sentiment < -0.1 else 0)) == target_direction
        )
        
        return agreements / len(other_sentiments)


class OptimizedHistoricalPatternMatcher:
    """Optimized pattern matcher with cached lookups"""
    
    def __init__(self):
        # Pre-computed impact multipliers for faster lookups
        self.impact_multipliers = {
            'earnings': {
                'strong_beat': 1.25, 'beat': 1.08, 'miss': 0.75, 'strong_miss': 0.55
            },
            'biotech': {
                'fda_approval': 1.8, 'trial_success': 1.4, 'trial_failure': 0.35, 'fda_rejection': 0.25
            },
            'analyst': {
                'upgrade': 1.15, 'downgrade': 0.85, 'initiate_buy': 1.06
            }
        }
        
        self.accuracy_rates = {
            'earnings': 0.72, 'biotech': 0.82, 'analyst': 0.62, 'general': 0.52
        }
        
    def find_historical_impact(self, news_type: str, sentiment_score: float) -> Tuple[float, float]:
        """Optimized historical impact lookup"""
        category = self._categorize_news_optimized(news_type, sentiment_score)
        multiplier = self.impact_multipliers.get(news_type, {}).get(category, 1.0)
        accuracy = self.accuracy_rates.get(news_type, 0.52)
        
        return multiplier, accuracy
    
    def _categorize_news_optimized(self, news_type: str, sentiment: float) -> str:
        """Optimized news categorization"""
        if news_type == 'earnings':
            if sentiment > 0.55:
                return 'strong_beat'
            elif sentiment > 0.15:
                return 'beat'
            elif sentiment < -0.55:
                return 'strong_miss'
            else:
                return 'miss'
        return 'general'


class OptimizedNewsQualityScorer:
    """Optimized news quality scorer with better performance"""
    
    def __init__(self):
        self.keyword_analyzer = OptimizedKeywordAnalyzer()
        self.source_validator = OptimizedMultiSourceValidator()
        self.pattern_matcher = OptimizedHistoricalPatternMatcher()
        
    def calculate_enhanced_sentiment(self, title: str, content: str, symbol: str,
                                   current_price: float, topic: str,
                                   source: str = "", recent_analyses: Optional[List] = None) -> OptimizedSentimentScore:
        """Optimized enhanced sentiment calculation"""
        
        text = f"{title} {content}".lower()
        
        # Optimized component calculations
        base_sentiment = self._calculate_base_sentiment_optimized(text)
        magnitude_score = self._calculate_magnitude_score_optimized(text, topic)
        urgency_score = self._calculate_urgency_score_optimized(text, title)
        credibility_score = self._calculate_credibility_score_optimized(text, source, content)
        sector_relevance = self._calculate_sector_relevance_optimized(text, topic)
        surprise_factor = self._calculate_surprise_factor_optimized(text)
        
        # Multi-source confirmation
        confirmation_score = self._get_confirmation_score_optimized(
            symbol, base_sentiment, topic, recent_analyses
        )
        
        # Historical pattern matching
        impact_multiplier, historical_accuracy = self.pattern_matcher.find_historical_impact(
            topic, base_sentiment
        )
        
        # Optimized weighted sentiment calculation
        final_sentiment = self._calculate_weighted_sentiment_optimized(
            base_sentiment, magnitude_score, credibility_score, 
            sector_relevance, confirmation_score, urgency_score, impact_multiplier
        )
        
        # Optimized quality confidence calculation
        quality_confidence = self._calculate_quality_confidence_optimized(
            credibility_score, confirmation_score, sector_relevance, 
            magnitude_score, historical_accuracy
        )
        
        return OptimizedSentimentScore(
            base_sentiment=base_sentiment,
            magnitude_score=magnitude_score,
            urgency_score=urgency_score,
            credibility_score=credibility_score,
            sector_relevance=sector_relevance,
            surprise_factor=surprise_factor,
            confirmation_score=confirmation_score,
            final_sentiment=final_sentiment,
            quality_confidence=quality_confidence
        )
    
    def _calculate_base_sentiment_optimized(self, text: str) -> float:
        """Optimized base sentiment calculation using frozenset intersections"""
        # Convert text to set of words for efficient intersection operations
        text_words = set(text.split())
        
        # Use frozenset intersections for fast counting
        positive_count = len(text_words & self.keyword_analyzer.power_positive_keywords) * 2
        negative_count = len(text_words & self.keyword_analyzer.power_negative_keywords) * 2
        
        # Add biotech catalysts
        positive_count += len(text_words & self.keyword_analyzer.biotech_catalysts['positive']) * 1.5
        negative_count += len(text_words & self.keyword_analyzer.biotech_catalysts['negative']) * 1.5
        
        # Add tech catalysts
        positive_count += len(text_words & self.keyword_analyzer.tech_catalysts['positive']) * 1.2
        negative_count += len(text_words & self.keyword_analyzer.tech_catalysts['negative']) * 1.2
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _calculate_magnitude_score_optimized(self, text: str, topic: str) -> float:
        """Optimized magnitude score calculation"""
        magnitude = 1.0
        
        # Check for magnitude amplifiers using dictionary lookup
        for word, multiplier in self.keyword_analyzer.magnitude_amplifiers.items():
            if word in text:
                magnitude = max(magnitude, multiplier)
        
        # Check percentage patterns using pre-compiled regex
        for pattern in self.keyword_analyzer.percentage_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and match[0].replace('.', '').isdigit():
                    pct = float(match[0])
                    if pct > 10:
                        magnitude = max(magnitude, 1.4)
                    elif pct > 5:
                        magnitude = max(magnitude, 1.2)
        
        # Topic-specific adjustments
        if topic == 'biotech' and any(word in text for word in [FDA_APPROVAL, 'breakthrough']):
            magnitude *= 1.4
        elif topic == 'earnings' and any(word in text for word in ['blowout', 'crush']):
            magnitude *= 1.25
        
        return min(magnitude, 2.0) / 2.0  # Normalize to 0-1
    
    def _calculate_urgency_score_optimized(self, text: str, title: str) -> float:
        """Optimized urgency score calculation"""
        urgency = 0.5
        
        # Dictionary-based lookup for urgency keywords
        for word, score in self.keyword_analyzer.urgency_keywords.items():
            if word in text or word in title:
                urgency = max(urgency, score)
        
        # Time sensitivity phrases
        for phrase, score in self.keyword_analyzer.time_sensitivity.items():
            if phrase in text:
                urgency = max(urgency, score)
        
        # Breaking news bonus
        title_lower = title.lower()
        if 'breaking' in title_lower or 'just in' in title_lower:
            urgency = min(urgency * 1.15, 1.0)
        
        return urgency
    
    def _calculate_credibility_score_optimized(self, text: str, source: str, content: str) -> float:
        """Optimized credibility score calculation"""
        credibility = 0.5
        source_lower = source.lower()
        
        # Source credibility using frozenset lookups
        source_weights = {'tier_1': 1.0, 'tier_2': 0.8, 'tier_3': 0.6, 'company_direct': 1.15}
        
        for tier, sources in self.keyword_analyzer.credible_sources.items():
            if any(credible_source in source_lower for credible_source in sources):
                credibility = max(credibility, source_weights[tier])
                break
        
        # Content credibility markers using frozenset intersections
        text_words = set(text.split())
        credibility_weights = {'high': 0.85, 'medium': 0.65, 'low': 0.25}
        
        for level, markers in self.keyword_analyzer.credibility_markers.items():
            if text_words & markers:
                if level == 'low':
                    credibility = min(credibility, credibility_weights[level])
                else:
                    credibility = max(credibility, credibility_weights[level])
        
        # Content length factor
        if len(content) > 400:
            credibility = min(credibility * 1.08, 1.0)
        elif len(content) < 80:
            credibility *= 0.85
        
        return credibility
    
    def _calculate_sector_relevance_optimized(self, text: str, topic: str) -> float:
        """Optimized sector relevance calculation"""
        relevance = 0.5
        
        if topic in self.keyword_analyzer.sector_keywords:
            sector_data = self.keyword_analyzer.sector_keywords[topic]
            text_words = set(text.split())
            
            # Use frozenset intersections for efficient counting
            high_impact_count = len(text_words & sector_data['high_impact'])
            catalyst_count = len(text_words & sector_data['catalysts'])
            
            if high_impact_count > 0:
                relevance = max(relevance, 0.75)
            if catalyst_count > 0:
                relevance = max(relevance, 0.85)
        
        return relevance
    
    def _calculate_surprise_factor_optimized(self, text: str) -> float:
        """Optimized surprise factor calculation"""
        surprise_indicators = frozenset([
            'unexpected', 'surprise', 'shocking', 'unprecedented', 'unusual',
            'rare', 'first time', 'never before', 'breaking news'
        ])
        
        text_words = set(text.split())
        if text_words & surprise_indicators:
            return 0.75
        
        return 0.5
    
    def _get_confirmation_score_optimized(self, symbol: str, base_sentiment: float, 
                                        topic: str, recent_analyses: Optional[List]) -> float:
        """Optimized confirmation score calculation"""
        if recent_analyses and len(recent_analyses) > 1:
            analysis_dict = {
                'symbol': symbol,
                'sentiment_score': base_sentiment,
                'timestamp': datetime.now(),
                'topic': topic
            }
            return self.source_validator.check_multi_source_confirmation(
                analysis_dict, recent_analyses[-10:]  # Only check last 10 for efficiency
            )
        return 0.5
    
    def _calculate_weighted_sentiment_optimized(self, base_sentiment: float, magnitude_score: float,
                                              credibility_score: float, sector_relevance: float,
                                              confirmation_score: float, urgency_score: float,
                                              impact_multiplier: float) -> float:
        """Optimized weighted sentiment calculation"""
        # Pre-defined weights for efficiency
        weights = (0.3, 0.18, 0.18, 0.14, 0.12, 0.08)  # base, magnitude, credibility, sector, confirmation, urgency
        
        weighted_sentiment = (
            base_sentiment * weights[0] +
            base_sentiment * magnitude_score * weights[1] +
            base_sentiment * credibility_score * weights[2] +
            base_sentiment * sector_relevance * weights[3] +
            base_sentiment * confirmation_score * weights[4] +
            base_sentiment * urgency_score * weights[5]
        ) * impact_multiplier
        
        return max(-1.0, min(1.0, weighted_sentiment))
    
    def _calculate_quality_confidence_optimized(self, credibility_score: float, confirmation_score: float,
                                              sector_relevance: float, magnitude_score: float,
                                              historical_accuracy: float) -> float:
        """Optimized quality confidence calculation"""
        # Pre-defined weights for efficiency
        weights = (0.28, 0.22, 0.18, 0.14, 0.18)  # credibility, confirmation, sector, magnitude, historical
        
        return (
            credibility_score * weights[0] +
            confirmation_score * weights[1] +
            sector_relevance * weights[2] +
            magnitude_score * weights[3] +
            historical_accuracy * weights[4]
        )


class OptimizedEnhancedNewsAnalyzer:
    """Optimized news analyzer with improved performance and memory management"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized model loading and caching"""
        self.fmp_loader = fmp_loader
        self.technical_analyzer = TechnicalAnalyzer(fmp_loader)
        self.technical_strategies = TechnicalStrategies(fmp_loader)
        self.gemini_analyzer = OptimizedGeminiNewsAnalyzer()
        self.quality_scorer = OptimizedNewsQualityScorer()
        self.recent_analyses_cache = []  # Limited cache for multi-source validation
        self.max_cache_size = 50  # Reduced for memory efficiency
        
        # Optimized device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.finbert_labels = ["positive", "negative", "neutral"]
        
        # Initialize FinBERT with optimizations
        self._load_finbert_optimized()
        
        # Initialize optimized keywords
        self._initialize_optimized_keywords()
        self._enhance_existing_keywords()
    
    def _load_finbert_optimized(self) -> None:
        """Load FinBERT model with optimized configuration"""
        try:
            log_info("Loading FinBERT model with optimizations...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                "ProsusAI/finbert", cache_dir="./cache"
            )
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert", cache_dir="./cache"
            )
            self.finbert_model.to(self.device)
            self.finbert_model.eval()
            
            # Optimize for inference if CUDA available
            if hasattr(torch, 'jit') and self.device == "cuda":
                self.finbert_model = torch.jit.optimize_for_inference(self.finbert_model)
            
            # Set model to half precision for memory efficiency if CUDA
            if self.device == "cuda":
                self.finbert_model = self.finbert_model.half()
            
            log_info(f"FinBERT loaded successfully on {self.device}")
            
        except Exception as e:
            log_error(f"Failed to load FinBERT: {e}")
            log_warning("Falling back to keyword-only analysis")
            self.finbert_model = None
            self.finbert_tokenizer = None
    
    def _initialize_optimized_keywords(self) -> None:
        """Initialize optimized keyword sets using frozensets for faster lookups"""
        self.positive_keywords = frozenset([
            'beats', 'beat', 'exceeds', 'exceed', 'raises', 'upgrade', 'approval', 'approved',
            'growth', 'strong', 'positive', 'success', 'breakthrough', 'innovation',
            'revenue increase', 'profit increase', 'buyback', 'dividend', 'expansion',
            'outperform', 'surge', 'rally', 'boost', 'gain', 'rise', 'soar', 'bullish',
            'record', 'milestone', 'achievement', 'partnership', 'collaboration', 'deal',
            'cheap', 'undervalued', 'good buy', 'bargain', 'discount', 'value',
            'attractive price', 'buying opportunity', 'oversold', WORTH_BUYING,
            TIME_TO_BUY, 'attractive valuation', 'compelling value', 'good value',
            'reasonable price', 'fair value', 'target price increase', 'price target raised',
            'buy rating', 'accumulate', 'overweight', 'recommend buy'
        ])
        
        self.negative_keywords = frozenset([
            'misses', 'miss', 'falls short', 'disappointing', 'decline', 'drop',
            'downgrade', 'concern', 'loss', 'cut', 'reduce', 'weak', 'struggle',
            'investigation', 'lawsuit', 'recall', 'bankruptcy', 'layoffs',
            'plunge', 'crash', 'fall', 'slump', 'tumble', 'sink', 'bearish',
            'warning', 'delay', 'setback', 'failure', 'reject', 'denied',
            'overvalued', 'expensive', 'overpriced', 'sell rating', 'avoid',
            'sell recommendation', 'underweight', 'price target cut',
            'target price lowered', 'poor value', 'too expensive', 'risky buy'
        ])
        
        self.biotech_keywords = frozenset([
            'fda', 'approval', 'phase', 'trial', 'clinical', 'drug', 'therapy',
            'treatment', 'efficacy', 'safety', 'regulatory', 'orphan drug',
            'breakthrough therapy', FAST_TRACK, PRIORITY_REVIEW, 'biologics',
            'pipeline', 'indication', 'endpoint', 'biomarker'
        ])
        
        # Pre-compile topic patterns for better performance
        self._topic_patterns = {
            'earnings': re.compile(r'\b(earnings|revenue|profit|eps|quarterly|guidance|beat|miss)\b', re.IGNORECASE),
            'biotech': re.compile(r'\b(fda|approval|phase|trial|clinical|drug|therapy|efficacy)\b', re.IGNORECASE),
            'ma': re.compile(r'\b(merger|acquisition|deal|buyout|takeover|acquire|merge)\b', re.IGNORECASE),
            'analyst': re.compile(r'\b(upgrade|downgrade|target|analyst|rating|price target)\b', re.IGNORECASE),
            'corporate_action': re.compile(r'\b(dividend|buyback|split|spinoff|distribution)\b', re.IGNORECASE),
            'business_development': re.compile(r'\b(contract|partnership|agreement|collaboration|alliance)\b', re.IGNORECASE)
        }
    
    def _enhance_existing_keywords(self) -> None:
        """Enhance existing keywords with optimized power keywords"""
        power_positive = frozenset([
            'crushes', 'smashes', 'demolishes', 'blowout', 'blockbuster', 'stellar',
            'phenomenal', 'explosive growth', 'cash flow surge', 'profit explosion',
            'market domination', 'game changer', 'revolutionary', 'unprecedented growth',
            'record-breaking', 'all-time high', 'massive beat', 'huge surprise'
        ])
        
        power_negative = frozenset([
            'catastrophic', 'devastating', 'collapse', 'free fall', 'plummeting',
            'cash crunch', 'liquidity crisis', 'massive losses', 'debt spiral',
            'losing market share', 'regulatory nightmare', 'investigation launched',
            'nightmare', 'cratering', 'imploding', 'hemorrhaging'
        ])
        
        # Use union for efficient set combination
        self.positive_keywords = self.positive_keywords | power_positive
        self.negative_keywords = self.negative_keywords | power_negative
        
        # Add biotech-specific keywords
        biotech_positive = frozenset([
            FDA_APPROVAL, 'breakthrough designation', FAST_TRACK, PRIORITY_REVIEW,
            'pdufa date', 'nda accepted', 'meets endpoints', 'statistically significant',
            'regulatory approval', 'cms approval', 'reimbursement approved'
        ])
        
        biotech_negative = frozenset([
            'fda rejection', 'complete response letter', 'crl', 'safety hold',
            'trial failure', 'missed endpoints', 'adverse events', 'trial halted',
            'regulatory delay', 'black box warning'
        ])
        
        self.biotech_keywords = self.biotech_keywords | biotech_positive | biotech_negative
        
        log_info(f"Enhanced keywords: {len(self.positive_keywords)} positive, {len(self.negative_keywords)} negative")
    
    def _get_finbert_sentiment_optimized(self, text: str) -> Tuple[float, float]:
        """Optimized FinBERT sentiment analysis with memory management"""
        if not self.finbert_model or not self.finbert_tokenizer:
            return 0.0, 0.0
        
        try:
            # Truncate text for efficiency
            text = text.strip()[:384]  # Reduced from 512 for better performance
            
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
            
            probs = predictions.cpu().float().numpy()[0]  # Convert to float32 for consistency
            pos_prob, neg_prob, _ = probs
            
            sentiment = float(pos_prob - neg_prob)
            confidence = float(max(probs))
            
            return sentiment, confidence
            
        except Exception as e:
            log_error(f"FinBERT analysis error: {e}")
            return 0.0, 0.0
    
    def _calculate_keyword_score_optimized(self, text: str) -> Tuple[float, float]:
        """Optimized keyword-based sentiment analysis using set operations"""
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Use set intersections for efficient counting
        positive_matches = len(text_words & self.positive_keywords)
        negative_matches = len(text_words & self.negative_keywords)
        biotech_matches = len(text_words & self.biotech_keywords)
        
        # Pre-compiled phrase patterns for efficiency
        positive_phrases = [
            'revenue increase', 'profit increase', 'breakthrough therapy', 'good buy',
            'buying opportunity', 'attractive price', WORTH_BUYING, TIME_TO_BUY,
            'attractive valuation', 'compelling value', 'good value', 'reasonable price',
            'target price raised', 'price target increase', 'buy rating'
        ]
        
        negative_phrases = [
            'falls short', 'price target cut', 'target price lowered', 'sell rating',
            'sell recommendation', 'poor value', 'too expensive', 'risky buy'
        ]
        
        # Efficient phrase checking with early termination
        clean_text = re.sub(r'[^\w\s]', '', text_lower)
        
        for phrase in positive_phrases:
            if phrase in clean_text:
                positive_matches += 1
        
        for phrase in negative_phrases:
            if phrase in clean_text:
                negative_matches += 1
        
        # Calculate sentiment
        total_keywords = positive_matches + negative_matches
        if total_keywords == 0:
            sentiment = 0.0
            confidence = 0.1
        else:
            sentiment = (positive_matches - negative_matches) / total_keywords
            confidence = min(total_keywords / 4.0, 1.0)  # Slightly reduced for efficiency
        
        # Biotech boost
        if biotech_matches > 0:
            confidence = min(confidence * 1.25, 1.0)
        
        return sentiment, confidence
    
    def _calculate_enhanced_sentiment_optimized(self, title: str, content: str, symbol: str,
                                              current_price: float, topic: str, 
                                              source: str = "") -> Tuple[float, float]:
        """Optimized enhanced sentiment analysis"""
        
        enhanced_score = self.quality_scorer.calculate_enhanced_sentiment(
            title=title, content=content, symbol=symbol,
            current_price=current_price, topic=topic,
            source=source, recent_analyses=self.recent_analyses_cache[-10:]  # Only last 10
        )
        
        # Efficient cache management
        current_time = datetime.now()
        self.recent_analyses_cache.append({
            'symbol': symbol,
            'sentiment_score': enhanced_score.final_sentiment,
            'timestamp': current_time,
            'topic': topic,
            'quality_confidence': enhanced_score.quality_confidence
        })
        
        # Keep cache size under control
        if len(self.recent_analyses_cache) > self.max_cache_size:
            # Remove oldest half
            self.recent_analyses_cache = self.recent_analyses_cache[-self.max_cache_size//2:]
        
        # Log high-quality signals efficiently
        if enhanced_score.quality_confidence > 0.68:
            log_info(f"HIGH QUALITY signal for {symbol}: "
                    f"final_sentiment={enhanced_score.final_sentiment:.3f}, "
                    f"quality={enhanced_score.quality_confidence:.3f}")
        
        return enhanced_score.final_sentiment, enhanced_score.quality_confidence
    
    @lru_cache(maxsize=500)  # Reduced cache size for memory efficiency
    def _detect_topic_optimized(self, text: str) -> str:
        """Optimized topic detection with cached regex patterns"""
        text_lower = text.lower()
        
        for topic, pattern in self._topic_patterns.items():
            if pattern.search(text_lower):
                return topic
        
        return 'general'
    
    def _enhanced_ensemble_score_optimized(self, finbert_sentiment: float, finbert_conf: float,
                                         enhanced_sentiment: float, enhanced_conf: float,
                                         gemini_sentiment: float, gemini_conf: float) -> Tuple[float, float]:
        """Optimized ensemble scoring with pre-computed weights"""
        
        # Pre-computed weights for efficiency
        finbert_weight = CONFIG.finbert_weight
        enhanced_weight = CONFIG.keyword_weight * 1.15  # Slight boost for enhanced
        gemini_weight = CONFIG.gemini_weight
        
        # Calculate weighted sentiment efficiently
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
        
        # Optimized confidence calculation
        components = [conf for conf in [finbert_conf, enhanced_conf, gemini_conf] if conf > 0]
        if not components:
            return ensemble_sentiment, 0.0
        
        base_confidence = sum(components) / len(components)
        quality_bonus = 1.0 + (enhanced_conf * 0.25)
        
        # Simplified agreement bonus
        sentiments = [s for s, c in [(finbert_sentiment, finbert_conf), 
                                   (enhanced_sentiment, enhanced_conf),
                                   (gemini_sentiment, gemini_conf)] if c > 0]
        
        agreement_bonus = 1.25 if len(sentiments) >= 2 and all(
            (s > 0) == (sentiments[0] > 0) for s in sentiments
        ) else 1.0
        
        gemini_bonus = 1.15 if gemini_conf > 0.65 else 1.0
        
        final_confidence = min(base_confidence * quality_bonus * agreement_bonus * gemini_bonus, 1.0)
        
        return ensemble_sentiment, final_confidence
    
    def _combine_news_technical_confidence_optimized(self, news_confidence: float, 
                                                   technical_analysis: Optional[TechnicalAnalysis],
                                                   sentiment_score: float,
                                                   strategy_signals: Optional[List[StrategySignal]] = None) -> float:
        """Optimized confidence combination"""
        if technical_analysis is None:
            return news_confidence * 0.65
        
        tech_confidence = technical_analysis.technical_confidence
        
        # Optimized momentum alignment
        momentum_alignment = 1.2 if (
            abs(sentiment_score) > 0.25 and abs(technical_analysis.momentum_score) > 0.15 and
            (sentiment_score > 0) == (technical_analysis.momentum_score > 0)
        ) else (0.75 if abs(sentiment_score) > 0.3 and abs(technical_analysis.momentum_score) > 0.2 and
                (sentiment_score > 0) != (technical_analysis.momentum_score > 0) else 1.0)
        
        volume_bonus = 1.08 if technical_analysis.volume_score > 0.65 else 1.0
        liquidity_factor = max(technical_analysis.liquidity_score, 0.25)
        
        # Optimized strategy bonus
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
        """Optimized news analysis with batch processing and memory management"""
        if news_df is None or news_df.empty:
            return []
        
        log_info(f"Analyzing {len(news_df)} news articles with OPTIMIZED Enhanced Sentiment + Technical Analysis")
        
        # Create optimized price lookup
        price_lookup = (
            current_prices.set_index('symbol').to_dict('index') 
            if current_prices is not None and not current_prices.empty 
            else {}
        )
        
        # Process articles with optimized memory management
        results = []
        for _, row in news_df.iterrows():
            try:
                analysis = self._process_single_article_optimized(row, price_lookup)
                if analysis:
                    results.append(analysis)
                    
                    if analysis.combined_confidence >= CONFIG.min_confidence_score:
                        self._log_high_confidence_result_optimized(analysis)
                        
            except Exception as e:
                symbol = row.get('symbol', 'UNKNOWN')
                log_error(f"Error analyzing news for {symbol}: {e}")
                continue
        
        return results
    
    def _process_single_article_optimized(self, row: pd.Series, price_lookup: Dict) -> Optional[EnhancedNewsAnalysis]:
        """Optimized single article processing"""
        # Extract and validate essential fields
        symbol = str(row.get('symbol', '')).strip()
        title = str(row.get('title', '')).strip()
        content = str(row.get('content', '')).strip()
        
        if not all([symbol, title, content]):
            return None
        
        # Get price data and topic
        price_data = price_lookup.get(symbol)
        current_price = float(price_data.get('lastSalePrice', 0)) if price_data else 0.0
        topic = self._detect_topic_optimized(f"{title} {content}")
        source = str(row.get('source', '')).strip()
        
        # Optimized sentiment analysis
        full_text = f"{title} {content}"
        
        # FinBERT analysis
        finbert_sentiment, finbert_conf = self._get_finbert_sentiment_optimized(full_text)
        
        # Enhanced sentiment analysis
        enhanced_sentiment, quality_confidence = self._calculate_enhanced_sentiment_optimized(
            title, content, symbol, current_price, topic, source
        )
        
        # Gemini analysis (with rate limiting)
        gemini_sentiment, gemini_conf, gemini_reasoning = self._perform_gemini_analysis_optimized(
            symbol, title, content, current_price, topic
        )
        
        # Optimized ensemble scoring
        final_sentiment, news_confidence = self._enhanced_ensemble_score_optimized(
            finbert_sentiment, finbert_conf,
            enhanced_sentiment, quality_confidence,
            gemini_sentiment, gemini_conf
        )
        
        # Technical analysis
        technical_analysis = self._perform_technical_analysis_optimized(symbol, price_data)
        
        # Strategy analysis
        strategy_signals, best_strategy = self._perform_strategy_analysis_optimized(
            symbol, current_price, technical_analysis, final_sentiment
        )
        
        # Combined confidence calculation
        combined_confidence = self._combine_news_technical_confidence_optimized(
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
            keyword_score=enhanced_sentiment,
            gemini_score=gemini_sentiment,
            gemini_confidence=gemini_conf,
            gemini_reasoning=gemini_reasoning,
            technical_analysis=technical_analysis,
            strategy_signals=strategy_signals or [],
            best_strategy=best_strategy,
            combined_confidence=combined_confidence
        )
    
    def _perform_gemini_analysis_optimized(self, symbol: str, title: str, content: str,
                                         current_price: float, topic: str) -> Tuple[float, float, str]:
        """Optimized Gemini analysis with better rate limiting"""
        if self.gemini_analyzer.enabled and current_price > 0:
            try:
                gemini_analysis = self.gemini_analyzer.analyze_news(
                    symbol=symbol, title=title[:150], content=content[:600],  # Truncate for efficiency
                    current_price=current_price, topic=topic
                )
                if gemini_analysis:
                    return (gemini_analysis.sentiment_score,
                           gemini_analysis.confidence,
                           gemini_analysis.reasoning[:100])  # Truncate reasoning
            except Exception as e:
                log_warning(f"Gemini analysis failed for {symbol}: {e}")
        
        return 0.0, 0.0, ""
    
    def _perform_technical_analysis_optimized(self, symbol: str, price_data) -> Optional[TechnicalAnalysis]:
        """Optimized technical analysis"""
        if price_data is not None:
            return self.technical_analyzer.analyze_symbol(symbol, pd.Series(price_data))
        return None
    
    def _perform_strategy_analysis_optimized(self, symbol: str, current_price: float,
                                           technical_analysis: Optional[TechnicalAnalysis],
                                           sentiment_score: float) -> Tuple[List[StrategySignal], Optional[StrategySignal]]:
        """Optimized strategy analysis"""
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
    
    def _log_high_confidence_result_optimized(self, analysis: EnhancedNewsAnalysis) -> None:
        """Optimized high confidence result logging"""
        # Optimized technical info formatting
        tech_info = ""
        if analysis.technical_analysis:
            tech_info = (f"tech_conf={analysis.technical_analysis.technical_confidence:.2f} "
                        f"momentum={analysis.technical_analysis.momentum_score:.2f} "
                        f"liquidity={analysis.technical_analysis.liquidity_score:.2f}")
        
        # Optimized gemini info formatting
        gemini_info = ""
        if analysis.gemini_confidence > 0:
            gemini_info = f"gemini={analysis.gemini_score:.2f}({analysis.gemini_confidence:.2f}) "
        
        # Optimized strategy info formatting
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
        
        # Log reasoning and strategy summary if available
        if analysis.gemini_reasoning:
            log_info(f"  Gemini: {analysis.gemini_reasoning[:120]}...")
        
        if analysis.best_strategy:
            strategy_summary = self.technical_strategies.format_strategy_summary(analysis.best_strategy)
            log_info(f"  Strategy: {strategy_summary}")
    
    def filter_high_confidence(self, analyses: List[EnhancedNewsAnalysis], 
                             use_combined_confidence: bool = True) -> List[EnhancedNewsAnalysis]:
        """Optimized high confidence filtering"""
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
            self._log_detailed_signal_analysis_optimized(analyses, confidence_attr, threshold)
        
        return high_conf
    
    def _log_detailed_signal_analysis_optimized(self, analyses: List[EnhancedNewsAnalysis], 
                                              confidence_attr: str, threshold: float) -> None:
        """Optimized detailed signal analysis logging"""
        log_info("=" * 60)
        log_info("DEBUG: OPTIMIZED SIGNAL ANALYSIS")
        log_info("=" * 60)
        
        if not analyses:
            log_info("DEBUG: No analyses to filter")
            return
        
        scores = [getattr(a, confidence_attr, 0) for a in analyses]
        max_score = max(scores) if scores else 0
        log_info(f"THRESHOLD: {threshold:.3f} | HIGHEST {confidence_attr.upper()}: {max_score:.3f}")
        
        # Only log top 5 for efficiency
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
        log_info("OPTIMIZATION SUGGESTIONS:")
        log_info(f"1. Lower threshold from {threshold} to 0.40 in config.py")
        log_info("2. Wait for stronger sentiment signals")
        log_info("3. Ensure market conditions are favorable")
        log_info("=" * 60)


# Create alias for backwards compatibility
EnhancedNewsAnalyzer = OptimizedEnhancedNewsAnalyzer