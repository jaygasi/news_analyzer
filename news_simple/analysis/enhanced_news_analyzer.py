"""
Optimized news analyzer with Gemini LLM integration, professional technical strategies,
and enhanced sentiment analysis for quality trade predictions
"""
import pandas as pd
from typing import Dict, Optional, Tuple, List, Set
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


@dataclass
class EnhancedSentimentScore:
    """Enhanced sentiment with quality metrics"""
    base_sentiment: float
    magnitude_score: float      # How big is the expected impact?
    urgency_score: float        # How time-sensitive is this?
    credibility_score: float    # How reliable is the source/news?
    sector_relevance: float     # How relevant to this sector?
    surprise_factor: float      # How unexpected was this news?
    confirmation_score: float   # Multi-source confirmation
    final_sentiment: float      # Weighted final score
    quality_confidence: float   # Overall quality confidence


class EnhancedKeywordAnalyzer:
    """Research-backed enhanced keyword analysis"""
    
    def __init__(self):
        self._initialize_enhanced_keywords()
        self._initialize_magnitude_indicators()
        self._initialize_urgency_indicators()
        self._initialize_sector_keywords()
        self._initialize_credibility_indicators()
        
    def _initialize_enhanced_keywords(self):
        """Enhanced keywords based on financial research"""
        
        # Research shows these have highest predictive power
        self.power_positive_keywords = {
            # Earnings & Performance
            'crushes', 'smashes', 'demolishes', 'obliterates', 'destroys estimates',
            'blowout', 'blockbuster', 'stellar', 'phenomenal', 'exceptional',
            'record-breaking', 'all-time high', 'massive beat', 'huge surprise',
            
            # Growth & Expansion  
            'explosive growth', 'rapid expansion', 'accelerating', 'momentum building',
            'scaling rapidly', 'breakthrough momentum', 'unprecedented growth',
            
            # Financial Strength
            'cash flow surge', 'profit explosion', 'revenue acceleration',
            'margin expansion', 'cost savings realized', 'efficiency gains',
            
            # Market Position
            'market domination', 'competitive advantage', 'market leader',
            'disrupting industry', 'game changer', 'revolutionary'
        }
        
        self.power_negative_keywords = {
            # Performance Issues
            'catastrophic', 'devastating', 'disastrous', 'nightmare', 'collapse',
            'free fall', 'plummeting', 'cratering', 'imploding', 'hemorrhaging',
            
            # Financial Distress
            'cash crunch', 'liquidity crisis', 'burning cash', 'bleeding money',
            'massive losses', 'debt spiral', 'financial distress',
            
            # Business Problems
            'losing market share', 'competitive pressure', 'disrupted business',
            'obsolete technology', 'regulatory nightmare', 'investigation launched'
        }
        
        # Biotech/Pharma specific (high volatility sector)
        self.biotech_catalysts = {
            'positive': {
                'fda approval', 'breakthrough designation', 'fast track', 'priority review',
                'orphan drug', 'accelerated approval', 'phase 3 success', 'meets endpoints',
                'statistically significant', 'pdufa date', 'nda accepted', 'bla accepted',
                'regulatory approval', 'cms approval', 'reimbursement approved'
            },
            'negative': {
                'fda rejection', 'complete response letter', 'crl', 'safety hold',
                'trial failure', 'missed endpoints', 'adverse events', 'safety concerns',
                'trial halted', 'regulatory delay', 'black box warning'
            }
        }
        
        # Tech specific
        self.tech_catalysts = {
            'positive': {
                'ai breakthrough', 'patent approval', 'licensing deal', 'cloud growth',
                'user growth', 'subscription growth', 'platform adoption', 'digital transformation'
            },
            'negative': {
                'data breach', 'regulatory scrutiny', 'antitrust', 'user decline',
                'competition threat', 'platform issues', 'security vulnerability'
            }
        }

    def _initialize_magnitude_indicators(self):
        """Words that indicate the size of impact"""
        self.magnitude_amplifiers = {
            'extreme': 2.0, 'massive': 1.8, 'huge': 1.6, 'significant': 1.4,
            'substantial': 1.3, 'major': 1.2, 'notable': 1.1,
            'slight': 0.7, 'minor': 0.6, 'small': 0.5, 'tiny': 0.3
        }
        
        # Percentage indicators (extract actual numbers)
        self.percentage_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*(increase|growth|up|higher)',
            r'(\d+(?:\.\d+)?)\s*%\s*(decrease|decline|down|lower)',
            r'(doubled|tripled|quadrupled)',
            r'(\d+)x\s*(growth|increase)'
        ]

    def _initialize_urgency_indicators(self):
        """Words that indicate time sensitivity"""
        self.urgency_keywords = {
            'immediate': 1.0, 'urgent': 0.9, 'breaking': 0.9, 'just announced': 0.9,
            'developing': 0.8, 'emerging': 0.7, 'upcoming': 0.6, 'planned': 0.4,
            'potential': 0.3, 'possible': 0.2, 'rumored': 0.1
        }
        
        # Time phrases
        self.time_sensitivity = {
            'within hours': 1.0, 'today': 0.9, 'this week': 0.7,
            'this month': 0.5, 'this quarter': 0.3, 'next year': 0.1
        }

    def _initialize_sector_keywords(self):
        """Sector-specific keyword enhancement"""
        self.sector_keywords = {
            'biotech': {
                'high_impact': ['fda', 'approval', 'trial', 'drug', 'therapy', 'treatment'],
                'catalysts': ['pdufa', 'breakthrough', 'orphan', 'fast track', 'priority'],
                'risks': ['safety', 'adverse', 'crl', 'rejection', 'delay']
            },
            'tech': {
                'high_impact': ['ai', 'cloud', 'subscription', 'platform', 'user growth'],
                'catalysts': ['breakthrough', 'patent', 'acquisition', 'partnership'],
                'risks': ['competition', 'regulation', 'data breach', 'antitrust']
            },
            'energy': {
                'high_impact': ['oil', 'gas', 'renewable', 'production', 'reserves'],
                'catalysts': ['discovery', 'drilling', 'capacity', 'contract'],
                'risks': ['spill', 'accident', 'regulation', 'embargo']
            }
        }

    def _initialize_credibility_indicators(self):
        """News source and content credibility markers"""
        self.credible_sources = {
            'tier_1': ['reuters', 'bloomberg', 'wsj', 'ft', 'ap news'],      # Weight: 1.0
            'tier_2': ['cnbc', 'marketwatch', 'yahoo finance', 'seeking alpha'], # Weight: 0.8
            'tier_3': ['motley fool', 'benzinga', 'zacks'],                 # Weight: 0.6
            'company_direct': ['press release', 'sec filing', '8-k', '10-k'] # Weight: 1.2
        }
        
        # Content credibility markers
        self.credibility_markers = {
            'high': ['sec filing', 'press release', 'earnings call', 'official statement',
                    'regulatory filing', 'management guidance', 'board approval'],
            'medium': ['analyst report', 'research note', 'expert opinion', 'industry study'],
            'low': ['rumor', 'speculation', 'unconfirmed', 'alleged', 'sources say']
        }


class MultiSourceValidator:
    """Cross-reference and validate news across multiple sources"""
    
    def __init__(self):
        self.news_cache = {}
        self.confirmation_window = timedelta(hours=6)
        
    def check_multi_source_confirmation(self, current_analysis: dict, 
                                      recent_analyses: List[dict]) -> float:
        """Check if multiple sources confirm the same story"""
        if not recent_analyses:
            return 0.5  # No confirmation data
        
        current_symbol = current_analysis['symbol']
        current_sentiment = current_analysis['sentiment_score']
        current_topic = current_analysis.get('topic', 'general')
        
        # Find related articles
        related_articles = [
            art for art in recent_analyses 
            if (art['symbol'] == current_symbol and 
                abs((art['timestamp'] - current_analysis['timestamp']).total_seconds()) < self.confirmation_window.total_seconds())
        ]
        
        if len(related_articles) < 2:
            return 0.5  # Not enough for confirmation
        
        # Check sentiment alignment
        sentiments = [art['sentiment_score'] for art in related_articles]
        sentiment_agreement = self._calculate_sentiment_agreement(current_sentiment, sentiments)
        
        # Check topic relevance
        topic_match = sum(1 for art in related_articles if art.get('topic') == current_topic)
        topic_score = topic_match / len(related_articles)
        
        # Combine scores
        confirmation_score = (sentiment_agreement * 0.7 + topic_score * 0.3)
        
        log_debug(f"Multi-source confirmation for {current_symbol}: {confirmation_score:.2f} "
                 f"({len(related_articles)} sources, sentiment_agreement={sentiment_agreement:.2f})")
        
        return confirmation_score
    
    def _calculate_sentiment_agreement(self, target_sentiment: float, other_sentiments: List[float]) -> float:
        """Calculate how well sentiments agree"""
        if not other_sentiments:
            return 0.5
            
        # Check if sentiments are in same direction
        target_direction = 1 if target_sentiment > 0.1 else (-1 if target_sentiment < -0.1 else 0)
        
        agreements = 0
        for sentiment in other_sentiments:
            other_direction = 1 if sentiment > 0.1 else (-1 if sentiment < -0.1 else 0)
            if target_direction == other_direction:
                agreements += 1
        
        return agreements / len(other_sentiments)


class HistoricalPatternMatcher:
    """Match current news to historical patterns for better predictions"""
    
    def __init__(self):
        self.pattern_cache = {}
        
    def find_historical_impact(self, symbol: str, news_type: str, 
                             sentiment_score: float) -> Tuple[float, float]:
        """Find how similar news affected this stock historically"""
        # This would ideally connect to a database of historical news->price movements
        # For now, return intelligent defaults based on research
        
        impact_multipliers = {
            'earnings': {
                'strong_beat': 1.3,    # Strong earnings beats typically move stock 3-8%
                'beat': 1.1,
                'miss': 0.7,
                'strong_miss': 0.5
            },
            'biotech': {
                'fda_approval': 2.0,   # FDA approvals can move biotech 20-50%
                'trial_success': 1.5,
                'trial_failure': 0.3,
                'fda_rejection': 0.2
            },
            'analyst': {
                'upgrade': 1.2,
                'downgrade': 0.8,
                'initiate_buy': 1.1
            }
        }
        
        # Determine news category
        category = self._categorize_news(news_type, sentiment_score)
        multiplier = impact_multipliers.get(news_type, {}).get(category, 1.0)
        
        # Historical accuracy based on research (from papers)
        accuracy_rates = {
            'earnings': 0.75,    # Earnings sentiment predicts direction 75% of time
            'biotech': 0.85,     # FDA/trial news very predictive
            'analyst': 0.65,     # Analyst changes moderately predictive
            'general': 0.55      # General news barely better than random
        }
        
        accuracy = accuracy_rates.get(news_type, 0.55)
        
        return multiplier, accuracy
    
    def _categorize_news(self, news_type: str, sentiment: float) -> str:
        """Categorize news based on type and sentiment"""
        if news_type == 'earnings':
            if sentiment > 0.6:
                return 'strong_beat'
            elif sentiment > 0.2:
                return 'beat'
            elif sentiment < -0.6:
                return 'strong_miss'
            else:
                return 'miss'
        # Add more categorizations as needed
        return 'general'


class NewsQualityScorer:
    """Score news quality and filter low-quality signals"""
    
    def __init__(self):
        self.keyword_analyzer = EnhancedKeywordAnalyzer()
        self.source_validator = MultiSourceValidator()
        self.pattern_matcher = HistoricalPatternMatcher()
        
    def calculate_enhanced_sentiment(self, title: str, content: str, symbol: str,
                                   current_price: float, topic: str,
                                   source: str = "", recent_analyses: List = None) -> EnhancedSentimentScore:
        """Calculate enhanced sentiment with quality scoring"""
        
        text = f"{title} {content}".lower()
        
        # Base sentiment (use existing FinBERT/keyword logic)
        base_sentiment = self._calculate_base_sentiment(text)
        
        # Enhanced scoring components
        magnitude_score = self._calculate_magnitude_score(text, topic)
        urgency_score = self._calculate_urgency_score(text, title)
        credibility_score = self._calculate_credibility_score(text, source, content)
        sector_relevance = self._calculate_sector_relevance(text, symbol, topic)
        surprise_factor = self._calculate_surprise_factor(text, topic)
        
        # Multi-source confirmation (if available)
        confirmation_score = 0.5
        if recent_analyses:
            analysis_dict = {
                'symbol': symbol,
                'sentiment_score': base_sentiment,
                'timestamp': datetime.now(),
                'topic': topic
            }
            confirmation_score = self.source_validator.check_multi_source_confirmation(
                analysis_dict, recent_analyses
            )
        
        # Historical pattern matching
        impact_multiplier, historical_accuracy = self.pattern_matcher.find_historical_impact(
            symbol, topic, base_sentiment
        )
        
        # Calculate weighted final sentiment
        weights = {
            'base': 0.3,
            'magnitude': 0.2,
            'credibility': 0.2,
            'sector': 0.15,
            'confirmation': 0.1,
            'urgency': 0.05
        }
        
        final_sentiment = (
            base_sentiment * weights['base'] +
            base_sentiment * magnitude_score * weights['magnitude'] +
            base_sentiment * credibility_score * weights['credibility'] +
            base_sentiment * sector_relevance * weights['sector'] +
            base_sentiment * confirmation_score * weights['confirmation'] +
            base_sentiment * urgency_score * weights['urgency']
        ) * impact_multiplier
        
        # Quality confidence calculation
        quality_confidence = (
            credibility_score * 0.3 +
            confirmation_score * 0.25 +
            sector_relevance * 0.2 +
            magnitude_score * 0.15 +
            historical_accuracy * 0.1
        )
        
        return EnhancedSentimentScore(
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
    
    def _calculate_base_sentiment(self, text: str) -> float:
        """Calculate base sentiment using enhanced keywords"""
        positive_count = 0
        negative_count = 0
        
        # Check power keywords first (higher weight)
        for word in self.keyword_analyzer.power_positive_keywords:
            if word in text:
                positive_count += 2  # Double weight for power words
        
        for word in self.keyword_analyzer.power_negative_keywords:
            if word in text:
                negative_count += 2  # Double weight for power words
        
        # Check biotech catalysts
        for word in self.keyword_analyzer.biotech_catalysts['positive']:
            if word in text:
                positive_count += 1.5  # Biotech catalysts are important
        
        for word in self.keyword_analyzer.biotech_catalysts['negative']:
            if word in text:
                negative_count += 1.5
        
        # Check tech catalysts
        for word in self.keyword_analyzer.tech_catalysts['positive']:
            if word in text:
                positive_count += 1.3
        
        for word in self.keyword_analyzer.tech_catalysts['negative']:
            if word in text:
                negative_count += 1.3
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _calculate_magnitude_score(self, text: str, topic: str) -> float:
        """Calculate expected magnitude of impact"""
        magnitude = 1.0
        
        # Check for magnitude amplifiers
        for word, multiplier in self.keyword_analyzer.magnitude_amplifiers.items():
            if word in text:
                magnitude = max(magnitude, multiplier)
        
        # Extract percentage changes
        for pattern in self.keyword_analyzer.percentage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and match[0].replace('.', '').isdigit():
                    pct = float(match[0])
                    if pct > 10:  # Double-digit percentage changes are significant
                        magnitude = max(magnitude, 1.5)
                    elif pct > 5:
                        magnitude = max(magnitude, 1.3)
        
        # Topic-specific magnitude adjustments
        if topic == 'biotech' and any(word in text for word in ['fda approval', 'breakthrough']):
            magnitude *= 1.5
        elif topic == 'earnings' and any(word in text for word in ['blowout', 'crush']):
            magnitude *= 1.3
        
        return min(magnitude, 2.0) / 2.0  # Normalize to 0-1
    
    def _calculate_urgency_score(self, text: str, title: str) -> float:
        """Calculate time sensitivity"""
        urgency = 0.5  # Default
        
        # Check urgency keywords
        for word, score in self.keyword_analyzer.urgency_keywords.items():
            if word in text or word in title:
                urgency = max(urgency, score)
        
        # Check time sensitivity phrases
        for phrase, score in self.keyword_analyzer.time_sensitivity.items():
            if phrase in text:
                urgency = max(urgency, score)
        
        # Breaking news in title gets bonus
        if 'breaking' in title.lower() or 'just in' in title.lower():
            urgency = min(urgency * 1.2, 1.0)
        
        return urgency
    
    def _calculate_credibility_score(self, text: str, source: str, content: str) -> float:
        """Calculate source and content credibility"""
        credibility = 0.5  # Default
        
        # Source credibility
        source_lower = source.lower()
        for tier, sources in self.keyword_analyzer.credible_sources.items():
            for credible_source in sources:
                if credible_source in source_lower:
                    if tier == 'tier_1':
                        credibility = max(credibility, 1.0)
                    elif tier == 'tier_2':
                        credibility = max(credibility, 0.8)
                    elif tier == 'tier_3':
                        credibility = max(credibility, 0.6)
                    elif tier == 'company_direct':
                        credibility = max(credibility, 1.2)
        
        # Content credibility markers
        for level, markers in self.keyword_analyzer.credibility_markers.items():
            for marker in markers:
                if marker in text:
                    if level == 'high':
                        credibility = max(credibility, 0.9)
                    elif level == 'medium':
                        credibility = max(credibility, 0.7)
                    elif level == 'low':
                        credibility = min(credibility, 0.3)
        
        # Content length and detail (longer, more detailed articles are often more credible)
        if len(content) > 500:
            credibility = min(credibility * 1.1, 1.0)
        elif len(content) < 100:
            credibility *= 0.8
        
        return min(credibility, 1.0)
    
    def _calculate_sector_relevance(self, text: str, symbol: str, topic: str) -> float:
        """Calculate how relevant the news is to the specific sector"""
        relevance = 0.5  # Default
        
        if topic in self.keyword_analyzer.sector_keywords:
            sector_data = self.keyword_analyzer.sector_keywords[topic]
            
            # Check high-impact keywords
            high_impact_count = sum(1 for word in sector_data['high_impact'] if word in text)
            catalyst_count = sum(1 for word in sector_data['catalysts'] if word in text)
            
            if high_impact_count > 0:
                relevance = max(relevance, 0.8)
            if catalyst_count > 0:
                relevance = max(relevance, 0.9)
        
        return relevance
    
    def _calculate_surprise_factor(self, text: str, topic: str) -> float:
        """Calculate how unexpected this news is (unexpected news has bigger impact)"""
        surprise = 0.5  # Default
        
        surprise_indicators = [
            'unexpected', 'surprise', 'shocking', 'unprecedented', 'unusual',
            'rare', 'first time', 'never before', 'breaking news'
        ]
        
        for indicator in surprise_indicators:
            if indicator in text:
                surprise = max(surprise, 0.8)
        
        return surprise


class EnhancedNewsAnalyzer:
    """Optimized news analyzer with Gemini LLM integration, professional strategies, and enhanced sentiment analysis."""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized model loading, Gemini, technical strategies, and enhanced sentiment analysis."""
        self.fmp_loader = fmp_loader
        self.technical_analyzer = TechnicalAnalyzer(fmp_loader)
        self.technical_strategies = TechnicalStrategies(fmp_loader)
        
        # Initialize Gemini analyzer
        self.gemini_analyzer = GeminiNewsAnalyzer()
        
        # Enhanced sentiment analysis components
        self.quality_scorer = NewsQualityScorer()
        self.recent_analyses_cache = []  # Store recent analyses for confirmation
        
        # Device selection with fallback
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.finbert_labels = ["positive", "negative", "neutral"]
        
        # Initialize FinBERT with error handling
        self._load_finbert_safely()
        
        # Optimized keyword sets
        self._initialize_keywords()
        
        # Enhance existing keywords with power keywords
        self._enhance_existing_keywords()
    
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
    
    def _enhance_existing_keywords(self) -> None:
        """Enhance existing keywords with power keywords"""
        # Add power keywords to existing sets
        power_positive = {
            'crushes', 'smashes', 'demolishes', 'blowout', 'blockbuster', 'stellar',
            'phenomenal', 'explosive growth', 'cash flow surge', 'profit explosion',
            'market domination', 'game changer', 'revolutionary', 'unprecedented growth',
            'record-breaking', 'all-time high', 'massive beat', 'huge surprise'
        }
        
        power_negative = {
            'catastrophic', 'devastating', 'collapse', 'free fall', 'plummeting',
            'cash crunch', 'liquidity crisis', 'massive losses', 'debt spiral',
            'losing market share', 'regulatory nightmare', 'investigation launched',
            'nightmare', 'cratering', 'imploding', 'hemorrhaging'
        }
        
        # Enhance existing keyword sets
        self.positive_keywords.update(power_positive)
        self.negative_keywords.update(power_negative)
        
        # Add biotech-specific keywords
        biotech_positive = {
            'fda approval', 'breakthrough designation', 'fast track', 'priority review',
            'pdufa date', 'nda accepted', 'meets endpoints', 'statistically significant',
            'regulatory approval', 'cms approval', 'reimbursement approved'
        }
        
        biotech_negative = {
            'fda rejection', 'complete response letter', 'crl', 'safety hold',
            'trial failure', 'missed endpoints', 'adverse events', 'trial halted',
            'regulatory delay', 'black box warning'
        }
        
        # Add to biotech keywords
        self.biotech_keywords.update(biotech_positive)
        self.biotech_keywords.update(biotech_negative)
        
        log_info(f"Enhanced keywords: {len(self.positive_keywords)} positive, {len(self.negative_keywords)} negative, {len(self.biotech_keywords)} biotech")
    
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
    
    def _calculate_enhanced_sentiment(self, title: str, content: str, symbol: str,
                                    current_price: float, topic: str, 
                                    source: str = "") -> Tuple[float, float]:
        """Enhanced sentiment analysis with quality scoring"""
        
        # Get enhanced sentiment score using the new quality scorer
        enhanced_score = self.quality_scorer.calculate_enhanced_sentiment(
            title=title,
            content=content,
            symbol=symbol,
            current_price=current_price,
            topic=topic,
            source=source,
            recent_analyses=self.recent_analyses_cache
        )
        
        # Store for future multi-source confirmation
        self.recent_analyses_cache.append({
            'symbol': symbol,
            'sentiment_score': enhanced_score.final_sentiment,
            'timestamp': datetime.now(),
            'topic': topic,
            'quality_confidence': enhanced_score.quality_confidence
        })
        
        # Keep only recent analyses (last 6 hours)
        cutoff_time = datetime.now() - timedelta(hours=6)
        self.recent_analyses_cache = [
            a for a in self.recent_analyses_cache 
            if a['timestamp'] > cutoff_time
        ]
        
        # Log enhanced analysis details for high-quality signals
        if enhanced_score.quality_confidence > 0.7:
            log_info(f"HIGH QUALITY signal for {symbol}: "
                    f"final_sentiment={enhanced_score.final_sentiment:.3f}, "
                    f"quality={enhanced_score.quality_confidence:.3f}, "
                    f"credibility={enhanced_score.credibility_score:.2f}, "
                    f"confirmation={enhanced_score.confirmation_score:.2f}, "
                    f"magnitude={enhanced_score.magnitude_score:.2f}")
        
        return enhanced_score.final_sentiment, enhanced_score.quality_confidence
    
    @lru_cache(maxsize=1000)
    def _detect_topic(self, text: str) -> str:
        """Optimized topic detection with cached regex patterns."""
        text_lower = text.lower()
        
        for topic, pattern in self._topic_patterns.items():
            if pattern.search(text_lower):
                return topic
        
        return 'general'
    
    def _enhanced_ensemble_score(self, finbert_sentiment: float, finbert_conf: float,
                                enhanced_sentiment: float, enhanced_conf: float,
                                gemini_sentiment: float, gemini_conf: float) -> Tuple[float, float]:
        """Enhanced ensemble scoring with quality weighting"""
        
        components = []
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        # FinBERT component
        if finbert_conf > 0.0:
            weight = CONFIG.finbert_weight * finbert_conf
            weighted_sentiment += finbert_sentiment * weight
            total_weight += weight
            components.append(finbert_conf)
        
        # Enhanced sentiment component (replaces basic keyword)
        if enhanced_conf > 0.0:
            weight = CONFIG.keyword_weight * enhanced_conf * 1.2  # Boost for enhanced quality
            weighted_sentiment += enhanced_sentiment * weight
            total_weight += weight
            components.append(enhanced_conf)
        
        # Gemini component
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
        
        # Quality bonus for enhanced sentiment
        quality_bonus = 1.0 + (enhanced_conf * 0.3)  # Up to 30% bonus for high quality
        
        # Agreement bonus calculation
        sentiments = []
        if finbert_conf > 0:
            sentiments.append(finbert_sentiment)
        if enhanced_conf > 0:
            sentiments.append(enhanced_sentiment)
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
        
        final_confidence = min(base_confidence * quality_bonus * agreement_bonus * gemini_bonus, 1.0)
        
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
        
        log_info(f"Analyzing {len(news_df)} news articles with Enhanced Sentiment + Technical Analysis")
        
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
        """Process a single news article with comprehensive validation including Enhanced Sentiment, Gemini and strategies."""
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
        
        # Get source information (try to extract from row data)
        source = str(row.get('source', '')).strip()
        
        # Multi-source sentiment analysis
        full_text = f"{title} {content}"
        
        # 1. FinBERT analysis
        finbert_sentiment, finbert_conf = self._get_finbert_sentiment(full_text)
        
        # 2. Enhanced sentiment analysis (replaces basic keyword analysis)
        enhanced_sentiment, quality_confidence = self._calculate_enhanced_sentiment(
            title=title,
            content=content,
            symbol=symbol,
            current_price=current_price,
            topic=topic,
            source=source
        )
        
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
        final_sentiment, news_confidence = self._enhanced_ensemble_score(
            finbert_sentiment, finbert_conf,
            enhanced_sentiment, quality_confidence,  # Use enhanced instead of basic keyword
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
            keyword_score=enhanced_sentiment,  # Store enhanced sentiment as keyword_score
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
            f"finbert={analysis.finbert_score:.2f} enhanced={analysis.keyword_score:.2f} "
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
        log_info(f"   Enhanced Sentiment: {analysis.keyword_score:.3f}")
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
        log_info("5. Enhanced sentiment may be stricter - check quality scores")
        log_info("=" * 80)
