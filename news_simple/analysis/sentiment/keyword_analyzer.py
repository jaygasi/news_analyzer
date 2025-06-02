"""
Optimized keyword analysis with performance improvements and expanded patterns
"""
import re
from typing import Dict, Set, Pattern
from functools import lru_cache


class KeywordAnalyzer:
    """High-performance keyword analysis with pre-compiled patterns and optimized lookups."""
    
    # Pre-compiled regex patterns for maximum performance
    _PERCENTAGE_PATTERNS = [
        re.compile(r'(\d+(?:\.\d+)?)\s*%\s*(?:increase|growth|up|higher|gain)', re.IGNORECASE),
        re.compile(r'(\d+(?:\.\d+)?)\s*%\s*(?:decrease|decline|down|lower|drop)', re.IGNORECASE),
        re.compile(r'\b(?:doubled|tripled|quadrupled)\b', re.IGNORECASE),
        re.compile(r'(\d+)x\s*(?:growth|increase|gain)', re.IGNORECASE)
    ]
    
    _SPAM_PATTERN = re.compile(
        r'\b(?:click here|ad:|advertisement|sponsored|free trial|limited time)\b', 
        re.IGNORECASE
    )
    
    def __init__(self):
        """Initialize with optimized keyword sets and patterns."""
        self._initialize_optimized_keywords()
        self._initialize_performance_indicators()
    
    def _initialize_optimized_keywords(self) -> None:
        """Initialize keyword sets optimized for fast lookups."""
        
        # Ultra-high impact keywords for maximum market movement
        self.power_positive_keywords = frozenset([
            'crushes', 'smashes', 'demolishes', 'obliterates', 'destroys estimates',
            'blowout', 'blockbuster', 'stellar', 'phenomenal', 'exceptional',
            'record-breaking', 'all-time high', 'massive beat', 'huge surprise',
            'explosive growth', 'rapid expansion', 'accelerating momentum',
            'breakthrough', 'revolutionary', 'game-changer', 'disruptive',
            'cash flow surge', 'profit explosion', 'revenue acceleration',
            'margin expansion', 'market domination', 'competitive advantage'
        ])
        
        self.power_negative_keywords = frozenset([
            'catastrophic', 'devastating', 'disastrous', 'nightmare', 'collapse',
            'free fall', 'plummeting', 'cratering', 'imploding', 'hemorrhaging',
            'cash crunch', 'liquidity crisis', 'burning cash', 'bleeding money',
            'massive losses', 'debt spiral', 'financial distress', 'bankruptcy',
            'investigation', 'fraud', 'scandal', 'lawsuit', 'regulatory action'
        ])
        
        # Sector-specific catalyst keywords
        self.sector_catalysts = {
            'biotech': {
                'positive': frozenset([
                    'fda approval', 'breakthrough designation', 'fast track', 'priority review',
                    'orphan drug', 'accelerated approval', 'phase 3 success', 'meets endpoints',
                    'statistically significant', 'pdufa date', 'nda accepted', 'bla accepted',
                    'clinical trial success', 'efficacy demonstrated', 'safety profile'
                ]),
                'negative': frozenset([
                    'fda rejection', 'complete response letter', 'crl', 'safety hold',
                    'trial failure', 'missed endpoints', 'adverse events', 'safety concerns',
                    'trial halted', 'regulatory delay', 'black box warning', 'recall'
                ])
            },
            'tech': {
                'positive': frozenset([
                    'ai breakthrough', 'patent approval', 'licensing deal', 'cloud growth',
                    'user growth', 'subscription growth', 'platform adoption',
                    'digital transformation', 'market expansion', 'innovation',
                    'product launch', 'partnership', 'acquisition target'
                ]),
                'negative': frozenset([
                    'data breach', 'regulatory scrutiny', 'antitrust', 'user decline',
                    'competition threat', 'platform issues', 'security vulnerability',
                    'privacy violation', 'service outage', 'cyber attack'
                ])
            },
            'financial': {
                'positive': frozenset([
                    'loan growth', 'deposit increase', 'credit quality', 'net interest margin',
                    'fee income', 'cost reduction', 'efficiency ratio', 'capital strength',
                    'dividend increase', 'share buyback', 'merger target'
                ]),
                'negative': frozenset([
                    'loan losses', 'credit deterioration', 'regulatory fine', 'stress test',
                    'capital shortfall', 'fraud investigation', 'compliance issue'
                ])
            }
        }
    
    def _initialize_performance_indicators(self) -> None:
        """Initialize magnitude and urgency indicators."""
        self.magnitude_multipliers = {
            'extreme': 2.2, 'massive': 1.9, 'huge': 1.7, 'significant': 1.5,
            'substantial': 1.4, 'major': 1.3, 'notable': 1.2, 'strong': 1.1,
            'slight': 0.8, 'minor': 0.7, 'small': 0.6, 'tiny': 0.4
        }
        
        self.urgency_multipliers = {
            'immediate': 1.0, 'urgent': 0.95, 'breaking': 0.95, 'just announced': 0.9,
            'developing': 0.85, 'emerging': 0.8, 'upcoming': 0.7, 'planned': 0.5,
            'potential': 0.4, 'possible': 0.3, 'rumored': 0.2, 'speculated': 0.1
        }
        
        self.time_sensitivity = {
            'within hours': 1.0, 'today': 0.95, 'this week': 0.8,
            'this month': 0.6, 'this quarter': 0.4, 'next year': 0.2
        }
    
    def calculate_base_sentiment(self, text: str) -> float:
        """Optimized base sentiment calculation using efficient set operations."""
        text_words = set(text.split())
        
        # Fast intersection operations with pre-computed sets
        positive_matches = 0
        negative_matches = 0
        
        # Power keywords (weighted higher)
        positive_matches += len(text_words & self.power_positive_keywords) * 2.5
        negative_matches += len(text_words & self.power_negative_keywords) * 2.5
        
        # Sector-specific catalysts
        for sector_data in self.sector_catalysts.values():
            # Check for multi-word phrases in text
            positive_matches += self._count_phrase_matches(text, sector_data['positive']) * 2.0
            negative_matches += self._count_phrase_matches(text, sector_data['negative']) * 2.0
        
        total_matches = positive_matches + negative_matches
        if total_matches == 0:
            return 0.0
        
        sentiment = (positive_matches - negative_matches) / total_matches
        return max(-1.0, min(1.0, sentiment))
    
    @lru_cache(maxsize=500)
    def _count_phrase_matches(self, text: str, phrases: frozenset) -> int:
        """Cached phrase matching for performance."""
        return sum(1 for phrase in phrases if phrase in text)
    
    def calculate_magnitude_score(self, text: str, topic: str) -> float:
        """Enhanced magnitude calculation with sector-specific adjustments."""
        magnitude = 1.0
        
        # Check magnitude amplifiers efficiently
        text_lower = text.lower()
        for amplifier, multiplier in self.magnitude_multipliers.items():
            if amplifier in text_lower:
                magnitude = max(magnitude, multiplier)
        
        # Optimized percentage pattern matching
        for pattern in self._PERCENTAGE_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    try:
                        pct = float(match[0])
                        if pct > 20:
                            magnitude = max(magnitude, 1.6)
                        elif pct > 10:
                            magnitude = max(magnitude, 1.4)
                        elif pct > 5:
                            magnitude = max(magnitude, 1.2)
                    except (ValueError, IndexError):
                        continue
        
        # Topic-specific magnitude adjustments
        topic_multipliers = {
            'biotech': 1.5,  # Higher volatility
            'earnings': 1.3,
            'ma': 1.4,       # M&A announcements
            'analyst': 1.1,
            'general': 1.0
        }
        
        magnitude *= topic_multipliers.get(topic, 1.0)
        
        # Normalize to 0-1 range
        return min(magnitude / 2.0, 1.0)
    
    def calculate_urgency_score(self, text: str, title: str) -> float:
        """Optimized urgency calculation with enhanced time sensitivity."""
        urgency = 0.5
        combined_text = f"{title} {text}".lower()
        
        # Fast urgency keyword lookup
        for keyword, score in self.urgency_multipliers.items():
            if keyword in combined_text:
                urgency = max(urgency, score)
        
        # Time sensitivity patterns
        for phrase, score in self.time_sensitivity.items():
            if phrase in combined_text:
                urgency = max(urgency, score)
        
        # Breaking news indicators
        breaking_indicators = ['breaking', 'just in', 'urgent', 'alert', 'flash']
        if any(indicator in title.lower() for indicator in breaking_indicators):
            urgency = min(urgency * 1.2, 1.0)
        
        # Market hours urgency (if during trading hours)
        market_hours_indicators = ['pre-market', 'after-hours', 'market open', 'closing bell']
        if any(indicator in combined_text for indicator in market_hours_indicators):
            urgency = min(urgency * 1.1, 1.0)
        
        return urgency
    
    def is_spam_content(self, text: str) -> bool:
        """Optimized spam detection with pre-compiled patterns."""
        return bool(self._SPAM_PATTERN.search(text))
    
    def get_sentiment_context(self, text: str, topic: str) -> Dict[str, float]:
        """Get comprehensive sentiment context for debugging."""
        return {
            'base_sentiment': self.calculate_base_sentiment(text),
            'magnitude_score': self.calculate_magnitude_score(text, topic),
            'urgency_score': self.calculate_urgency_score(text, ''),
            'is_spam': float(self.is_spam_content(text)),
            'text_length': len(text),
            'word_count': len(text.split())
        }