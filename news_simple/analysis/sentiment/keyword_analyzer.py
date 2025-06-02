"""
Optimized keyword analysis with pre-compiled patterns and efficient lookups
"""
import re
from typing import Dict, List, Tuple, Set
from functools import lru_cache


class KeywordAnalyzer:
    """Optimized keyword analysis with pre-compiled patterns and efficient lookups"""
    
    def __init__(self):
        self._initialize_keywords()
        self._initialize_indicators()
        self._compile_patterns()
    
    def _initialize_keywords(self) -> None:
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
        
        # Sector-specific catalysts
        self.biotech_catalysts = {
            'positive': frozenset([
                'fda approval', 'breakthrough designation', 'fast track', 'priority review',
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
    
    def _initialize_indicators(self) -> None:
        """Initialize magnitude and urgency indicators"""
        self.magnitude_amplifiers = {
            'extreme': 2.0, 'massive': 1.8, 'huge': 1.6, 'significant': 1.4,
            'substantial': 1.3, 'major': 1.2, 'notable': 1.1,
            'slight': 0.7, 'minor': 0.6, 'small': 0.5, 'tiny': 0.3
        }
        
        self.urgency_keywords = {
            'immediate': 1.0, 'urgent': 0.9, 'breaking': 0.9, 'just announced': 0.9,
            'developing': 0.8, 'emerging': 0.7, 'upcoming': 0.6, 'planned': 0.4,
            'potential': 0.3, 'possible': 0.2, 'rumored': 0.1
        }
        
        self.time_sensitivity = {
            'within hours': 1.0, 'today': 0.9, 'this week': 0.7,
            'this month': 0.5, 'this quarter': 0.3, 'next year': 0.1
        }
    
    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for better performance"""
        self.percentage_patterns = [
            re.compile(r'(\d+(?:\.\d+)?)\s*%\s*(increase|growth|up|higher)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*%\s*(decrease|decline|down|lower)', re.IGNORECASE),
            re.compile(r'(doubled|tripled|quadrupled)', re.IGNORECASE),
            re.compile(r'(\d+)x\s*(growth|increase)', re.IGNORECASE)
        ]
        
        self.spam_pattern = re.compile(
            r'\b(click here|ad:|advertisement|sponsored)\b', 
            re.IGNORECASE
        )
    
    def calculate_base_sentiment(self, text: str) -> float:
        """Calculate base sentiment using keyword matching"""
        text_words = set(text.lower().split())
        
        # Use frozenset intersections for fast counting
        positive_count = len(text_words & self.power_positive_keywords) * 2
        negative_count = len(text_words & self.power_negative_keywords) * 2
        
        # Add sector-specific catalysts
        positive_count += len(text_words & self.biotech_catalysts['positive']) * 1.5
        negative_count += len(text_words & self.biotech_catalysts['negative']) * 1.5
        
        positive_count += len(text_words & self.tech_catalysts['positive']) * 1.2
        negative_count += len(text_words & self.tech_catalysts['negative']) * 1.2
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def calculate_magnitude_score(self, text: str, topic: str) -> float:
        """Calculate magnitude score from text indicators"""
        magnitude = 1.0
        
        # Check for magnitude amplifiers
        for word, multiplier in self.magnitude_amplifiers.items():
            if word in text:
                magnitude = max(magnitude, multiplier)
        
        # Check percentage patterns
        for pattern in self.percentage_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and match[0].replace('.', '').isdigit():
                    pct = float(match[0])
                    if pct > 10:
                        magnitude = max(magnitude, 1.4)
                    elif pct > 5:
                        magnitude = max(magnitude, 1.2)
        
        # Topic-specific adjustments
        if topic == 'biotech' and any(word in text for word in ['fda approval', 'breakthrough']):
            magnitude *= 1.4
        elif topic == 'earnings' and any(word in text for word in ['blowout', 'crush']):
            magnitude *= 1.25
        
        return min(magnitude, 2.0) / 2.0  # Normalize to 0-1
    
    def calculate_urgency_score(self, text: str, title: str) -> float:
        """Calculate urgency score from timing indicators"""
        urgency = 0.5
        
        # Check urgency keywords
        for word, score in self.urgency_keywords.items():
            if word in text or word in title:
                urgency = max(urgency, score)
        
        # Check time sensitivity
        for phrase, score in self.time_sensitivity.items():
            if phrase in text:
                urgency = max(urgency, score)
        
        # Breaking news bonus
        title_lower = title.lower()
        if 'breaking' in title_lower or 'just in' in title_lower:
            urgency = min(urgency * 1.15, 1.0)
        
        return urgency
    
    def is_spam_content(self, text: str) -> bool:
        """Check if content appears to be spam"""
        return bool(self.spam_pattern.search(text))