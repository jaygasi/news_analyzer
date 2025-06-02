"""
Credibility scoring for news sources and content
"""
from typing import Dict, Set


class CredibilityScorer:
    """Evaluates credibility of news sources and content"""
    
    def __init__(self):
        self._initialize_source_rankings()
        self._initialize_credibility_markers()
    
    def _initialize_source_rankings(self) -> None:
        """Initialize source credibility rankings"""
        self.credible_sources = {
            'tier_1': frozenset(['reuters', 'bloomberg', 'wsj', 'ft', 'ap news']),
            'tier_2': frozenset(['cnbc', 'marketwatch', 'yahoo finance', 'seeking alpha']),
            'tier_3': frozenset(['motley fool', 'benzinga', 'zacks']),
            'company_direct': frozenset(['press release', 'sec filing', '8-k', '10-k'])
        }
        
        self.source_weights = {
            'tier_1': 1.0,
            'tier_2': 0.8,
            'tier_3': 0.6,
            'company_direct': 1.15
        }
    
    def _initialize_credibility_markers(self) -> None:
        """Initialize credibility markers in content"""
        self.credibility_markers = {
            'high': frozenset([
                'sec filing', 'press release', 'earnings call', 'official statement',
                'regulatory filing', 'management guidance', 'board approval'
            ]),
            'medium': frozenset([
                'analyst report', 'research note', 'expert opinion', 'industry study'
            ]),
            'low': frozenset([
                'rumor', 'speculation', 'unconfirmed', 'alleged', 'sources say'
            ])
        }
        
        self.credibility_weights = {
            'high': 0.85,
            'medium': 0.65,
            'low': 0.25
        }
    
    def calculate_credibility_score(self, text: str, source: str, content: str) -> float:
        """Calculate overall credibility score"""
        credibility = 0.5
        source_lower = source.lower()
        
        # Source credibility
        for tier, sources in self.credible_sources.items():
            if any(credible_source in source_lower for credible_source in sources):
                credibility = max(credibility, self.source_weights[tier])
                break
        
        # Content credibility markers
        text_words = set(text.split())
        
        for level, markers in self.credibility_markers.items():
            if text_words & markers:
                if level == 'low':
                    credibility = min(credibility, self.credibility_weights[level])
                else:
                    credibility = max(credibility, self.credibility_weights[level])
        
        # Content length factor
        if len(content) > 400:
            credibility = min(credibility * 1.08, 1.0)
        elif len(content) < 80:
            credibility *= 0.85
        
        return credibility
    
    def get_source_tier(self, source: str) -> str:
        """Get the tier of a news source"""
        source_lower = source.lower()
        
        for tier, sources in self.credible_sources.items():
            if any(credible_source in source_lower for credible_source in sources):
                return tier
        
        return 'unknown'