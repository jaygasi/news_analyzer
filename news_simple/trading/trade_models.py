"""
Trade data models and related structures
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime, timezone


@dataclass
class EnhancedTrade:
    """Enhanced trade record with comprehensive data"""
    id: str
    symbol: str
    side: str
    entry_price: float
    position_size: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    
    # Trade execution status
    trade_type: str = "LIVE"
    execution_status: str = "PENDING"
    rejection_reason: str = ""
    
    # News analysis data
    news_title: str = ""
    news_confidence: float = 0.0
    combined_confidence: float = 0.0
    finbert_score: float = 0.0
    keyword_score: float = 0.0
    topic: str = ""
    
    # Technical analysis data
    technical_confidence: float = 0.0
    liquidity_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    rsi: float = 50.0
    price_trend: str = ""
    bid_ask_spread: float = 0.0
    
    # Market condition data
    market_regime: str = ""
    market_stress: float = 0.0
    time_score: float = 0.0
    entry_timing: str = ""


@dataclass
class ProcessedArticle:
    """Processed article record with JSON-serializable fields"""
    article_hash: str
    symbol: str
    title_hash: str
    processed_time: str  # Store as ISO string
    sentiment_score: float
    combined_confidence: float
    was_traded: int = 0  # Use int instead of bool
    trade_side: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedArticle':
        """Create from dictionary with safe type conversion"""
        return cls(
            article_hash=str(data.get('article_hash', '')),
            symbol=str(data.get('symbol', '')),
            title_hash=str(data.get('title_hash', '')),
            processed_time=str(data.get('processed_time', datetime.now(timezone.utc).isoformat())),
            sentiment_score=float(data.get('sentiment_score', 0.0)),
            combined_confidence=float(data.get('combined_confidence', 0.0)),
            was_traded=int(data.get('was_traded', 0)),
            trade_side=str(data.get('trade_side', ''))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def processed_datetime(self) -> datetime:
        """Get processed time as datetime object"""
        try:
            return datetime.fromisoformat(self.processed_time.replace('Z', '+00:00'))
        except:
            return datetime.now(timezone.utc)