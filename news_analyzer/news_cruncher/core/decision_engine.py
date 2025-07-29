from __future__ import annotations

"""
Decision engine that combines news and technical analysis - Enhanced with dynamic price fields
Python 3.13.3 compatible
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from analysis.multi_llm_analyzer import DirectionalPrediction
from analysis.technical_analyzer_simple import TechnicalSignal
from config import Config
from utils.simple_logger import log_debug, log_info, log_warning


@dataclass
class TradingDecision:
    """Final trading decision with all supporting data including configurable price tracking"""
    ticker: str
    decision: str  # 'LONG', 'SHORT', 'NONE'
    confidence: float  # 0.0 to 1.0
    reasoning: str
    
    # Supporting analysis
    news_prediction: Optional[DirectionalPrediction] = None
    technical_signal: Optional[TechnicalSignal] = None
    
    # Detailed scores
    news_score: float = 0.0
    technical_score: float = 0.0
    combined_score: float = 0.0
    
    # Multi-source specific data (if available)
    sources_used: List[str] = field(default_factory=list)
    source_weights: Dict[str, float] = field(default_factory=dict)
    individual_predictions: List[Dict[str, Any]] = field(default_factory=list)
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    article_count: int = 0
    analysis_timestamp: str = ""
    
    # Entry price tracking
    recommendation_price: Optional[float] = None
    recommendation_timestamp: Optional[datetime] = None
    
    # Dynamic price tracking fields - will be set dynamically by price tracker
    price_tracking_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_checkpoint_price(self, index: int) -> Optional[float]:
        """Get price for a specific checkpoint index (0=first, 1=second, etc.)"""
        checkpoint_info = Config.get_checkpoint_info()
        if index < len(checkpoint_info):
            field_prefix = checkpoint_info[index]['field_prefix']
            return self.price_tracking_data.get(field_prefix)
        return None
    
    def get_checkpoint_change_pct(self, index: int) -> Optional[float]:
        """Get price change percentage for a specific checkpoint"""
        checkpoint_info = Config.get_checkpoint_info()
        if index < len(checkpoint_info):
            field_prefix = checkpoint_info[index]['field_prefix']
            return self.price_tracking_data.get(f"{field_prefix}_change_pct")
        return None
    
    def get_all_checkpoint_data(self) -> Dict[str, Any]:
        """Get all checkpoint data in a structured format"""
        result = {}
        checkpoint_info = Config.get_checkpoint_info()
        
        for i, checkpoint in enumerate(checkpoint_info):
            field_prefix = checkpoint['field_prefix']
            label = checkpoint['short_label']
            
            result[label] = {
                'price': self.price_tracking_data.get(field_prefix),
                'timestamp': self.price_tracking_data.get(f"{field_prefix}_timestamp"),
                'change_pct': self.price_tracking_data.get(f"{field_prefix}_change_pct"),
                'field_prefix': field_prefix
            }
        
        return result
    def set_checkpoint_price(self, index: int, price: float) -> None:
        """Set price for a specific checkpoint index (0=first, 1=second, etc.)"""
        from config import Config
        checkpoint_info = Config.get_checkpoint_info()
        if index < len(checkpoint_info):
            field_prefix = checkpoint_info[index]['field_prefix']
            self.price_tracking_data[field_prefix] = price
    
    def set_checkpoint_change_pct(self, index: int, change_pct: float) -> None:
        """Set price change percentage for a specific checkpoint"""
        from config import Config
        checkpoint_info = Config.get_checkpoint_info()
        if index < len(checkpoint_info):
            field_prefix = checkpoint_info[index]['field_prefix']
            self.price_tracking_data[f"{field_prefix}_change_pct"] = change_pct
    
    def set_checkpoint_timestamp(self, index: int, timestamp) -> None:
        """Set timestamp for a specific checkpoint"""
        from config import Config
        checkpoint_info = Config.get_checkpoint_info()
        if index < len(checkpoint_info):
            field_prefix = checkpoint_info[index]['field_prefix']
            # Convert datetime to ISO string if needed
            if hasattr(timestamp, 'isoformat'):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)
            self.price_tracking_data[f"{field_prefix}_timestamp"] = timestamp_str
    # Legacy property methods for backward compatibility with existing code
    @property
    def price_check1(self) -> Optional[float]:
        """Get first checkpoint price (dynamic based on config)"""
        return self.get_checkpoint_price(0)
    
    @property 
    def price_check2(self) -> Optional[float]:
        """Get second checkpoint price (dynamic based on config)"""
        return self.get_checkpoint_price(1)
    
    @property
    def price_close(self) -> Optional[float]:
        """Get close price"""
        return self.get_checkpoint_price(2)
    
    @property
    def price_check1_change_pct(self) -> Optional[float]:
        """Get first checkpoint change percentage"""
        return self.get_checkpoint_change_pct(0)
    
    @property
    def price_check2_change_pct(self) -> Optional[float]:
        """Get second checkpoint change percentage"""
        return self.get_checkpoint_change_pct(1)
    
    @property
    def price_close_change_pct(self) -> Optional[float]:
        """Get close price change percentage"""
        return self.get_checkpoint_change_pct(2)


# You can add other classes like DecisionEngine here if needed,
# but the main issue was the TradingDecision dataclass definition
