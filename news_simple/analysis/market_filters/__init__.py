"""
Market filters module - import all filter components
"""
from .market_condition_filter import MarketConditionFilter, MarketConditions
from .news_quality_filter import NewsQualityFilter
from .price_action_filter import PriceActionFilter
from .portfolio_risk_filter import PortfolioRiskFilter
from .entry_timing_optimizer import EntryTimingOptimizer

# Create aliases for backwards compatibility
MarketFilter = MarketConditionFilter

__all__ = [
    'MarketConditionFilter',
    'MarketConditions',
    'NewsQualityFilter', 
    'PriceActionFilter',
    'PortfolioRiskFilter',
    'EntryTimingOptimizer',
    'MarketFilter'  # Alias
]