"""
Backwards compatibility module - re-exports all market filter components
"""
# Import all components for backwards compatibility
from .market_filters.market_condition_filter import MarketConditionFilter, MarketConditions
from .market_filters.news_quality_filter import NewsQualityFilter
from .market_filters.price_action_filter import PriceActionFilter
from .market_filters.portfolio_risk_filter import PortfolioRiskFilter
from .market_filters.entry_timing_optimizer import EntryTimingOptimizer

# Create aliases for backwards compatibility
MarketFilter = MarketConditionFilter
OptimizedMarketFilter = MarketConditionFilter
OptimizedNewsQualityFilter = NewsQualityFilter
OptimizedPriceActionFilter = PriceActionFilter
OptimizedPortfolioRiskFilter = PortfolioRiskFilter
OptimizedEntryTimingOptimizer = EntryTimingOptimizer

__all__ = [
    'MarketConditionFilter',
    'MarketConditions',
    'NewsQualityFilter', 
    'PriceActionFilter',
    'PortfolioRiskFilter',
    'EntryTimingOptimizer',
    # Aliases for backwards compatibility
    'MarketFilter',
    'OptimizedMarketFilter',
    'OptimizedNewsQualityFilter',
    'OptimizedPriceActionFilter', 
    'OptimizedPortfolioRiskFilter',
    'OptimizedEntryTimingOptimizer'
]