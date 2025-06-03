"""
Enhanced data loaders module with specialized FMP loaders
"""
from .enhanced_fmp_loader import EnhancedFMPLoader
from .news_data_loader import NewsDataLoader
from .market_data_loader import MarketDataLoader
from .social_data_loader import SocialDataLoader
from .analyst_data_loader import AnalystDataLoader
from .corporate_data_loader import CorporateDataLoader
from .price_data_loader import PriceDataLoader

# For backward compatibility
SimpleFMPLoader = EnhancedFMPLoader

__all__ = [
    'EnhancedFMPLoader',
    'SimpleFMPLoader',  # Backward compatibility alias
    'NewsDataLoader',
    'MarketDataLoader',
    'SocialDataLoader',
    'AnalystDataLoader',
    'CorporateDataLoader',
    'PriceDataLoader'
]