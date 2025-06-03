"""
Enhanced FMP loader that orchestrates all specialized data loaders
"""
import pandas as pd
from typing import Optional, List, Dict, Any
from utils.simple_logger import log_info, log_debug
from .news_data_loader import NewsDataLoader
from .market_data_loader import MarketDataLoader
from .social_data_loader import SocialDataLoader
from .analyst_data_loader import AnalystDataLoader
from .corporate_data_loader import CorporateDataLoader
from .price_data_loader import PriceDataLoader


class EnhancedFMPLoader:
    """Enhanced FMP loader that orchestrates all specialized loaders."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize all specialized loaders."""
        self.api_key = api_key
        
        # Initialize all specialized loaders
        self.news_loader = NewsDataLoader(api_key)
        self.market_loader = MarketDataLoader(api_key)
        self.social_loader = SocialDataLoader(api_key)
        self.analyst_loader = AnalystDataLoader(api_key)
        self.corporate_loader = CorporateDataLoader(api_key)
        self.price_loader = PriceDataLoader(api_key)
        
        log_info("Enhanced FMP loader initialized with all specialized loaders")
    
    def get_comprehensive_news(self) -> Optional[pd.DataFrame]:
        """Get comprehensive news from all sources with minimal deduplication."""
        try:
            all_articles = []
            
            # Collect from all news sources
            news_sources = [
                ("News", self.news_loader.get_comprehensive_news),
                ("Market Events", self.market_loader.get_market_events),
                ("Social Sentiment", self.social_loader.get_social_sentiment_news),
                ("Analyst Changes", self.analyst_loader.get_analyst_changes),
                ("Corporate News", self.corporate_loader.get_corporate_news),
            ]
            
            for source_name, source_func in news_sources:
                try:
                    articles = source_func()
                    if articles:
                        all_articles.extend(articles)
                        log_debug(f"{source_name}: {len(articles)} articles")
                except Exception as e:
                    log_debug(f"Error getting {source_name}: {e}")
            
            if not all_articles:
                log_debug("No articles from any news source")
                return None
            
            # Create DataFrame and process with minimal deduplication
            df = pd.DataFrame(all_articles)
            df = self._process_comprehensive_news_minimal(df)
            
            log_info(f"Total comprehensive news: {len(df)} articles from {df['source'].nunique()} sources")
            return df
            
        except Exception as e:
            log_debug(f"Error in comprehensive news: {e}")
            return None
    
    def _process_comprehensive_news_minimal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process comprehensive news data with minimal filtering to preserve fresh content."""
        if df.empty:
            return df
        
        try:
            # FIX: Create explicit copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Only remove exact duplicates - no similarity filtering at this level
            df = df.drop_duplicates(subset=['symbol', 'title'], keep='first')
            
            # Process datetime - Fixed SettingWithCopyWarning
            if 'publishedDate' in df.columns:
                df.loc[:, 'publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce', utc=True)
            
            # Create combined content field - Fixed SettingWithCopyWarning
            if 'title' in df.columns and 'text' in df.columns:
                df.loc[:, 'content'] = df['title'].astype(str) + " " + df['text'].astype(str)
            
            # Ensure source field - Fixed SettingWithCopyWarning
            if 'source' not in df.columns:
                df.loc[:, 'source'] = 'unknown'
            
            # Filter only essential missing data
            essential_filters = (
                df['symbol'].notna() & 
                (df['symbol'].str.strip() != '') &
                df['title'].notna() & 
                (df['title'].str.strip() != '') &
                (df['title'].str.len() >= 5)  # Minimum title length
            )
            
            filtered_df = df[essential_filters].copy()  # Explicit copy
            
            # Sort by publication date to prioritize fresh content
            if 'publishedDate' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('publishedDate', ascending=False, na_position='last')
            
            log_debug(f"Minimal processing: {len(df)} → {len(filtered_df)} articles (removed only essential missing data)")
            
            return filtered_df
            
        except Exception as e:
            log_debug(f"Error processing comprehensive news: {e}")
            return df
    
    # Delegate methods to appropriate loaders
    def get_stock_screener(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Get stock screener."""
        return self.price_loader.get_stock_screener(limit)
    
    def get_real_time_prices(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get real-time prices."""
        return self.price_loader.get_real_time_prices(symbols)
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data."""
        return self.price_loader.get_historical_data(symbol, days)
    
    def get_social_sentiment_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get social sentiment for symbol - DISABLED to avoid 429 errors."""
        # Temporarily disabled to reduce API load
        log_debug(f"Social sentiment lookup disabled to avoid rate limits for {symbol}")
        return None
    
    def get_analyst_estimates_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get analyst estimates for symbol - DISABLED to avoid 429 errors."""
        # Temporarily disabled to reduce API load
        log_debug(f"Analyst estimates lookup disabled to avoid rate limits for {symbol}")
        return None
    
    def get_price_target_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price target for symbol - DISABLED to avoid 429 errors."""
        # Temporarily disabled to reduce API load
        log_debug(f"Price target lookup disabled to avoid rate limits for {symbol}")
        return None
    
    def set_universe_cache(self, universe: List[str]) -> None:
        """Set universe cache for all loaders that need it."""
        self.news_loader.set_universe_cache(universe)
    
    # Backward compatibility
    def get_news_rss(self) -> Optional[pd.DataFrame]:
        """Backward compatibility method."""
        return self.get_comprehensive_news()