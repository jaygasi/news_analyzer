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
        """Get comprehensive news from all sources."""
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
            
            # Create DataFrame and process
            df = pd.DataFrame(all_articles)
            df = self._process_comprehensive_news(df)
            
            if not df.empty:
                log_info(f"Total comprehensive news: {len(df)} articles from {df['source'].nunique()} sources")
            
            return df if not df.empty else None
            
        except Exception as e:
            log_debug(f"Error in comprehensive news: {e}")
            return None
    
    def _process_comprehensive_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process comprehensive news data with proper DataFrame operations."""
        if df.empty:
            return df
        
        try:
            # Make a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Enhanced deduplication
            df = self._enhanced_deduplicate_news(df)
            
            # Process datetime safely
            if 'publishedDate' in df.columns:
                df.loc[:, 'publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce', utc=True)
            
            # Create combined content field safely
            if 'title' in df.columns and 'text' in df.columns:
                df.loc[:, 'content'] = df['title'].astype(str) + " " + df['text'].astype(str)
            
            # Ensure source field exists
            if 'source' not in df.columns:
                df.loc[:, 'source'] = 'unknown'
            else:
                # Fill missing source values
                df.loc[:, 'source'] = df['source'].fillna('unknown')
            
            # Filter essential data
            essential_filters = (
                df['symbol'].notna() & 
                (df['symbol'].str.strip() != '') &
                df['title'].notna() & 
                (df['title'].str.strip() != '')
            )
            
            filtered_df = df[essential_filters].copy()
            return filtered_df
            
        except Exception as e:
            log_debug(f"Error processing comprehensive news: {e}")
            return df
    
    def _enhanced_deduplicate_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced deduplication with content similarity - improved for incremental fetching."""
        if df.empty:
            return df
        
        try:
            # Remove exact duplicates first
            df = df.drop_duplicates(subset=['symbol', 'title'], keep='first')
            
            # For incremental fetching, be less aggressive with similarity removal
            # since we should be getting fewer, more relevant articles
            if len(df) > 100:  # Only apply similarity filtering if we have many articles
                df = self._remove_similar_content(df)
            
            return df
            
        except Exception as e:
            log_debug(f"Error in deduplication: {e}")
            return df
    
    def _remove_similar_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove articles with similar content - optimized for incremental processing."""
        if 'title' not in df.columns:
            return df
        
        try:
            unique_indices = []
            processed_content = {}  # symbol -> set of content fingerprints
            
            for idx, row in df.iterrows():
                title = str(row['title']).lower().strip()
                symbol = str(row.get('symbol', '')).upper()
                source = str(row.get('source', 'unknown'))
                
                # Create content fingerprint
                title_words = set(word for word in title.split() if len(word) > 3)
                
                if symbol not in processed_content:
                    processed_content[symbol] = set()
                
                # Check for similarity within the same symbol
                is_similar = False
                similarity_threshold = 0.85  # High threshold - only remove very similar content
                
                for existing_words in processed_content[symbol]:
                    if title_words and existing_words:
                        similarity = len(title_words & existing_words) / len(title_words | existing_words)
                        if similarity > similarity_threshold:
                            is_similar = True
                            break
                
                if not is_similar:
                    unique_indices.append(idx)
                    processed_content[symbol].add(frozenset(title_words))
            
            return df.loc[unique_indices].copy()
            
        except Exception as e:
            log_debug(f"Error removing similar content: {e}")
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
        """Get social sentiment for symbol."""
        return self.social_loader.get_social_sentiment_for_symbol(symbol)
    
    def get_analyst_estimates_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get analyst estimates for symbol."""
        return self.analyst_loader.get_analyst_estimates_for_symbol(symbol)
    
    def get_price_target_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price target for symbol."""
        return self.analyst_loader.get_price_target_for_symbol(symbol)
    
    def set_universe_cache(self, universe: List[str]) -> None:
        """Set universe cache for all loaders that need it."""
        self.news_loader.set_universe_cache(universe)
    
    def force_refresh_news(self, hours_back: int = 1) -> None:
        """Force refresh news sources to get fresh content."""
        self.news_loader.force_refresh_from_time(hours_back)
        log_info(f"Forced news refresh: reset to {hours_back} hours back")
    
    # Backward compatibility
    def get_news_rss(self) -> Optional[pd.DataFrame]:
        """Backward compatibility method."""
        return self.get_comprehensive_news()