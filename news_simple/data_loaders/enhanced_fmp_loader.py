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
            
            log_info(f"Total comprehensive news: {len(df)} articles from {df['source'].nunique()} sources")
            return df
            
        except Exception as e:
            log_debug(f"Error in comprehensive news: {e}")
            return None
    
    def _process_comprehensive_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process comprehensive news data."""
        if df.empty:
            return df
        
        try:
            # Enhanced deduplication
            df = self._enhanced_deduplicate_news(df)
            
            # Process datetime
            if 'publishedDate' in df.columns:
                df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce', utc=True)
            
            # Create combined content field
            if 'title' in df.columns and 'text' in df.columns:
                df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
            
            # Ensure source field
            if 'source' not in df.columns:
                df['source'] = 'unknown'
            
            # Filter essential data
            essential_filters = (
                df['symbol'].notna() & 
                (df['symbol'].str.strip() != '') &
                df['title'].notna() & 
                (df['title'].str.strip() != '')
            )
            
            return df[essential_filters]
            
        except Exception as e:
            log_debug(f"Error processing comprehensive news: {e}")
            return df
    
    def _enhanced_deduplicate_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced deduplication with content similarity."""
        if df.empty:
            return df
        
        try:
            # Remove exact duplicates
            df = df.drop_duplicates(subset=['symbol', 'title'], keep='first')
            
            # Less aggressive similarity removal for diverse sources
            if len(df) <= 1200:  # Increased threshold for comprehensive data
                df = self._remove_similar_content(df)
            
            return df
            
        except Exception as e:
            log_debug(f"Error in deduplication: {e}")
            return df
    
    def _remove_similar_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove articles with similar content - optimized for diverse sources."""
        if 'title' not in df.columns:
            return df
        
        try:
            unique_indices = []
            seen_content = set()
            
            for idx, row in df.iterrows():
                title = str(row['title']).lower().strip()
                symbol = str(row.get('symbol', '')).upper()
                source = str(row.get('source', ''))
                
                # Create content fingerprint - consider source
                title_words = set(word for word in title.split() if len(word) > 3)
                content_key = (symbol, source, frozenset(title_words))
                
                # Check similarity - more lenient for different sources
                is_similar = False
                for existing_symbol, existing_source, existing_words in seen_content:
                    if (existing_symbol == symbol and 
                        title_words and existing_words):
                        
                        # Higher threshold if different sources
                        threshold = 0.9 if existing_source != source else 0.8
                        similarity = len(title_words & existing_words) / len(title_words | existing_words)
                        
                        if similarity > threshold:
                            is_similar = True
                            break
                
                if not is_similar:
                    unique_indices.append(idx)
                    seen_content.add(content_key)
                    
                    # Prevent memory growth
                    if len(seen_content) > 500:
                        seen_content = set(list(seen_content)[-250:])
            
            return df.loc[unique_indices]
            
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
    
    # Backward compatibility
    def get_news_rss(self) -> Optional[pd.DataFrame]:
        """Backward compatibility method."""
        return self.get_comprehensive_news()