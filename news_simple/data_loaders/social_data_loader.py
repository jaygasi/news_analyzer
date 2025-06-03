"""
Social data loader for FMP API - handles social sentiment endpoints
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.simple_logger import log_debug
from .base_fmp_loader import BaseFMPLoader


class SocialDataLoader(BaseFMPLoader):
    """Specialized loader for social sentiment data."""
    
    def get_social_sentiment_news(self) -> List[Dict[str, Any]]:
        """Get social sentiment data as news signals."""
        try:
            articles = []
            
            # Trending social sentiment
            trending_data = self._make_request("social-sentiments/trending", {"limit": 20}, use_v4=True)
            
            if trending_data and isinstance(trending_data, list):
                for sentiment in trending_data:
                    if 'symbol' in sentiment and sentiment['symbol']:
                        sentiment_score = sentiment.get('sentiment', 0)
                        sentiment_label = 'Bullish' if sentiment_score > 0 else 'Bearish' if sentiment_score < 0 else 'Neutral'
                        
                        transformed = {
                            'symbol': sentiment['symbol'].strip().upper(),
                            'title': f"Social Sentiment: {sentiment['symbol']} - {sentiment_label}",
                            'text': f"Social media sentiment for {sentiment['symbol']}: {sentiment_label} (score: {sentiment_score:.2f})",
                            'url': '',
                            'publishedDate': datetime.now().isoformat(),
                            'site': 'social_sentiment',
                            'source': 'social_sentiment',
                            'social_sentiment_score': sentiment_score,
                            'social_sentiment_label': sentiment_label
                        }
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Social sentiment not available: {e}")
            return []
    
    def get_social_sentiment_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get social sentiment for specific symbol."""
        try:
            data = self._make_request(f"social-sentiments/{symbol}", use_v4=True)
            
            if data and isinstance(data, list) and data:
                return data[0]  # Return most recent sentiment
            
            return None
            
        except Exception as e:
            log_debug(f"Social sentiment for {symbol} not available: {e}")
            return None
    
    def get_general_news_sentiment(self) -> List[Dict[str, Any]]:
        """Get general news sentiment."""
        try:
            articles = []
            
            # General news sentiment
            general_data = self._make_request("general-news", {"limit": 20}, use_v4=True)
            
            if general_data and isinstance(general_data, list):
                for news in general_data:
                    transformed = {
                        'symbol': 'SPY',  # General market news
                        'title': news.get('title', 'Market News'),
                        'text': news.get('text', ''),
                        'url': news.get('url', ''),
                        'publishedDate': news.get('publishedDate', ''),
                        'site': 'general_news',
                        'source': 'general_news'
                    }
                    articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"General news sentiment not available: {e}")
            return []
    
    def get_forex_news(self) -> List[Dict[str, Any]]:
        """Get forex news."""
        try:
            articles = []
            
            forex_data = self._make_request("forex_news", {"limit": 10})
            
            if forex_data and isinstance(forex_data, list):
                for news in forex_data:
                    transformed = {
                        'symbol': 'SPY',  # Use SPY for forex market impact
                        'title': f"Forex: {news.get('title', 'Currency News')}",
                        'text': news.get('text', ''),
                        'url': news.get('url', ''),
                        'publishedDate': news.get('publishedDate', ''),
                        'site': 'forex_news',
                        'source': 'forex_news'
                    }
                    articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Forex news not available: {e}")
            return []
    
    def get_crypto_news(self) -> List[Dict[str, Any]]:
        """Get crypto news."""
        try:
            articles = []
            
            crypto_data = self._make_request("crypto_news", {"limit": 10})
            
            if crypto_data and isinstance(crypto_data, list):
                for news in crypto_data:
                    transformed = {
                        'symbol': 'SPY',  # Use SPY for crypto market impact on stocks
                        'title': f"Crypto: {news.get('title', 'Cryptocurrency News')}",
                        'text': news.get('text', ''),
                        'url': news.get('url', ''),
                        'publishedDate': news.get('publishedDate', ''),
                        'site': 'crypto_news',
                        'source': 'crypto_news'
                    }
                    articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Crypto news not available: {e}")
            return []