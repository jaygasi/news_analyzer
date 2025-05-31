"""
Simplified FMP data loader
"""
import requests
import pandas as pd
from typing import Optional, List, Dict, Any
import time
from utils.simple_logger import log_info, log_error

class SimpleFMPLoader:
    """Simplified FMP data loader"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _rate_limit(self):
        """Simple rate limiting"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make API request with error handling"""
        self._rate_limit()
        
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log_error(f"FMP API error: {e}")
            return None
    
    def get_stock_screener(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Get stock screener results"""
        data = self._make_request("stock-screener", {
            "marketCapMoreThan": 100000000,  # $100M+
            "priceMoreThan": 2,
            "priceLowerThan": 500,
            "volumeMoreThan": 50000,
            "isActivelyTrading": "true",
            "exchange": "NYSE,NASDAQ",
            "limit": limit
        })
        
        if data:
            return pd.DataFrame(data)
        return None
    
    def get_real_time_prices(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get real-time prices for symbols"""
        if not symbols:
            return None
            
        symbol_str = ",".join(symbols[:100])  # Limit to 100 symbols
        data = self._make_request(f"stock/full/real-time-price/{symbol_str}")
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty and 'symbol' in df.columns:
                return df
        return None
    
    def get_news_rss(self) -> Optional[pd.DataFrame]:
        """Get latest news from RSS feed"""
        data = self._make_request("../v4/stock-news-sentiments-rss-feed", {"page": 0})
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                # Clean and standardize
                if 'publishedDate' in df.columns:
                    df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
                if 'title' in df.columns and 'text' in df.columns:
                    df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
                return df
        return None