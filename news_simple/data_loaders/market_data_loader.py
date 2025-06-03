"""
Market data loader for FMP API - handles market events and economic data
"""
from datetime import datetime
from typing import List, Dict, Any
from utils.simple_logger import log_debug
from .base_fmp_loader import BaseFMPLoader


class MarketDataLoader(BaseFMPLoader):
    """Specialized loader for market events and economic data."""
    
    def get_market_events(self) -> List[Dict[str, Any]]:
        """Get all market events."""
        all_events = []
        
        event_sources = [
            ("IPO calendar", self._get_ipo_calendar),
            ("Stock splits", self._get_stock_splits),
            ("Dividends", self._get_dividends),
            ("Economic calendar", self._get_economic_calendar),
        ]
        
        for source_name, source_func in event_sources:
            try:
                events = source_func()
                if events:
                    all_events.extend(events)
                    log_debug(f"{source_name}: {len(events)} events")
            except Exception as e:
                log_debug(f"Error getting {source_name}: {e}")
        
        return all_events
    
    def _get_ipo_calendar(self) -> List[Dict[str, Any]]:
        """Get IPO calendar."""
        try:
            articles = []
            
            ipo_data = self._make_request("ipo_calendar", {"limit": 20})
            if ipo_data and isinstance(ipo_data, list):
                for ipo in ipo_data:
                    if 'symbol' in ipo and ipo['symbol']:
                        transformed = {
                            'symbol': ipo['symbol'].strip().upper(),
                            'title': f"IPO: {ipo['symbol']} - {ipo.get('company', 'Unknown')}",
                            'text': f"IPO scheduled for {ipo.get('date', 'TBD')} - Price range: ${ipo.get('priceFrom', 'N/A')}-${ipo.get('priceTo', 'N/A')}",
                            'url': '',
                            'publishedDate': ipo.get('date', ''),
                            'site': 'ipo_calendar',
                            'source': 'market_events',
                            'event_type': 'ipo'
                        }
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"IPO calendar not available: {e}")
            return []
    
    def _get_stock_splits(self) -> List[Dict[str, Any]]:
        """Get stock split calendar."""
        try:
            articles = []
            
            split_data = self._make_request("stock_split_calendar", {"limit": 20})
            if split_data and isinstance(split_data, list):
                for split in split_data:
                    if 'symbol' in split and split['symbol']:
                        transformed = {
                            'symbol': split['symbol'].strip().upper(),
                            'title': f"Stock Split: {split['symbol']} - {split.get('numerator', '?')}:{split.get('denominator', '?')}",
                            'text': f"Stock split {split.get('numerator', '?')} for {split.get('denominator', '?')} on {split.get('date', 'TBD')}",
                            'url': '',
                            'publishedDate': split.get('date', ''),
                            'site': 'stock_splits',
                            'source': 'market_events',
                            'event_type': 'split'
                        }
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Stock splits not available: {e}")
            return []
    
    def _get_dividends(self) -> List[Dict[str, Any]]:
        """Get dividend calendar."""
        try:
            articles = []
            
            div_data = self._make_request("dividend_calendar", {"limit": 30})
            if div_data and isinstance(div_data, list):
                for div in div_data:
                    if 'symbol' in div and div['symbol']:
                        transformed = {
                            'symbol': div['symbol'].strip().upper(),
                            'title': f"Dividend: {div['symbol']} - ${div.get('dividend', 0):.2f}",
                            'text': f"Dividend of ${div.get('dividend', 0):.2f} ex-date: {div.get('date', 'TBD')}",
                            'url': '',
                            'publishedDate': div.get('date', ''),
                            'site': 'dividends',
                            'source': 'market_events',
                            'event_type': 'dividend'
                        }
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Dividends not available: {e}")
            return []
    
    def _get_economic_calendar(self) -> List[Dict[str, Any]]:
        """Get economic calendar."""
        try:
            articles = []
            
            econ_data = self._make_request("economic_calendar", {"limit": 10})
            if econ_data and isinstance(econ_data, list):
                for event in econ_data:
                    # Create general market event (no specific symbol)
                    transformed = {
                        'symbol': 'SPY',  # Use SPY as proxy for market events
                        'title': f"Economic Event: {event.get('event', 'Unknown')}",
                        'text': f"Economic indicator: {event.get('event', 'Unknown')} on {event.get('date', 'TBD')}",
                        'url': '',
                        'publishedDate': event.get('date', ''),
                        'site': 'economic_calendar',
                        'source': 'market_events',
                        'event_type': 'economic'
                    }
                    articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Economic calendar not available: {e}")
            return []