"""
Corporate data loader for FMP API - handles SEC filings, insider trading, and corporate actions
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any
from utils.simple_logger import log_debug
from .base_fmp_loader import BaseFMPLoader


class CorporateDataLoader(BaseFMPLoader):
    """Specialized loader for corporate filings and insider data."""
    
    def get_corporate_news(self) -> List[Dict[str, Any]]:
        """Get all corporate-related news."""
        all_corporate = []
        
        corporate_sources = [
            ("SEC filings", self._get_sec_filings),
            ("Insider trading", self._get_insider_trading),
            ("Senate trading", self._get_senate_trading),
            ("Earnings calendar", self._get_earnings_calendar),
        ]
        
        for source_name, source_func in corporate_sources:
            try:
                items = source_func()
                if items:
                    all_corporate.extend(items)
                    log_debug(f"{source_name}: {len(items)} items")
            except Exception as e:
                log_debug(f"Error getting {source_name}: {e}")
        
        return all_corporate
    
    def _get_sec_filings(self) -> List[Dict[str, Any]]:
        """Get SEC filings as news."""
        try:
            articles = []
            
            # Get recent SEC filings
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            params = {
                "from": start_date,
                "to": end_date,
                "limit": 50
            }
            
            data = self._make_request("sec_filings", params)
            
            if data and isinstance(data, list):
                for filing in data[:30]:  # Limit to 30 most recent
                    if 'symbol' in filing and filing['symbol']:
                        transformed = {
                            'symbol': filing['symbol'].strip().upper(),
                            'title': f"SEC Filing: {filing.get('type', 'Unknown')} - {filing['symbol']}",
                            'text': f"SEC filing {filing.get('type', '')} filed on {filing.get('fillingDate', '')}. {filing.get('description', '')}",
                            'url': filing.get('finalLink', ''),
                            'publishedDate': filing.get('fillingDate', ''),
                            'site': 'sec_filings',
                            'source': 'sec_filings',
                            'filing_type': filing.get('type', ''),
                            'filing_date': filing.get('fillingDate', '')
                        }
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"SEC filings not available: {e}")
            return []
    
    def _get_insider_trading(self) -> List[Dict[str, Any]]:
        """Get insider trading as news."""
        try:
            articles = []
            
            params = {"limit": 50}
            data = self._make_request("insider-trading", params, use_v4=True)
            
            if data and isinstance(data, list):
                for trade in data[:20]:  # Limit to 20 most recent
                    if 'symbol' in trade and trade['symbol']:
                        trade_type = trade.get('transactionType', 'Trade')
                        shares = trade.get('securitiesTransacted', 0)
                        
                        transformed = {
                            'symbol': trade['symbol'].strip().upper(),
                            'title': f"Insider {trade_type}: {trade['symbol']} - {shares} shares",
                            'text': f"Insider {trade_type} by {trade.get('reportingName', 'Unknown')} of {shares} shares at ${trade.get('price', 0):.2f}",
                            'url': '',
                            'publishedDate': trade.get('filingDate', ''),
                            'site': 'insider_trading',
                            'source': 'insider_trading',
                            'transaction_type': trade_type,
                            'shares_transacted': shares,
                            'insider_name': trade.get('reportingName', ''),
                            'transaction_price': trade.get('price', 0)
                        }
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Insider trading not available: {e}")
            return []
    
    def _get_senate_trading(self) -> List[Dict[str, Any]]:
        """Get senate trading as news."""
        try:
            articles = []
            
            params = {"limit": 30}
            data = self._make_request("senate-trading", params, use_v4=True)
            
            if data and isinstance(data, list):
                for trade in data[:10]:  # Limit to 10 most recent
                    if 'symbol' in trade and trade['symbol']:
                        transformed = {
                            'symbol': trade['symbol'].strip().upper(),
                            'title': f"Senate Trade: {trade['symbol']} - {trade.get('representative', 'Unknown')}",
                            'text': f"Senate trade by {trade.get('representative', 'Unknown')} in {trade['symbol']} on {trade.get('dateRecieved', '')}",
                            'url': '',
                            'publishedDate': trade.get('dateRecieved', ''),
                            'site': 'senate_trading',
                            'source': 'senate_trading',
                            'representative': trade.get('representative', ''),
                            'transaction_date': trade.get('dateRecieved', '')
                        }
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Senate trading not available: {e}")
            return []
    
    def _get_earnings_calendar(self) -> List[Dict[str, Any]]:
        """Get earnings calendar as news."""
        try:
            articles = []
            
            # Get recent earnings calendar
            params = {"limit": 30}
            data = self._make_request("earning_calendar", params)
            
            if data and isinstance(data, list):
                for earning in data[:15]:  # Limit to 15 most recent
                    if 'symbol' in earning and earning['symbol']:
                        transformed = {
                            'symbol': earning['symbol'].strip().upper(),
                            'title': f"Earnings Call: {earning['symbol']} - Q{earning.get('quarter', '?')} {earning.get('year', '')}",
                            'text': f"Earnings call for {earning['symbol']} scheduled for {earning.get('date', 'TBD')}. EPS estimate: ${earning.get('epsEstimated', 'N/A')}",
                            'url': '',
                            'publishedDate': earning.get('date', ''),
                            'site': 'earnings_calendar',
                            'source': 'earnings_calendar',
                            'earnings_date': earning.get('date', ''),
                            'eps_estimated': earning.get('epsEstimated', 0),
                            'revenue_estimated': earning.get('revenueEstimated', 0)
                        }
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Earnings calendar not available: {e}")
            return []
    
    def get_company_outlook(self, symbol: str) -> List[Dict[str, Any]]:
        """Get company outlook and guidance."""
        try:
            data = self._make_request(f"company-outlook/{symbol}")
            
            if data and isinstance(data, dict):
                return [data]  # Return as list for consistency
            
            return []
            
        except Exception as e:
            log_debug(f"Company outlook for {symbol} not available: {e}")
            return []