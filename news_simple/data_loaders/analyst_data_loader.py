"""
Analyst data loader for FMP API - handles analyst upgrades, downgrades, and estimates
"""
from typing import List, Dict, Any, Optional
from utils.simple_logger import log_debug
from .base_fmp_loader import BaseFMPLoader


class AnalystDataLoader(BaseFMPLoader):
    """Specialized loader for analyst data and recommendations."""
    
    def get_analyst_changes(self) -> List[Dict[str, Any]]:
        """Get analyst upgrades/downgrades and price targets."""
        try:
            articles = []
            
            # Get recent upgrades/downgrades
            params = {"limit": 50}
            upgrades_data = self._make_request("upgrades-downgrades", params)
            
            if upgrades_data and isinstance(upgrades_data, list):
                for change in upgrades_data[:30]:  # Limit to 30 most recent
                    if 'symbol' in change and change['symbol']:
                        grade_change = change.get('gradeNew', '') + ' (from ' + change.get('gradePrevious', '') + ')'
                        
                        transformed = {
                            'symbol': change['symbol'].strip().upper(),
                            'title': f"Analyst {change.get('action', 'Update')}: {change['symbol']} - {grade_change}",
                            'text': f"{change.get('company', 'Unknown')} {change.get('action', 'updated')} by {change.get('analystCompany', 'Unknown')} to {grade_change}",
                            'url': '',
                            'publishedDate': change.get('publishedDate', ''),
                            'site': 'analyst_changes',
                            'source': 'analyst_changes',
                            'analyst_action': change.get('action', ''),
                            'analyst_grade_new': change.get('gradeNew', ''),
                            'analyst_grade_previous': change.get('gradePrevious', ''),
                            'analyst_company': change.get('analystCompany', '')
                        }
                        articles.append(transformed)
            
            # Get price targets
            price_targets_data = self._make_request("price-target-consensus", params)
            
            if price_targets_data and isinstance(price_targets_data, list):
                for target in price_targets_data[:20]:  # Limit to 20
                    if 'symbol' in target and target['symbol']:
                        target_price = target.get('targetConsensus', 0)
                        
                        transformed = {
                            'symbol': target['symbol'].strip().upper(),
                            'title': f"Price Target: {target['symbol']} - ${target_price:.2f} consensus",
                            'text': f"Analyst consensus price target for {target['symbol']}: ${target_price:.2f}",
                            'url': '',
                            'publishedDate': target.get('date', ''),
                            'site': 'price_targets',
                            'source': 'price_targets',
                            'price_target': target_price,
                            'target_high': target.get('targetHigh', 0),
                            'target_low': target.get('targetLow', 0)
                        }
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Analyst data not available: {e}")
            return []
    
    def get_analyst_estimates_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get analyst estimates for specific symbol."""
        try:
            data = self._make_request(f"analyst-estimates/{symbol}")
            
            if data and isinstance(data, list) and data:
                return data[0]  # Return most recent estimates
            
            return None
            
        except Exception as e:
            log_debug(f"Analyst estimates for {symbol} not available: {e}")
            return None
    
    def get_price_target_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price target for specific symbol."""
        try:
            data = self._make_request(f"price-target/{symbol}")
            
            if data and isinstance(data, list) and data:
                return data[0]  # Return most recent price target
            
            return None
            
        except Exception as e:
            log_debug(f"Price target for {symbol} not available: {e}")
            return None
    
    def get_analyst_recommendations(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get analyst recommendations for specific symbol."""
        try:
            data = self._make_request(f"analyst-stock-recommendations/{symbol}")
            
            if data and isinstance(data, list) and data:
                return data[0]  # Return most recent recommendations
            
            return None
            
        except Exception as e:
            log_debug(f"Analyst recommendations for {symbol} not available: {e}")
            return None
    
    def get_earnings_call_transcript(self, symbol: str, quarter: int, year: int) -> Optional[Dict[str, Any]]:
        """Get earnings call transcript for specific symbol and quarter."""
        try:
            data = self._make_request(f"earning_call_transcript/{symbol}", {
                "quarter": quarter,
                "year": year
            })
            
            if data and isinstance(data, list) and data:
                return data[0]  # Return transcript
            
            return None
            
        except Exception as e:
            log_debug(f"Earnings transcript for {symbol} Q{quarter} {year} not available: {e}")
            return None
    
    def get_earnings_surprises(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get earnings surprises for specific symbol."""
        try:
            data = self._make_request(f"earnings-surprises/{symbol}")
            
            if data and isinstance(data, list):
                return data[:5]  # Return last 5 quarters
            
            return None
            
        except Exception as e:
            log_debug(f"Earnings surprises for {symbol} not available: {e}")
            return None