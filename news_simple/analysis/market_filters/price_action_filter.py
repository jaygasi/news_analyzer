"""
Price action filtering for trading signals
"""
import pandas as pd
from typing import List
from config import CONFIG
from utils.simple_logger import log_warning


class PriceActionFilter:
    """Filter symbols based on price action criteria"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with data loader"""
        self.fmp_loader = fmp_loader
    
    def filter_price_action(self, symbols: List[str], current_prices: pd.DataFrame) -> List[str]:
        """Filter symbols based on price action criteria"""
        if not symbols or current_prices is None or current_prices.empty:
            return symbols
        
        try:
            # Apply price and volume filters
            price_mask = (
                (current_prices['lastSalePrice'] >= CONFIG.min_price) &
                (current_prices['lastSalePrice'] <= CONFIG.max_price) &
                (current_prices['volume'] >= CONFIG.min_volume)
            )
            
            valid_prices = current_prices[price_mask]
            valid_symbols = set(valid_prices['symbol'].tolist())
            
            # Filter input symbols
            filtered_symbols = [symbol for symbol in symbols if symbol in valid_symbols]
            
            return filtered_symbols
            
        except Exception as e:
            log_warning(f"Error in price action filtering: {e}")
            return symbols
    
    def validate_price_data(self, symbol: str, price_data: pd.Series) -> bool:
        """Validate price data for a single symbol"""
        try:
            price = float(price_data.get('lastSalePrice', 0))
            volume = float(price_data.get('volume', 0))
            
            return (
                CONFIG.min_price <= price <= CONFIG.max_price and
                volume >= CONFIG.min_volume
            )
            
        except (ValueError, TypeError):
            return False
    
    def get_price_statistics(self, current_prices: pd.DataFrame) -> dict:
        """Get price action statistics for analysis"""
        if current_prices is None or current_prices.empty:
            return {}
        
        try:
            stats = {
                'total_symbols': len(current_prices),
                'avg_price': current_prices['lastSalePrice'].mean(),
                'median_price': current_prices['lastSalePrice'].median(),
                'avg_volume': current_prices['volume'].mean(),
                'median_volume': current_prices['volume'].median(),
                'price_range': {
                    'min': current_prices['lastSalePrice'].min(),
                    'max': current_prices['lastSalePrice'].max()
                },
                'volume_range': {
                    'min': current_prices['volume'].min(),
                    'max': current_prices['volume'].max()
                }
            }
            
            # Filter statistics
            price_filtered = current_prices[
                (current_prices['lastSalePrice'] >= CONFIG.min_price) &
                (current_prices['lastSalePrice'] <= CONFIG.max_price)
            ]
            
            volume_filtered = current_prices[
                current_prices['volume'] >= CONFIG.min_volume
            ]
            
            both_filtered = current_prices[
                (current_prices['lastSalePrice'] >= CONFIG.min_price) &
                (current_prices['lastSalePrice'] <= CONFIG.max_price) &
                (current_prices['volume'] >= CONFIG.min_volume)
            ]
            
            stats['filtered_counts'] = {
                'price_filtered': len(price_filtered),
                'volume_filtered': len(volume_filtered),
                'both_filtered': len(both_filtered),
                'filter_rate': len(both_filtered) / len(current_prices) if len(current_prices) > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            log_warning(f"Error calculating price statistics: {e}")
            return {}