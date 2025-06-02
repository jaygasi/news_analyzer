"""
Portfolio risk management and filtering
"""
from typing import Tuple
from config import CONFIG
from utils.simple_logger import log_warning


class PortfolioRiskFilter:
    """Manage portfolio risk and position limits"""
    
    def __init__(self, trader) -> None:
        """Initialize with trader reference"""
        self.trader = trader
        self._max_positions = 8
        self._max_symbol_positions = 1
        self._max_total_exposure_multiplier = 8
    
    def check_portfolio_limits(self, symbol: str, position_size: float) -> Tuple[bool, str]:
        """Check if new position would violate portfolio limits"""
        try:
            active_positions = self.trader.get_active_positions()
            
            # Check maximum positions
            if len(active_positions) >= self._max_positions:
                return False, f"max_positions_exceeded_{self._max_positions}"
            
            # Check symbol concentration
            symbol_count = sum(1 for trade in active_positions if trade.symbol == symbol)
            if symbol_count >= self._max_symbol_positions:
                return False, f"symbol_concentration_limit_{self._max_symbol_positions}"
            
            # Check total exposure
            total_exposure = sum(trade.position_size for trade in active_positions)
            max_exposure = CONFIG.position_size * self._max_total_exposure_multiplier
            if total_exposure + position_size > max_exposure:
                return False, f"total_exposure_limit_{max_exposure:.0f}"
            
            return True, "within_limits"
            
        except Exception as e:
            log_warning(f"Error checking portfolio limits: {e}")
            return False, f"check_error: {e}"
    
    def calculate_portfolio_risk(self) -> dict:
        """Calculate current portfolio risk metrics"""
        try:
            active_positions = self.trader.get_active_positions()
            
            if not active_positions:
                return {
                    'total_positions': 0,
                    'total_exposure': 0,
                    'position_utilization': 0,
                    'exposure_utilization': 0,
                    'symbol_concentration': {},
                    'sector_concentration': {}
                }
            
            total_exposure = sum(trade.position_size for trade in active_positions)
            max_exposure = CONFIG.position_size * self._max_total_exposure_multiplier
            
            # Symbol concentration
            symbol_counts = {}
            for trade in active_positions:
                symbol_counts[trade.symbol] = symbol_counts.get(trade.symbol, 0) + 1
            
            # Topic/sector concentration (if available)
            sector_counts = {}
            for trade in active_positions:
                topic = getattr(trade, 'topic', 'unknown')
                sector_counts[topic] = sector_counts.get(topic, 0) + 1
            
            return {
                'total_positions': len(active_positions),
                'total_exposure': total_exposure,
                'position_utilization': len(active_positions) / self._max_positions,
                'exposure_utilization': total_exposure / max_exposure,
                'symbol_concentration': symbol_counts,
                'sector_concentration': sector_counts,
                'max_positions': self._max_positions,
                'max_exposure': max_exposure
            }
            
        except Exception as e:
            log_warning(f"Error calculating portfolio risk: {e}")
            return {}
    
    def get_available_capacity(self) -> dict:
        """Get available portfolio capacity"""
        try:
            active_positions = self.trader.get_active_positions()
            total_exposure = sum(trade.position_size for trade in active_positions)
            max_exposure = CONFIG.position_size * self._max_total_exposure_multiplier
            
            return {
                'available_positions': max(0, self._max_positions - len(active_positions)),
                'available_exposure': max(0, max_exposure - total_exposure),
                'can_trade': len(active_positions) < self._max_positions and total_exposure < max_exposure
            }
            
        except Exception as e:
            log_warning(f"Error calculating available capacity: {e}")
            return {
                'available_positions': 0,
                'available_exposure': 0,
                'can_trade': False
            }