"""
Position management utilities
"""
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timezone
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning
from .trade_models import EnhancedTrade
from analysis.market_filters import MarketConditionFilter


class PositionManager:
    """Manage trading positions and exits"""
    
    def __init__(self, fmp_loader):
        self.fmp_loader = fmp_loader
        self.market_filter = MarketConditionFilter(fmp_loader)
        self.active_trades = {}
    
    def add_position(self, trade: EnhancedTrade) -> None:
        """Add a new position"""
        self.active_trades[trade.symbol] = trade
    
    def remove_position(self, symbol: str) -> Optional[EnhancedTrade]:
        """Remove a position"""
        return self.active_trades.pop(symbol, None)
    
    def get_active_positions(self) -> List[EnhancedTrade]:
        """Get all active positions"""
        return list(self.active_trades.values())
    
    def check_exits(self, current_prices: pd.DataFrame) -> None:
        """Check all positions for exit conditions"""
        if current_prices is None or current_prices.empty or not self.active_trades:
            return
        
        try:
            market_conditions = self.market_filter.get_market_conditions()
            stress_multiplier = self._calculate_stress_multiplier(market_conditions)
            
            trades_to_close = []
            
            for symbol, trade in self.active_trades.items():
                try:
                    if self._check_trade_exit(trade, current_prices, stress_multiplier):
                        trades_to_close.append(symbol)
                except Exception as e:
                    log_error(f"Error checking exit for {symbol}: {e}")
            
            for symbol in trades_to_close:
                self.active_trades.pop(symbol, None)
                
        except Exception as e:
            log_error(f"Error in exit checking: {e}")
    
    def _calculate_stress_multiplier(self, market_conditions) -> float:
        """Calculate stress-based exit multiplier"""
        stress_level = market_conditions.market_stress_level
        
        if stress_level > 0.8:
            return 0.65
        elif stress_level > 0.6:
            return 0.8
        else:
            return 1.0
    
    def _check_trade_exit(self, trade: EnhancedTrade, current_prices: pd.DataFrame, 
                         stress_multiplier: float) -> bool:
        """Check if a trade should be exited"""
        current_price = self._get_current_price(trade.symbol, current_prices)
        if current_price <= 0:
            return False
        
        pnl_pct = self._calculate_pnl_percentage(trade, current_price)
        stop_loss_pct, take_profit_pct = self._calculate_dynamic_levels(trade, stress_multiplier)
        exit_reason = self._determine_exit_reason(pnl_pct, stop_loss_pct, take_profit_pct, stress_multiplier)
        
        if exit_reason:
            self._execute_exit(trade, current_price, exit_reason, pnl_pct)
            return True
        
        return False
    
    def _get_current_price(self, symbol: str, current_prices: pd.DataFrame) -> float:
        """Get current price for symbol"""
        try:
            price_row = current_prices[current_prices['symbol'] == symbol]
            if price_row.empty:
                return 0.0
            
            current_price = float(price_row.iloc[0].get('lastSalePrice', 0))
            return current_price if current_price > 0 else 0.0
            
        except (ValueError, IndexError, KeyError) as e:
            log_error(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    def _calculate_pnl_percentage(self, trade: EnhancedTrade, current_price: float) -> float:
        """Calculate P&L percentage"""
        if trade.side == 'long':
            return (current_price - trade.entry_price) / trade.entry_price
        else:
            return (trade.entry_price - current_price) / trade.entry_price
    
    def _calculate_dynamic_levels(self, trade: EnhancedTrade, stress_multiplier: float) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels"""
        stop_loss_pct = CONFIG.stop_loss_pct * stress_multiplier
        take_profit_pct = CONFIG.take_profit_pct
        
        # Adjust for liquidity
        if trade.liquidity_score < 0.5:
            stop_loss_pct *= 0.85
            take_profit_pct *= 0.85
        
        # Adjust for confidence
        if trade.combined_confidence > 0.8:
            take_profit_pct *= 1.15
        
        return stop_loss_pct, take_profit_pct
    
    def _determine_exit_reason(self, pnl_pct: float, stop_loss_pct: float, 
                             take_profit_pct: float, stress_multiplier: float) -> Optional[str]:
        """Determine if trade should exit and why"""
        if pnl_pct <= -stop_loss_pct:
            return f'stop_loss_{stress_multiplier:.1f}x'
        elif pnl_pct >= take_profit_pct:
            return 'take_profit'
        elif stress_multiplier < 0.8 and pnl_pct > 0.015:
            return 'market_stress_protect'
        
        return None
    
    def _execute_exit(self, trade: EnhancedTrade, current_price: float, 
                     exit_reason: str, pnl_pct: float) -> None:
        """Execute trade exit"""
        trade.exit_price = current_price
        trade.exit_time = datetime.now(timezone.utc)
        trade.exit_reason = exit_reason
        trade.pnl = trade.position_size * pnl_pct
        
        duration_minutes = (trade.exit_time - trade.entry_time).total_seconds() / 60
        log_info(f"[EXIT] {trade.side} {trade.symbol} @ ${current_price:.2f} "
                f"P&L=${trade.pnl:.2f} ({exit_reason}) | duration={duration_minutes:.1f}min")
    
    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary statistics"""
        if not self.active_trades:
            return {
                'position_count': 0,
                'total_exposure': 0,
                'symbols': [],
                'average_confidence': 0,
                'topics': {}
            }
        
        trades = list(self.active_trades.values())
        
        total_exposure = sum(trade.position_size for trade in trades)
        avg_confidence = sum(trade.combined_confidence for trade in trades) / len(trades)
        
        # Topic distribution
        topics = {}
        for trade in trades:
            topic = trade.topic or 'unknown'
            topics[topic] = topics.get(topic, 0) + 1
        
        return {
            'position_count': len(trades),
            'total_exposure': total_exposure,
            'symbols': [trade.symbol for trade in trades],
            'average_confidence': avg_confidence,
            'topics': topics
        }