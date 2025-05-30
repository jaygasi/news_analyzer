"""
Optimized price tracker with advanced caching and real-time processing
"""
import asyncio
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
import pandas as pd
import numpy as np

from data_loaders.fmp_data_loader import FmpDataLoader
from utils.log_utils import logi, logw, loge, logd
from utils.market_calendar import MarketCalendar
from utils.performance_monitor import register_component_performance
from config import CONFIG


@dataclass
class PriceUpdate:
    """Immutable price update record"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_sale_time: Optional[datetime] = None


@dataclass
class MarketData:
    """Consolidated market data for a symbol"""
    symbol: str
    current_price: float
    volume: int
    bid_ask_spread: float
    price_changes: deque = field(default_factory=lambda: deque(maxlen=100))
    volume_changes: deque = field(default_factory=lambda: deque(maxlen=100))
    last_update: datetime = field(default_factory=datetime.now)


class PriceCache:
    """High-performance price cache with automatic cleanup"""
    
    def __init__(self, max_age_hours: int = 24, max_symbols: int = 10000):
        self.max_age = timedelta(hours=max_age_hours)
        self.max_symbols = max_symbols
        self._data: Dict[str, MarketData] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def update_price(self, update: PriceUpdate) -> None:
        """Update price data with automatic cleanup"""
        with self._lock:
            if update.symbol not in self._data:
                self._data[update.symbol] = MarketData(
                    symbol=update.symbol,
                    current_price=update.price,
                    volume=update.volume,
                    bid_ask_spread=0.0
                )
            
            market_data = self._data[update.symbol]
            
            # Calculate bid-ask spread if available
            if update.bid and update.ask and update.bid > 0:
                market_data.bid_ask_spread = (update.ask - update.bid) / update.bid
            
            # Track price changes
            if market_data.current_price != update.price:
                change = (update.price - market_data.current_price) / market_data.current_price
                market_data.price_changes.append(change)
                market_data.current_price = update.price
            
            # Track volume changes
            if market_data.volume != update.volume:
                if market_data.volume > 0:
                    volume_change = (update.volume - market_data.volume) / market_data.volume
                    market_data.volume_changes.append(volume_change)
                market_data.volume = update.volume
            
            market_data.last_update = update.timestamp
            self._access_times[update.symbol] = update.timestamp
            
            # Cleanup if needed
            if len(self._data) > self.max_symbols:
                self._cleanup_old_data()
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data for symbol"""
        with self._lock:
            if symbol in self._data:
                self._access_times[symbol] = datetime.now()
                return self._data[symbol]
            return None
    
    def get_all_symbols(self) -> Set[str]:
        """Get all tracked symbols"""
        with self._lock:
            return set(self._data.keys())
    
    def get_recent_updates(self, max_age_minutes: int = 5) -> Dict[str, MarketData]:
        """Get recently updated market data"""
        cutoff = datetime.now() - timedelta(minutes=max_age_minutes)
        with self._lock:
            return {
                symbol: data for symbol, data in self._data.items()
                if data.last_update >= cutoff
            }
    
    def _cleanup_old_data(self) -> None:
        """Remove old or least recently accessed data"""
        now = datetime.now()
        cutoff = now - self.max_age
        
        # Remove expired data
        expired_symbols = [
            symbol for symbol, data in self._data.items()
            if data.last_update < cutoff
        ]
        
        for symbol in expired_symbols:
            del self._data[symbol]
            self._access_times.pop(symbol, None)
        
        # If still too many symbols, remove least recently accessed
        if len(self._data) > self.max_symbols:
            sorted_symbols = sorted(
                self._access_times.items(),
                key=lambda x: x[1]
            )
            
            excess_count = len(self._data) - self.max_symbols
            for symbol, _ in sorted_symbols[:excess_count]:
                self._data.pop(symbol, None)
                self._access_times.pop(symbol, None)


class OptimizedPriceTracker:
    """High-performance price tracker with real-time processing"""
    
    def __init__(self, fmp_api_key: str, update_interval: float = 1.0):
        self.fmp_data_loader = FmpDataLoader(fmp_api_key)
        self.update_interval = update_interval
        self.market_calendar = MarketCalendar('NYSE')
        
        # Core data structures
        self.price_cache = PriceCache()
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # State management
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        self._last_fetch_time: Optional[datetime] = None
        
        # Performance tracking
        self.fetch_count = 0
        self.error_count = 0
        self.last_update_count = 0
        
        # Thread safety
        self._state_lock = threading.RLock()
    
    def subscribe_to_updates(self, symbol: str, callback: Callable[[PriceUpdate], None]) -> None:
        """Subscribe to price updates for a symbol"""
        with self._state_lock:
            self.subscribers[symbol].append(callback)
    
    def unsubscribe_from_updates(self, symbol: str, callback: Callable[[PriceUpdate], None]) -> None:
        """Unsubscribe from price updates"""
        with self._state_lock:
            if symbol in self.subscribers and callback in self.subscribers[symbol]:
                self.subscribers[symbol].remove(callback)
    
    def _notify_subscribers(self, update: PriceUpdate) -> None:
        """Notify subscribers of price updates"""
        callbacks = self.subscribers.get(update.symbol, [])
        for callback in callbacks:
            try:
                callback(update)
            except Exception as e:
                loge(f"Error in price update callback for {update.symbol}: {e}")
    
    async def _fetch_prices_batch(self) -> bool:
        """Fetch prices in optimized batch"""
        try:
            start_time = time.time()
            
            # Fetch real-time prices asynchronously
            prices_df = await asyncio.to_thread(
                self.fmp_data_loader.fetch_realtime_prices
            )
            
            if prices_df is None or len(prices_df) == 0:
                self.error_count += 1
                if self.error_count % 10 == 0:
                    logw(f"No price data received (error #{self.error_count})")
                return False
            
            # Process updates efficiently
            update_count = 0
            for _, row in prices_df.iterrows():
                try:
                    update = PriceUpdate(
                        symbol=row['symbol'],
                        timestamp=datetime.now(),
                        price=float(row.get('lastSalePrice', 0)),
                        volume=int(row.get('volume', 0)),
                        bid=float(row.get('bidPrice', 0)) if row.get('bidPrice') else None,
                        ask=float(row.get('askPrice', 0)) if row.get('askPrice') else None,
                        last_sale_time=pd.to_datetime(row.get('lastUpdated'), unit='ms', errors='coerce')
                    )
                    
                    if update.price > 0:  # Valid price
                        self.price_cache.update_price(update)
                        self._notify_subscribers(update)
                        update_count += 1
                        
                except (ValueError, TypeError) as e:
                    logd(f"Invalid price data for {row.get('symbol', 'unknown')}: {e}")
                    continue
            
            # Update performance metrics
            self.fetch_count += 1
            self.error_count = 0  # Reset on success
            self.last_update_count = update_count
            self._last_fetch_time = datetime.now()
            
            fetch_time = time.time() - start_time
            
            # Register performance metrics
            register_component_performance(
                'price_tracker',
                fetch_time=fetch_time,
                updates_processed=update_count,
                total_fetches=self.fetch_count,
                error_rate=self.error_count / max(self.fetch_count, 1)
            )
            
            if self.fetch_count % 100 == 0:
                logi(f"📊 Price tracker: {self.fetch_count} fetches, "
                     f"{update_count} updates in {fetch_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.error_count += 1
            loge(f"Error fetching prices: {e}")
            return False
    
    def _is_market_open(self) -> bool:
        """Check if market is open (with caching)"""
        try:
            return self.market_calendar.is_market_open_now(extended_hours=True)
        except Exception:
            # Default to open if check fails
            return True
    
    async def _price_tracking_loop(self) -> None:
        """Main price tracking loop"""
        logi("🚀 Starting optimized price tracking...")
        consecutive_errors = 0
        
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # Adaptive interval based on market status
                if self._is_market_open():
                    interval = self.update_interval
                else:
                    interval = self.update_interval * 5  # Slower during closed hours
                
                # Fetch price updates
                success = await self._fetch_prices_batch()
                
                if success:
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    
                    # Exponential backoff on consecutive errors
                    if consecutive_errors > 3:
                        backoff_interval = min(interval * (2 ** (consecutive_errors - 3)), 60)
                        logw(f"Multiple errors, backing off for {backoff_interval:.1f}s")
                        interval = backoff_interval
                
                # Wait for next iteration
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue loop
                    
            except asyncio.CancelledError:
                logi("📊 Price tracker cancelled")
                break
            except Exception as e:
                consecutive_errors += 1
                loge(f"Critical error in price tracker: {e}")
                
                # Emergency recovery
                if consecutive_errors > 10:
                    loge("🚨 Too many critical errors, emergency pause")
                    await asyncio.sleep(60)
                    consecutive_errors = 0
                else:
                    await asyncio.sleep(self.update_interval)
        
        logi("⏹️  Price tracker stopped")
    
    async def start(self) -> None:
        """Start price tracking"""
        if self.is_running:
            return
        
        with self._state_lock:
            self.is_running = True
            self._shutdown_event.clear()
        
        await self._price_tracking_loop()
    
    def stop(self) -> None:
        """Stop price tracking"""
        with self._state_lock:
            if not self.is_running:
                return
            
            logi("🛑 Stopping price tracker...")
            self.is_running = False
            self._shutdown_event.set()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        market_data = self.price_cache.get_market_data(symbol)
        return market_data.current_price if market_data else None
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get complete market data for symbol"""
        return self.price_cache.get_market_data(symbol)
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get current prices for all tracked symbols"""
        return {
            symbol: data.current_price
            for symbol, data in self.price_cache.get_recent_updates().items()
        }
    
    def get_price_changes(self, symbol: str, lookback_periods: int = 10) -> List[float]:
        """Get recent price changes for symbol"""
        market_data = self.price_cache.get_market_data(symbol)
        if market_data:
            return list(market_data.price_changes)[-lookback_periods:]
        return []
    
    def calculate_momentum(self, symbol: str, periods: int = 5) -> Optional[float]:
        """Calculate momentum for symbol"""
        changes = self.get_price_changes(symbol, periods)
        if len(changes) >= periods:
            return sum(changes) / len(changes)
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._state_lock:
            total_symbols = len(self.price_cache.get_all_symbols())
            recent_updates = len(self.price_cache.get_recent_updates())
            
            return {
                'total_fetches': self.fetch_count,
                'error_count': self.error_count,
                'last_update_count': self.last_update_count,
                'total_symbols_tracked': total_symbols,
                'recent_active_symbols': recent_updates,
                'last_fetch_time': self._last_fetch_time.isoformat() if self._last_fetch_time else None,
                'is_running': self.is_running,
                'success_rate': (self.fetch_count - self.error_count) / max(self.fetch_count, 1)
            }
    
    # Backward compatibility methods
    def get_prices(self) -> pd.DataFrame:
        """Get prices as DataFrame for backward compatibility"""
        recent_data = self.price_cache.get_recent_updates()
        
        if not recent_data:
            return pd.DataFrame()
        
        rows = []
        for symbol, data in recent_data.items():
            rows.append({
                'symbol': symbol,
                'lastSalePrice': data.current_price,
                'volume': data.volume,
                'lastUpdated': int(data.last_update.timestamp() * 1000),
                'bidAskSpread': data.bid_ask_spread
            })
        
        return pd.DataFrame(rows)


# Maintain singleton pattern for backward compatibility
class PriceTracker:
    """Backward compatible singleton wrapper"""
    _instance: Optional[OptimizedPriceTracker] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Extract API key from config if not provided
            fmp_api_key = kwargs.get('fmp_api_key') or CONFIG.get_api_key('fmp') or ""
            update_interval = kwargs.get('data_collection_interval', CONFIG.data.price_tick_interval * 10)
            
            cls._instance = OptimizedPriceTracker(fmp_api_key, update_interval)
        return cls._instance
    
    def __getattr__(self, name):
        """Delegate all attribute access to the optimized instance"""
        return getattr(self._instance, name)