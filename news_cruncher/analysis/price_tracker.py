"""
Price tracking functionality for trading decisions - FIXED to avoid double price fetching
Python 3.13.3 compatible
"""
import asyncio
import pytz
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_error, log_debug, log_warning
from config import Config


class PriceTracker:
    """Handles price fetching and market timing logic"""
    
    def __init__(self, fmp_loader: BaseFMPLoader) -> None:
        """Initialize price tracker"""
        self.fmp_loader = fmp_loader
        self.est_tz = pytz.timezone('US/Eastern')
        
    def is_market_hours(self, dt: datetime) -> bool:
        """Check if datetime is during market hours (9:30am-4:00pm EST)"""
        est_dt = dt.astimezone(self.est_tz)
        
        # Check if weekend
        if est_dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check time range (9:30am-4:00pm EST)
        market_open = est_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = est_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= est_dt <= market_close
    
    def calculate_tracking_schedule(self, recommendation_time: datetime) -> List[Tuple[str, datetime]]:
        """Calculate intelligent price check schedule using configurable intervals"""
        schedule = []
        rec_time_est = recommendation_time.astimezone(self.est_tz)
        
        log_debug(f"Calculating tracking schedule from {rec_time_est} EST")
        
        # Use configurable intervals from Config
        time_check1 = rec_time_est + timedelta(minutes=Config.PRICE_CHECK_1_MINUTES)
        time_check2 = rec_time_est + timedelta(minutes=Config.PRICE_CHECK_2_MINUTES)
        time_close = rec_time_est.replace(
            hour=Config.CLOSE_PRICE_HOUR, 
            minute=Config.CLOSE_PRICE_MINUTE, 
            second=0, 
            microsecond=0
        )
        
        # Apply intelligent fallback logic
        if self.is_market_hours(time_check1):
            schedule.append((f"{Config.PRICE_CHECK_1_MINUTES}m", time_check1))
            log_debug(f"Added {Config.PRICE_CHECK_1_MINUTES}m checkpoint: {time_check1}")
            
        # Only add second check if it's before close time
        if (self.is_market_hours(time_check2) and 
            time_check2.time() < time_close.time()):
            schedule.append((f"{Config.PRICE_CHECK_2_MINUTES}m", time_check2))
            log_debug(f"Added {Config.PRICE_CHECK_2_MINUTES}m checkpoint: {time_check2}")
            
        # Add close price if recommendation was made before close time on a weekday
        if (rec_time_est.time() < time_close.time() and 
            rec_time_est.weekday() < 5):
            schedule.append(("close", time_close))
            log_debug(f"Added close checkpoint: {time_close}")
            
        log_info(f"Price tracking schedule: {len(schedule)} checkpoints for {rec_time_est}")
        return schedule
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Fetch current price using FMP API with enhanced error handling"""
        try:
            log_debug(f"Fetching current price for {ticker}")
            data = self.fmp_loader.make_request(f"quote/{ticker}")
            
            if data and isinstance(data, list) and len(data) > 0:
                price_data = data[0]
                price = float(price_data.get('price', 0))
                
                if price > 0:
                    log_debug(f"✅ Fetched price for {ticker}: ${price:.2f}")
                    return price
                else:
                    log_warning(f"❌ Invalid price for {ticker}: {price}")
                    return None
            else:
                log_warning(f"❌ No price data returned for {ticker}: {data}")
                return None
                
        except Exception as e:
            log_error(f"❌ Error fetching price for {ticker}: {e}")
            return None


class TrackingScheduler:
    """Background scheduler for monitoring price checkpoints - FIXED to avoid double fetching"""
    
    def __init__(self, price_tracker: PriceTracker, csv_logger) -> None:
        """Initialize tracking scheduler"""
        self.price_tracker = price_tracker
        self.csv_logger = csv_logger
        self.pending_tracks: Dict[str, Any] = {}  # {decision_id: TradingDecision}
        self.running = True
        
    def add_tracking(self, decision) -> None:
        """Add a LONG/SHORT decision for price tracking - FIXED to avoid double price fetch"""
        if decision.decision not in ['LONG', 'SHORT']:
            log_debug(f"Skipping tracking for {decision.ticker} - not LONG/SHORT decision")
            return
        
        # ======================================================================
        # FIXED: Check if entry price is already set (from main.py)
        # ======================================================================
        current_price = None
        if hasattr(decision, 'recommendation_price') and decision.recommendation_price:
            current_price = decision.recommendation_price
            log_info(f"📊 Using existing entry price for {decision.ticker}: ${current_price:.2f}")
        else:
            # Fallback: fetch price if not already set
            log_warning(f"⚠️ No entry price set for {decision.ticker}, fetching now...")
            current_price = self.price_tracker.get_current_price(decision.ticker)
            
            if current_price:
                decision.recommendation_price = current_price
                decision.recommendation_timestamp = datetime.now(timezone.utc)
                log_info(f"📊 Fetched fallback entry price for {decision.ticker}: ${current_price:.2f}")
            else:
                log_error(f"❌ Could not get entry price for {decision.ticker}, skipping tracking")
                return
        
        # Calculate schedule
        schedule = self.price_tracker.calculate_tracking_schedule(
            decision.recommendation_timestamp
        )
        decision.tracking_schedule = [target_time for _, target_time in schedule]
        
        # Initialize price tracking fields
        decision.price_45m = None
        decision.price_45m_timestamp = None
        decision.price_45m_change_pct = None
        decision.price_1hr = None
        decision.price_1hr_timestamp = None
        decision.price_1hr_change_pct = None
        decision.price_close = None
        decision.price_close_timestamp = None
        decision.price_close_change_pct = None
        decision.tracking_completed = False
        decision.tracking_status = "pending"
        
        # Store for tracking
        decision_id = f"{decision.ticker}_{decision.recommendation_timestamp.isoformat()}"
        self.pending_tracks[decision_id] = decision
        
        log_info(f"📊 Added price tracking for {decision.ticker}: ${current_price:.2f} baseline, {len(schedule)} checkpoints")
        log_debug(f"📅 Tracking schedule: {[t.strftime('%H:%M:%S') for t in decision.tracking_schedule]}")
    
    async def check_pending_tracks(self) -> None:
        """Check for due price reads and execute them"""
        if not self.pending_tracks:
            return
            
        now = datetime.now(timezone.utc)
        completed_tracks = []
        updated_tracks = []
        
        log_debug(f"Checking {len(self.pending_tracks)} pending tracks at {now}")
        
        for decision_id, decision in list(self.pending_tracks.items()):
            try:
                updated = await self._check_decision_schedule(decision, now)
                if updated:
                    updated_tracks.append(decision.ticker)
                    # Update CSV with new price data
                    self.csv_logger.update_decision_prices(decision)
                    
                if decision.tracking_completed:
                    completed_tracks.append(decision_id)
                    log_info(f"✅ Completed price tracking for {decision.ticker}")
                    
            except Exception as e:
                log_error(f"Error checking track {decision_id}: {e}")
        
        # Log updates
        if updated_tracks:
            log_info(f"📈 Updated prices for: {', '.join(updated_tracks)}")
        
        # Clean up completed tracks
        for decision_id in completed_tracks:
            completed_decision = self.pending_tracks.pop(decision_id)
            log_info(f"🏁 Removed completed tracking for {completed_decision.ticker}")
    
    async def _check_decision_schedule(self, decision, now: datetime) -> bool:
        """Check and execute due price checkpoints with enhanced logging"""
        updated = False
        
        # Check 45m checkpoint
        if decision.price_45m is None and len(decision.tracking_schedule) > 0:
            checkpoint_time = decision.tracking_schedule[0]
            if now >= checkpoint_time:
                log_debug(f"⏰ 45m checkpoint due for {decision.ticker}")
                price = self.price_tracker.get_current_price(decision.ticker)
                if price:
                    decision.price_45m = price
                    decision.price_45m_timestamp = now
                    decision.price_45m_change_pct = self._calculate_change_pct(
                        decision.recommendation_price, price
                    )
                    updated = True
                    log_info(f"📈 45m: {decision.ticker} ${price:.2f} ({decision.price_45m_change_pct:+.2f}%)")
                else:
                    log_error(f"❌ Failed to get 45m price for {decision.ticker}")
        
        # Check 1hr checkpoint
        if decision.price_1hr is None and len(decision.tracking_schedule) > 1:
            checkpoint_time = decision.tracking_schedule[1]
            if now >= checkpoint_time:
                log_debug(f"⏰ 1hr checkpoint due for {decision.ticker}")
                price = self.price_tracker.get_current_price(decision.ticker)
                if price:
                    decision.price_1hr = price
                    decision.price_1hr_timestamp = now
                    decision.price_1hr_change_pct = self._calculate_change_pct(
                        decision.recommendation_price, price
                    )
                    updated = True
                    log_info(f"📈 1hr: {decision.ticker} ${price:.2f} ({decision.price_1hr_change_pct:+.2f}%)")
                else:
                    log_error(f"❌ Failed to get 1hr price for {decision.ticker}")
        
        # Check close checkpoint
        if decision.price_close is None and len(decision.tracking_schedule) > 2:
            checkpoint_time = decision.tracking_schedule[2]
            if now >= checkpoint_time:
                log_debug(f"⏰ Close checkpoint due for {decision.ticker}")
                price = self.price_tracker.get_current_price(decision.ticker)
                if price:
                    decision.price_close = price
                    decision.price_close_timestamp = now
                    decision.price_close_change_pct = self._calculate_change_pct(
                        decision.recommendation_price, price
                    )
                    decision.tracking_completed = True
                    decision.tracking_status = "completed"
                    updated = True
                    log_info(f"📈 Close: {decision.ticker} ${price:.2f} ({decision.price_close_change_pct:+.2f}%)")
                else:
                    log_error(f"❌ Failed to get close price for {decision.ticker}")
        
        # Check if tracking should be completed due to timeout
        if not decision.tracking_completed and len(decision.tracking_schedule) > 0:
            last_checkpoint = decision.tracking_schedule[-1]
            hours_past_last = (now - last_checkpoint).total_seconds() / 3600
            
            if hours_past_last > 2:  # 2 hours past last checkpoint
                log_warning(f"⏰ Timing out tracking for {decision.ticker} ({hours_past_last:.1f}h past last checkpoint)")
                decision.tracking_completed = True
                decision.tracking_status = "partial"
                updated = True
        
        return updated
    
    def _calculate_change_pct(self, original_price: float, new_price: float) -> float:
        """Calculate percentage change"""
        if original_price and original_price > 0:
            return ((new_price - original_price) / original_price) * 100
        return 0.0
    
    async def run_scheduler(self) -> None:
        """Main scheduler loop - runs as background task with enhanced logging"""
        log_info("🕐 Price tracking scheduler started")
        
        check_count = 0
        
        while self.running:
            try:
                check_count += 1
                
                # Log scheduler status every 10 checks (50 minutes with 5-minute intervals)
                if check_count % 10 == 1:
                    pending_count = len(self.pending_tracks)
                    log_info(f"🕐 Scheduler check #{check_count}: {pending_count} positions being tracked")
                    
                    if pending_count > 0:
                        # Show status of active tracks
                        for decision_id, decision in list(self.pending_tracks.items())[:5]:  # Show first 5
                            status_parts = []
                            if decision.price_45m:
                                status_parts.append(f"45m: {decision.price_45m_change_pct:+.1f}%")
                            if decision.price_1hr:
                                status_parts.append(f"1hr: {decision.price_1hr_change_pct:+.1f}%")
                            if decision.price_close:
                                status_parts.append(f"close: {decision.price_close_change_pct:+.1f}%")
                            
                            status = " | ".join(status_parts) if status_parts else "pending all"
                            log_info(f"  📊 {decision.ticker}: {status}")
                
                await self.check_pending_tracks()
                
                # Use configurable check interval
                await asyncio.sleep(Config.PRICE_TRACKER_CHECK_INTERVAL * 60)  # Convert minutes to seconds
                
            except Exception as e:
                log_error(f"Scheduler error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
        
        log_info("🕐 Price tracking scheduler stopped")
