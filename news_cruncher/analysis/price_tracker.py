"""
Price tracking functionality for trading decisions with configurable intervals
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
    """Handles price fetching and market timing logic with configurable intervals"""
    
    def __init__(self, fmp_loader: BaseFMPLoader) -> None:
        """Initialize price tracker"""
        self.fmp_loader = fmp_loader
        self.est_tz = pytz.timezone('US/Eastern')
        
        # Log current configuration
        checkpoint_info = Config.get_checkpoint_info()
        intervals = [f"{info['short_label']} ({info.get('minutes', 'close')})" for info in checkpoint_info]
        log_info(f"Price tracker initialized with configurable intervals: {', '.join(intervals)}")
        
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
        
        # Get checkpoint configuration
        checkpoint_info = Config.get_checkpoint_info()
        
        for checkpoint in checkpoint_info:
            label = checkpoint['short_label']
            minutes = checkpoint.get('minutes')
            
            if minutes is not None:
                # Regular interval checkpoint
                check_time = rec_time_est + timedelta(minutes=minutes)
                
                if self.is_market_hours(check_time):
                    schedule.append((label, check_time))
                    log_debug(f"Added {label} checkpoint: {check_time}")
                else:
                    log_debug(f"Skipping {label} checkpoint: outside market hours")
            else:
                # Close time checkpoint (special case)
                close_time = rec_time_est.replace(
                    hour=Config.CLOSE_PRICE_HOUR, 
                    minute=Config.CLOSE_PRICE_MINUTE, 
                    second=0, 
                    microsecond=0
                )
                
                # Add close price if recommendation was made before close time on a weekday
                if (rec_time_est.time() < close_time.time() and 
                    rec_time_est.weekday() < 5):
                    schedule.append((label, close_time))
                    log_debug(f"Added {label} checkpoint: {close_time}")
                else:
                    log_debug(f"Skipping {label} checkpoint: recommendation too late or weekend")
        
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
    """Background scheduler for monitoring price checkpoints with configurable intervals"""
    
    def __init__(self, price_tracker: PriceTracker, csv_logger) -> None:
        """Initialize tracking scheduler"""
        self.price_tracker = price_tracker
        self.csv_logger = csv_logger
        self.pending_tracks: Dict[str, Any] = {}  # {decision_id: TradingDecision}
        self.running = True
        
        # Log current configuration
        checkpoint_info = Config.get_checkpoint_info()
        log_info(f"Tracking scheduler initialized with {len(checkpoint_info)} configurable checkpoints")
        
    def add_tracking(self, decision) -> None:
        """Add a LONG/SHORT decision for price tracking with configurable intervals"""
        if decision.decision not in ['LONG', 'SHORT']:
            log_debug(f"Skipping tracking for {decision.ticker} - not LONG/SHORT decision")
            return
        
        # Check if entry price is already set (from main.py)
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
        
        # Calculate schedule with configurable intervals
        schedule = self.price_tracker.calculate_tracking_schedule(
            decision.recommendation_timestamp
        )
        decision.tracking_schedule = [target_time for _, target_time in schedule]
        
        # Initialize tracking fields
        decision.tracking_completed = False
        decision.tracking_status = "pending"
        
        # Store for tracking
        decision_id = f"{decision.ticker}_{decision.recommendation_timestamp.isoformat()}"
        self.pending_tracks[decision_id] = decision
        
        # Log with dynamic interval information
        checkpoint_labels = [label for label, _ in schedule]
        log_info(f"📊 Added price tracking for {decision.ticker}: ${current_price:.2f} baseline, {len(schedule)} checkpoints: {', '.join(checkpoint_labels)}")
        log_debug(f"📅 Tracking schedule: {[t.strftime('%H:%M:%S') for t in decision.tracking_schedule]}")
    
    async def check_pending_tracks(self) -> None:
        """Check for due price reads and execute them with configurable intervals"""
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
        
        # Log updates with dynamic labels
        if updated_tracks:
            log_info(f"📈 Updated prices for: {', '.join(updated_tracks)}")
        
        # Clean up completed tracks
        for decision_id in completed_tracks:
            completed_decision = self.pending_tracks.pop(decision_id)
            log_info(f"🏁 Removed completed tracking for {completed_decision.ticker}")
    
    async def _check_decision_schedule(self, decision, now: datetime) -> bool:
        """Check and execute due price checkpoints with configurable intervals"""
        updated = False
        checkpoint_info = Config.get_checkpoint_info()
        
        # Check each configured checkpoint
        for i, checkpoint in enumerate(checkpoint_info):
            # Check if we have a scheduled time for this checkpoint
            if i >= len(decision.tracking_schedule):
                continue
                
            checkpoint_time = decision.tracking_schedule[i]
            label = checkpoint['short_label']
            
            # Check if this checkpoint is due and hasn't been processed yet
            current_price = decision.get_checkpoint_price(i)
            
            if current_price is None and now >= checkpoint_time:
                log_debug(f"⏰ {label} checkpoint due for {decision.ticker}")
                price = self.price_tracker.get_current_price(decision.ticker)
                
                if price:
                    change_pct = self._calculate_change_pct(decision.recommendation_price, price)
                    decision.set_checkpoint_price(i, price, now, change_pct)
                    updated = True
                    
                    # Use dynamic logging
                    log_message = Config.format_price_checkpoint_log(decision.ticker, i, price, change_pct)
                    log_info(log_message)
                else:
                    log_error(f"❌ Failed to get {label} price for {decision.ticker}")
        
        # Check if tracking should be completed
        if not decision.tracking_completed:
            # Check if all scheduled checkpoints are done or if we're past the last checkpoint
            all_checkpoints_done = True
            for i in range(len(decision.tracking_schedule)):
                if decision.get_checkpoint_price(i) is None:
                    all_checkpoints_done = False
                    break
            
            if all_checkpoints_done:
                decision.tracking_completed = True
                decision.tracking_status = "completed"
                updated = True
                log_info(f"🏁 All checkpoints completed for {decision.ticker}")
            elif len(decision.tracking_schedule) > 0:
                # Check timeout condition
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
        """Main scheduler loop with enhanced configurable interval logging"""
        checkpoint_info = Config.get_checkpoint_info()
        intervals_summary = ", ".join([f"{info['short_label']}" for info in checkpoint_info])
        
        log_info(f"🕐 Price tracking scheduler started with intervals: {intervals_summary}")
        
        check_count = 0
        
        while self.running:
            try:
                check_count += 1
                
                # Log scheduler status every 10 checks with configurable interval info
                if check_count % 10 == 1:
                    pending_count = len(self.pending_tracks)
                    log_info(f"🕐 Scheduler check #{check_count}: {pending_count} positions being tracked")
                    log_debug(f"🔧 Current configuration: {Config.PRICE_CHECK_1_MINUTES}m, {Config.PRICE_CHECK_2_MINUTES}m, close at {Config.CLOSE_PRICE_HOUR:02d}:{Config.CLOSE_PRICE_MINUTE:02d}")
                    
                    if pending_count > 0:
                        # Show status of active tracks with dynamic labels
                        for decision_id, decision in list(self.pending_tracks.items())[:5]:  # Show first 5
                            status_parts = []
                            
                            for i, checkpoint in enumerate(checkpoint_info):
                                price = decision.get_checkpoint_price(i)
                                change_pct = decision.get_checkpoint_change_pct(i)
                                
                                if price is not None and change_pct is not None:
                                    status_parts.append(f"{checkpoint['short_label']}: {change_pct:+.1f}%")
                            
                            status = " | ".join(status_parts) if status_parts else "pending all"
                            log_info(f"  📊 {decision.ticker}: {status}")
                
                await self.check_pending_tracks()
                
                # Use configurable check interval
                await asyncio.sleep(Config.PRICE_TRACKER_CHECK_INTERVAL * 60)  # Convert minutes to seconds
                
            except Exception as e:
                log_error(f"Scheduler error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
        
        log_info("🕐 Price tracking scheduler stopped")
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of current tracking status with configurable interval info"""
        checkpoint_info = Config.get_checkpoint_info()
        
        summary = {
            'total_pending_tracks': len(self.pending_tracks),
            'configuration': {
                'check_intervals': [f"{info['short_label']} ({info.get('minutes', 'close')})" for info in checkpoint_info],
                'check1_minutes': Config.PRICE_CHECK_1_MINUTES,
                'check2_minutes': Config.PRICE_CHECK_2_MINUTES,
                'close_time': f"{Config.CLOSE_PRICE_HOUR:02d}:{Config.CLOSE_PRICE_MINUTE:02d} EST",
                'check_frequency': f"{Config.PRICE_TRACKER_CHECK_INTERVAL} minutes"
            },
            'active_positions': []
        }
        
        for decision_id, decision in self.pending_tracks.items():
            position_info = {
                'ticker': decision.ticker,
                'decision': decision.decision,
                'entry_price': decision.recommendation_price,
                'tracking_status': decision.tracking_status,
                'checkpoints_completed': []
            }
            
            # Check which checkpoints are completed
            for i, checkpoint in enumerate(checkpoint_info):
                price = decision.get_checkpoint_price(i)
                change_pct = decision.get_checkpoint_change_pct(i)
                
                if price is not None:
                    position_info['checkpoints_completed'].append({
                        'label': checkpoint['short_label'],
                        'price': price,
                        'change_pct': change_pct
                    })
            
            summary['active_positions'].append(position_info)
        
        return summary