"""
Price tracking functionality for trading decisions with configurable intervals
Python 3.13.3 compatible - OPTIMIZED VERSION
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
        intervals = [info['short_label'] for info in checkpoint_info]
        log_info(f"Price tracker initialized with configurable intervals: {', '.join(intervals)}")
        
    def is_market_hours(self, dt: datetime) -> bool:
        """Check if datetime is during market hours with proper timezone handling"""
        # Ensure we're working with timezone-aware datetime
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        est_dt = dt.astimezone(self.est_tz)
        
        # Check if weekend
        if est_dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Proper market hours check (9:30am-4:00pm EST)
        market_open = est_dt.replace(
            hour=Config.MARKET_OPEN_HOUR, 
            minute=Config.MARKET_OPEN_MINUTE, 
            second=0, microsecond=0
        )
        market_close = est_dt.replace(
            hour=Config.MARKET_CLOSE_HOUR, 
            minute=Config.MARKET_CLOSE_MINUTE, 
            second=0, microsecond=0
        )
        
        return market_open <= est_dt <= market_close
    
    def calculate_tracking_schedule(self, recommendation_time: datetime) -> List[Tuple[str, datetime]]:
        """Calculate realistic tracking schedule based on market hours"""
        
        if recommendation_time.tzinfo is None:
            recommendation_time = recommendation_time.replace(tzinfo=timezone.utc)
        
        rec_time_est = recommendation_time.astimezone(self.est_tz)
        standard_checkpoints_config = Config.get_checkpoint_info()
        
        log_info(f"Calculating realistic tracking schedule from: {rec_time_est.strftime('%Y-%m-%d %H:%M:%S')} EST")

        if self.is_market_hours(rec_time_est):
            return self._schedule_during_market_hours(rec_time_est, standard_checkpoints_config)
        else:
            return self._schedule_outside_market_hours(rec_time_est, standard_checkpoints_config)

    def _schedule_during_market_hours(self, rec_time_est: datetime, checkpoints_config: List[Dict]) -> List[Tuple[str, datetime]]:
        """Handle scheduling during market hours"""
        log_info("   📈 SCENARIO: During market hours")
        schedule = []
        
        # Get market timing
        market_close_est = rec_time_est.replace(
            hour=getattr(Config, 'MARKET_CLOSE_HOUR', 16),
            minute=getattr(Config, 'MARKET_CLOSE_MINUTE', 0),
            second=0, microsecond=0
        )
        close_price_est = rec_time_est.replace(
            hour=getattr(Config, 'CLOSE_PRICE_HOUR', 15),
            minute=getattr(Config, 'CLOSE_PRICE_MINUTE', 30),
            second=0, microsecond=0
        )
        
        # Handle checkpoint 1 and 2
        for i in range(2):
            cp_config = checkpoints_config[i]
            label = cp_config['short_label']
            minutes_offset = cp_config['minutes']
            
            potential_target = rec_time_est + timedelta(minutes=minutes_offset)
            final_target = self._adjust_target_for_close(potential_target, close_price_est, market_close_est, label)
            
            schedule.append((label, final_target.astimezone(timezone.utc)))

        # Handle close price
        close_target = self._get_close_target(rec_time_est, close_price_est)
        close_label = checkpoints_config[2]['short_label']
        schedule.append((close_label, close_target.astimezone(timezone.utc)))
        
        return schedule

    def _schedule_outside_market_hours(self, rec_time_est: datetime, checkpoints_config: List[Dict]) -> List[Tuple[str, datetime]]:
        """Handle scheduling outside market hours"""
        log_info("   🌙 SCENARIO: Outside market hours - entry at next market open")
        schedule = []
        
        next_open_est = self._get_next_market_open_time(rec_time_est)
        log_info(f"   📅 Next market open: {next_open_est.strftime('%Y-%m-%d %H:%M')} EST")

        # Handle checkpoint 1 and 2
        for i in range(2):
            cp_config = checkpoints_config[i]
            label = cp_config['short_label']
            minutes_offset = cp_config['minutes']
            
            target_time = next_open_est + timedelta(minutes=minutes_offset)
            schedule.append((label, target_time.astimezone(timezone.utc)))
            log_info(f"   ✅ {label}: {target_time.strftime('%H:%M')} EST (open + {minutes_offset}min)")

        # Close price same day as open
        same_day_close = next_open_est.replace(
            hour=getattr(Config, 'CLOSE_PRICE_HOUR', 15),
            minute=getattr(Config, 'CLOSE_PRICE_MINUTE', 30),
            second=0, microsecond=0
        )
        close_label = checkpoints_config[2]['short_label']
        schedule.append((close_label, same_day_close.astimezone(timezone.utc)))
        log_info(f"   🏁 {close_label}: {same_day_close.strftime('%H:%M')} EST (same day)")
        
        return schedule

    def _adjust_target_for_close(self, potential_target: datetime, close_price_est: datetime, 
                                market_close_est: datetime, label: str) -> datetime:
        """Adjust target time if it falls after close"""
        if potential_target > market_close_est:
            log_info(f"   ⏰ {label} (as close): Using close time - checkpoint after market close")
            return close_price_est
        elif potential_target > close_price_est:
            log_info(f"   ⏰ {label} (at close): Using close time - checkpoint after target close")
            return close_price_est
        else:
            log_info(f"   ✅ {label}: Normal timing at {potential_target.strftime('%H:%M')} EST")
            return potential_target

    def _get_close_target(self, rec_time_est: datetime, close_price_est: datetime) -> datetime:
        """Get appropriate close target time"""
        if close_price_est <= rec_time_est:
            # Move to next trading day
            next_close = close_price_est + timedelta(days=1)
            while next_close.weekday() >= 5:
                next_close += timedelta(days=1)
            log_info(f"   🏁 Close: Next trading day {next_close.strftime('%Y-%m-%d %H:%M')} EST")
            return next_close
        else:
            log_info(f"   🏁 Close: Today {close_price_est.strftime('%H:%M')} EST")
            return close_price_est

    def is_extended_hours(self, dt: datetime) -> bool:
        """Check if datetime is during extended hours (4:00am-8:00pm EST)"""
        if dt.tzinfo is None:
            dt_utc = dt.replace(tzinfo=timezone.utc)
            est_dt = dt_utc.astimezone(self.est_tz)
        elif dt.tzinfo != self.est_tz:
            est_dt = dt.astimezone(self.est_tz)
        else:
            est_dt = dt

        if est_dt.weekday() >= 5:  # Weekend
            return False
            
        return Config.EXTENDED_OPEN_HOUR <= est_dt.hour < Config.EXTENDED_CLOSE_HOUR

    def _get_next_market_open_time(self, from_time_est: datetime) -> datetime:
        """Calculate the next valid market open time in EST"""
        next_open_est = from_time_est.replace(
            hour=getattr(Config, 'MARKET_OPEN_HOUR', 9),
            minute=getattr(Config, 'MARKET_OPEN_MINUTE', 30),
            second=0, microsecond=0
        )

        while next_open_est.weekday() >= 5 or next_open_est <= from_time_est:
            next_open_est += timedelta(days=1)
            next_open_est = next_open_est.replace(
                hour=getattr(Config, 'MARKET_OPEN_HOUR', 9),
                minute=getattr(Config, 'MARKET_OPEN_MINUTE', 30),
                second=0, microsecond=0
            )
        return next_open_est
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Fetch current price with proper error handling"""
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
                log_warning(f"❌ No price data returned for {ticker}")
                return None
                
        except Exception as e:
            log_error(f"❌ Error fetching price for {ticker}: {e}")
            return None


class TrackingScheduler:
    """Background scheduler for monitoring price checkpoints"""
  
    def __init__(self, price_tracker: PriceTracker, csv_logger) -> None:
        """Initialize tracking scheduler with learning integration"""
        self.price_tracker = price_tracker
        self.csv_logger = csv_logger
        self.pending_tracks: Dict[str, Any] = {}
        self.running = True
        
        # Initialize learning system for feedback
        self.learning_system = None
        self._initialize_learning_system()
        
        checkpoint_info = Config.get_checkpoint_info()
        log_info(f"Tracking scheduler initialized with {len(checkpoint_info)} configurable checkpoints")

    def _initialize_learning_system(self) -> None:
        """Initialize learning system for providing feedback"""
        try:
            # Only initialize if learning feedback is enabled
            if getattr(Config, 'ENABLE_LEARNING_FEEDBACK', True):
                from tools.model_learning import MultiModalLearningSystem
                self.learning_system = MultiModalLearningSystem()
                log_info("🎓 Price tracker connected to learning system for feedback")
            else:
                log_debug("📚 Learning feedback disabled in price tracker")
        except Exception as e:
            log_warning(f"Learning system integration failed in price tracker: {e}")
            self.learning_system = None
        
    def add_tracking(self, decision) -> None:
        """Add a LONG/SHORT decision for price tracking"""
        if decision.decision not in ['LONG', 'SHORT']:
            log_debug(f"Skipping tracking for {decision.ticker} - not LONG/SHORT decision")
            return
        
        # Ensure we have entry price
        current_price = self._ensure_entry_price(decision)
        if current_price is None:
            log_error(f"❌ Could not get entry price for {decision.ticker}, skipping tracking")
            return
        
        # Calculate realistic schedule
        schedule = self.price_tracker.calculate_tracking_schedule(decision.recommendation_timestamp)
        decision.tracking_schedule = [target_time for _, target_time in schedule]
        
        # Initialize tracking
        decision.tracking_completed = False
        decision.tracking_status = "pending"
        
        # Store for tracking
        decision_id = f"{decision.ticker}_{decision.recommendation_timestamp.isoformat()}"
        self.pending_tracks[decision_id] = decision
        
        checkpoint_labels = [label for label, _ in schedule]
        log_info(f"📊 Added realistic tracking for {decision.ticker}: ${current_price:.2f} baseline, "
                f"checkpoints: {', '.join(checkpoint_labels)}")

    def _ensure_entry_price(self, decision) -> Optional[float]:
        """Ensure decision has entry price"""
        if hasattr(decision, 'recommendation_price') and decision.recommendation_price is not None:
            return decision.recommendation_price
        
        # Fetch current price as fallback
        log_warning(f"⚠️ No entry price set for {decision.ticker}, fetching current price...")
        current_price = self.price_tracker.get_current_price(decision.ticker)
        
        if current_price:
            decision.recommendation_price = current_price
            decision.recommendation_timestamp = datetime.now(timezone.utc)
            log_info(f"📊 Set entry price for {decision.ticker}: ${current_price:.2f}")
            return current_price
        
        return None
    
    async def check_pending_tracks(self) -> None:
        """Check for due price reads and execute them with learning feedback"""
        if not self.pending_tracks:
            return
            
        now = datetime.now(timezone.utc)
        completed_tracks = []
        updated_tracks = []
        learning_insights = []
        
        for decision_id, decision in list(self.pending_tracks.items()):
            try:
                updated = await self._check_decision_schedule(decision, now)
                if updated:
                    updated_tracks.append(decision.ticker)
                    self.csv_logger.update_decision_prices(decision)
                    
                if decision.tracking_completed:
                    completed_tracks.append(decision_id)
                    
                    # Collect learning insight for this completion
                    if self.learning_system:
                        insight = self._generate_completion_learning_insight(decision)
                        if insight:
                            learning_insights.append(insight)
                    
                    log_info(f"✅ Completed realistic tracking for {decision.ticker}")
                    
            except Exception as e:
                log_error(f"Error checking track {decision_id}: {e}")
        
        if updated_tracks:
            log_info(f"📈 Updated prices for: {', '.join(updated_tracks)}")
        
        # Log learning insights from completed trades
        if learning_insights:
            log_info("🎓 Learning Insights from Completed Trades:")
            for insight in learning_insights[:3]:  # Limit to top 3
                log_info(f"   💡 {insight}")
        
        # Clean up completed tracks
        for decision_id in completed_tracks:
            self.pending_tracks.pop(decision_id)

    def _generate_completion_learning_insight(self, decision) -> Optional[str]:
        """Generate learning insight for completed trade"""
        try:
            metrics = self._calculate_trade_metrics(decision)
            if not metrics:
                return None
            
            ticker = decision.ticker
            decision_type = metrics['decision_type']
            
            if decision_type in ['LONG', 'SHORT']:
                opportunity_cost = metrics['opportunity_cost']
                optimal_exit = metrics['optimal_exit_time']
                
                if opportunity_cost > 1.0:
                    return f"{ticker}: {opportunity_cost:.1f}% cost, {optimal_exit} was optimal"
                else:
                    return f"{ticker}: Near-optimal exit ({optimal_exit})"
            
            elif decision_type == 'NONE':
                if metrics['should_have_traded']:
                    missed = metrics['missed_opportunity']
                    direction = 'LONG' if metrics['max_long_opportunity'] > metrics['max_short_opportunity'] else 'SHORT'
                    return f"{ticker}: Missed {direction} opportunity ({missed:.1f}%)"
            
            return None
            
        except Exception as e:
            log_debug(f"Learning insight generation failed: {e}")
            return None
    
    async def _check_decision_schedule(self, decision, now: datetime) -> bool:
        """Check and execute due price checkpoints with learning feedback"""
        updated = False
        checkpoint_info = Config.get_checkpoint_info()
        
        for i, checkpoint in enumerate(checkpoint_info):
            if i < len(decision.tracking_schedule):
                scheduled_time = decision.tracking_schedule[i]
                
                # Ensure UTC comparison
                if isinstance(scheduled_time, datetime) and scheduled_time.tzinfo:
                    scheduled_utc = scheduled_time.astimezone(timezone.utc)
                else:
                    scheduled_utc = scheduled_time.replace(tzinfo=timezone.utc)
                
                # Check if it's time and we don't have price yet
                if now >= scheduled_utc and decision.get_checkpoint_price(i) is None:
                    price = self.price_tracker.get_current_price(decision.ticker)
                    
                    if price:
                        decision.set_checkpoint_price(i, price)
                        
                        if decision.recommendation_price:
                            change_pct = self._calculate_change_pct(decision.recommendation_price, price)
                            decision.set_checkpoint_change_pct(i, change_pct)
                            
                            log_info(f"📊 {checkpoint['short_label']} update for {decision.ticker}: "
                                   f"${price:.2f} ({change_pct:+.2f}%)")
                        
                        decision.set_checkpoint_timestamp(i, now)
                        updated = True
        
        # Check completion and provide learning feedback
        completed_checkpoints = sum(1 for i in range(len(checkpoint_info)) 
                                  if decision.get_checkpoint_price(i) is not None)
        
        if completed_checkpoints == len(checkpoint_info):
            decision.tracking_completed = True
            decision.tracking_status = "completed"
            updated = True
            
            # Process completed trade for learning feedback
            await self._process_completed_trade_for_learning(decision)
            
        elif completed_checkpoints > 0:
            decision.tracking_status = "partial"
            updated = True
        
        return updated

    async def _process_completed_trade_for_learning(self, decision) -> None:
        """Process completed trade and provide feedback to learning system"""
        try:
            if not self.learning_system:
                return

            # Calculate comprehensive trade performance metrics
            trade_metrics = self._calculate_trade_metrics(decision) 
            
            # Log learning insights
            if trade_metrics:
                self._log_trade_learning_insights(decision, trade_metrics)
                
                # Trigger learning update if conditions are met
                await self._trigger_learning_update_if_needed(decision, trade_metrics)

        except Exception as e:
            log_debug(f"Learning feedback processing failed for {decision.ticker}: {e}")

    def _calculate_trade_metrics(self, decision) -> Optional[Dict[str, Any]]:
        """Calculate comprehensive trade performance metrics"""
        try:
            checkpoint_info = Config.get_checkpoint_info()
            
            # Get all checkpoint changes
            checkpoint_changes = []
            checkpoint_profits = []
            
            for i, checkpoint in enumerate(checkpoint_info):
                change_pct = decision.get_checkpoint_change_pct(i)
                if change_pct is not None:
                    checkpoint_changes.append(change_pct)
                    
                    # Calculate profit based on decision type
                    if decision.decision == 'LONG':
                        profit = change_pct
                    elif decision.decision == 'SHORT':
                        profit = -change_pct  # Inverted for SHORT
                    else:
                        profit = 0
                    
                    checkpoint_profits.append(profit)

            if len(checkpoint_changes) < 3:  # Need all checkpoints
                return None

            # Calculate key metrics
            if decision.decision in ['LONG', 'SHORT']:
                max_profit = max(checkpoint_profits)
                actual_profit = checkpoint_profits[2]  # Close price profit
                opportunity_cost = max(0, max_profit - actual_profit)
                
                # Optimal exit timing
                optimal_exit_index = checkpoint_profits.index(max_profit)
                optimal_exit_time = ['checkpoint1', 'checkpoint2', 'close'][optimal_exit_index]
                
                return {
                    'decision_type': decision.decision,
                    'checkpoint_changes': checkpoint_changes,
                    'checkpoint_profits': checkpoint_profits,
                    'max_profit': max_profit,
                    'actual_profit': actual_profit,
                    'opportunity_cost': opportunity_cost,
                    'optimal_exit_time': optimal_exit_time,
                    'was_profitable': actual_profit > 0,
                    'exit_efficiency': (actual_profit / max_profit) if max_profit > 0 else 0
                }
            else:
                # For NONE decisions, calculate missed opportunities
                max_long_opportunity = max(checkpoint_changes)
                max_short_opportunity = max([-x for x in checkpoint_changes])
                missed_opportunity = max(max_long_opportunity, max_short_opportunity)
                
                return {
                    'decision_type': 'NONE',
                    'checkpoint_changes': checkpoint_changes,
                    'max_long_opportunity': max_long_opportunity,
                    'max_short_opportunity': max_short_opportunity,
                    'missed_opportunity': missed_opportunity,
                    'should_have_traded': missed_opportunity > getattr(Config, 'MISSED_OPPORTUNITY_THRESHOLD', 1.0)
                }

        except Exception as e:
            log_debug(f"Trade metrics calculation failed: {e}")
            return None

    def _log_trade_learning_insights(self, decision, metrics: Dict[str, Any]) -> None:
        """Log learning insights from completed trade"""
        try:
            ticker = decision.ticker
            decision_type = metrics['decision_type']
            
            if decision_type in ['LONG', 'SHORT']:
                opportunity_cost = metrics['opportunity_cost']
                optimal_exit = metrics['optimal_exit_time']
                actual_profit = metrics['actual_profit']
                exit_efficiency = metrics['exit_efficiency']
                
                if getattr(Config, 'OPPORTUNITY_COST_LOGGING_ENABLED', True):
                    if opportunity_cost > 0:
                        log_info(f"💰 {ticker} {decision_type}: {opportunity_cost:.2f}% opportunity cost "
                                f"(optimal exit: {optimal_exit}, efficiency: {exit_efficiency:.1%})")
                    else:
                        log_info(f"🎯 {ticker} {decision_type}: Optimal exit achieved ({actual_profit:+.2f}%)")
                
                # Warning for high opportunity cost
                warning_threshold = getattr(Config, 'OPPORTUNITY_COST_WARNING_THRESHOLD', 3.0)
                if opportunity_cost > warning_threshold:
                    log_warning(f"⚠️ {ticker}: High opportunity cost ({opportunity_cost:.2f}%) - "
                               f"consider {optimal_exit} exit strategy")
            
            elif decision_type == 'NONE':
                missed_opportunity = metrics['missed_opportunity']
                should_have_traded = metrics['should_have_traded']
                
                if should_have_traded:
                    best_direction = 'LONG' if metrics['max_long_opportunity'] > metrics['max_short_opportunity'] else 'SHORT'
                    log_info(f"📊 {ticker} NONE: {missed_opportunity:.2f}% missed opportunity "
                            f"(should have been {best_direction})")

        except Exception as e:
            log_debug(f"Learning insights logging failed: {e}")

    async def _trigger_learning_update_if_needed(self, decision, metrics: Dict[str, Any]) -> None:
        """Trigger learning system update if conditions are met"""
        try:
            if not self.learning_system:
                return

            # Check if we should trigger learning update
            should_trigger = False
            
            # Real-time learning trigger conditions
            if getattr(Config, 'ENABLE_REAL_TIME_LEARNING_UPDATES', False):
                # Count recent completed trades
                recent_completed = self._count_recent_completed_trades()
                batch_size = getattr(Config, 'REAL_TIME_LEARNING_BATCH_SIZE', 5)
                
                if recent_completed >= batch_size:
                    should_trigger = True
                    log_info(f"🎓 Triggering real-time learning update: {recent_completed} recent trades completed")
            
            # High opportunity cost trigger
            if decision.decision in ['LONG', 'SHORT']:
                opportunity_cost = metrics['opportunity_cost']
                if opportunity_cost > getattr(Config, 'OPPORTUNITY_COST_WARNING_THRESHOLD', 3.0):
                    should_trigger = True
                    log_info(f"🎓 Triggering learning update due to high opportunity cost: {opportunity_cost:.2f}%")
            
            # Standard trigger check
            if not should_trigger and self.learning_system.check_training_trigger():
                should_trigger = True
                log_info("🎓 Standard learning trigger conditions met")
            
            if should_trigger:
                # Run learning update in background
                success = self.learning_system.run_adaptive_learning()
                if success:
                    log_info("✅ Learning system updated with new trade data")
                else:
                    log_warning("⚠️ Learning system update encountered issues")

        except Exception as e:
            log_debug(f"Learning update trigger failed: {e}")

    def _count_recent_completed_trades(self) -> int:
        """Count recently completed trades for real-time learning trigger"""
        try:
            # Simple implementation - count completed trades in pending_tracks
            # In production, you might want to check the CSV file for recent completions
            completed_count = 0
            
            for decision in self.pending_tracks.values():
                if getattr(decision, 'tracking_status', '') == 'completed':
                    completed_count += 1
            
            return completed_count
            
        except Exception as e:
            log_debug(f"Recent trades count failed: {e}")
            return 0
    
    def _calculate_change_pct(self, original_price: float, new_price: float) -> float:
        """Calculate percentage change (enhanced with learning context)"""
        if original_price and original_price > 0:
            change_pct = ((new_price - original_price) / original_price) * 100
            
            # Log significant changes for learning insights
            if abs(change_pct) > 2.0:  # Significant movement
                log_debug(f"📈 Significant price movement: {change_pct:+.2f}%")
            
            return change_pct
        return 0.0
    
    async def run_scheduler(self) -> None:
        """Main scheduler loop"""
        checkpoint_info = Config.get_checkpoint_info()
        intervals_summary = ", ".join([f"{info['short_label']}" for info in checkpoint_info])
        
        log_info(f"🕐 Realistic price tracking scheduler started with intervals: {intervals_summary}")
        
        check_count = 0
        
        while self.running:
            try:
                check_count += 1
                
                if check_count % 10 == 1:
                    pending_count = len(self.pending_tracks)
                    log_info(f"🕐 Scheduler check #{check_count}: {pending_count} positions tracked")
                    
                    if pending_count > 0:
                        for decision_id, decision in list(self.pending_tracks.items())[:5]:
                            status_parts = []
                            
                            for i, checkpoint in enumerate(checkpoint_info):
                                change_pct = decision.get_checkpoint_change_pct(i)
                                if change_pct is not None:
                                    status_parts.append(f"{checkpoint['short_label']}: {change_pct:+.1f}%")
                            
                            status = " | ".join(status_parts) if status_parts else "pending"
                            log_info(f"  📊 {decision.ticker}: {status}")
                
                await self.check_pending_tracks()
                await asyncio.sleep(Config.PRICE_TRACKER_CHECK_INTERVAL * 60)
                
            except Exception as e:
                log_error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
        
        log_info("🕐 Realistic price tracking scheduler stopped")

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning feedback from price tracking"""
        try:
            summary = {
                'learning_integration': {
                    'enabled': self.learning_system is not None,
                    'real_time_updates': getattr(Config, 'ENABLE_REAL_TIME_LEARNING_UPDATES', False),
                    'opportunity_cost_logging': getattr(Config, 'OPPORTUNITY_COST_LOGGING_ENABLED', True)
                },
                'tracking_performance': {
                    'total_tracked': len(self.pending_tracks),
                    'completed_trades': len([d for d in self.pending_tracks.values() 
                                           if getattr(d, 'tracking_status', '') == 'completed'])
                }
            }
            
            # Add opportunity cost statistics
            if self.learning_system:
                try:
                    # Get recent learning insights
                    learning_recommendations = self.learning_system.get_exit_strategy_recommendations()
                    if 'error' not in learning_recommendations:
                        summary['learning_insights'] = {
                            'total_trades_analyzed': learning_recommendations.get('total_trades_analyzed', 0),
                            'long_strategy': learning_recommendations.get('long_strategy', ('close', 0)),
                            'short_strategy': learning_recommendations.get('short_strategy', ('close', 0))
                        }
                except Exception as e:
                    summary['learning_insights'] = {'error': str(e)}
            
            return summary
            
        except Exception as e:
            return {'error': f'Learning summary failed: {e}'}
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of current tracking status"""
        checkpoint_info = Config.get_checkpoint_info()
        
        return {
            'total_pending_tracks': len(self.pending_tracks),
            'configuration': {
                'intervals': [f"{info['short_label']}" for info in checkpoint_info],
                'realistic_logic': True,
                'no_overnight_holds': True
            },
            'active_positions': [
                {
                    'ticker': decision.ticker,
                    'decision': decision.decision,
                    'entry_price': decision.recommendation_price,
                    'status': decision.tracking_status
                }
                for decision in self.pending_tracks.values()
            ]
        }
