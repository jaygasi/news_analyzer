"""
Enhanced trader with comprehensive filtering system
"""
import pandas as pd
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import csv
from pathlib import Path
from config import CONFIG
from analysis.enhanced_news_analyzer import EnhancedNewsAnalysis
from analysis.technical_analyzer import TechnicalAnalysis
from analysis.market_filters import (
    MarketFilter, NewsQualityFilter, PriceActionFilter, 
    PortfolioRiskFilter, EntryTimingOptimizer
)
from utils.simple_logger import log_info, log_error, log_warning

@dataclass
class EnhancedTrade:
    """Enhanced trade record with all filter data"""
    id: str
    symbol: str
    side: str
    entry_price: float
    position_size: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    
    # News analysis data
    news_title: str = ""
    news_confidence: float = 0.0
    combined_confidence: float = 0.0
    finbert_score: float = 0.0
    keyword_score: float = 0.0
    topic: str = ""
    
    # Technical analysis data
    technical_confidence: float = 0.0
    liquidity_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    rsi: float = 50.0
    price_trend: str = ""
    bid_ask_spread: float = 0.0
    
    # Market condition data
    market_regime: str = ""
    market_stress: float = 0.0
    time_score: float = 0.0
    entry_timing: str = ""

class EnhancedTrader:
    """Enhanced trader with comprehensive filtering"""
    
    def __init__(self, fmp_loader):
        self.fmp_loader = fmp_loader
        self.active_trades: Dict[str, EnhancedTrade] = {}
        self.trade_log_file = CONFIG.trade_log_path / CONFIG.trade_log_file
        self.trade_counter = 0
        
        # Initialize filters
        self.market_filter = MarketFilter(fmp_loader)
        self.news_filter = NewsQualityFilter()
        self.price_filter = PriceActionFilter(fmp_loader)
        self.portfolio_filter = PortfolioRiskFilter(self)
        self.timing_optimizer = EntryTimingOptimizer()
        
        self._setup_trade_log()
    
    def _setup_trade_log(self):
        """Setup comprehensive trade log CSV file"""
        if not self.trade_log_file.exists():
            with open(self.trade_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    # Trade basics
                    'trade_id', 'symbol', 'side', 'entry_price', 'position_size',
                    'entry_time', 'exit_price', 'exit_time', 'exit_reason', 'pnl',
                    
                    # News analysis
                    'news_title', 'news_confidence', 'combined_confidence',
                    'finbert_score', 'keyword_score', 'topic',
                    
                    # Technical analysis
                    'technical_confidence', 'liquidity_score', 'momentum_score',
                    'volume_score', 'rsi', 'price_trend', 'bid_ask_spread',
                    
                    # Market conditions
                    'market_regime', 'market_stress', 'time_score', 'entry_timing'
                ])
    
    def _log_trade(self, trade: EnhancedTrade):
        """Log comprehensive trade data to CSV"""
        with open(self.trade_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                # Trade basics
                trade.id, trade.symbol, trade.side, trade.entry_price,
                trade.position_size, trade.entry_time, trade.exit_price,
                trade.exit_time, trade.exit_reason, trade.pnl,
                
                # News analysis
                trade.news_title, trade.news_confidence, trade.combined_confidence,
                trade.finbert_score, trade.keyword_score, trade.topic,
                
                # Technical analysis
                trade.technical_confidence, trade.liquidity_score, trade.momentum_score,
                trade.volume_score, trade.rsi, trade.price_trend, trade.bid_ask_spread,
                
                # Market conditions
                trade.market_regime, trade.market_stress, trade.time_score, trade.entry_timing
            ])
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        self.trade_counter += 1
        return f"T{datetime.now().strftime('%Y%m%d')}_{self.trade_counter:04d}"
    
    def _validate_trade_conditions(self, analysis: EnhancedNewsAnalysis) -> bool:
        """Enhanced trade validation with technical filters"""
        tech = analysis.technical_analysis
        if tech is None:
            log_info(f"No technical data for {analysis.symbol} - skipping")
            return False
        
        # Liquidity requirements
        if tech.liquidity_score < 0.3:
            log_info(f"Low liquidity for {analysis.symbol}: {tech.liquidity_score:.2f}")
            return False
        
        # Spread requirements
        if tech.bid_ask_spread > 0.05:  # 5% max spread
            log_info(f"Wide spread for {analysis.symbol}: {tech.bid_ask_spread:.2f}")
            return False
        
        # Volume requirements
        if tech.volume_score < 0.2:  # Minimum volume activity
            log_info(f"Low volume for {analysis.symbol}: {tech.volume_score:.2f}")
            return False
        
        # Technical confidence requirement
        if tech.technical_confidence < 0.3:
            log_info(f"Low technical confidence for {analysis.symbol}: {tech.technical_confidence:.2f}")
            return False
        
        return True
    
    def pre_filter_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Apply news quality filters before analysis"""
        if news_df is None or news_df.empty:
            return news_df
        
        # Apply news quality filters
        filtered_news = self.news_filter.filter_news_quality(news_df)
        
        original_count = len(news_df)
        filtered_count = len(filtered_news) if filtered_news is not None else 0
        
        if filtered_count < original_count:
            log_info(f"News pre-filter: {original_count} → {filtered_count} articles")
        
        return filtered_news
    
    def _determine_position_size(self, analysis: EnhancedNewsAnalysis) -> float:
        """Enhanced dynamic position sizing"""
        base_size = CONFIG.position_size
        
        # Start with combined confidence
        confidence_multiplier = analysis.combined_confidence
        
        # Technical adjustments (only if technical_analysis exists)
        if analysis.technical_analysis:
            # Reduce for high volatility
            volatility_factor = 1.0 - (analysis.technical_analysis.volatility_score * 0.2)
            
            # Reduce for poor liquidity
            liquidity_factor = max(analysis.technical_analysis.liquidity_score, 0.5)
            
            # Boost for strong momentum alignment
            momentum_boost = 1.0
            if (analysis.sentiment_score > 0 and analysis.technical_analysis.momentum_score > 0.3) or \
               (analysis.sentiment_score < 0 and analysis.technical_analysis.momentum_score < -0.3):
                momentum_boost = 1.1
            
            confidence_multiplier *= volatility_factor * liquidity_factor * momentum_boost
        
        # Topic-based adjustments
        topic_multipliers = {
            'earnings': 1.1,     # Earnings often have strong moves
            'biotech': 1.2,      # Biotech can have explosive moves
            'ma': 1.15,          # M&A often has defined outcomes
            'analyst': 0.9,      # Analyst calls can be noisy
            'general': 0.95      # General news lower conviction
        }
        
        topic_mult = topic_multipliers.get(analysis.topic, 1.0)
        confidence_multiplier *= topic_mult
        
        # Position size bounds
        final_size = base_size * max(
            CONFIG.min_position_multiplier, 
            min(CONFIG.max_position_multiplier, confidence_multiplier)
        )
        
        return final_size
    
    def process_news_signals(self, analyses: List[EnhancedNewsAnalysis], current_prices: pd.DataFrame):
        """Process news signals with comprehensive filtering"""
        if current_prices is None or current_prices.empty:
            return
        
        # 1. Check market conditions
        market_conditions = self.market_filter.get_market_conditions()
        should_trade, market_reason = self.market_filter.should_trade_now(market_conditions)
        
        if not should_trade:
            log_info(f"Market conditions unfavorable: {market_reason}")
            return
        
        log_info(f"Market conditions favorable: regime={market_conditions.market_regime}, "
                f"stress={market_conditions.market_stress_level:.2f}, "
                f"time_score={market_conditions.time_of_day_score:.2f}")
        
        # 2. Apply price action filters
        candidate_symbols = [a.symbol for a in analyses]
        good_symbols = self.price_filter.filter_price_action(candidate_symbols, current_prices)
        
        # Filter analyses to only good symbols
        filtered_analyses = [a for a in analyses if a.symbol in good_symbols]
        
        if len(filtered_analyses) < len(analyses):
            log_info(f"Price action filter: {len(analyses)} → {len(filtered_analyses)} signals")
        
        trades_created = 0
        
        for analysis in filtered_analyses:
            symbol = analysis.symbol
            
            # Skip if no technical analysis available
            if analysis.technical_analysis is None:
                log_warning(f"Skipping {symbol} - no technical analysis")
                continue
            
            # 3. Check portfolio limits
            # First determine trade direction and size
            if (analysis.sentiment_score > 0.25 and 
                analysis.combined_confidence >= CONFIG.min_confidence_score):
                side = 'long'
            elif (analysis.sentiment_score < -0.25 and 
                  analysis.combined_confidence >= CONFIG.min_confidence_score):
                side = 'short'
            else:
                continue
            
            position_size = self._determine_position_size(analysis)
            
            # Check portfolio risk limits
            within_limits, risk_reason = self.portfolio_filter.check_portfolio_limits(
                symbol, side, position_size
            )
            
            if not within_limits:
                log_info(f"Portfolio limit violation for {symbol}: {risk_reason}")
                continue
            
            # 4. Get current price and validate
            price_row = current_prices[current_prices['symbol'] == symbol]
            if price_row.empty:
                continue
            
            current_price = price_row.iloc[0].get('lastSalePrice', 0)
            if current_price <= 0:
                continue
            
            # 5. Check entry timing
            should_enter, timing_reason = self.timing_optimizer.should_enter_now(
                symbol, analysis, current_price
            )
            
            if not should_enter:
                log_info(f"Entry timing not optimal for {symbol}: {timing_reason}")
                continue
            
            # 6. Final technical validation with momentum alignment
            sentiment = analysis.sentiment_score
            momentum = analysis.technical_analysis.momentum_score
            
            # Skip if sentiment and momentum strongly disagree
            if sentiment > 0.5 and momentum < -0.3:
                log_info(f"Skipping {symbol}: bullish news but bearish momentum")
                continue
            elif sentiment < -0.5 and momentum > 0.3:
                log_info(f"Skipping {symbol}: bearish news but bullish momentum")
                continue
            
            # 7. Create trade with safe attribute access
            entry_price = current_price * (1.001 if side == 'long' else 0.999)
            
            trade = EnhancedTrade(
                id=self._generate_trade_id(),
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                entry_time=datetime.now(),
                
                # News data
                news_title=analysis.title[:100],
                news_confidence=analysis.confidence,
                combined_confidence=analysis.combined_confidence,
                finbert_score=analysis.finbert_score,
                keyword_score=analysis.keyword_score,
                topic=analysis.topic,
                
                # Technical data (safe access)
                technical_confidence=analysis.technical_analysis.technical_confidence,
                liquidity_score=analysis.technical_analysis.liquidity_score,
                momentum_score=analysis.technical_analysis.momentum_score,
                volume_score=analysis.technical_analysis.volume_score,
                rsi=analysis.technical_analysis.rsi,
                price_trend=analysis.technical_analysis.price_trend,
                bid_ask_spread=analysis.technical_analysis.bid_ask_spread,
                
                # Market conditions
                market_regime=market_conditions.market_regime,
                market_stress=market_conditions.market_stress_level,
                time_score=market_conditions.time_of_day_score,
                entry_timing=timing_reason
            )
            
            # Add to active trades
            self.active_trades[symbol] = trade
            trades_created += 1
            
            # Log entry
            self._log_trade(trade)
            log_info(f"✅ TRADE ENTRY: {side} {symbol} @ ${entry_price:.2f} "
                    f"size=${position_size:.0f} | "
                    f"Combined Conf: {analysis.combined_confidence:.2f} | "
                    f"Market: {market_conditions.market_regime} | "
                    f"Timing: {timing_reason}")
        
        # Cleanup stale entries
        self.timing_optimizer.cleanup_stale_entries()
        
        if trades_created > 0:
            log_info(f"🎯 Created {trades_created} high-quality trades from {len(analyses)} signals")
        elif len(analyses) > 0:
            log_info(f"⚠️  No trades created from {len(analyses)} signals due to filters")
    
    def check_exits(self, current_prices: pd.DataFrame):
        """Enhanced exit logic with market condition considerations"""
        if current_prices is None or current_prices.empty:
            return
        
        # Check market stress for early exits
        market_conditions = self.market_filter.get_market_conditions()
        stress_exit_multiplier = 1.0
        
        if market_conditions.market_stress_level > 0.7:
            stress_exit_multiplier = 0.8  # Tighter stops in stressed markets
            log_info("High market stress - tightening stops")
        
        to_close = []
        
        for symbol, trade in self.active_trades.items():
            price_row = current_prices[current_prices['symbol'] == symbol]
            if price_row.empty:
                continue
            
            current_price = price_row.iloc[0].get('lastSalePrice', 0)
            if current_price <= 0:
                continue
            
            # Calculate P&L
            if trade.side == 'long':
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            else:
                pnl_pct = (trade.entry_price - current_price) / trade.entry_price
            
            # Dynamic stop/target adjustments
            stop_loss_pct = CONFIG.stop_loss_pct * stress_exit_multiplier
            take_profit_pct = CONFIG.take_profit_pct
            
            # Adjust based on original trade quality
            if trade.liquidity_score < 0.5:
                stop_loss_pct *= 0.8  # Tighter stops for low liquidity
                take_profit_pct *= 0.8
            
            if trade.combined_confidence > 0.9:
                take_profit_pct *= 1.2  # Let high-confidence trades run
            
            exit_reason = None
            
            # Check exit conditions
            if pnl_pct <= -stop_loss_pct:
                exit_reason = f'stop_loss_{stress_exit_multiplier:.1f}x'
            elif pnl_pct >= take_profit_pct:
                exit_reason = 'take_profit'
            elif market_conditions.market_stress_level > 0.8 and pnl_pct > 0.02:
                exit_reason = 'market_stress_profit_protect'
            
            if exit_reason:
                # Close trade
                trade.exit_price = current_price
                trade.exit_time = datetime.now()
                trade.exit_reason = exit_reason
                trade.pnl = (trade.position_size * pnl_pct)
                
                # Log exit
                self._log_trade(trade)
                to_close.append(symbol)
                
                log_info(f"💰 TRADE EXIT: {trade.side} {symbol} @ ${current_price:.2f} "
                        f"P&L: ${trade.pnl:.2f} ({exit_reason}) | "
                        f"Duration: {(trade.exit_time - trade.entry_time).total_seconds()/60:.1f}min")
        
        # Remove closed trades
        for symbol in to_close:
            del self.active_trades[symbol]
    
    def get_active_positions(self) -> List[EnhancedTrade]:
        """Get list of active trades"""
        return list(self.active_trades.values())
    
    def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        if not self.trade_log_file.exists():
            return 0.0
        
        today = datetime.now().date()
        total_pnl = 0.0
        
        try:
            df = pd.read_csv(self.trade_log_file)
            if 'exit_time' in df.columns and 'pnl' in df.columns:
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
                today_trades = df[df['exit_time'].dt.date == today]
                total_pnl = today_trades['pnl'].fillna(0).sum()
        except Exception as e:
            log_error(f"Error calculating daily P&L: {e}")
        
        return total_pnl
    
    def get_filter_performance(self) -> Dict:
        """Analyze filter effectiveness"""
        if not self.trade_log_file.exists():
            return {}
        
        try:
            df = pd.read_csv(self.trade_log_file)
            completed_trades = df[df['exit_time'].notna()]
            
            if completed_trades.empty:
                return {}
            
            # Performance by market regime
            regime_perf = completed_trades.groupby('market_regime')['pnl'].agg(['count', 'sum', 'mean'])
            
            # Performance by entry timing
            timing_perf = completed_trades.groupby('entry_timing')['pnl'].agg(['count', 'sum', 'mean'])
            
            # Performance by technical confidence buckets
            completed_trades['tech_conf_bucket'] = pd.cut(
                completed_trades['technical_confidence'], 
                bins=[0, 0.3, 0.6, 1.0], 
                labels=['low', 'medium', 'high']
            )
            tech_perf = completed_trades.groupby('tech_conf_bucket')['pnl'].agg(['count', 'sum', 'mean'])
            
            return {
                'regime_performance': regime_perf.to_dict(),
                'timing_performance': timing_perf.to_dict(),
                'technical_confidence_performance': tech_perf.to_dict(),
                'avg_market_stress': completed_trades['market_stress'].mean(),
                'avg_time_score': completed_trades['time_score'].mean()
            }
            
        except Exception as e:
            log_error(f"Error analyzing filter performance: {e}")
            return {}