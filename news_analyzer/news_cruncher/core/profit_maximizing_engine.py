"""
Profit Maximizing Engine - Core decision engine focused on profit optimization
Python 3.13.3 compatible
"""
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from config import Config
from utils.simple_logger import log_info, log_error, log_debug, log_warning


@dataclass
class ProfitPrediction:
    """Multi-horizon profit prediction with uncertainty quantification"""
    ticker: str
    
    # Profit predictions at multiple time horizons (in percentage points)
    profit_15min: float = 0.0
    profit_1h: float = 0.0  
    profit_4h: float = 0.0
    profit_eod: float = 0.0
    
    # Risk assessment
    downside_risk: float = 0.0      # Maximum expected loss
    upside_potential: float = 0.0   # Maximum expected gain
    volatility_forecast: float = 0.0 # Expected price volatility
    
    # Timing optimization
    optimal_entry_delay: int = 0     # Minutes to wait before entering
    optimal_exit_time: int = 0       # Minutes to optimal exit
    stop_loss_level: float = 0.0     # Exit if loss exceeds this %
    take_profit_level: float = 0.0   # Exit if profit exceeds this %
    
    # Confidence and uncertainty
    prediction_confidence: float = 0.0        # How sure we are (0-1)
    model_uncertainty: float = 0.0            # Model's uncertainty estimate
    similar_historical_cases: int = 0         # Number of similar past cases
    epistemic_uncertainty: float = 0.0       # Knowledge uncertainty
    aleatoric_uncertainty: float = 0.0       # Data uncertainty
    
    # Market context
    market_regime: str = "unknown"            # Current market regime
    regime_confidence: float = 0.0            # Confidence in regime detection
    
    # Supporting analysis
    news_impact_score: float = 0.0
    technical_momentum: float = 0.0
    earnings_catalyst: bool = False
    
    # Metadata
    analysis_timestamp: datetime = None
    model_version: str = "v1.0"


@dataclass  
class TradingDecision:
    """Enhanced trading decision with profit optimization"""
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD', 'WAIT'
    
    # Position sizing and risk management
    position_size: float = 0.0              # Percentage of portfolio
    kelly_fraction: float = 0.0             # Kelly criterion result
    risk_adjusted_size: float = 0.0         # Final position size after risk adjustment
    
    # Entry and exit strategy
    recommended_entry_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_hold_time: int = 0                  # Maximum minutes to hold
    
    # Profit expectations
    expected_profit: float = 0.0            # Expected profit percentage
    expected_return_1h: float = 0.0
    expected_return_4h: float = 0.0
    expected_return_eod: float = 0.0
    
    # Risk metrics
    value_at_risk: float = 0.0              # VaR at 95% confidence
    maximum_drawdown: float = 0.0           # Maximum expected drawdown
    sharpe_ratio_forecast: float = 0.0      # Expected Sharpe ratio
    
    # Decision rationale
    primary_catalyst: str = ""
    confidence_level: float = 0.0
    reasoning: str = ""
    
    # Supporting predictions
    profit_prediction: Optional[ProfitPrediction] = None
    
    # Metadata
    decision_timestamp: datetime = None
    article_count: int = 0


class ProfitMaximizingEngine:
    """Main decision engine that focuses on profit optimization rather than direction accuracy"""
    
    def __init__(self, multi_horizon_predictor, market_regime_detector, uncertainty_estimator):
        """Initialize the profit maximizing engine"""
        self.multi_horizon_predictor = multi_horizon_predictor
        self.market_regime_detector = market_regime_detector
        self.uncertainty_estimator = uncertainty_estimator
        
        # Load configuration
        self.profit_weights = Config.get_profit_weights()
        self.risk_params = Config.get_risk_parameters()
        self.trading_thresholds = Config.get_trading_thresholds()
        
        # Performance tracking
        self.decisions_made = 0
        self.profitable_decisions = 0
        self.total_expected_profit = 0.0
        
        log_info("🎯 Profit Maximizing Engine initialized")
        log_debug(f"   Profit weights: {self.profit_weights}")
        log_debug(f"   Risk parameters: {self.risk_params}")
        log_debug(f"   Trading thresholds: {self.trading_thresholds}")

    async def make_profit_decision(self, 
                                 ticker: str,
                                 profit_prediction: ProfitPrediction,
                                 uncertainty: Dict[str, Any],
                                 risk_metrics: Dict[str, Any],
                                 exit_strategy: Dict[str, Any],
                                 market_regime: Dict[str, Any],
                                 articles: List[Dict[str, Any]]) -> Optional[TradingDecision]:
        """
        Make a trading decision focused on profit maximization
        
        Returns None if no profitable opportunity is identified
        """
        try:
            log_debug(f"Making profit decision for {ticker}")
            
            # Step 1: Calculate weighted expected profit
            weighted_profit = self._calculate_weighted_expected_profit(profit_prediction)
            
            # Step 2: Apply market regime adjustments
            regime_adjusted_profit = self._apply_market_regime_adjustments(
                weighted_profit, market_regime, profit_prediction
            )
            
            # Step 3: Check if opportunity meets minimum thresholds
            if not self._meets_profit_thresholds(regime_adjusted_profit, profit_prediction, uncertainty):
                log_debug(f"❌ {ticker}: Does not meet profit thresholds")
                return None
            
            # Step 4: Calculate optimal position size using Kelly criterion
            kelly_fraction = self._calculate_kelly_fraction(profit_prediction, uncertainty)
            
            # Step 5: Apply risk management constraints
            risk_adjusted_size = self._apply_risk_constraints(kelly_fraction, risk_metrics)
            
            # Step 6: Determine entry/exit strategy
            entry_strategy = self._determine_entry_strategy(profit_prediction, market_regime)
            exit_strategy_enhanced = self._enhance_exit_strategy(exit_strategy, profit_prediction)
            
            # Step 7: Calculate risk metrics
            risk_assessment = self._calculate_risk_metrics(
                profit_prediction, risk_adjusted_size, uncertainty
            )
            
            # Step 8: Determine action (BUY/SELL based on profit direction)
            action = self._determine_optimal_action(profit_prediction, market_regime)
            
            # Step 9: Build comprehensive decision
            decision = TradingDecision(
                ticker=ticker,
                action=action,
                position_size=risk_adjusted_size,
                kelly_fraction=kelly_fraction,
                risk_adjusted_size=risk_adjusted_size,
                expected_profit=regime_adjusted_profit,
                expected_return_1h=profit_prediction.profit_1h,
                expected_return_4h=profit_prediction.profit_4h,
                expected_return_eod=profit_prediction.profit_eod,
                value_at_risk=risk_assessment['var_95'],
                maximum_drawdown=risk_assessment['max_drawdown'],
                sharpe_ratio_forecast=risk_assessment['expected_sharpe'],
                confidence_level=profit_prediction.prediction_confidence,
                primary_catalyst=self._identify_primary_catalyst(profit_prediction, articles),
                reasoning=self._generate_profit_reasoning(
                    ticker, profit_prediction, market_regime, uncertainty, risk_assessment
                ),
                profit_prediction=profit_prediction,
                decision_timestamp=datetime.now(),
                article_count=len(articles),
                max_hold_time=exit_strategy_enhanced['optimal_hold_time'],
                stop_loss_price=exit_strategy_enhanced['stop_loss_price'],
                take_profit_price=exit_strategy_enhanced['take_profit_price']
            )
            
            # Update tracking
            self.decisions_made += 1
            if regime_adjusted_profit > 0:
                self.profitable_decisions += 1
                self.total_expected_profit += regime_adjusted_profit
            
            log_info(f"🎯 {ticker}: {action} decision - Expected profit: {regime_adjusted_profit:+.2%}, "
                    f"Position: {risk_adjusted_size:.1%}, Confidence: {profit_prediction.prediction_confidence:.1%}")
            
            return decision
            
        except Exception as e:
            log_error(f"Error making profit decision for {ticker}: {e}")
            return None

    def _calculate_weighted_expected_profit(self, prediction: ProfitPrediction) -> float:
        """Calculate weighted expected profit across all time horizons"""
        weighted_profit = (
            prediction.profit_15min * self.profit_weights['15min'] +
            prediction.profit_1h * self.profit_weights['1h'] +
            prediction.profit_4h * self.profit_weights['4h'] +
            prediction.profit_eod * self.profit_weights['eod']
        )
        
        log_debug(f"Weighted profit calculation: "
                 f"15min={prediction.profit_15min:.3f}*{self.profit_weights['15min']:.2f} + "
                 f"1h={prediction.profit_1h:.3f}*{self.profit_weights['1h']:.2f} + "
                 f"4h={prediction.profit_4h:.3f}*{self.profit_weights['4h']:.2f} + "
                 f"eod={prediction.profit_eod:.3f}*{self.profit_weights['eod']:.2f} = "
                 f"{weighted_profit:.3f}")
        
        return weighted_profit

    def _apply_market_regime_adjustments(self, 
                                       base_profit: float, 
                                       market_regime: Dict[str, Any],
                                       prediction: ProfitPrediction) -> float:
        """Apply market regime-specific adjustments to expected profit"""
        regime = market_regime.get('regime', 'unknown')
        confidence = market_regime.get('confidence', 0.5)
        
        # Get regime multipliers from config
        regime_config = Config.get_market_regime_config()
        
        multiplier = 1.0
        if regime == 'bull_market':
            multiplier = regime_config['bull_market_multiplier']
        elif regime == 'bear_market':
            multiplier = regime_config['bear_market_multiplier']
        elif regime == 'high_volatility':
            # High volatility can be good for short-term trades but risky for longer holds
            if prediction.optimal_exit_time <= 60:  # Short-term trades
                multiplier = 1.1
            else:
                multiplier = 0.9
        elif regime == 'low_volatility':
            # Low volatility is generally more predictable
            multiplier = 1.05
        
        # Apply confidence weighting
        confidence_weight = confidence if confidence > 0.5 else 0.5
        adjusted_multiplier = 1.0 + (multiplier - 1.0) * confidence_weight
        
        adjusted_profit = base_profit * adjusted_multiplier
        
        log_debug(f"Market regime adjustment: {regime} (conf={confidence:.2f}) -> "
                 f"multiplier={adjusted_multiplier:.3f}, profit: {base_profit:.3f} -> {adjusted_profit:.3f}")
        
        return adjusted_profit

    def _meets_profit_thresholds(self, 
                                expected_profit: float,
                                prediction: ProfitPrediction,
                                uncertainty: Dict[str, Any]) -> bool:
        """Check if the opportunity meets minimum profit thresholds"""
        
        # Check minimum expected profit
        if abs(expected_profit) < self.trading_thresholds['min_expected_profit']:
            log_debug(f"Failed profit threshold: {abs(expected_profit):.3f} < {self.trading_thresholds['min_expected_profit']:.3f}")
            return False
        
        # Check prediction confidence
        if prediction.prediction_confidence < self.trading_thresholds['min_prediction_confidence']:
            log_debug(f"Failed confidence threshold: {prediction.prediction_confidence:.3f} < {self.trading_thresholds['min_prediction_confidence']:.3f}")
            return False
        
        # Check model uncertainty
        if prediction.model_uncertainty > self.trading_thresholds['max_model_uncertainty']:
            log_debug(f"Failed uncertainty threshold: {prediction.model_uncertainty:.3f} > {self.trading_thresholds['max_model_uncertainty']:.3f}")
            return False
        
        # Check similar historical cases
        if prediction.similar_historical_cases < self.trading_thresholds['min_historical_cases']:
            log_debug(f"Failed historical cases threshold: {prediction.similar_historical_cases} < {self.trading_thresholds['min_historical_cases']}")
            return False
        
        # Calculate expected Sharpe ratio
        if prediction.volatility_forecast > 0:
            expected_sharpe = expected_profit / prediction.volatility_forecast
            if expected_sharpe < self.trading_thresholds['min_sharpe_ratio']:
                log_debug(f"Failed Sharpe threshold: {expected_sharpe:.3f} < {self.trading_thresholds['min_sharpe_ratio']:.3f}")
                return False
        
        return True

    def _calculate_kelly_fraction(self, 
                                prediction: ProfitPrediction,
                                uncertainty: Dict[str, Any]) -> float:
        """Calculate optimal position size using Kelly criterion"""
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds (upside/downside ratio), p = win probability, q = loss probability
        
        # Estimate win probability from prediction confidence and uncertainty
        base_win_prob = prediction.prediction_confidence
        
        # Adjust for uncertainty
        uncertainty_penalty = prediction.model_uncertainty * 0.2  # Reduce confidence by uncertainty
        adjusted_win_prob = max(0.5, base_win_prob - uncertainty_penalty)
        
        # Calculate odds ratio
        if prediction.downside_risk > 0:
            odds_ratio = prediction.upside_potential / prediction.downside_risk
        else:
            odds_ratio = 2.0  # Default conservative ratio
        
        # Kelly fraction calculation
        loss_prob = 1.0 - adjusted_win_prob
        kelly_fraction = (odds_ratio * adjusted_win_prob - loss_prob) / odds_ratio
        
        # Apply Kelly multiplier for conservatism
        kelly_fraction *= Config.KELLY_MULTIPLIER
        
        # Clamp to configured limits
        kelly_fraction = max(Config.MIN_KELLY_FRACTION, 
                           min(Config.MAX_KELLY_FRACTION, kelly_fraction))
        
        log_debug(f"Kelly calculation: win_prob={adjusted_win_prob:.3f}, odds={odds_ratio:.2f}, "
                 f"kelly_raw={kelly_fraction/Config.KELLY_MULTIPLIER:.3f}, "
                 f"kelly_final={kelly_fraction:.3f}")
        
        return kelly_fraction

    def _apply_risk_constraints(self, 
                              kelly_fraction: float,
                              risk_metrics: Dict[str, Any]) -> float:
        """Apply portfolio-level risk constraints to position size"""
        
        # Start with Kelly fraction
        position_size = kelly_fraction
        
        # Apply maximum position size constraint
        position_size = min(position_size, Config.MAX_POSITION_SIZE)
        
        # Apply portfolio risk constraint
        # This would need to access current portfolio state
        current_portfolio_risk = risk_metrics.get('current_portfolio_risk', 0.0)
        additional_risk = risk_metrics.get('additional_risk', position_size * 0.1)
        
        if current_portfolio_risk + additional_risk > Config.MAX_PORTFOLIO_RISK:
            # Scale down position to stay within portfolio risk limit
            available_risk = Config.MAX_PORTFOLIO_RISK - current_portfolio_risk
            if available_risk > 0:
                risk_scale_factor = available_risk / additional_risk
                position_size = min(position_size, kelly_fraction * risk_scale_factor)
            else:
                position_size = 0.0  # No available risk capacity
        
        # Apply volatility-based scaling
        volatility_adjustment = risk_metrics.get('volatility_adjustment', 1.0)
        position_size *= volatility_adjustment
        
        log_debug(f"Risk constraints: kelly={kelly_fraction:.3f} -> "
                 f"max_pos={min(kelly_fraction, Config.MAX_POSITION_SIZE):.3f} -> "
                 f"portfolio_risk={position_size:.3f}")
        
        return max(0.0, position_size)

    def _determine_entry_strategy(self, 
                                prediction: ProfitPrediction,
                                market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal entry strategy"""
        
        entry_delay = prediction.optimal_entry_delay
        
        # Adjust entry timing based on market regime
        regime = market_regime.get('regime', 'unknown')
        if regime == 'high_volatility':
            # In high volatility, wait for better entry points
            entry_delay = max(entry_delay, 5)  # At least 5 minutes
        elif regime == 'low_volatility':
            # In low volatility, enter quickly before opportunity disappears
            entry_delay = min(entry_delay, 2)  # Max 2 minutes
        
        return {
            'entry_delay_minutes': entry_delay,
            'entry_type': 'market' if entry_delay <= 1 else 'limit',
            'entry_price_offset': 0.001 if entry_delay > 1 else 0.0  # Small offset for limit orders
        }

    def _enhance_exit_strategy(self, 
                             exit_strategy: Dict[str, Any],
                             prediction: ProfitPrediction) -> Dict[str, Any]:
        """Enhance exit strategy with profit-focused optimizations"""
        
        # Base exit timing from optimizer
        optimal_hold_time = exit_strategy.get('optimal_hold_time', prediction.optimal_exit_time)
        
        # Dynamic stop loss based on prediction
        stop_loss_pct = max(Config.DEFAULT_STOP_LOSS, prediction.downside_risk * 0.8)
        
        # Dynamic take profit based on prediction
        take_profit_pct = min(Config.DEFAULT_TAKE_PROFIT, prediction.upside_potential * 0.9)
        
        return {
            'optimal_hold_time': optimal_hold_time,
            'stop_loss_price': None,  # Will be set when entry price is known
            'take_profit_price': None,  # Will be set when entry price is known
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'trailing_stop_enabled': prediction.volatility_forecast > 0.02,  # Enable for volatile stocks
            'trailing_stop_distance': prediction.volatility_forecast * 0.5
        }

    def _calculate_risk_metrics(self, 
                              prediction: ProfitPrediction,
                              position_size: float,
                              uncertainty: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        
        # Value at Risk (95% confidence)
        volatility = prediction.volatility_forecast
        if volatility > 0:
            var_95 = position_size * volatility * 1.645  # 95th percentile
        else:
            var_95 = position_size * 0.02  # Default 2% VaR
        
        # Maximum expected drawdown
        max_drawdown = position_size * max(prediction.downside_risk, 0.01)
        
        # Expected Sharpe ratio
        if volatility > 0:
            expected_sharpe = prediction.profit_1h / volatility  # Use 1h as representative
        else:
            expected_sharpe = 0.0
        
        return {
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'expected_sharpe': expected_sharpe,
            'volatility': volatility,
            'uncertainty_score': prediction.model_uncertainty
        }

    def _determine_optimal_action(self, 
                                prediction: ProfitPrediction,
                                market_regime: Dict[str, Any]) -> str:
        """Determine optimal action based on profit prediction"""
        
        # Weighted profit across horizons
        weighted_profit = self._calculate_weighted_expected_profit(prediction)
        
        # Action thresholds (can be made configurable)
        strong_buy_threshold = 0.02   # 2% expected profit
        buy_threshold = 0.01          # 1% expected profit
        sell_threshold = -0.01        # -1% expected profit (short opportunity)
        strong_sell_threshold = -0.02 # -2% expected profit (strong short)
        
        if weighted_profit >= strong_buy_threshold:
            return 'BUY'  # Strong buy signal
        elif weighted_profit >= buy_threshold:
            return 'BUY'  # Regular buy signal
        elif weighted_profit <= strong_sell_threshold:
            return 'SELL'  # Strong sell/short signal
        elif weighted_profit <= sell_threshold:
            return 'SELL'  # Regular sell/short signal
        else:
            return 'HOLD'  # No clear profit opportunity

    def _identify_primary_catalyst(self, 
                                 prediction: ProfitPrediction,
                                 articles: List[Dict[str, Any]]) -> str:
        """Identify the primary catalyst for the trading opportunity"""
        
        catalysts = []
        
        # Check earnings catalyst
        if prediction.earnings_catalyst:
            catalysts.append("earnings_event")
        
        # Check news impact
        if prediction.news_impact_score > 0.7:
            catalysts.append("high_impact_news")
        elif prediction.news_impact_score > 0.4:
            catalysts.append("moderate_news")
        
        # Check technical momentum
        if prediction.technical_momentum > 0.6:
            catalysts.append("technical_momentum")
        
        # Check market regime
        if prediction.market_regime in ['bull_market', 'bear_market']:
            catalysts.append(f"market_regime_{prediction.market_regime}")
        
        # Default to news if no clear catalyst
        if not catalysts:
            catalysts.append("news_analysis")
        
        return catalysts[0]  # Return primary catalyst

    def _generate_profit_reasoning(self, 
                                 ticker: str,
                                 prediction: ProfitPrediction,
                                 market_regime: Dict[str, Any],
                                 uncertainty: Dict[str, Any],
                                 risk_assessment: Dict[str, Any]) -> str:
        """Generate comprehensive reasoning for the trading decision"""
        
        reasoning_parts = []
        
        # Profit expectation
        weighted_profit = self._calculate_weighted_expected_profit(prediction)
        reasoning_parts.append(f"Expected profit: {weighted_profit:+.2%}")
        
        # Time horizon breakdown
        reasoning_parts.append(
            f"Horizon breakdown: 15min={prediction.profit_15min:+.2%}, "
            f"1h={prediction.profit_1h:+.2%}, 4h={prediction.profit_4h:+.2%}, "
            f"eod={prediction.profit_eod:+.2%}"
        )
        
        # Confidence and uncertainty
        reasoning_parts.append(
            f"Confidence: {prediction.prediction_confidence:.1%} "
            f"(uncertainty: {prediction.model_uncertainty:.1%})"
        )
        
        # Risk assessment
        reasoning_parts.append(
            f"Risk: VaR={risk_assessment['var_95']:.2%}, "
            f"Sharpe={risk_assessment['expected_sharpe']:.2f}"
        )
        
        # Market context
        regime = market_regime.get('regime', 'unknown')
        reasoning_parts.append(f"Market regime: {regime}")
        
        # Similar cases
        if prediction.similar_historical_cases > 0:
            reasoning_parts.append(f"Similar historical cases: {prediction.similar_historical_cases}")
        
        return "; ".join(reasoning_parts)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the decision engine"""
        
        win_rate = (self.profitable_decisions / self.decisions_made) if self.decisions_made > 0 else 0.0
        avg_expected_profit = (self.total_expected_profit / self.profitable_decisions) if self.profitable_decisions > 0 else 0.0
        
        return {
            'decisions_made': self.decisions_made,
            'profitable_decisions': self.profitable_decisions,
            'win_rate': win_rate,
            'total_expected_profit': self.total_expected_profit,
            'avg_expected_profit': avg_expected_profit,
            'engine_version': '1.0',
            'profit_focused': True
        }

    def update_from_realized_outcome(self, 
                                   ticker: str,
                                   decision: TradingDecision,
                                   actual_profit: float,
                                   hold_time: int) -> None:
        """Update engine based on realized trading outcomes"""
        try:
            # This method would be called by the learning system
            # to provide feedback on actual vs predicted profits
            
            prediction_error = decision.expected_profit - actual_profit
            timing_error = decision.max_hold_time - hold_time
            
            log_debug(f"Realized outcome for {ticker}: "
                     f"predicted={decision.expected_profit:+.2%}, "
                     f"actual={actual_profit:+.2%}, "
                     f"error={prediction_error:+.2%}")
            
            # This feedback would be used to improve future predictions
            # Implementation would depend on the learning system integration
            
        except Exception as e:
            log_error(f"Error updating from realized outcome: {e}")
