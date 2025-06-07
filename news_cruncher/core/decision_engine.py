"""
Decision engine that combines news and technical analysis - Enhanced with dynamic price fields
Python 3.13.3 compatible
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from analysis.multi_llm_analyzer import DirectionalPrediction
from analysis.technical_analyzer_simple import TechnicalSignal
from config import Config
from utils.simple_logger import log_debug, log_info, log_warning


@dataclass
class TradingDecision:
    """Final trading decision with all supporting data including configurable price tracking"""
    ticker: str
    decision: str  # 'LONG', 'SHORT', 'NONE'
    confidence: float  # 0.0 to 1.0
    reasoning: str
    
    # Supporting analysis
    news_prediction: Optional[DirectionalPrediction] = None
    technical_signal: Optional[TechnicalSignal] = None
    
    # Detailed scores
    news_score: float = 0.0
    technical_score: float = 0.0
    combined_score: float = 0.0
    
    # Multi-source specific data (if available)
    sources_used: List[str] = None
    source_weights: Dict[str, float] = None
    individual_predictions: List[Dict[str, Any]] = None
    weighted_scores: Dict[str, float] = None
    
    # Metadata
    article_count: int = 0
    analysis_timestamp: str = ""
    
    # Entry price tracking
    recommendation_price: Optional[float] = None
    recommendation_timestamp: Optional[datetime] = None
    
    # Dynamic price tracking fields based on configuration
    # These will be set dynamically based on Config.PRICE_CHECK_1_MINUTES, etc.
    price_tracking_data: Dict[str, Any] = field(default_factory=dict)
    
    tracking_schedule: List[datetime] = field(default_factory=list)
    tracking_completed: bool = False
    tracking_status: str = "pending"
    
    def __post_init__(self):
        """Initialize dynamic price tracking fields based on configuration"""
        if not self.price_tracking_data:
            self.price_tracking_data = {}
            
            # Initialize dynamic price fields based on current configuration
            checkpoint_info = Config.get_checkpoint_info()
            
            for checkpoint in checkpoint_info:
                field_prefix = checkpoint['field_prefix']
                
                # Initialize price, timestamp, and change_pct for each checkpoint
                self.price_tracking_data[field_prefix] = None
                self.price_tracking_data[f"{field_prefix}_timestamp"] = None
                self.price_tracking_data[f"{field_prefix}_change_pct"] = None
    
    def get_checkpoint_price(self, checkpoint_index: int) -> Optional[float]:
        """Get price for a specific checkpoint index"""
        checkpoint_info = Config.get_checkpoint_info()
        if 0 <= checkpoint_index < len(checkpoint_info):
            field_prefix = checkpoint_info[checkpoint_index]['field_prefix']
            return self.price_tracking_data.get(field_prefix)
        return None
    
    def set_checkpoint_price(self, checkpoint_index: int, price: float, 
                           timestamp: datetime, change_pct: float) -> None:
        """Set price data for a specific checkpoint index"""
        checkpoint_info = Config.get_checkpoint_info()
        if 0 <= checkpoint_index < len(checkpoint_info):
            field_prefix = checkpoint_info[checkpoint_index]['field_prefix']
            
            self.price_tracking_data[field_prefix] = price
            self.price_tracking_data[f"{field_prefix}_timestamp"] = timestamp
            self.price_tracking_data[f"{field_prefix}_change_pct"] = change_pct
    
    def get_checkpoint_change_pct(self, checkpoint_index: int) -> Optional[float]:
        """Get percentage change for a specific checkpoint index"""
        checkpoint_info = Config.get_checkpoint_info()
        if 0 <= checkpoint_index < len(checkpoint_info):
            field_prefix = checkpoint_info[checkpoint_index]['field_prefix']
            return self.price_tracking_data.get(f"{field_prefix}_change_pct")
        return None
    
    def get_all_checkpoint_data(self) -> Dict[str, Any]:
        """Get all checkpoint data in a structured format"""
        result = {}
        checkpoint_info = Config.get_checkpoint_info()
        
        for i, checkpoint in enumerate(checkpoint_info):
            field_prefix = checkpoint['field_prefix']
            label = checkpoint['short_label']
            
            result[label] = {
                'price': self.price_tracking_data.get(field_prefix),
                'timestamp': self.price_tracking_data.get(f"{field_prefix}_timestamp"),
                'change_pct': self.price_tracking_data.get(f"{field_prefix}_change_pct"),
                'field_prefix': field_prefix
            }
        
        return result
    
    # Legacy property methods for backward compatibility with existing code
    @property
    def price_check1(self) -> Optional[float]:
        """Get first checkpoint price (dynamic based on config)"""
        return self.get_checkpoint_price(0)
    
    @property 
    def price_check2(self) -> Optional[float]:
        """Get second checkpoint price (dynamic based on config)"""
        return self.get_checkpoint_price(1)
    
    @property
    def price_close(self) -> Optional[float]:
        """Get close price"""
        return self.get_checkpoint_price(2)
    
    @property
    def price_check1_change_pct(self) -> Optional[float]:
        """Get first checkpoint change percentage"""
        return self.get_checkpoint_change_pct(0)
    
    @property
    def price_check2_change_pct(self) -> Optional[float]:
        """Get second checkpoint change percentage"""
        return self.get_checkpoint_change_pct(1)
    
    @property
    def price_close_change_pct(self) -> Optional[float]:
        """Get close price change percentage"""
        return self.get_checkpoint_change_pct(2)


class DecisionEngine:
    """Enhanced decision engine with configurable price tracking field support"""
    
    def __init__(self) -> None:
        """Initialize decision engine with more relaxed thresholds"""
        # LOWERED THRESHOLDS FOR BETTER LOGGING
        self.min_confidence = 0.6  # Lowered from 0.7
        
        # Weights for combining signals
        self.news_weight = 0.7  # News is primary driver
        self.technical_weight = 0.3  # Technical provides confirmation
        
        # Minimum scores required (also lowered)
        self.min_news_confidence = 0.5  # Lowered from 0.6
        self.min_technical_strength = 0.4
        
        # Log the current configuration for dynamic price tracking
        checkpoint_info = Config.get_checkpoint_info()
        intervals = [f"{info['short_label']}" for info in checkpoint_info]
        
        log_info(f"Decision engine initialized with relaxed thresholds: min_confidence={self.min_confidence}, min_news_confidence={self.min_news_confidence}")
        log_info(f"Price tracking intervals: {', '.join(intervals)}")
    
    def make_decision(self, ticker: str, news_prediction: Optional[DirectionalPrediction], 
                     technical_signal: Optional[TechnicalSignal], 
                     article_count: int = 0) -> TradingDecision:
        """Make trading decision combining news and technical analysis"""
        
        log_debug(f"Making decision for {ticker}: news={news_prediction is not None}, tech={technical_signal is not None}")
        
        # Calculate individual scores
        news_score = self._calculate_news_score(news_prediction)
        technical_score = self._calculate_technical_score(technical_signal)
        
        log_debug(f"{ticker}: news_score={news_score:.3f}, technical_score={technical_score:.3f}")
        
        # Combine scores
        combined_score = self._combine_scores(news_score, technical_score)
        
        log_debug(f"{ticker}: combined_score={combined_score:.3f}")
        
        # Make decision
        decision, confidence, reasoning = self._determine_final_decision(
            ticker, news_prediction, technical_signal, news_score, technical_score, combined_score
        )
        
        log_debug(f"{ticker}: final_decision={decision}, confidence={confidence:.3f}")
        
        # Extract multi-source data if available
        sources_used = []
        source_weights = {}
        individual_predictions = []
        weighted_scores = {}
        
        if news_prediction and hasattr(news_prediction, 'individual_predictions'):
            if news_prediction.individual_predictions:
                for pred in news_prediction.individual_predictions:
                    sources_used.append(pred.source)
                    individual_predictions.append({
                        'source': pred.source,
                        'direction': pred.direction,
                        'confidence': pred.confidence,
                        'reasoning': pred.reasoning[:100],
                        'raw_score': pred.raw_score
                    })
            
            if hasattr(news_prediction, 'source_weights') and news_prediction.source_weights:
                source_weights = news_prediction.source_weights
            
            if hasattr(news_prediction, 'weighted_scores') and news_prediction.weighted_scores:
                weighted_scores = news_prediction.weighted_scores
        
        return TradingDecision(
            ticker=ticker,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            news_prediction=news_prediction,
            technical_signal=technical_signal,
            news_score=news_score,
            technical_score=technical_score,
            combined_score=combined_score,
            sources_used=sources_used,
            source_weights=source_weights,
            individual_predictions=individual_predictions,
            weighted_scores=weighted_scores,
            article_count=article_count,
            analysis_timestamp=self._get_timestamp()
        )
    
    def _calculate_news_score(self, news_prediction: Optional[DirectionalPrediction]) -> float:
        """Calculate normalized news score (-1.0 to 1.0)"""
        if not news_prediction:
            log_debug("No news prediction available")
            return 0.0
        
        # Convert direction to numeric score
        direction_multiplier = {
            'BUY': 1.0,
            'SELL': -1.0,
            'NEUTRAL': 0.0
        }.get(news_prediction.direction, 0.0)
        
        # Scale by confidence
        score = direction_multiplier * news_prediction.confidence
        
        log_debug(f"News score calculation: {news_prediction.direction} * {news_prediction.confidence:.3f} = {score:.3f}")
        
        return max(-1.0, min(1.0, score))
    
    def _calculate_technical_score(self, technical_signal: Optional[TechnicalSignal]) -> float:
        """Calculate normalized technical score (-1.0 to 1.0)"""
        if not technical_signal:
            log_debug("No technical signal available")
            return 0.0
        
        # Convert direction to numeric score
        direction_multiplier = {
            'BUY': 1.0,
            'SELL': -1.0,
            'NEUTRAL': 0.0
        }.get(technical_signal.direction, 0.0)
        
        # Scale by strength
        score = direction_multiplier * technical_signal.strength
        
        log_debug(f"Technical score calculation: {technical_signal.direction} * {technical_signal.strength:.3f} = {score:.3f}")
        
        return max(-1.0, min(1.0, score))
    
    def _combine_scores(self, news_score: float, technical_score: float) -> float:
        """Combine news and technical scores with weights"""
        combined = (news_score * self.news_weight) + (technical_score * self.technical_weight)
        final_combined = max(-1.0, min(1.0, combined))
        
        log_debug(f"Score combination: ({news_score:.3f} * {self.news_weight}) + ({technical_score:.3f} * {self.technical_weight}) = {final_combined:.3f}")
        
        return final_combined
    
    def _determine_final_decision(self, ticker: str, news_prediction: Optional[DirectionalPrediction],
                                technical_signal: Optional[TechnicalSignal], news_score: float,
                                technical_score: float, combined_score: float) -> tuple[str, float, str]:
        """Determine final trading decision with extensive debugging"""
        
        # Check if we have minimum required data
        if not news_prediction:
            log_debug(f"{ticker}: No news analysis available")
            return 'NONE', 0.0, "No news analysis available"
        
        # Check minimum confidence thresholds
        log_debug(f"{ticker}: Checking news confidence {news_prediction.confidence:.3f} >= {self.min_news_confidence}")
        if news_prediction.confidence < self.min_news_confidence:
            log_warning(f"{ticker}: News confidence too low: {news_prediction.confidence:.2f} < {self.min_news_confidence}")
            return 'NONE', news_prediction.confidence, f"News confidence too low: {news_prediction.confidence:.2f}"
        
        # Check for conflicting signals
        if technical_signal and self._signals_conflict(news_prediction, technical_signal):
            conflict_confidence = abs(combined_score) * 0.7  # Reduce confidence for conflicts
            log_debug(f"{ticker}: Signals conflict, reduced confidence: {conflict_confidence:.3f}")
            if conflict_confidence < self.min_confidence:
                log_warning(f"{ticker}: Conflict confidence too low: {conflict_confidence:.3f}")
                return 'NONE', conflict_confidence, "News and technical analysis conflict"
        
        # Determine direction based on combined score (LOWERED THRESHOLDS)
        log_debug(f"{ticker}: Determining direction from combined_score: {combined_score:.3f}")
        
        if combined_score > 0.2:  # Lowered from 0.5
            decision = 'LONG'
            confidence = min(abs(combined_score), 1.0)
            log_debug(f"{ticker}: LONG decision with confidence {confidence:.3f}")
        elif combined_score < -0.2:  # Lowered from -0.5
            decision = 'SHORT'
            confidence = min(abs(combined_score), 1.0)
            log_debug(f"{ticker}: SHORT decision with confidence {confidence:.3f}")
        else:
            decision = 'NONE'
            confidence = 1.0 - abs(combined_score)
            log_debug(f"{ticker}: NEUTRAL decision with confidence {confidence:.3f}")
        
        # Boost confidence for multi-source agreement
        if hasattr(news_prediction, 'individual_predictions') and news_prediction.individual_predictions:
            source_count = len(news_prediction.individual_predictions)
            log_debug(f"{ticker}: {source_count} sources contributed to prediction")
            if source_count >= 3:  # 3+ sources agreeing
                old_confidence = confidence
                confidence = min(confidence * 1.1, 1.0)  # 10% boost
                log_debug(f"{ticker}: Multi-source boost: {old_confidence:.3f} -> {confidence:.3f}")
            elif source_count >= 5:  # 5+ sources agreeing
                old_confidence = confidence
                confidence = min(confidence * 1.2, 1.0)  # 20% boost
                log_debug(f"{ticker}: High multi-source boost: {old_confidence:.3f} -> {confidence:.3f}")
        
        # Final confidence check
        log_debug(f"{ticker}: Final confidence check: {confidence:.3f} >= {self.min_confidence}")
        if confidence < self.min_confidence:
            log_warning(f"{ticker}: Final confidence too low: {confidence:.2f} < {self.min_confidence}")
            decision = 'NONE'
            reasoning = f"Combined confidence too low: {confidence:.2f}"
        else:
            reasoning = self._create_reasoning(news_prediction, technical_signal, combined_score)
            log_info(f"✅ {ticker}: {decision} decision with confidence {confidence:.3f} - SHOULD BE LOGGED")
        
        return decision, confidence, reasoning
    
    def _signals_conflict(self, news_prediction: DirectionalPrediction, 
                         technical_signal: TechnicalSignal) -> bool:
        """Check if news and technical signals conflict"""
        news_direction = news_prediction.direction
        tech_direction = technical_signal.direction
        
        # Consider conflicting if they point in opposite directions
        conflicting_pairs = [
            ('BUY', 'SELL'),
            ('SELL', 'BUY')
        ]
        
        is_conflict = (news_direction, tech_direction) in conflicting_pairs
        log_debug(f"Signal conflict check: {news_direction} vs {tech_direction} = {is_conflict}")
        
        return is_conflict
    
    def _create_reasoning(self, news_prediction: Optional[DirectionalPrediction],
                         technical_signal: Optional[TechnicalSignal], 
                         combined_score: float) -> str:
        """Create human-readable reasoning for the decision"""
        reasoning_parts = []
        
        if news_prediction:
            reasoning_parts.append(f"News: {news_prediction.direction} ({news_prediction.confidence:.2f} confidence)")
            
            # Add multi-source details if available
            if hasattr(news_prediction, 'individual_predictions') and news_prediction.individual_predictions:
                source_count = len(news_prediction.individual_predictions)
                reasoning_parts.append(f"Sources: {source_count} services")
                
                # Show top contributing sources
                if hasattr(news_prediction, 'weighted_scores') and news_prediction.weighted_scores:
                    top_sources = sorted(news_prediction.weighted_scores.items(), 
                                       key=lambda x: abs(x[1]), reverse=True)[:3]
                    source_info = ", ".join([f"{src}({score:.2f})" for src, score in top_sources])
                    reasoning_parts.append(f"Top contributors: {source_info}")
            
            if news_prediction.reasoning:
                reasoning_parts.append(f"News reasoning: {news_prediction.reasoning[:100]}")
        
        if technical_signal:
            reasoning_parts.append(f"Technical: {technical_signal.direction} ({technical_signal.strength:.2f} strength)")
            if technical_signal.reasoning:
                reasoning_parts.append(f"Technical reasoning: {technical_signal.reasoning[:100]}")
        
        reasoning_parts.append(f"Combined score: {combined_score:.2f}")
        
        return " | ".join(reasoning_parts)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def batch_process_decisions(self, ticker_analyses: Dict[str, Dict[str, Any]]) -> List[TradingDecision]:
        """Process multiple tickers and return decisions (with debug logging)"""
        decisions = []
        high_confidence_decisions = []
        
        log_info(f"Processing {len(ticker_analyses)} ticker analyses...")
        
        for ticker, analysis_data in ticker_analyses.items():
            news_prediction = analysis_data.get('news_prediction')
            technical_signal = analysis_data.get('technical_signal')
            article_count = analysis_data.get('article_count', 0)
            
            decision = self.make_decision(ticker, news_prediction, technical_signal, article_count)
            decisions.append(decision)
            
            # Check if this should be a high-confidence decision
            if decision.decision != 'NONE' and decision.confidence >= self.min_confidence:
                high_confidence_decisions.append(decision)
                log_info(f"✅ HIGH-CONFIDENCE: {ticker} - {decision.decision} ({decision.confidence:.3f})")
            else:
                log_debug(f"❌ LOW-CONFIDENCE: {ticker} - {decision.decision} ({decision.confidence:.3f}) - Reason: {decision.reasoning[:100]}")
        
        # Sort by confidence (highest first)
        high_confidence_decisions.sort(key=lambda x: x.confidence, reverse=True)
        
        log_info(f"Decision processing complete: {len(decisions)} total, {len(high_confidence_decisions)} high-confidence")
        
        if high_confidence_decisions:
            log_info("High-confidence decisions summary:")
            for decision in high_confidence_decisions[:5]:  # Show top 5
                log_info(f"  {decision.ticker}: {decision.decision} ({decision.confidence:.3f})")
        else:
            log_warning("❌ NO HIGH-CONFIDENCE DECISIONS GENERATED!")
            log_info("All decisions summary:")
            for decision in decisions[:10]:  # Show top 10
                log_info(f"  {decision.ticker}: {decision.decision} ({decision.confidence:.3f}) - {decision.reasoning[:50]}")
        
        return high_confidence_decisions
    
    def get_decision_statistics(self, decisions: List[TradingDecision]) -> Dict[str, Any]:
        """Get statistics about trading decisions"""
        if not decisions:
            return {}
        
        total_decisions = len(decisions)
        long_decisions = sum(1 for d in decisions if d.decision == 'LONG')
        short_decisions = sum(1 for d in decisions if d.decision == 'SHORT')
        
        avg_confidence = sum(d.confidence for d in decisions) / total_decisions
        avg_news_score = sum(abs(d.news_score) for d in decisions) / total_decisions
        avg_technical_score = sum(abs(d.technical_score) for d in decisions) / total_decisions
        
        # Multi-source statistics
        total_sources_used = sum(len(d.sources_used) for d in decisions if d.sources_used)
        avg_sources_per_decision = total_sources_used / total_decisions if total_decisions > 0 else 0
        
        # Source usage frequency
        source_usage = {}
        for decision in decisions:
            if decision.sources_used:
                for source in decision.sources_used:
                    source_usage[source] = source_usage.get(source, 0) + 1
        
        return {
            'total_decisions': total_decisions,
            'long_decisions': long_decisions,
            'short_decisions': short_decisions,
            'avg_confidence': avg_confidence,
            'avg_news_score': avg_news_score,
            'avg_technical_score': avg_technical_score,
            'min_confidence': min(d.confidence for d in decisions),
            'max_confidence': max(d.confidence for d in decisions),
            'avg_sources_per_decision': avg_sources_per_decision,
            'source_usage_frequency': source_usage,
            'total_sources_used': total_sources_used
        }