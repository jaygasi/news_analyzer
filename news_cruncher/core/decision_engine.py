"""
Decision engine that combines news and technical analysis
Python 3.13.3 compatible
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from analysis.multi_llm_analyzer import DirectionalPrediction
from analysis.technical_analyzer_simple import TechnicalSignal
from config import Config
from utils.simple_logger import log_debug, log_info


@dataclass
class TradingDecision:
    """Final trading decision with all supporting data"""
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
    
    # Metadata
    article_count: int = 0
    analysis_timestamp: str = ""


class DecisionEngine:
    """Combine news and technical analysis to make trading decisions"""
    
    def __init__(self) -> None:
        """Initialize decision engine"""
        self.min_confidence = Config.MIN_CONFIDENCE_THRESHOLD
        
        # Weights for combining signals
        self.news_weight = 0.7  # News is primary driver
        self.technical_weight = 0.3  # Technical provides confirmation
        
        # Minimum scores required
        self.min_news_confidence = 0.6
        self.min_technical_strength = 0.4
    
    def make_decision(self, ticker: str, news_prediction: Optional[DirectionalPrediction], 
                     technical_signal: Optional[TechnicalSignal], 
                     article_count: int = 0) -> TradingDecision:
        """Make trading decision combining news and technical analysis"""
        
        # Calculate individual scores
        news_score = self._calculate_news_score(news_prediction)
        technical_score = self._calculate_technical_score(technical_signal)
        
        # Combine scores
        combined_score = self._combine_scores(news_score, technical_score)
        
        # Make decision
        decision, confidence, reasoning = self._determine_final_decision(
            ticker, news_prediction, technical_signal, news_score, technical_score, combined_score
        )
        
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
            article_count=article_count,
            analysis_timestamp=self._get_timestamp()
        )
    
    def _calculate_news_score(self, news_prediction: Optional[DirectionalPrediction]) -> float:
        """Calculate normalized news score (-1.0 to 1.0)"""
        if not news_prediction:
            return 0.0
        
        # Convert direction to numeric score
        direction_multiplier = {
            'BUY': 1.0,
            'SELL': -1.0,
            'NEUTRAL': 0.0
        }.get(news_prediction.direction, 0.0)
        
        # Scale by confidence
        score = direction_multiplier * news_prediction.confidence
        
        return max(-1.0, min(1.0, score))
    
    def _calculate_technical_score(self, technical_signal: Optional[TechnicalSignal]) -> float:
        """Calculate normalized technical score (-1.0 to 1.0)"""
        if not technical_signal:
            return 0.0
        
        # Convert direction to numeric score
        direction_multiplier = {
            'BUY': 1.0,
            'SELL': -1.0,
            'NEUTRAL': 0.0
        }.get(technical_signal.direction, 0.0)
        
        # Scale by strength
        score = direction_multiplier * technical_signal.strength
        
        return max(-1.0, min(1.0, score))
    
    def _combine_scores(self, news_score: float, technical_score: float) -> float:
        """Combine news and technical scores with weights"""
        combined = (news_score * self.news_weight) + (technical_score * self.technical_weight)
        return max(-1.0, min(1.0, combined))
    
    def _determine_final_decision(self, ticker: str, news_prediction: Optional[DirectionalPrediction],
                                technical_signal: Optional[TechnicalSignal], news_score: float,
                                technical_score: float, combined_score: float) -> tuple[str, float, str]:
        """Determine final trading decision"""
        
        # Check if we have minimum required data
        if not news_prediction:
            return 'NONE', 0.0, "No news analysis available"
        
        # Check minimum confidence thresholds
        if news_prediction.confidence < self.min_news_confidence:
            return 'NONE', news_prediction.confidence, f"News confidence too low: {news_prediction.confidence:.2f}"
        
        # Check for conflicting signals
        if technical_signal and self._signals_conflict(news_prediction, technical_signal):
            conflict_confidence = abs(combined_score) * 0.7  # Reduce confidence for conflicts
            if conflict_confidence < self.min_confidence:
                return 'NONE', conflict_confidence, "News and technical analysis conflict"
        
        # Determine direction
        if combined_score > 0.5:
            decision = 'LONG'
            confidence = min(abs(combined_score), 1.0)
        elif combined_score < -0.5:
            decision = 'SHORT'
            confidence = min(abs(combined_score), 1.0)
        else:
            decision = 'NONE'
            confidence = 1.0 - abs(combined_score)
        
        # Final confidence check
        if confidence < self.min_confidence:
            decision = 'NONE'
            reasoning = f"Combined confidence too low: {confidence:.2f}"
        else:
            reasoning = self._create_reasoning(news_prediction, technical_signal, combined_score)
        
        log_debug(f"{ticker}: {decision} (conf: {confidence:.2f}, news: {news_score:.2f}, tech: {technical_score:.2f})")
        
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
        
        return (news_direction, tech_direction) in conflicting_pairs
    
    def _create_reasoning(self, news_prediction: Optional[DirectionalPrediction],
                         technical_signal: Optional[TechnicalSignal], 
                         combined_score: float) -> str:
        """Create human-readable reasoning for the decision"""
        reasoning_parts = []
        
        if news_prediction:
            reasoning_parts.append(f"News: {news_prediction.direction} ({news_prediction.confidence:.2f} confidence)")
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
        """Process multiple tickers and return only high-confidence decisions"""
        decisions = []
        
        for ticker, analysis_data in ticker_analyses.items():
            news_prediction = analysis_data.get('news_prediction')
            technical_signal = analysis_data.get('technical_signal')
            article_count = analysis_data.get('article_count', 0)
            
            decision = self.make_decision(ticker, news_prediction, technical_signal, article_count)
            
            # Only keep high-confidence decisions
            if decision.decision != 'NONE' and decision.confidence >= self.min_confidence:
                decisions.append(decision)
        
        # Sort by confidence (highest first)
        decisions.sort(key=lambda x: x.confidence, reverse=True)
        
        if decisions:
            log_info(f"Generated {len(decisions)} high-confidence trading decisions")
        
        return decisions
    
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
        
        return {
            'total_decisions': total_decisions,
            'long_decisions': long_decisions,
            'short_decisions': short_decisions,
            'avg_confidence': avg_confidence,
            'avg_news_score': avg_news_score,
            'avg_technical_score': avg_technical_score,
            'min_confidence': min(d.confidence for d in decisions),
            'max_confidence': max(d.confidence for d in decisions)
        }