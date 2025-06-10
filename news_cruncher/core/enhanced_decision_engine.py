"""
Enhanced Decision Engine with 3-way scoring: News + Earnings + Technical
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass, field
from core.decision_engine import TradingDecision, DirectionalPrediction, TechnicalSignal
from utils.simple_logger import log_info, log_debug, log_warning, log_error
from config import Config


@dataclass 
class EnhancedTradingDecision(TradingDecision):
    """Extended trading decision with earnings analysis"""
    earnings_analysis: Optional['EarningsAnalysis'] = None
    earnings_score: float = 0.0
    has_earnings_event: bool = False
    scoring_method: str = "2-way"  # "2-way" or "3-way"
    
    def get_enhanced_reasoning(self) -> str:
        """Get detailed reasoning including earnings analysis"""
        parts = [self.reasoning]
        
        if self.has_earnings_event and self.earnings_analysis:
            earnings_part = f"Earnings: {self.earnings_analysis.direction} "
            earnings_part += f"({self.earnings_analysis.overall_score:+.2f} score) | "
            earnings_part += self.earnings_analysis.get_reasoning()
            parts.append(earnings_part)
        
        return " | ".join(parts)


class EnhancedDecisionEngine:
    """Enhanced decision engine with earnings integration"""
    
    # REPLACE the entire __init__ method with this:

    def __init__(self):
        """Initialize decision engine with ALL values from config (no hardcoding!)"""
        from config import Config
        
        # FIXED: All scoring weights now come from config/environment
        self.news_weight_2way = Config.NEWS_WEIGHT_2WAY           # Was: 0.70 (hardcoded)
        self.technical_weight_2way = Config.TECHNICAL_WEIGHT_2WAY # Was: 0.30 (hardcoded)
        
        self.news_weight_3way = Config.NEWS_WEIGHT_3WAY           # Was: 0.40 (hardcoded)
        self.earnings_weight_3way = Config.EARNINGS_WEIGHT_3WAY   # Was: 0.30 (hardcoded)
        self.technical_weight_3way = Config.TECHNICAL_WEIGHT_3WAY # Was: 0.30 (hardcoded)
        
        # FIXED: All confidence thresholds now come from config/environment
        self.min_confidence = Config.MIN_CONFIDENCE_THRESHOLD     # Was: 0.6 (hardcoded)
        self.min_news_confidence = Config.MIN_NEWS_CONFIDENCE     # Was: 0.5 (hardcoded)
        self.min_earnings_confidence = Config.MIN_EARNINGS_CONFIDENCE # Was: 0.6 (hardcoded)
        # ADD these lines to read the new config values:
        self.combined_score_long_threshold = Config.COMBINED_SCORE_LONG_THRESHOLD
        self.combined_score_short_threshold = Config.COMBINED_SCORE_SHORT_THRESHOLD
        self.default_neutral_confidence = Config.DEFAULT_NEUTRAL_CONFIDENCE
        self.max_confidence_limit = Config.MAX_CONFIDENCE_LIMIT
        self.confidence_boost_factor = Config.CONFIDENCE_BOOST_FACTOR
        self.consensus_confidence_boost = Config.CONSENSUS_CONFIDENCE_BOOST
        
        # Validation: Ensure weights add up correctly
        two_way_total = self.news_weight_2way + self.technical_weight_2way
        three_way_total = self.news_weight_3way + self.earnings_weight_3way + self.technical_weight_3way
        
        if abs(two_way_total - 1.0) > 0.01:
            log_warning(f"2-way weights don't sum to 1.0: {two_way_total:.3f}")
        
        if abs(three_way_total - 1.0) > 0.01:
            log_warning(f"3-way weights don't sum to 1.0: {three_way_total:.3f}")
        
        # Enhanced logging with actual values from config
        log_info(f"Enhanced decision engine initialized (ALL values from config):")
        log_info(f"  2-way weights: news={self.news_weight_2way}, technical={self.technical_weight_2way}")
        log_info(f"  3-way weights: news={self.news_weight_3way}, earnings={self.earnings_weight_3way}, technical={self.technical_weight_3way}")
        log_info(f"  Confidence thresholds: overall={self.min_confidence}, news={self.min_news_confidence}, earnings={self.min_earnings_confidence}")
        
        # Log the new configuration values
        log_info(f"  Decision thresholds: long>{self.combined_score_long_threshold}, short<{self.combined_score_short_threshold}")
        log_info(f"  Confidence settings: default_neutral={self.default_neutral_confidence}, max_limit={self.max_confidence_limit}")
        log_info(f"  Boost factors: confidence={self.confidence_boost_factor}, consensus={self.consensus_confidence_boost}")
        
    def make_enhanced_decision(self, ticker: str, 
                             news_prediction: Optional[DirectionalPrediction],
                             earnings_analysis: Optional['EarningsAnalysis'],
                             technical_signal: Optional[TechnicalSignal],
                             article_count: int = 0) -> EnhancedTradingDecision:
        """Make trading decision with optional earnings analysis"""
        
        log_debug(f"Making enhanced decision for {ticker}: "
                 f"news={news_prediction is not None}, "
                 f"earnings={earnings_analysis is not None}, "
                 f"tech={technical_signal is not None}")
        
        # Calculate individual scores
        news_score = self._calculate_news_score(news_prediction)
        earnings_score = self._calculate_earnings_score(earnings_analysis)
        technical_score = self._calculate_technical_score(technical_signal)
        
        # Determine if we use 2-way or 3-way scoring
        has_earnings = earnings_analysis is not None and earnings_analysis.confidence >= self.min_earnings_confidence
        scoring_method = "3-way" if has_earnings else "2-way"
        
        log_debug(f"{ticker}: scores - news={news_score:.3f}, earnings={earnings_score:.3f}, "
                 f"tech={technical_score:.3f}, method={scoring_method}")
        
        # Combine scores based on method
        if has_earnings:
            combined_score = self._combine_scores_3way(news_score, earnings_score, technical_score)
        else:
            combined_score = self._combine_scores_2way(news_score, technical_score)
        
        log_debug(f"{ticker}: combined_score={combined_score:.3f}")
        
        # Make decision
        decision, confidence, reasoning = self._determine_enhanced_decision(
            ticker, news_prediction, earnings_analysis, technical_signal,
            news_score, earnings_score, technical_score, combined_score, scoring_method
        )
        
        log_debug(f"{ticker}: final_decision={decision}, confidence={confidence:.3f}")
        
        # Create enhanced decision object
        base_decision = self._create_base_decision(
            ticker, decision, confidence, reasoning, news_prediction, technical_signal,
            news_score, technical_score, combined_score, article_count
        )
        
        # Convert to enhanced decision
        enhanced_decision = EnhancedTradingDecision(
            **base_decision.__dict__,
            earnings_analysis=earnings_analysis,
            earnings_score=earnings_score,
            has_earnings_event=earnings_analysis is not None,
            scoring_method=scoring_method
        )
        
        return enhanced_decision
    
    def _calculate_news_score(self, news_prediction: Optional[DirectionalPrediction]) -> float:
        """Calculate normalized news score (-1.0 to 1.0)"""
        if not news_prediction:
            return 0.0
        
        direction_multiplier = {
            'BUY': 1.0,
            'SELL': -1.0,
            'NEUTRAL': 0.0
        }.get(news_prediction.direction, 0.0)
        
        score = direction_multiplier * news_prediction.confidence
        return max(-1.0, min(1.0, score))
    
    def _calculate_earnings_score(self, earnings_analysis: Optional['EarningsAnalysis']) -> float:
        """Calculate normalized earnings score (-1.0 to 1.0)"""
        if not earnings_analysis:
            return 0.0
        
        # Use the overall score from earnings analysis, already normalized
        return earnings_analysis.overall_score
    
    def _calculate_technical_score(self, technical_signal: Optional[TechnicalSignal]) -> float:
        """Calculate normalized technical score (-1.0 to 1.0)"""
        if not technical_signal:
            return 0.0
        
        direction_multiplier = {
            'BUY': 1.0,
            'SELL': -1.0,
            'NEUTRAL': 0.0
        }.get(technical_signal.direction, 0.0)
        
        score = direction_multiplier * technical_signal.strength
        return max(-1.0, min(1.0, score))
    
    def _combine_scores_2way(self, news_score: float, technical_score: float) -> float:
        """Combine news and technical scores (traditional method)"""
        combined = (news_score * self.news_weight_2way) + (technical_score * self.technical_weight_2way)
        return max(-1.0, min(1.0, combined))
    
    def _combine_scores_3way(self, news_score: float, earnings_score: float, technical_score: float) -> float:
        """Combine news, earnings, and technical scores"""
        combined = (
            (news_score * self.news_weight_3way) +
            (earnings_score * self.earnings_weight_3way) +
            (technical_score * self.technical_weight_3way)
        )
        return max(-1.0, min(1.0, combined))
    
    def _determine_enhanced_decision(self, ticker: str, news_prediction: Optional[DirectionalPrediction],
                                   earnings_analysis: Optional['EarningsAnalysis'],
                                   technical_signal: Optional[TechnicalSignal],
                                   news_score: float, earnings_score: float, technical_score: float,
                                   combined_score: float, scoring_method: str) -> tuple[str, float, str]:
        """Determine final trading decision with enhanced logic"""
        
        # Check minimum requirements
        if not news_prediction:
            return 'NONE', 0.0, "No news analysis available"
        
        if news_prediction.confidence < self.min_news_confidence:
            return 'NONE', news_prediction.confidence, f"News confidence too low: {news_prediction.confidence:.2f}"
        
        # For 3-way scoring, also check earnings confidence
        if scoring_method == "3-way" and earnings_analysis:
            if earnings_analysis.confidence < self.min_earnings_confidence:
                log_debug(f"{ticker}: Earnings confidence too low, falling back to 2-way scoring")
                # Recalculate with 2-way scoring
                combined_score = self._combine_scores_2way(news_score, technical_score)
                scoring_method = "2-way"
        
        # Determine direction and confidence
        # Determine direction and confidence - USING CONFIG VALUES
        if combined_score > self.combined_score_long_threshold:
            decision = 'LONG'
            base_confidence = min(self.max_confidence_limit, abs(combined_score) + self.confidence_boost_factor)
        elif combined_score < self.combined_score_short_threshold:
            decision = 'SHORT'
            base_confidence = min(self.max_confidence_limit, abs(combined_score) + self.confidence_boost_factor)
        else:
            decision = 'NONE'
            # FIXED: Use config value instead of hardcoded 0.5
            base_confidence = news_prediction.confidence if news_prediction else self.default_neutral_confidence
        # Adjust confidence based on consensus
        final_confidence = self._calculate_consensus_confidence(
            news_prediction, earnings_analysis, technical_signal, 
            base_confidence, scoring_method
        )
        
        # IMPROVED: More nuanced confidence checking using config values
        if final_confidence < self.min_confidence:
            # Provide more specific feedback
            if news_prediction and news_prediction.confidence < self.min_news_confidence:
                return 'NONE', final_confidence, f"News confidence too low: {news_prediction.confidence:.2f} (min: {self.min_news_confidence})"
            elif base_confidence < Config.NEURAL_WEAK_PREDICTION_THRESHOLD:
                return 'NONE', final_confidence, f"Base prediction too weak: {base_confidence:.2f}"
            else:
                return 'NONE', final_confidence, f"Combined confidence too low: {final_confidence:.2f} (min: {self.min_confidence})"
        
        # Build reasoning
        reasoning = self._build_enhanced_reasoning(
            news_prediction, earnings_analysis, technical_signal, 
            news_score, earnings_score, technical_score, scoring_method
        )
        if abs(final_confidence - 0.500) < Config.SUSPICIOUS_CONFIDENCE_THRESHOLD:
            log_warning(f"⚠️ {ticker}: SUSPICIOUS 0.500 confidence detected!")
            log_warning(f"   News prediction: {news_prediction.direction if news_prediction else 'None'} "
                    f"(conf: {news_prediction.confidence if news_prediction else 'N/A'})")
            log_warning(f"   Combined score: {combined_score:.3f}")
            log_warning(f"   Decision path: {decision}")
            log_warning(f"   This suggests model is defaulting instead of making real predictions")
        
        # Enhanced logging for all decisions when debug mode enabled
        if hasattr(Config, 'DECISION_DEBUG_MODE') and Config.DECISION_DEBUG_MODE:
            debug_info = [
                f"{ticker}: {decision} (conf: {final_confidence:.3f})",
                f"combined_score: {combined_score:.3f}",
                f"method: {scoring_method}"
            ]
            
            if news_prediction:
                debug_info.append(f"news: {news_prediction.direction} ({news_prediction.confidence:.3f})")
            if earnings_analysis:
                debug_info.append(f"earnings: {earnings_analysis.direction} ({earnings_analysis.confidence:.3f})")
            if technical_signal:
                debug_info.append(f"tech: {technical_signal.direction} ({technical_signal.strength:.3f})")
                
            log_debug(" | ".join(debug_info))
        
        return decision, final_confidence, reasoning
    
    def _calculate_consensus_confidence(self, news_prediction: Optional[DirectionalPrediction],
                                      earnings_analysis: Optional['EarningsAnalysis'],
                                      technical_signal: Optional[TechnicalSignal],
                                      base_confidence: float, scoring_method: str) -> float:
        """Calculate confidence based on consensus between signals"""
        
        if scoring_method == "2-way":
            # Traditional 2-way consensus
            if technical_signal and news_prediction:
                if self._signals_agree(news_prediction.direction, technical_signal.direction):
                    return min(0.95, base_confidence * 1.1)  # Boost for consensus
                else:
                    return base_confidence * 0.8  # Reduce for conflict
            return base_confidence
        
        else:  # 3-way scoring
            signals = []
            confidences = []
            
            if news_prediction:
                signals.append(news_prediction.direction)
                confidences.append(news_prediction.confidence)
            
            if earnings_analysis:
                signals.append(earnings_analysis.direction)
                confidences.append(earnings_analysis.confidence)
            
            if technical_signal:
                signals.append(technical_signal.direction)
                confidences.append(technical_signal.strength)
            
            # Count agreements
            buy_count = signals.count('BUY')
            sell_count = signals.count('SELL')
            neutral_count = signals.count('NEUTRAL')
            
            total_signals = len(signals)
            max_agreement = max(buy_count, sell_count, neutral_count)
            
            # Consensus bonus/penalty
            consensus_ratio = max_agreement / total_signals if total_signals > 0 else 0
            
            if consensus_ratio >= 0.67:  # 2/3 agreement
                return min(0.95, base_confidence * 1.15)  # Strong consensus boost
            elif consensus_ratio >= 0.5:  # Majority agreement
                return min(0.90, base_confidence * 1.05)  # Slight consensus boost
            else:  # Conflict
                return base_confidence * 0.85  # Conflict penalty
    
    def _signals_agree(self, signal1: str, signal2: str) -> bool:
        """Check if two signals agree in direction"""
        if signal1 == signal2:
            return True
        # NEUTRAL can agree with anything (no conflict)
        if signal1 == 'NEUTRAL' or signal2 == 'NEUTRAL':
            return True
        return False
    
    def _build_enhanced_reasoning(self, news_prediction: Optional[DirectionalPrediction],
                                earnings_analysis: Optional['EarningsAnalysis'],
                                technical_signal: Optional[TechnicalSignal],
                                news_score: float, earnings_score: float, technical_score: float,
                                scoring_method: str) -> str:
        """Build comprehensive reasoning string"""
        
        parts = []
        
        # News component
        if news_prediction:
            parts.append(f"News: {news_prediction.direction} ({news_prediction.confidence:.2f} confidence)")
        
        # Earnings component
        if earnings_analysis and scoring_method == "3-way":
            parts.append(f"Earnings: {earnings_analysis.direction} ({earnings_analysis.confidence:.2f} confidence)")
        
        # Technical component
        if technical_signal:
            parts.append(f"Technical: {technical_signal.direction} ({technical_signal.strength:.2f} strength)")
        
        # Scoring method
        parts.append(f"Method: {scoring_method} scoring")
        
        # Combined scores
        if scoring_method == "3-way":
            parts.append(f"Scores: news={news_score:+.2f}, earnings={earnings_score:+.2f}, tech={technical_score:+.2f}")
        else:
            parts.append(f"Scores: news={news_score:+.2f}, tech={technical_score:+.2f}")
        
        return " | ".join(parts)
    
    def _create_base_decision(self, ticker: str, decision: str, confidence: float, reasoning: str,
                            news_prediction: Optional[DirectionalPrediction],
                            technical_signal: Optional[TechnicalSignal],
                            news_score: float, technical_score: float, combined_score: float,
                            article_count: int) -> TradingDecision:
        """Create base TradingDecision object"""
        
        # Extract additional data for compatibility
        sources_used = []
        if news_prediction and hasattr(news_prediction, 'individual_predictions'):
            if news_prediction.individual_predictions:
                sources_used = [pred.source for pred in news_prediction.individual_predictions]
        
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
            article_count=article_count,
            analysis_timestamp=datetime.now(timezone.utc)
        )
    
    def batch_process_enhanced_decisions(self, ticker_analyses: Dict[str, Dict[str, Any]], 
                                       earnings_analyses: Dict[str, 'EarningsAnalysis']) -> list[EnhancedTradingDecision]:
        """Process multiple ticker analyses with earnings integration"""
        decisions = []
        
        for ticker, analysis in ticker_analyses.items():
            news_prediction = analysis.get('news_prediction')
            technical_signal = analysis.get('technical_signal')
            article_count = analysis.get('article_count', 0)
            
            # Get earnings analysis if available
            earnings_analysis = earnings_analyses.get(ticker)
            
            # Make enhanced decision
            decision = self.make_enhanced_decision(
                ticker, news_prediction, earnings_analysis, technical_signal, article_count
            )
            
            decisions.append(decision)
        
        # Log summary
        total_decisions = len(decisions)
        earnings_decisions = len([d for d in decisions if d.has_earnings_event])
        three_way_decisions = len([d for d in decisions if d.scoring_method == "3-way"])
        
        log_info(f"📊 Enhanced decision summary: {total_decisions} total, "
                f"{earnings_decisions} with earnings events, {three_way_decisions} using 3-way scoring")
        
        return decisions