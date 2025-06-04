from datetime import datetime, timezone # Import datetime and timezone
from typing import Dict, Any, List, Tuple # Import types for better hinting
from config import DECISION_CONFIDENCE_THRESHOLD
from utils.utils import get_logger

logger = get_logger(__name__)

class DecisionEngine:
    """
    Combines news-based directional analysis and technical analysis
    to make a single, high-confidence LONG/SHORT/NEUTRAL decision per ticker.
    """
    def __init__(self, confidence_threshold: float = DECISION_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def _determine_initial_signal(self, news_score: float, technical_score: float) -> Tuple[str, float, List[str]]:
        """
        Determines the initial directional signal and its confidence based on news and technical scores.
        Returns (decision, confidence, reasons).
        """
        decision = "NEUTRAL"
        confidence = 0.0
        reason_summary: List[str] = []

        if news_score > 0 and technical_score > 0:
            decision = "LONG"
            confidence = abs(news_score * 0.6 + technical_score * 0.4) # Combine scores for confidence
            reason_summary.append(f"News indicates positive impact ({news_score:.2f}).")
            reason_summary.append(f"Technical indicators are bullish ({technical_score:.2f}).")
        elif news_score < 0 and technical_score < 0:
            decision = "SHORT"
            confidence = abs(news_score * 0.6 + technical_score * 0.4)
            reason_summary.append(f"News indicates negative impact ({news_score:.2f}).")
            reason_summary.append(f"Technical indicators are bearish ({technical_score:.2f}).")
        else:
            # Mixed signals or one is neutral
            # Prioritize stronger signal if one is significant (e.g., abs > 0.1)
            combined_score = (news_score * 0.6) + (technical_score * 0.4)
            
            if abs(news_score) > abs(technical_score) and abs(news_score) > 0.1:
                # News is stronger and non-neutral
                if news_score > 0:
                    decision = "LONG" if combined_score > 0.05 else "NEUTRAL"
                    confidence = abs(combined_score)
                    reason_summary.append(f"Strong positive news ({news_score:.2f}) despite mixed technicals ({technical_score:.2f}).")
                else:
                    decision = "SHORT" if combined_score < -0.05 else "NEUTRAL"
                    confidence = abs(combined_score)
                    reason_summary.append(f"Strong negative news ({news_score:.2f}) despite mixed technicals ({technical_score:.2f}).")
            elif abs(technical_score) > abs(news_score) and abs(technical_score) > 0.1:
                # Technicals are stronger and non-neutral
                if technical_score > 0:
                    decision = "LONG" if combined_score > 0.05 else "NEUTRAL"
                    confidence = abs(combined_score)
                    reason_summary.append(f"Strong bullish technicals ({technical_score:.2f}) despite mixed news ({news_score:.2f}).")
                else:
                    decision = "SHORT" if combined_score < -0.05 else "NEUTRAL"
                    confidence = abs(combined_score)
                    reason_summary.append(f"Strong bearish technicals ({technical_score:.2f}) despite mixed news ({news_score:.2f}).")
            else:
                # No strong signal dominates, or both are weak/neutral
                decision = "NEUTRAL"
                confidence = 0.0
                reason_summary.append(f"Mixed news ({news_score:.2f}) and technical ({technical_score:.2f}) signals, resulting in neutral.")
        
        # Ensure confidence is within [0, 1] range
        confidence = min(1.0, max(0.0, confidence))
        
        return decision, confidence, reason_summary

    def make_decision(self, ticker: str, news_analysis: Dict[str, Any], technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes a single decision (LONG/SHORT/NEUTRAL) for a given ticker
        based on combined news and technical analysis.
        """
        news_score = news_analysis.get('score', 0.0)
        news_reason = news_analysis.get('reason', 'N/A') # Details for debugging
        
        tech_score = technical_analysis.get('technical_score', 0.0)
        # tech_trend = technical_analysis.get('trend_direction', 'NEUTRAL') # Removed unused variable
        tech_reason = technical_analysis.get('reason', 'N/A') # Details for debugging

        # Determine initial signal and confidence
        decision, confidence, reason_summary = self._determine_initial_signal(news_score, tech_score)

        # Filter for "solid winners" based on confidence threshold
        if confidence < self.confidence_threshold and decision != "NEUTRAL":
            original_decision = decision
            decision = "NEUTRAL"
            logger.info(f"Decision for {ticker} downgraded from {original_decision} to NEUTRAL due to low confidence ({confidence:.2f} < {self.confidence_threshold:.2f}).")
            reason_summary.append("Decision downgraded due to confidence below threshold.") # Fixed S3457

        final_reason = " ".join(reason_summary).strip()
        if not final_reason:
            final_reason = "Analysis completed, but no strong signals for action."


        decision_output = {
            "timestamp": datetime.now(timezone.utc).isoformat(), # Add timestamp here for CSV log
            "ticker": ticker,
            "decision": decision,
            "confidence": confidence,
            "news_score": news_score,
            "technical_score": tech_score,
            "reasoning": final_reason,
            "news_details": news_reason, # Keep news details for debugging
            "technical_details": tech_reason, # Keep technical details for debugging
        }

        logger.info(f"Decision for {ticker}: {decision} (Confidence: {confidence:.2f}, News: {news_score:.2f}, Tech: {tech_score:.2f})")
        return decision_output