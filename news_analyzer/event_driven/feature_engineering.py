# --- feature_engineering.py ---
from typing import Dict, Any, Tuple
import logging
from SHARED_CONSTANTS import NUMERICAL_FEATURE_NAMES
from config import KEYWORD_WEIGHTS, COMPLEX_KEYWORD_RULES
import re


def _extract_sentiment_features(sentiment_data: Dict[str, Any]) -> Dict[str, float]:
    """Extracts sentiment-related features."""
    finbert = sentiment_data.get("finbert_sentiment", {})
    finbert_tone = sentiment_data.get("finbert_tone_sentiment", {})
    return {
        "finbert_score": finbert.get("score", 0.5)
        * (1 if finbert.get("label") == "positive" else -1),
        "finbert_tone_score": finbert_tone.get("score", 0.5)
        * (1 if finbert_tone.get("label") == "POSITIVE" else -1),
    }


def _extract_other_activity_features(
    news_text: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extracts features by calculating a weighted keyword score.
    This now includes simple keyword matching and complex rule-based matching.
    Returns the features dictionary and a dictionary of matched keywords and their weights.
    """
    features = {}
    matched_keywords = {}
    news_text_lower = news_text.lower()

    # 1. Simple Keyword Matching (existing logic)
    for keyword, weight in KEYWORD_WEIGHTS.items():
        if keyword in news_text_lower:
            matched_keywords[keyword] = weight

    # 2. Complex Rule-Based Matching (NEW LOGIC)
    # Split text into sentences for contextual matching.
    sentences = re.split(r'[.!?]+', news_text_lower)

    for rule_name, rule_data in COMPLEX_KEYWORD_RULES.items():
        # Skip if this rule has already been matched by a simple keyword
        if rule_name in matched_keywords:
            continue

        for sentence in sentences:
            if rule_data.get("logic") == "AND":
                # Check if at least one item from EACH set is in the sentence
                if all(any(term in sentence for term in term_set) for term_set in rule_data["sets"]):
                    matched_keywords[rule_name] = rule_data["weight"]
                    break  # Move to the next rule once this one is matched

    # 3. Calculate final score
    raw_keyword_score = sum(matched_keywords.values())

    # Normalize by article length (number of words) to avoid bias from longer articles.
    # Add a small epsilon (1e-6) to prevent division by zero for empty or very short texts.
    num_words = len(news_text.split()) + 1e-6
    normalized_keyword_score = raw_keyword_score / num_words
    features["keyword_score"] = normalized_keyword_score

    return features, matched_keywords


def _extract_fmp_features(details: Dict[str, Any]) -> Dict[str, float]:
    """Extracts fundamental and quote features directly from the FMP data payload."""
    features = {}
    
    # FMP profile data is at the top level of the 'details' dict
    features["market_cap"] = float(details.get("marketCap", 0.0))
    features["beta"] = float(details.get("beta", 0.0))

    # FMP quote data is in a sub-dictionary
    quote = details.get("fmp_quote", {})
    features["volume"] = float(quote.get("volume", 0.0))
    features["change_percent"] = float(quote.get("changesPercentage", 0.0))

    return features


def prepare_numerical_features(
    sentiment_analysis_results: Dict[str, Any],
    news_text: str,
    news_item: Dict[str, Any],
    details: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Engineers the final numerical features using only reliable data sources.
    """
    features = {}
    features.update(_extract_sentiment_features(sentiment_analysis_results))

    # Add news-specific features
    features["alpha_vantage_relevance"] = news_item.get("alpha_vantage_relevance", 0.0)
    features["alpha_vantage_sentiment"] = news_item.get("alpha_vantage_sentiment", 0.0)

    # Add keyword-based features
    other_features, matched_keywords = _extract_other_activity_features(news_text)
    features.update(other_features)

    # Add features from FMP data
    fmp_features = _extract_fmp_features(details)
    features.update(fmp_features)

    # Ensure final feature set is ordered and complete according to the master list
    final_features = {name: features.get(name, 0.0) for name in NUMERICAL_FEATURE_NAMES}
    
    return final_features, matched_keywords