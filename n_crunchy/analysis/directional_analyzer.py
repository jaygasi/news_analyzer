from typing import List, Dict, Tuple, Any # Import Any
from config import DIRECTIONAL_KEYWORDS
from utils.utils import get_logger, clean_text_for_analysis

logger = get_logger(__name__) # Initialize logger

class DirectionalAnalyzer:
    """
    Determines the directional impact (up/down) of news on a ticker
    using keyword-based analysis and integrating potential AI scores.
    """
    def __init__(self):
        self.directional_keywords = DIRECTIONAL_KEYWORDS

    def _perform_keyword_analysis(self, article: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Performs keyword-based analysis on a single article.
        Returns article score and list of matched keywords.
        """
        content = article.get('content') or article.get('text') or article.get('title', '')
        cleaned_content = clean_text_for_analysis(content)
        
        article_score = 0.0
        article_matched_keywords = []

        for category, keywords_dict in self.directional_keywords.items():
            for keyword, score_impact in keywords_dict.items():
                if keyword.lower() in cleaned_content:
                    article_score += score_impact
                    article_matched_keywords.append(f"'{keyword}' ({score_impact})")
        return article_score, article_matched_keywords

    def analyze_news_directional_impact(self, news_articles: List[Dict[str, Any]], ticker: str) -> Tuple[float, str]:
        """
        Analyzes a list of news articles for a given ticker to determine
        a combined directional impact score and a reason.

        Combines keyword matching and aggregates scores.
        """
        if not news_articles:
            return 0.0, "No relevant news articles found."

        total_score = 0.0
        all_matched_keywords = []

        for article in news_articles:
            article_score, matched_keywords = self._perform_keyword_analysis(article)
            total_score += article_score
            if matched_keywords:
                all_matched_keywords.extend(matched_keywords)

        # Aggregate reason
        reason = "No strong directional signals found."
        if all_matched_keywords:
            reason = f"Identified keywords: {', '.join(sorted(list(set(all_matched_keywords))))}."

        # Normalize total_score by the number of articles
        avg_score = total_score / len(news_articles) if news_articles else 0.0

        # Confidence for news score (simple heuristic based on average score and keyword count)
        # Add a small value to len(news_articles) to avoid division by zero if it somehow ends up 0
        news_confidence = min(1.0, abs(avg_score) * 0.5 + (len(all_matched_keywords) / (len(news_articles) * 5.0 + 0.001)) * 0.5)

        logger.info(f"Directional analysis for {ticker}: Score={avg_score:.2f}, Confidence={news_confidence:.2f}, Reason: {reason}")
        
        return avg_score, reason