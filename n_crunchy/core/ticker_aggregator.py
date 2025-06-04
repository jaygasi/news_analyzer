from collections import defaultdict
from datetime import datetime, timezone # Import datetime and timezone
from utils.utils import get_logger

logger = get_logger(__name__)

class TickerAggregator:
    """
    Groups news articles by ticker symbol.
    """
    def __init__(self):
        # No specific initialization needed for this class.
        # It primarily provides a stateless aggregation method.
        pass

    def aggregate_by_ticker(self, articles: list[dict]) -> dict[str, list[dict]]:
        """
        Aggregates a list of articles into a dictionary where keys are ticker symbols
        and values are lists of articles associated with that ticker.

        Articles can be associated with multiple tickers.
        """
        ticker_buckets = defaultdict(list)
        # The 'processed_article_ids' set was removed as deduplication is handled
        # upstream in NewsFetcher and actual tracking in ArticleTracker.

        for article in articles:
            # The 'article_unique_id' was removed as it was not used within this class.
            # Article identification for tracking is now handled by fmp_article_id and article_title_hash
            # which are added to the article dict by the NewsFetcher.

            # The 'extracted_tickers' field should be populated by NewsFetcher
            associated_tickers = article.get('extracted_tickers', [])
            
            if not associated_tickers:
                logger.debug(f"Article '{article.get('title', 'N/A')}' has no associated tickers. Skipping aggregation for this article.")
                continue

            for ticker in associated_tickers:
                # Ensure the ticker is valid before adding to a bucket
                if isinstance(ticker, str) and ticker.strip():
                    ticker_buckets[ticker.strip().upper()].append(article)
                else:
                    logger.warning(f"Invalid ticker '{ticker}' found for article '{article.get('title', 'N/A')}'. Skipping.")
        
        logger.info(f"Aggregated {len(articles)} articles into {len(ticker_buckets)} ticker buckets.")
        # Optional: Log counts per ticker
        # for ticker, articles_list in ticker_buckets.items():
        #     logger.debug(f"  {ticker}: {len(articles_list)} articles")

        return dict(ticker_buckets)