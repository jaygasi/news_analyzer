import re
import concurrent.futures
from datetime import datetime, timedelta, timezone # Import timezone
# FIX: Changed from `from ..data_loaders.base_fmp_loader` to `from .base_fmp_loader`
from .base_fmp_loader import BaseFMPLoader 
from database.article_tracker import ArticleTracker
from config import NEWS_AGE_LIMIT_DAYS, FMP_ENDPOINTS, MAX_CONCURRENT_API_CALLS
from utils.utils import get_logger, extract_timestamp_from_fmp_news, generate_article_hash, normalize_ticker, load_ner_model, clean_text_for_analysis
from utils.exceptions import TickerResolutionError, DataProcessingError

logger = get_logger(__name__)

class NewsFetcher:
    """
    Fetches financial news from FMP APIs, handles deduplication,
    and extracts ticker symbols using NER.
    """
    def __init__(self):
        self.fmp_loader = BaseFMPLoader()
        self.article_tracker = ArticleTracker()
        self.nlp = load_ner_model() # Load SpaCy model once
        self.company_name_to_ticker = self._load_company_name_to_ticker_mapping() # Cache mapping

    def _load_company_name_to_ticker_mapping(self) -> dict[str, str]:
        """Loads and caches a mapping from company names to ticker symbols from FMP."""
        logger.info("Loading supported tickers and company names from FMP for NER mapping...")
        stock_list = self.fmp_loader.get_stock_list()
        
        mapping = {}
        if stock_list:
            for stock in stock_list:
                symbol = stock.get('symbol')
                company_name = stock.get('name')
                if symbol and company_name:
                    normalized_symbol = normalize_ticker(symbol)
                    # Store multiple variations: full name, common abbreviations if known
                    mapping[clean_text_for_analysis(company_name)] = normalized_symbol
                    # Add common variants if needed, e.g., "Apple Inc." -> "Apple"
                    if "inc." in company_name.lower():
                        mapping[clean_text_for_analysis(company_name.lower().replace("inc.", "").strip())] = normalized_symbol
            logger.info(f"Cached {len(mapping)} company name to ticker mappings.")
        else:
            logger.warning("Failed to load supported tickers from FMP. Company name resolution may be limited.")
        return mapping

    def _filter_and_deduplicate_articles(self, articles: list[dict], last_processed_timestamp: datetime | None) -> list[dict]:
        """
        Filters out articles older than the last processed timestamp (if any)
        and deduplicates using the article tracker.
        """
        filtered_articles = []
        new_article_count = 0
        total_articles = len(articles)

        for article in articles:
            # FMP doesn't always provide unique 'id' across all news types,
            # so we use a robust hash for deduplication.
            url = article.get('url', '')
            title = article.get('title', '')
            
            # Generate a robust hash for deduplication using title + url
            article_hash = generate_article_hash(title, url)
            
            # Use timezone.utc for consistency
            current_datetime_str = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
            article_id_fallback = f"{article.get('symbol', 'N/A')}_{current_datetime_str}"
            
            # Check if already processed using title_hash primarily, fallback to FMP-derived ID
            if self.article_tracker.is_article_processed(article_id=article_id_fallback, title_hash=article_hash):
                logger.debug(f"Skipping already processed article (hash: {article_hash}): {title}")
                continue

            # Check timestamp against last processed (if available)
            article_timestamp = extract_timestamp_from_fmp_news(article)
            if article_timestamp:
                if last_processed_timestamp and article_timestamp <= last_processed_timestamp:
                    logger.debug(f"Skipping old article (published: {article_timestamp}): {title}")
                    continue
                article['parsed_timestamp'] = article_timestamp # Add parsed timestamp for later use
            else:
                logger.warning(f"Could not extract timestamp for article: {title}. Relying solely on hash deduplication.")
                article['parsed_timestamp'] = datetime.now(timezone.utc) # Assign current time as fallback (S6903 fix)

            # Add fallback ID and robust hash to article for later tracking
            article['fmp_article_id'] = article_id_fallback
            article['article_title_hash'] = article_hash
            
            filtered_articles.append(article)
            new_article_count += 1
        
        logger.info(f"Fetched {total_articles} articles. Identified {new_article_count} new/unprocessed articles.")
        return filtered_articles

    def _get_tickers_from_fmp_fields(self, article: dict) -> set[str]:
        """Extracts tickers from FMP's 'symbol' and 'tickers' fields."""
        tickers = set()
        
        fmp_symbol = article.get('symbol')
        if fmp_symbol:
            normalized_symbol = normalize_ticker(fmp_symbol)
            if normalized_symbol in self.company_name_to_ticker.values():
                tickers.add(normalized_symbol)
            else:
                logger.debug(f"FMP symbol '{fmp_symbol}' not found in supported tickers list. Skipping direct add.")

        explicit_tickers_str = article.get('tickers')
        if explicit_tickers_str and isinstance(explicit_tickers_str, str):
            for t in explicit_tickers_str.split(','):
                normalized_t = normalize_ticker(t)
                if normalized_t in self.company_name_to_ticker.values():
                    tickers.add(normalized_t)
        return tickers

    def _get_tickers_from_ner(self, article: dict) -> set[str]:
        """Extracts tickers using Named Entity Recognition (NER) on article content."""
        tickers = set()
        content = article.get('title', '') + " " + (article.get('content') or article.get('text', ''))
        
        # Only process if content is substantial to avoid empty doc parsing
        if content.strip():
            doc = self.nlp(content)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    cleaned_entity = clean_text_for_analysis(ent.text)
                    mapped_ticker = self.company_name_to_ticker.get(cleaned_entity)
                    if mapped_ticker:
                        tickers.add(mapped_ticker)
        return tickers

    def _extract_tickers_from_article(self, article: dict) -> list[str]:
        """
        Orchestrates ticker extraction from an FMP news article using both FMP fields
        and Named Entity Recognition (NER).
        """
        all_tickers = set()
        
        # Get tickers from FMP's explicit fields
        all_tickers.update(self._get_tickers_from_fmp_fields(article))

        # Get tickers from NER
        all_tickers.update(self._get_tickers_from_ner(article))

        # Ensure that only valid tickers are returned (redundant check but safe)
        valid_tickers = [t for t in list(all_tickers) if t in self.company_name_to_ticker.values()]
        return valid_tickers

    def _fetch_endpoint_articles(self, endpoint_key: str, start_date: datetime, last_processed_timestamp: datetime | None) -> list[dict]:
        """Helper to fetch articles for a single endpoint and apply deduplication."""
        endpoint_url = FMP_ENDPOINTS.get(endpoint_key)
        if not endpoint_url:
            logger.warning(f"Unknown FMP news endpoint key: {endpoint_key}. Skipping.")
            return []

        limit_param = 100 # Default limit for fetching
        
        # Prepare date parameters if the endpoint supports them
        from_date_str = start_date.strftime('%Y-%m-%d')
        to_date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d') # S6903 fix

        raw_articles = []
        try:
            # Explicitly pass parameters to get_news to avoid Pylance type inference issues
            if endpoint_key == "stock_news":
                raw_articles = self.fmp_loader.get_news(
                    endpoint_key, 
                    limit=limit_param, 
                    from_date=from_date_str, 
                    to_date=to_date_str
                )
            else:
                # Other endpoints might not support date ranges in the same way, rely on limit and local filtering
                raw_articles = self.fmp_loader.get_news(
                    endpoint_key, 
                    limit=limit_param
                )

            if raw_articles:
                logger.info(f"Received {len(raw_articles)} articles from {endpoint_key}.")
                new_articles_for_endpoint = self._filter_and_deduplicate_articles(raw_articles, last_processed_timestamp)
                
                for article in new_articles_for_endpoint:
                    article['source_endpoint'] = endpoint_key
                    article['extracted_tickers'] = self._extract_tickers_from_article(article)
                return new_articles_for_endpoint
            else:
                logger.warning(f"No articles fetched from {endpoint_key} or API returned empty data.")
        except Exception as e:
            logger.error(f"Error fetching articles from {endpoint_key}: {e}", exc_info=True)
        return []

    def fetch_latest_news(self) -> list[dict]:
        """
        Fetches latest financial news from various FMP endpoints concurrently,
        filters out already processed articles, and adds ticker information.
        """
        all_new_articles = []
        last_processed_timestamp = self.article_tracker.get_last_processed_timestamp()
        
        if not last_processed_timestamp:
            start_date = datetime.now(timezone.utc) - timedelta(days=NEWS_AGE_LIMIT_DAYS) # S6903 fix
            logger.info(f"No last processed timestamp found. Fetching news since {start_date.isoformat()}.")
        else:
            # Fetch slightly older to ensure no gaps, e.g., if news was published right at the boundary
            start_date = last_processed_timestamp - timedelta(hours=1) 
            logger.info(f"Last processed timestamp: {last_processed_timestamp.isoformat()}. Fetching news since {start_date.isoformat()}.")

        news_endpoints_to_fetch = [
            "stock_news",
            "press_releases",
            "earnings_call_transcript",
            "analyst_estimates"
        ]

        # Use ThreadPoolExecutor for concurrent API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_API_CALLS) as executor:
            future_to_endpoint = {
                executor.submit(self._fetch_endpoint_articles, endpoint_key, start_date, last_processed_timestamp): endpoint_key
                for endpoint_key in news_endpoints_to_fetch
            }
            for future in concurrent.futures.as_completed(future_to_endpoint):
                endpoint_key = future_to_endpoint[future]
                try:
                    articles_from_endpoint = future.result()
                    if articles_from_endpoint: # Ensure results are not None or empty list from helper
                        all_new_articles.extend(articles_from_endpoint)
                except Exception as exc:
                    logger.error(f'Fetching from {endpoint_key} generated an exception: {exc}', exc_info=True)
        
        logger.info(f"Total new unique articles fetched across all sources: {len(all_new_articles)}")
        return all_new_articles