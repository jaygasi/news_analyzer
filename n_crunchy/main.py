import time
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

# All imports from the 'crunchy' package must be absolute
from data_loaders.news_fetcher import NewsFetcher
from core.ticker_aggregator import TickerAggregator
from analysis.multi_llm_analyzer import MultiLLMAnalyzer
from analysis.directional_analyzer import DirectionalAnalyzer
from analysis.technical_analyzer_simple import TechnicalAnalyzer
from core.decision_engine import DecisionEngine
from output.csv_logger import CSVLogger
from database.article_tracker import ArticleTracker
from config import WAIT_TIME_SECONDS, NEWS_AGE_LIMIT_DAYS, DECISION_CONFIDENCE_THRESHOLD, DEFAULT_TECHNICAL_INDICATORS_SETTINGS
from utils.utils import get_logger, generate_article_hash
from utils.exceptions import MissingConfigError, DataProcessingError, FMPAPIError, LLMAPIError, TickerResolutionError

logger = get_logger(__name__)

class PaperTradingApp:
    def __init__(self, args: argparse.Namespace):
        self.news_age_limit_days = args.news_age_limit_days
        self.confidence_threshold = args.confidence_threshold
        self.wait_time_seconds = args.wait_time_seconds

        self.news_fetcher = NewsFetcher()
        self.ticker_aggregator = TickerAggregator()
        self.multi_llm_analyzer = MultiLLMAnalyzer()
        self.directional_analyzer = DirectionalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.decision_engine = DecisionEngine(confidence_threshold=self.confidence_threshold)
        self.csv_logger = CSVLogger()
        self.article_tracker = ArticleTracker()
        self.technical_analysis_settings = DEFAULT_TECHNICAL_INDICATORS_SETTINGS

    def _process_ticker_news_analysis(self, ticker: str, articles_for_ticker: List[Dict[str, Any]]) -> Dict[str, Any]:
        llm_analysis_scores: List[float] = []
        llm_analysis_reasons: List[str] = []

        for article in articles_for_ticker:
            llm_score, llm_reason, llm_model = self.multi_llm_analyzer.analyze_news_with_llms(article, ticker)
            llm_analysis_scores.append(llm_score)
            llm_analysis_reasons.append(f"({llm_model}) {llm_reason}")
        
        avg_llm_score = sum(llm_analysis_scores) / len(llm_analysis_scores) if llm_analysis_scores else 0.0
        combined_llm_reason = " | ".join(llm_analysis_reasons)

        news_score_keyword_based, news_reason_keyword_based = self.directional_analyzer.analyze_news_directional_impact(articles_for_ticker, ticker)
        
        final_news_score = (news_score_keyword_based * 0.7) + (avg_llm_score * 0.3)
        final_news_reason = f"Keyword: {news_reason_keyword_based}. AI: {combined_llm_reason}"
        
        return {
            "score": final_news_score,
            "reason": final_news_reason
        }

    def _track_processed_articles_for_ticker(self, ticker: str, articles_for_ticker: List[Dict[str, Any]], processed_article_hashes_in_cycle: Set[str]):
        for article in articles_for_ticker:
            article_id = article.get('fmp_article_id')
            url = article.get('url', '')
            title = article.get('title', '')
            title_hash = article.get('article_title_hash')
            article_timestamp = article.get('parsed_timestamp', datetime.now(timezone.utc))

            if not isinstance(article_id, str):
                article_id = f"unknown_id_{title_hash if isinstance(title_hash, str) else hash(f'{title}{url}')}"

            if not isinstance(title_hash, str):
                title_hash = generate_article_hash(title, url)

            if title_hash and title_hash not in processed_article_hashes_in_cycle:
                try:
                    self.article_tracker.add_processed_article(
                        article_id=article_id,
                        url=url,
                        title=title,
                        title_hash=title_hash,
                        ticker=ticker,
                        article_timestamp=article_timestamp
                    )
                    processed_article_hashes_in_cycle.add(title_hash)
                except DataProcessingError as e:
                    logger.error(f"Failed to track article {title_hash}: {e}")
            else:
                logger.debug(f"Article {title_hash} for {ticker} already tracked in this cycle or previous run. Skipping add.")

    def run_single_cycle(self):
        logger.info("--- Starting new analysis cycle ---")
        
        try:
            logger.info("Step 1: Fetching latest news...")
            raw_new_articles = self.news_fetcher.fetch_latest_news()
            
            if not raw_new_articles:
                logger.info("No new articles to process in this cycle.")
                return

            logger.info("Step 2: Grouping articles by ticker...")
            ticker_to_articles = self.ticker_aggregator.aggregate_by_ticker(raw_new_articles)
            
            if not ticker_to_articles:
                logger.info("No tickers found in the fetched articles. Exiting cycle.")
                return

            processed_article_hashes_in_cycle: Set[str] = set()

            logger.info(f"Step 3: Analyzing {len(ticker_to_articles)} tickers and making decisions...")
            for ticker, articles_for_ticker in ticker_to_articles.items():
                logger.info(f"Processing ticker: {ticker} with {len(articles_for_ticker)} articles.")
                
                news_analysis_output = self._process_ticker_news_analysis(ticker, articles_for_ticker)

                technical_analysis_output = self.technical_analyzer.analyze(ticker, settings=self.technical_analysis_settings)

                decision_output = self.decision_engine.make_decision(
                    ticker, news_analysis_output, technical_analysis_output
                )

                # FIX: Only log decisions that are not 'NEUTRAL'
                if decision_output['decision'] != 'NEUTRAL':
                    self.csv_logger.log_decision(decision_output)
                else:
                    logger.info(f"Skipping CSV log for NEUTRAL decision for {ticker}.")

                self._track_processed_articles_for_ticker(ticker, articles_for_ticker, processed_article_hashes_in_cycle)
            
            logger.info(f"Finished cycle. Processed {len(processed_article_hashes_in_cycle)} unique articles across all tickers.")

        except (MissingConfigError, FMPAPIError, LLMAPIError, TickerResolutionError, DataProcessingError) as e:
            logger.critical(f"A critical error occurred during the analysis cycle: {e}. Skipping to next cycle.", exc_info=True)
        except Exception as e:
            logger.critical(f"An unexpected error occurred during the analysis cycle: {e}", exc_info=True)
        finally:
            logger.info("--- Analysis cycle completed ---")

    def run_continously(self):
        logger.info(f"Application starting in continuous mode. Will run every {self.wait_time_seconds} seconds.")
        while True:
            self.run_single_cycle()
            logger.info(f"Waiting for {self.wait_time_seconds} seconds before next cycle...")
            time.sleep(self.wait_time_seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a paper trading news analysis application.")
    parser.add_argument(
        "--news-age-limit-days",
        type=int,
        default=NEWS_AGE_LIMIT_DAYS,
        help=f"Number of days back to fetch news if no last processed timestamp (default: {NEWS_AGE_LIMIT_DAYS})."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DECISION_CONFIDENCE_THRESHOLD,
        help=f"Minimum confidence for a 'solid winner' decision (default: {DECISION_CONFIDENCE_THRESHOLD})."
    )
    parser.add_argument(
        "--wait-time-seconds",
        type=int,
        default=WAIT_TIME_SECONDS,
        help=f"Time to wait between analysis cycles in seconds (default: {WAIT_TIME_SECONDS})."
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run only a single analysis cycle and then exit."
    )

    args = parser.parse_args()

    app = PaperTradingApp(args)
    
    if args.run_once:
        app.run_single_cycle()
    else:
        try:
            app.run_continously()
        except KeyboardInterrupt:
            logger.info("Application stopped by user (KeyboardInterrupt).")
        except Exception as e:
            logger.critical(f"Application encountered an unhandled error: {e}", exc_info=True)