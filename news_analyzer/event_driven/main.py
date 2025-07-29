import asyncio
import logging
import argparse
import signal
from datetime import datetime, timezone
import json
import functools
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from config import (
    NEWS_FETCH_LIMIT,
    MIN_MARKET_CAP_USD,
    MIN_AVG_DAILY_VOLUME,
    ALLOWED_EXCHANGES,
    MAX_TICKER_FILTERING_LOGS,
)
from data_loader import FMPDataLoader, TickerFilterEngine
from utils import generate_news_hash
from sentiment_analyzer import SentimentAnalyzer
from ml_models import ProfitMaximizingModel
from feature_engineering import prepare_numerical_features
import database
from logger_config import setup_logging

# --- Global state for interruption handling ---
pause_event = asyncio.Event()


def signal_handler():
    """Sets the pause event when a signal (like Ctrl+C) is received."""
    print("\n--- Interruption Signal Received ---")
    print("Pausing after the current long-running operation. Please wait...")
    pause_event.set()


class InterruptedByUserError(Exception):
    """Custom exception for graceful shutdown from the pause menu."""
    pass


async def check_for_pause():
    """An awaitable that checks if a pause is requested and handles user interaction."""
    if pause_event.is_set():
        pause_event.clear()
        print("\n" + "=" * 50 + "\n⏸️  EXECUTION PAUSED\n" + "=" * 50)
        choice = await asyncio.to_thread(
            input, "Type 'c' to continue or 'e' to exit: "
        )
        if choice == "e":
            raise InterruptedByUserError("Exit requested by user from pause menu.")
        print("...Resuming execution.")

def _get_ticker_from_news_item(item: Dict[str, Any]) -> Optional[str]:
    """Extracts the ticker symbol from a raw news item."""
    return (item.get("symbol") or "").upper()


async def initialize_system() -> Dict[str, Any]:
    """Asynchronously initializes all system components."""
    database.init_db()
    logging.info(f"Configuration: NEWS_FETCH_LIMIT is set to {NEWS_FETCH_LIMIT}")

    sentiment_analyzer = SentimentAnalyzer()
    profit_model = ProfitMaximizingModel()

    init_tasks = [sentiment_analyzer.initialize(), profit_model.initialize()]
    await asyncio.gather(*init_tasks)

    components = {
        "fmp_loader": FMPDataLoader(),
        "sentiment_analyzer": sentiment_analyzer,
        "profit_model": profit_model,
        "processed_cache": database.load_cache_set("processed_news", "id"),
        "failed_tickers_cache": database.load_cache_set("failed_tickers", "ticker"),
        "fundamental_filter_log_tracker": {"count": 0, "lock": asyncio.Lock()},
        "ticker_filter": TickerFilterEngine(),
        "ticker_to_name_map": {},
    }
    if not profit_model.is_trained:
        logging.warning("Model not found. Please run train_model.py with backfilled data.")
    return components


def get_prediction_for_item(
    item: Dict[str, Any],
    components: Dict[str, Any],
    ticker_details: Dict[str, Any],
    sentiment_analysis_results: dict,
) -> Optional[Tuple]:
    """Generates a prediction for a single news item."""
    news_text = (item.get("text", "") + " " + item.get("title", "")).strip()
    if not news_text:
        return None

    try:
        numerical_features, matched_keywords = prepare_numerical_features(
            sentiment_analysis_results, news_text, item, ticker_details
        )
        prediction = components["profit_model"].predict(news_text, numerical_features)
        return prediction, news_text, numerical_features, matched_keywords
    except Exception as e:
        logging.error(f"Error during prediction for item '{item.get('title', 'N/A')[:70]}...': {e}", exc_info=False)
        return None


def aggregate_predictions(predictions: List[tuple]) -> Optional[tuple]:
    """Selects the best prediction from a list based on confidence."""
    if not predictions:
        return None
    return max(predictions, key=lambda p: p[0].get("predicted_confidence", 0))


async def _get_and_group_news_by_ticker(
    loader: FMPDataLoader, components: Dict[str, Any]
) -> Dict[str, List[Dict]]:
    """Fetches all raw news and groups them by ticker."""
    all_raw_items = await loader.get_all_activity(
        ticker_filter=components["ticker_filter"],
        processed_cache=components["processed_cache"],
    )
    logging.info(f"Fetched {len(all_raw_items)} new, unprocessed news items.")

    items_by_ticker = defaultdict(list)
    for item in all_raw_items:
        corrected_ticker = _get_ticker_from_news_item(item)
        if corrected_ticker:
            items_by_ticker[corrected_ticker].append(item)

    logging.info(f"Found {len(items_by_ticker)} unique tickers with associated news.")
    return items_by_ticker


def _check_single_ticker_fundamentals(ticker: str, profile: Dict[str, Any]) -> List[str]:
    """Checks a single ticker's profile against fundamental criteria."""
    reasons = []
    if profile.get("marketCap", 0) < MIN_MARKET_CAP_USD:
        reasons.append("Market Cap")
    if profile.get("averageVolume", 0) < MIN_AVG_DAILY_VOLUME:
        reasons.append("Avg Volume")
    
    allowed_exchanges_lower = {exchange.lower() for exchange in ALLOWED_EXCHANGES}
    exchange_short = (profile.get("exchangeShortName") or "").lower()
    exchange_full = (profile.get("exchange") or "").lower()
    if not (exchange_short in allowed_exchanges_lower or exchange_full in allowed_exchanges_lower):
        reasons.append("Exchange")
    return reasons


async def _apply_fundamental_filters(
    unique_tickers: List[str],
    all_profiles: Dict[str, Dict],
    components: Dict[str, Any],
) -> Dict[str, Dict]:
    """Applies fundamental filtering logic to a dictionary of fetched profiles."""
    passed_tickers = {}
    log_tracker = components["fundamental_filter_log_tracker"]
    failed_cache = components["failed_tickers_cache"]

    for ticker in unique_tickers:
        if database.is_failed_ticker(ticker, failed_cache):
            continue

        profile = all_profiles.get(ticker)
        if not profile:
            database.add_failed_ticker(ticker)
            continue

        reasons = _check_single_ticker_fundamentals(ticker, profile)

        if reasons:
            async with log_tracker["lock"]:
                if log_tracker["count"] < MAX_TICKER_FILTERING_LOGS:
                    logging.info(f"Filtering out {ticker} due to: {'; '.join(reasons)}")
                    log_tracker["count"] += 1
        else:
            passed_tickers[ticker] = profile

    logging.info(f"{len(passed_tickers)} tickers passed fundamental checks.")
    return passed_tickers


async def _fetch_and_prepare_ticker_data(
    loader: FMPDataLoader, components: Dict[str, Any]
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """Orchestrates the data fetching and preparation pipeline."""
    items_by_ticker = await _get_and_group_news_by_ticker(loader, components)
    if not items_by_ticker:
        return {}, {}

    unique_tickers = list(items_by_ticker.keys())
    logging.info(f"Performing fundamental checks for {len(unique_tickers)} tickers...")
    
    # Fetch all company profiles in a single, efficient batch request instead of one by one.
    # This is much faster and avoids hitting API rate limits.
    all_profiles = await loader.get_batch_profiles(unique_tickers)
    logging.info(f"Successfully fetched {len(all_profiles)} profiles from FMP.")

    passed_tickers = await _apply_fundamental_filters(unique_tickers, all_profiles, components)
    if not passed_tickers:
        return {}, {}

    quotes = await loader.get_batch_quotes(list(passed_tickers.keys()))
    details_cache = {
        ticker: {**details, "fmp_quote": quotes.get(ticker, {})}
        for ticker, details in passed_tickers.items()
        if quotes.get(ticker, {}).get("price")
    }
    
    logging.info(f"Fetched quotes and combined details for {len(details_cache)} tickers.")
    final_items_by_ticker = {t: i for t, i in items_by_ticker.items() if t in details_cache}
    return final_items_by_ticker, details_cache


async def _run_batch_sentiment_analysis(
    items_by_ticker: Dict[str, List[Dict]],
    details_cache: Dict[str, Dict],
    components: Dict[str, Any],
) -> Dict[str, Dict]:
    """Collects all texts and runs a single batch sentiment analysis."""
    texts_for_sentiment = []
    metadata_for_sentiment = []
    for ticker, items in items_by_ticker.items():
        if ticker in details_cache:
            for item in items:
                news_text = (item.get("text", "") + " " + item.get("title", "")).strip()
                if news_text:
                    texts_for_sentiment.append(news_text)
                    metadata_for_sentiment.append(item)

    if not texts_for_sentiment:
        logging.info("No new texts to analyze for sentiment.")
        return {}

    logging.info(f"Submitting {len(texts_for_sentiment)} articles for batch sentiment analysis...")
    sentiment_func = components["sentiment_analyzer"].analyze_sentiments_batch
    sentiment_results_list = await asyncio.to_thread(sentiment_func, texts_for_sentiment)

    sentiment_map = {
        generate_news_hash(item): result
        for item, result in zip(metadata_for_sentiment, sentiment_results_list)
    }
    logging.info("Batch sentiment analysis complete.")
    return sentiment_map


async def _create_and_run_prediction_tasks(
    items_by_ticker: Dict[str, List[Dict]],
    details_cache: Dict[str, Dict],
    sentiment_map: Dict[str, Dict],
    components: Dict[str, Any],
) -> Dict[str, List[tuple]]:
    """Creates and runs prediction tasks concurrently."""
    prediction_coroutines = []
    task_metadata = []
    for ticker, items in items_by_ticker.items():
        if ticker in details_cache:
            for item in items:
                item_hash = generate_news_hash(item)
                if item_hash in sentiment_map:
                    coro = asyncio.to_thread(
                        get_prediction_for_item,
                        item,
                        components,
                        details_cache[ticker],
                        sentiment_map[item_hash],
                    )
                    prediction_coroutines.append(coro)
                    task_metadata.append({"ticker": ticker, "item": item})

    if not prediction_coroutines:
        logging.info("No prediction tasks to run.")
        return defaultdict(list)

    logging.info(f"Submitted {len(prediction_coroutines)} articles for concurrent prediction analysis...")
    results = await asyncio.gather(*prediction_coroutines)

    predictions_by_ticker = defaultdict(list)
    processed_hashes = []
    for i, result_tuple in enumerate(results):
        metadata = task_metadata[i]
        ticker, item = metadata["ticker"], metadata["item"]
        processed_hashes.append(generate_news_hash(metadata["item"]))
        if result_tuple and result_tuple[0]:
            predictions_by_ticker[ticker].append(result_tuple)

    if processed_hashes:
        database.add_news_hashes_batch(processed_hashes)

    return predictions_by_ticker


async def _generate_predictions_for_tickers(
    items_by_ticker: Dict[str, List[Dict]],
    details_cache: Dict[str, Dict],
    components: Dict[str, Any],
) -> Dict[str, List[tuple]]:
    """Orchestrates sentiment and prediction generation."""
    sentiment_map = await _run_batch_sentiment_analysis(items_by_ticker, details_cache, components)
    return await _create_and_run_prediction_tasks(items_by_ticker, details_cache, sentiment_map, components)


def _prepare_trade_log_data(
    ticker: str,
    final_decision: Dict[str, Any],
    news_text: str,
    numerical_features: Optional[Dict[str, float]],
    details_cache: Dict[str, Dict],
    timestamp: str,
    db_column_names: List[str],
) -> Dict[str, Any]:
    """Prepares a dictionary of data for logging a trade."""
    log_data = {
        "timestamp": timestamp,
        "ticker": ticker,
        "decision": final_decision["direction"],
        "entry_price": details_cache[ticker]["fmp_quote"].get("price"),
        "predicted_confidence": final_decision["predicted_confidence"],
        "predicted_volatility": final_decision["predicted_volatility"],
        "news_text": news_text,
        "feature_model_explanation": json.dumps(final_decision.get("explanation", [])),
    }
    if numerical_features:
        for feature_name, value in numerical_features.items():
            if f"feature_{feature_name}" in db_column_names:
                log_data[f"feature_{feature_name}"] = value
    return log_data


def _log_final_decisions(
    predictions_by_ticker: Dict[str, List[tuple]],
    details_cache: Dict[str, Dict],
    db_column_names: List[str],
):
    """Aggregates predictions and logs the final decisions to the database."""
    trades_to_log_with_keywords = []
    current_timestamp = datetime.now(timezone.utc).isoformat()

    for ticker, predictions in predictions_by_ticker.items():
        aggregated_result = aggregate_predictions(predictions)
        if aggregated_result:
            final_decision, news_text, numerical_features, matched_keywords = aggregated_result
            trade_log_data = _prepare_trade_log_data(
                ticker, final_decision, news_text, numerical_features, 
                details_cache, current_timestamp, db_column_names
            )
            trades_to_log_with_keywords.append(
                {"trade_data": trade_log_data, "keywords": matched_keywords}
            )

    if not trades_to_log_with_keywords:
        return

    with database.get_db_connection() as conn:
        all_trades_to_log = [item["trade_data"] for item in trades_to_log_with_keywords]
        trade_ids = database.log_trades_batch(all_trades_to_log, conn)

        all_keyword_matches_to_log = []
        for i, trade_id in enumerate(trade_ids):
            if trade_id is not None:
                keywords = trades_to_log_with_keywords[i]["keywords"]
                for keyword, weight in keywords.items():
                    all_keyword_matches_to_log.append(
                        {"trade_id": trade_id, "keyword": keyword, "weight": weight}
                    )
        
        if all_keyword_matches_to_log:
            database.log_keyword_matches_batch(all_keyword_matches_to_log, conn)
        
        conn.commit()
        logging.info(f"Successfully logged {len(trade_ids)} trades and their keyword matches.")


async def main(run_once: bool):
    setup_logging()
    logging.info("Starting Event-Driven Financial News Analysis & Trading System")

    signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    fmp_loader = None
    try:
        components = await initialize_system()
        fmp_loader = components["fmp_loader"]
        db_column_names = database.get_trades_column_names()

        while True:
            await check_for_pause()
            logging.info(f"\n--- Starting Prediction Cycle at {datetime.now(timezone.utc).isoformat()} ---")

            items_by_ticker, details_cache = await _fetch_and_prepare_ticker_data(
                fmp_loader, components
            )

            if not items_by_ticker:
                logging.info("No new news items or valid tickers found. Waiting...")
                await asyncio.sleep(60)
                continue

            predictions_by_ticker = await _generate_predictions_for_tickers(
                items_by_ticker, details_cache, components
            )

            _log_final_decisions(
                predictions_by_ticker, details_cache, db_column_names
            )

            if run_once:
                logging.info("Run-once mode enabled. The application will exit after this cycle.")
                break
            else:
                logging.info("Prediction cycle completed. Waiting for next run...")
                await asyncio.sleep(60)

    except InterruptedByUserError as e:
        logging.info(f"Exiting: {e}")
    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}", exc_info=True)
    finally:
        if fmp_loader:
            await fmp_loader.close()
            logging.info("FMP data loader session closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the financial news analysis and trading system."
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run the prediction cycle only once and then exit.",
    )
    args = parser.parse_args()

    asyncio.run(main(run_once=args.run_once))
