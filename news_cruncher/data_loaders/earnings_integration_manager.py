from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from data_loaders.earnings_transcript_fetcher import EarningsTranscriptFetcher
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_error, log_debug, log_warning # Add other loggers if needed
from config import Config # Assuming Config is used for EARNINGS_LOOKBACK_DAYS etc.
import json # Added for failed cache handling

class EarningsIntegrationManager:
    """Simplified earnings integration manager"""
    
    def __init__(self, fmp_api_key: str):  # CHANGED: Take API key directly
        """Initialize earnings integration manager"""
        self.fmp_api_key = fmp_api_key
        self.earnings_fetcher = EarningsTranscriptFetcher(fmp_api_key)  # Single instance
        self.earnings_cache: Dict[str, Any] = {} # Cache for earnings calendar data
        self.cache_expiry: Optional[datetime] = None # Type hint for cache_expiry
        
        log_info("✅ Earnings integration manager initialized")
    
    def _refresh_earnings_cache(self) -> None:
        """Refresh earnings calendar cache if expired"""
        now = datetime.now(timezone.utc)
        
        if (self.cache_expiry is None or 
            now > self.cache_expiry or 
            not self.earnings_cache):
            
            log_info("🔄 Refreshing earnings calendar cache...")
            
            # Calculate date range
            start_date = now - timedelta(days=Config.EARNINGS_LOOKBACK_DAYS)
            end_date = now + timedelta(days=Config.EARNINGS_LOOKAHEAD_DAYS)
            
            # Make API request directly (remove NewsFetcher dependency)
            # MODIFIED: Use self.earnings_fetcher for the API call
            try:
                earnings_data = self.earnings_fetcher.make_request("earning_calendar", {
                    "from": start_date.strftime('%Y-%m-%d'),
                    "to": end_date.strftime('%Y-%m-%d')
                })
                
                if earnings_data and isinstance(earnings_data, list):
                    self.earnings_cache = {
                        item['symbol']: item for item in earnings_data if item.get('symbol')
                    }
                    self.cache_expiry = now + timedelta(hours=Config.EARNINGS_CACHE_HOURS)
                    log_info(f"✅ Earnings calendar cache refreshed with {len(self.earnings_cache)} entries.")
                else:
                    log_info("⚠️ No data returned for earnings calendar, cache not updated.")
                    # Potentially set a shorter expiry to retry sooner
                    self.cache_expiry = now + timedelta(minutes=Config.EARNINGS_CALENDAR_RETRY_MINUTES)

            except Exception as e:
                log_info(f"❌ Error refreshing earnings calendar cache: {e}")
                # Potentially set a shorter expiry to retry sooner
                self.cache_expiry = now + timedelta(minutes=Config.EARNINGS_CALENDAR_RETRY_MINUTES)
    
    def _load_failed_cache(self) -> Dict[str, str]:
        """Load the cache of tickers for which earnings transcript fetching failed."""
        failed_cache_file = Config.DATA_DIR / Config.FAILED_EARNINGS_CACHE_FILENAME
        failed_cache: Dict[str, str] = {}
        try:
            if failed_cache_file.exists():
                with open(failed_cache_file, 'r') as f:
                    loaded_data = json.load(f)
                    # Ensure data is in the expected format (str: str for timestamp)
                    if isinstance(loaded_data, dict):
                        failed_cache = {k: str(v) for k, v in loaded_data.items() if isinstance(k, str) and isinstance(v, str)}
                    else:
                        log_warning(f"Failed earnings cache file {failed_cache_file} has unexpected format. Reinitializing.")
        except Exception as e:
            log_debug(f"Could not load failed earnings cache: {e}. Initializing an empty cache.")
        return failed_cache

    def _save_failed_cache(self, cache: Dict[str, str]) -> None:
        """Save the cache of tickers for which earnings transcript fetching failed."""
        failed_cache_file = Config.DATA_DIR / Config.FAILED_EARNINGS_CACHE_FILENAME
        try:
            with open(failed_cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            log_debug(f"Could not save failed earnings cache: {e}")

    def analyze_tickers_for_earnings(self, ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze tickers for earnings events with improved caching"""
        log_info(f"📅 Analyzing {len(ticker_buckets)} tickers for earnings events...")
        
        # Refresh earnings calendar cache
        self._refresh_earnings_cache()
        
        earnings_analyses = {}
        processed_count = 0
        max_to_process = Config.MAX_EARNINGS_EVENTS_PER_CYCLE
        
        failed_cache = self._load_failed_cache()
        list_of_tickers_to_process = list(ticker_buckets.keys())
        
        for ticker in list_of_tickers_to_process:
            if processed_count >= max_to_process:
                log_info(f"⏱️ Reached earnings processing limit ({max_to_process})")
                break
                
            # Check if this ticker failed recently (within 24 hours)
            cache_key = f"{ticker}_{datetime.now().strftime('%Y-%m-%d')}"
            if cache_key in failed_cache:
                hours_since_failure = (datetime.now(timezone.utc) - datetime.fromisoformat(failed_cache[cache_key])).total_seconds() / 3600
                if hours_since_failure < Config.FAILED_EARNINGS_CACHE_EXPIRY_HOURS:
                    log_debug(f"⏭️ Skipping {ticker} - failed recently (cached)")
                    continue

            try:
                all_fetched_analyses_for_ticker: List[EarningsAnalysis] = []
                
                if ticker in self.earnings_cache: # Check calendar
                    log_info(f"📊 Processing earnings for {ticker} (found in calendar)...")
                    current_transcript_analyses = self.earnings_fetcher.fetch_recent_transcripts(
                        ticker, lookback_quarters=Config.EARNINGS_TRANSCRIPT_LOOKBACK_QUARTERS
                    )
                    if current_transcript_analyses:
                        all_fetched_analyses_for_ticker.extend(current_transcript_analyses)
                else: # Fallback: Try to fetch earnings data even if not in calendar
                    log_debug(f"📊 Attempting fallback earnings fetch for {ticker} (not in calendar)...")
                    # Use a potentially shorter lookback for fallback
                    fallback_transcript_analyses = self.earnings_fetcher.fetch_recent_transcripts(
                        ticker, lookback_quarters=2 
                    )
                    if fallback_transcript_analyses:
                        all_fetched_analyses_for_ticker.extend(fallback_transcript_analyses)

                # Filter for analyses that actually have content
                valid_analyses = [
                    analysis for analysis in all_fetched_analyses_for_ticker if analysis.transcript_length > 0
                ]

                if valid_analyses:
                    # Success - remove from failed cache if present
                    if cache_key in failed_cache:
                        del failed_cache[cache_key]
                    
                    earnings_articles = []
                    for analysis_obj in valid_analyses: # Process all valid analyses
                        articles_from_transcript = self.earnings_fetcher.create_earnings_articles(analysis_obj)
                        earnings_articles.extend(articles_from_transcript)
                    
                    if earnings_articles: # Ensure articles were actually generated
                        earnings_analyses[ticker] = {
                            'analyses': valid_analyses, # Store the list of EarningsAnalysis objects
                            'articles': earnings_articles
                        }
                        log_info(f"✅ Successfully processed earnings for {ticker}: {len(earnings_articles)} articles from {len(valid_analyses)} transcripts.")
                        processed_count += 1
                    else:
                        # This case means transcripts were found but create_earnings_articles returned nothing.
                        log_warning(f"⚠️ Transcripts found for {ticker} but no articles generated.")
                        failed_cache[cache_key] = datetime.now(timezone.utc).isoformat()
                else:
                    # No valid transcripts found (either none fetched or all were empty)
                    log_warning(f"⚠️ No recent transcripts with content found for {ticker} (calendar or fallback).")
                    # Cache the failure
                    failed_cache[cache_key] = datetime.now(timezone.utc).isoformat()
                    
            except Exception as e:
                log_error(f"Error processing earnings for {ticker}: {e}")
                failed_cache[cache_key] = datetime.now(timezone.utc).isoformat()
        
        self._save_failed_cache(failed_cache)
        
        log_info(f"📅 Earnings analysis complete: {len(earnings_analyses)} tickers with earnings data")
        return earnings_analyses