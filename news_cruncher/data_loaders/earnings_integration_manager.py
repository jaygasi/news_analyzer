from typing import Dict, Any
from datetime import datetime, timezone, timedelta
from data_loaders.earnings_transcript_fetcher import EarningsTranscriptFetcher
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info # Add other loggers if needed
from config import Config # Assuming Config is used for EARNINGS_LOOKBACK_DAYS etc.

class EarningsIntegrationManager:
    """Simplified earnings integration manager"""
    
    def __init__(self, fmp_api_key: str):  # CHANGED: Take API key directly
        """Initialize earnings integration manager"""
        self.fmp_api_key = fmp_api_key
        self.earnings_fetcher = EarningsTranscriptFetcher(fmp_api_key)  # Single instance
        self.earnings_cache: Dict[str, Any] = {} # Cache for earnings calendar data
        self.cache_expiry: datetime = None
        
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
            try:
                # api_loader is used here to make the request, ensuring it uses the FMP base URL and API key logic
                api_loader = BaseFMPLoader(self.fmp_api_key) 
                
                earnings_data = api_loader.make_request("earning_calendar", {
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
                    self.cache_expiry = now + timedelta(minutes=30) 

            except Exception as e:
                log_info(f"❌ Error refreshing earnings calendar cache: {e}")
                # Potentially set a shorter expiry to retry sooner
                self.cache_expiry = now + timedelta(minutes=30)
    
    def analyze_tickers_for_earnings(self, ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze tickers for earnings events and return earnings data
        This is the missing method causing the AttributeError
        """
        log_info(f"📅 Analyzing {len(ticker_buckets)} tickers for earnings events...")
        
        # Refresh earnings calendar cache
        self._refresh_earnings_cache()
        
        earnings_analyses = {}
        processed_count = 0
        max_to_process = Config.MAX_EARNINGS_EVENTS_PER_CYCLE
        
        for ticker, articles in ticker_buckets.items():
            if processed_count >= max_to_process:
                log_info(f"⏱️ Reached earnings processing limit ({max_to_process})")
                break
                
            # Check if ticker has earnings event
            if ticker in self.earnings_cache:
                try:
                    log_info(f"📊 Processing earnings for {ticker}...")
                    
                    # Fetch recent transcripts
                    transcript_analyses = self.earnings_fetcher.fetch_recent_transcripts(
                        ticker, lookback_quarters=2
                    )
                    
                    if transcript_analyses:
                        # Convert to articles format for integration
                        earnings_articles = []
                        for analysis in transcript_analyses:
                            articles_from_transcript = self.earnings_fetcher.create_earnings_articles(analysis)
                            earnings_articles.extend(articles_from_transcript)
                        
                        if earnings_articles:
                            earnings_analyses[ticker] = {
                                'analyses': transcript_analyses,
                                'articles': earnings_articles,
                                'earnings_date': self.earnings_cache[ticker].get('date', ''),
                                'quarter': self.earnings_cache[ticker].get('quarter', ''),
                                'year': self.earnings_cache[ticker].get('year', '')
                            }
                            log_info(f"✅ Generated {len(earnings_articles)} earnings articles for {ticker}")
                            processed_count += 1
                        else:
                            log_info(f"⚠️ No usable earnings data for {ticker}")
                    else:
                        log_info(f"⚠️ No recent transcripts found for {ticker}")
                        
                except Exception as e:
                    log_error(f"❌ Error processing earnings for {ticker}: {e}")
                    continue
            else:
                log_debug(f"No earnings event found for {ticker}")
        
        log_info(f"📅 Earnings analysis complete: {len(earnings_analyses)} tickers with earnings data")
        return earnings_analyses