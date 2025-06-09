# 📈 Earnings Transcript Optimization Guide
*Eliminate Wasted API Calls & Focus on Quality Tickers*

## 🎯 **SOLUTIONS**

### **SOLUTION 1: Add Transcript Caching**

**FILE**: `data_loaders/earnings_transcript_fetcher.py`
**ACTION**: Add caching methods to the `EarningsTranscriptFetcher` class

```python
# ADD these methods to EarningsTranscriptFetcher class

def __init__(self, api_key: str):
    # ... existing code ...
    
    # NEW: Add transcript caching
    self.transcript_cache = {}  # Cache for successful transcripts
    self.failed_transcript_cache = set()  # Cache for failed fetches
    self.cache_file = Path("data/transcript_cache.json")
    self.failed_cache_file = Path("data/failed_transcripts.json")
    self._load_caches()

def _load_caches(self):
    """Load transcript caches from disk"""
    try:
        # Load successful transcript cache
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                # Filter expired entries (older than 7 days)
                current_time = time.time()
                self.transcript_cache = {
                    key: value for key, value in cache_data.items()
                    if current_time - value.get('cached_at', 0) < 7 * 24 * 3600
                }
        
        # Load failed transcript cache
        if self.failed_cache_file.exists():
            with open(self.failed_cache_file, 'r') as f:
                failed_data = json.load(f)
                # Filter expired entries (older than 30 days)
                current_time = time.time()
                self.failed_transcript_cache = {
                    key for key, timestamp in failed_data.items()
                    if current_time - timestamp < 30 * 24 * 3600
                }
        
        log_info(f"📋 Loaded transcript cache: {len(self.transcript_cache)} successful, {len(self.failed_transcript_cache)} failed")
    except Exception as e:
        log_error(f"Error loading transcript caches: {e}")

def _save_caches(self):
    """Save transcript caches to disk"""
    try:
        # Ensure data directory exists
        self.cache_file.parent.mkdir(exist_ok=True)
        
        # Save successful cache
        with open(self.cache_file, 'w') as f:
            json.dump(self.transcript_cache, f)
        
        # Save failed cache with timestamps
        failed_with_timestamps = {
            ticker: time.time() for ticker in self.failed_transcript_cache
        }
        with open(self.failed_cache_file, 'w') as f:
            json.dump(failed_with_timestamps, f)
            
    except Exception as e:
        log_error(f"Error saving transcript caches: {e}")

def _get_cache_key(self, ticker: str, year: int, quarter: int) -> str:
    """Generate cache key for transcript"""
    return f"{ticker}_{year}_Q{quarter}"

def _is_transcript_cached_as_failed(self, ticker: str, year: int, quarter: int) -> bool:
    """Check if we've already tried and failed to get this transcript"""
    cache_key = self._get_cache_key(ticker, year, quarter)
    return cache_key in self.failed_transcript_cache

def _cache_failed_transcript(self, ticker: str, year: int, quarter: int):
    """Cache a failed transcript attempt"""
    cache_key = self._get_cache_key(ticker, year, quarter)
    self.failed_transcript_cache.add(cache_key)
    log_debug(f"🚫 Cached failed transcript: {cache_key}")

# MODIFY the existing fetch_earnings_transcript method:
def fetch_earnings_transcript(self, ticker: str, year: int, quarter: int) -> Optional[EarningsAnalysis]:
    """
    Fetch and analyze earnings call transcript for specific quarter (WITH CACHING)
    """
    cache_key = self._get_cache_key(ticker, year, quarter)
    
    # Check failed cache first
    if self._is_transcript_cached_as_failed(ticker, year, quarter):
        log_debug(f"⏭️ Skipping {ticker} Q{quarter} {year} (cached as unavailable)")
        return None
    
    # Check successful cache
    if cache_key in self.transcript_cache:
        cached_data = self.transcript_cache[cache_key]
        log_debug(f"📋 Using cached transcript for {ticker} Q{quarter} {year}")
        return EarningsAnalysis(**cached_data['analysis'])
    
    try:
        log_info(f"Fetching earnings transcript for {ticker} Q{quarter} {year}")
        
        # Fetch transcript from FMP API (existing code)
        endpoint = f"earning_call_transcript/{ticker}"
        params = {'year': year, 'quarter': quarter}
        
        transcript_data = self.make_request(endpoint, params)
        
        if not transcript_data or not isinstance(transcript_data, list) or len(transcript_data) == 0:
            log_warning(f"No transcript data found for {ticker} Q{quarter} {year}")
            self._cache_failed_transcript(ticker, year, quarter)
            self._save_caches()  # Save immediately
            return None
        
        # Process the transcript (existing code)
        transcript_info = transcript_data[0]
        
        if not transcript_info.get('content'):
            log_warning(f"Empty transcript content for {ticker} Q{quarter} {year}")
            self._cache_failed_transcript(ticker, year, quarter)
            self._save_caches()
            return None
        
        full_transcript = transcript_info['content']
        date = transcript_info.get('date', '')
        
        log_info(f"Processing transcript for {ticker} ({len(full_transcript)} characters)")
        
        # Analyze the transcript
        analysis = self._analyze_transcript(
            ticker=ticker,
            date=date,
            quarter=str(quarter),
            year=str(year),
            transcript=full_transcript
        )
        
        # Cache successful result
        self.transcript_cache[cache_key] = {
            'analysis': {
                'ticker': analysis.ticker,
                'date': analysis.date,
                'quarter': analysis.quarter,
                'year': analysis.year,
                'transcript': analysis.transcript
            },
            'cached_at': time.time()
        }
        self._save_caches()
        
        return analysis
        
    except Exception as e:
        log_error(f"Error fetching earnings transcript for {ticker}: {e}")
        self._cache_failed_transcript(ticker, year, quarter)
        self._save_caches()
        return None
```

---

### **SOLUTION 2: Improve Ticker Selection**

**FILE**: `data_loaders/news_fetcher.py`
**ACTION**: Replace `_get_priority_tickers_for_transcripts` method with better filtering

```python
def _get_priority_tickers_for_transcripts(self) -> List[str]:
    """Get high-quality tickers most likely to have earnings transcripts"""
    try:
        from data_loaders.earnings_integration_manager import EarningsIntegrationManager
        
        # Use earnings manager to get calendar tickers
        earnings_manager = EarningsIntegrationManager(self)
        earnings_manager._refresh_earnings_cache()
        
        if not earnings_manager.earnings_cache:
            return self._get_fallback_transcript_tickers()
        
        # Get all calendar tickers
        all_calendar_tickers = list(earnings_manager.earnings_cache.keys())
        
        # ENHANCED FILTERING for quality tickers
        high_quality_tickers = []
        
        for ticker in all_calendar_tickers:
            # Skip if already processed and cached as failed
            if hasattr(self.earnings_fetcher, '_is_any_quarter_cached_as_failed'):
                if self.earnings_fetcher._is_any_quarter_cached_as_failed(ticker):
                    continue
            
            # Filter criteria for transcript-likely tickers
            if self._is_transcript_worthy_ticker(ticker):
                high_quality_tickers.append(ticker)
        
        # Sort by transcript likelihood (market cap proxy)
        high_quality_tickers = self._sort_by_transcript_likelihood(high_quality_tickers)
        
        # Limit to a reasonable number
        max_tickers = min(Config.MAX_EARNINGS_EVENTS_PER_CYCLE, 50)  # Cap at 50
        limited_tickers = high_quality_tickers[:max_tickers]
        
        log_info(f"📅 Selected {len(limited_tickers)} high-quality tickers for transcripts")
        log_info(f"   🎯 Top candidates: {', '.join(limited_tickers[:10])}")
        
        return limited_tickers
        
    except Exception as e:
        log_error(f"Error getting priority tickers: {e}")
        return self._get_fallback_transcript_tickers()

def _is_transcript_worthy_ticker(self, ticker: str) -> bool:
    """Determine if a ticker is likely to have earnings transcripts"""
    
    # Length filter (most major companies have short tickers)
    if len(ticker) > 5:
        return False
    
    # Character filters
    ticker_upper = ticker.upper()
    
    # Skip international exchanges
    international_suffixes = ['.L', '.SS', '.HE', '.V', '.NZ', '.TO', '.F', '.MI', '.PA']
    if any(suffix in ticker_upper for suffix in international_suffixes):
        return False
    
    # Skip numeric tickers (often ETFs or special securities)
    if any(char.isdigit() for char in ticker_upper):
        return False
    
    # Skip obvious penny stock patterns
    penny_patterns = ['OOTB', 'OOTC', 'MQTD', 'MQTO']
    if any(pattern in ticker_upper for pattern in penny_patterns):
        return False
    
    # Skip foreign ticker patterns (ends with F)
    if ticker_upper.endswith('F') and len(ticker) >= 5:
        return False
    
    # Skip warrant/rights patterns
    if any(suffix in ticker_upper for suffix in ['W', 'WS', 'WT', 'R', 'RT']):
        return False
    
    # Prefer common major company patterns
    # Single letter tickers (often major companies)
    if len(ticker) == 1:
        return True
    
    # 2-4 letter tickers (most major companies)
    if 2 <= len(ticker) <= 4 and ticker.isalpha():
        return True
    
    # 5 letter tickers - be more selective
    if len(ticker) == 5:
        # Known major company patterns
        if ticker_upper in ['GOOGL', 'GOOG', 'BERKB', 'BRKB']:
            return True
        # Reject most 5-letter tickers (often smaller companies)
        return False
    
    return False

def _sort_by_transcript_likelihood(self, tickers: List[str]) -> List[str]:
    """Sort tickers by likelihood of having transcripts (market cap proxy)"""
    
    # Known major companies that definitely have transcripts
    tier_1_companies = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK', 'BRKB',
        'JNJ', 'JPM', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE',
        'CRM', 'NFLX', 'KO', 'PEP', 'TMO', 'COST', 'ABBV', 'WMT', 'MRK', 'ACN',
        'LLY', 'AMD', 'ORCL', 'CSCO', 'IBM', 'QCOM', 'INTC', 'COP', 'PFE', 'NKE',
        'XOM', 'CVX', 'WFC', 'AMAT', 'CAT', 'GE', 'F', 'GM', 'BA', 'MMM'
    ]
    
    # S&P 500-like companies (likely to have transcripts)
    tier_2_patterns = [
        # Tech companies
        'CRM', 'NOW', 'SNOW', 'PLTR', 'ZM', 'DOCU', 'OKTA', 'SPLK',
        # Financial
        'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'CB', 'TFC', 'USB', 'PNC',
        # Healthcare
        'UNH', 'CVS', 'ANTM', 'CI', 'HUM', 'AET', 'BMY', 'GILD', 'BIIB',
        # Industrial
        'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD', 'DE', 'EMR', 'ITW',
        # Consumer
        'AMGN', 'SBUX', 'MCD', 'NKE', 'LULU', 'TGT', 'LOW', 'BKNG'
    ]
    
    def get_priority(ticker):
        ticker_upper = ticker.upper()
        
        if ticker_upper in tier_1_companies:
            return 1
        elif ticker_upper in tier_2_patterns:
            return 2
        elif len(ticker) <= 4:  # Shorter tickers often major companies
            return 3
        else:
            return 4
    
    # Sort by priority, then alphabetically for consistency
    return sorted(tickers, key=lambda t: (get_priority(t), t))

def _get_fallback_transcript_tickers(self) -> List[str]:
    """Fallback list of major companies for transcript analysis"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK',
        'JNJ', 'JPM', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE',
        'CRM', 'NFLX', 'KO', 'PEP', 'TMO', 'COST', 'ABBV', 'WMT', 'MRK',
        'LLY', 'AMD', 'ORCL', 'CSCO', 'IBM', 'QCOM', 'INTC', 'PFE', 'NKE'
    ]
```

---

### **SOLUTION 3: Add Failed Ticker Methods to EarningsTranscriptFetcher**

**FILE**: `data_loaders/earnings_transcript_fetcher.py`
**ACTION**: Add these helper methods to the `EarningsTranscriptFetcher` class

```python
def _is_any_quarter_cached_as_failed(self, ticker: str) -> bool:
    """Check if ticker has ANY quarters cached as failed (likely doesn't do transcripts)"""
    for year in [2024, 2025]:
        for quarter in [1, 2, 3, 4]:
            if self._is_transcript_cached_as_failed(ticker, year, quarter):
                return True
    return False

def get_cache_stats(self) -> Dict[str, Any]:
    """Get cache statistics for monitoring"""
    return {
        'successful_transcripts_cached': len(self.transcript_cache),
        'failed_transcripts_cached': len(self.failed_transcript_cache),
        'cache_efficiency': len(self.transcript_cache) / (len(self.transcript_cache) + len(self.failed_transcript_cache)) if (len(self.transcript_cache) + len(self.failed_transcript_cache)) > 0 else 0
    }

def clear_failed_cache(self):
    """Clear failed transcript cache (for testing or reset)"""
    self.failed_transcript_cache.clear()
    if self.failed_cache_file.exists():
        self.failed_cache_file.unlink()
    log_info("🗑️ Cleared failed transcript cache")
```

---

### **SOLUTION 4: Configuration Optimization**

**FILE**: `.env`
**ACTION**: Add these optimized settings

```bash
# EARNINGS OPTIMIZATION SETTINGS

# Reduce transcript processing to quality tickers only
MAX_EARNINGS_EVENTS_PER_CYCLE=25  # Reduced from 150

# Extended caching to reduce API calls
EARNINGS_CACHE_HOURS=12  # Increased from 6

# Quality focus settings
ENABLE_FUNDAMENTAL_FILTERING=true
MIN_MARKET_CAP=500000000  # Focus on larger companies (more likely to have transcripts)
```

---

## 📊 **EXPECTED IMPROVEMENTS**

### **Before Optimization:**
- 🔄 **API Calls**: 300+ per cycle
- ⏱️ **Time**: 120+ seconds for transcripts
- 🎯 **Success Rate**: ~20% 
- 💸 **Efficiency**: Poor

### **After Optimization:**
- 🔄 **API Calls**: ~25-50 per cycle (80% reduction)
- ⏱️ **Time**: ~30 seconds for transcripts (75% reduction)
- 🎯 **Success Rate**: ~60-80% (3x improvement)
- 💸 **Efficiency**: Excellent

---

## 🚀 **IMPLEMENTATION STEPS**

1. **Add caching to transcript fetcher** - Eliminate repeat API calls
2. **Improve ticker selection** - Focus on quality companies
3. **Update configuration** - Optimize settings
4. **Test the improvements** - Monitor cache hit rates

### **Verification Commands:**

```bash
# Check cache effectiveness
ls -la data/transcript_cache.json data/failed_transcripts.json

# Monitor logs for cache hits
python main.py | grep "cached\|Skipping\|Selected.*high-quality"

# Test cache clearing (if needed)
python -c "
from data_loaders.earnings_transcript_fetcher import EarningsTranscriptFetcher
from config import Config
fetcher = EarningsTranscriptFetcher(Config.FMP_API_KEY)
print(fetcher.get_cache_stats())
"
```

**Result: Your system will be much faster, cheaper to run, and focus on companies that actually provide valuable earnings insights!** 🎯