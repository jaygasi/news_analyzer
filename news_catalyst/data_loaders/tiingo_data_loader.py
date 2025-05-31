# Fix for Tiingo API 403 Forbidden Error

# 1. Add better error handling and fallback logic to TiingoDataLoader
# Update: data_loaders/tiingo_data_loader.py

class OptimizedTiingoDataLoader:
    def __init__(self, api_key: str, rate_limit: int = 500):
        if not api_key or not api_key.strip():
            logw("Tiingo API key is empty - Tiingo features will be disabled")
            self.api_key = None
            self.is_available = False
            return
        
        self.api_key = api_key.strip()
        self.base_url = "https://api.tiingo.com"
        self.rate_limiter = TiingoRateLimiter(max_calls=rate_limit)
        self.is_available = True
        
        # Test API key validity on initialization
        self._test_api_connection()
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Tiingo-Python-Client/1.0'
        })
    
    def _test_api_connection(self) -> bool:
        """Test if Tiingo API key is valid and has required permissions."""
        try:
            test_url = f"{self.base_url}/api/test"
            test_params = {'token': self.api_key}
            
            response = requests.get(test_url, params=test_params, timeout=10)
            
            if response.status_code == 200:
                logi("✅ Tiingo API connection successful")
                return True
            elif response.status_code == 403:
                logw("⚠️ Tiingo API key lacks required permissions - disabling Tiingo")
                self.is_available = False
                return False
            else:
                logw(f"⚠️ Tiingo API test failed with status {response.status_code}")
                self.is_available = False
                return False
                
        except Exception as e:
            logw(f"⚠️ Tiingo API test failed: {e} - disabling Tiingo")
            self.is_available = False
            return False
    
    def fetch_latest_news_articles(self, news_article_limit: int = 50,
                                 cache_data: bool = False, cache_dir: str = 'cache') -> Optional[pd.DataFrame]:
        """
        Fetch latest news articles with improved error handling and fallback.
        """
        if not self.is_available:
            logd("Tiingo API not available, skipping news fetch")
            return None
            
        # Try different endpoints if main news endpoint fails
        endpoints_to_try = [
            f"{self.base_url}/tiingo/news",
            f"{self.base_url}/tiingo/fundamentals/news"  # Alternative endpoint
        ]
        
        start_date = datetime.today()
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date = datetime.today() + timedelta(days=1)
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        for endpoint in endpoints_to_try:
            try:
                params = {
                    'startDate': start_date_str,
                    'endDate': end_date_str,
                    'limit': min(news_article_limit, 100),  # Reduce limit
                    'token': self.api_key
                }
                
                response = requests.get(endpoint, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        return self._process_news_data(data, cache_data, cache_dir)
                elif response.status_code == 403:
                    loge(f"Tiingo API 403 Forbidden - check API key permissions")
                    self.is_available = False
                    return None
                elif response.status_code == 429:
                    logw(f"Tiingo rate limit hit - backing off")
                    time.sleep(60)
                    continue
                else:
                    logw(f"Tiingo endpoint {endpoint} failed with status {response.status_code}")
                    continue
                    
            except Exception as e:
                loge(f"Error with Tiingo endpoint {endpoint}: {e}")
                continue
        
        # All endpoints failed
        logw("All Tiingo endpoints failed - disabling Tiingo for this session")
        self.is_available = False
        return None

# 2. Update news event tracker to handle Tiingo failures gracefully
# Update: event_trackers/news_event_tracker.py

class OptimizedNewsEventTracker:
    async def _fetch_and_process_news(self) -> bool:
        """Enhanced fetch with better Tiingo error handling."""
        try:
            # Fetch news from FMP RSS feed (primary source - always try this first)
            news_df = await asyncio.to_thread(
                self.fmp_data_loader.fetch_stock_news_rss_feed
            )
            
            # Only try Tiingo if it's available and working
            if (self.tiingo_data_loader and 
                hasattr(self.tiingo_data_loader, 'is_available') and 
                self.tiingo_data_loader.is_available):
                
                try:
                    logd("Attempting to fetch Tiingo news...")
                    tiingo_news_df = await asyncio.to_thread(
                        self.tiingo_data_loader.fetch_latest_news_articles,
                        50  # Reduced limit
                    )
                    
                    if tiingo_news_df is not None and not tiingo_news_df.empty:
                        if news_df is not None and not news_df.empty:
                            news_df = pd.concat([news_df, tiingo_news_df], axis=0, ignore_index=True)
                        else:
                            news_df = tiingo_news_df
                        logd(f"Successfully added {len(tiingo_news_df)} Tiingo articles")
                    else:
                        logd("No Tiingo news returned")
                        
                except Exception as e:
                    logw(f"Tiingo news fetch failed, continuing with FMP only: {e}")
                    # Don't fail the entire process, just continue with FMP
            else:
                logd("Tiingo not available, using FMP news only")
            
            # Rest of the processing remains the same...
            news_df = self._validate_and_prepare_news_data(news_df)
            if news_df is None or news_df.empty:
                return False
            
            logd(f"{len(news_df)} total news articles returned")
            
            # Save for review
            self._save_news_for_review(news_df)
            
            # Process articles
            articles_sent = self._process_news_articles(news_df)
            
            # Update statistics
            self.total_articles_processed += len(news_df)
            self.valid_articles_sent += articles_sent
            
            if articles_sent > 0:
                logi(f"Processed {len(news_df)} articles, sent {articles_sent} for analysis")
            
            return True
            
        except Exception as e:
            loge(f"Error in news fetch and process cycle: {e}")
            return False

# 3. Improve Tiingo initialization in main.py
# Update the Tiingo initialization to handle missing/invalid keys

async def initialize_components(self) -> bool:
    """Enhanced component initialization with better Tiingo handling."""
    try:
        # Get API keys
        fmp_api_key = self.config.get_api_key('fmp')
        tiingo_api_key = self.config.get_api_key('tiingo')
        
        if not fmp_api_key:
            loge("FMP API key required but not configured")
            return False
        
        # Initialize Tiingo with graceful fallback
        tiingo_data_loader = None
        if tiingo_api_key:
            try:
                tiingo_data_loader = TiingoDataLoader(tiingo_api_key)
                if not tiingo_data_loader.is_available:
                    logw("Tiingo API key invalid or lacks permissions - continuing without Tiingo")
                    tiingo_data_loader = None
            except Exception as e:
                logw(f"Tiingo initialization failed: {e} - continuing without Tiingo")
                tiingo_data_loader = None
        else:
            logi("No Tiingo API key provided - using FMP news only")
        
        # Event tracking with optional Tiingo
        news_event_tracker = NewsEventTracker(
            fmp_api_key=fmp_api_key,
            tiingo_api_key=tiingo_api_key if tiingo_data_loader else "",
            data_collection_interval=int(self.config.data.news_check_interval)
        )
        
        # Rest of initialization...
        
    except Exception as e:
        loge(f"Failed to initialize components: {e}")
        return False