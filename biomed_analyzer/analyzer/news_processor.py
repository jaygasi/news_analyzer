# biomed_analyzer/analyzer/news_processor.py
import time
from datetime import datetime, timedelta, date 
from typing import List, Optional, Dict, Set, Tuple, Union, Any
import re

from .finnhub_client import FinnhubClient
from .fmp_client import FMPClient
from .llm_services import get_llm_service, BaseLLM
from .models import StockPrediction, LLMAnalysisResult
from .config import (
    RESEARCH_NEWS_KEYWORDS, DEFAULT_LLM, MIN_CONFIDENCE_THRESHOLD, NEWS_LOOKBACK_DAYS,
    FDA_NEWS_WINDOW_PAST_DAYS, FDA_NEWS_WINDOW_FUTURE_DAYS
)
from .utils import logger

class NewsProcessor:
    def __init__(self, finnhub_client: Optional[FinnhubClient], fmp_client: Optional[FMPClient], llm_service_name: str = DEFAULT_LLM):
        self.finnhub = finnhub_client
        self.fmp = fmp_client
        try:
            self.llm: BaseLLM = get_llm_service(llm_service_name)
        except ValueError as e:
            logger.critical(f"Failed to initialize LLM service: {e}")
            raise
        
        self._processed_news_symbol_pairs: Set[Tuple[Union[int, str], str]] = set()
        
        # Load biomedical symbols during initialization from both sources
        try:
            finnhub_symbols = self.finnhub.get_cached_biomed_symbols() if self.finnhub else set()
            fmp_symbols = self.fmp.get_cached_biomed_symbols() if self.fmp else set()
            self._biomed_symbols_set: Set[str] = finnhub_symbols.union(fmp_symbols)
            
            if not self._biomed_symbols_set:
                logger.warning("Biomedical symbols set is initially empty from both sources. Will rely on dynamic profile checks.")
            else:
                logger.info(f"Loaded {len(self._biomed_symbols_set)} biomedical symbols ({len(finnhub_symbols)} from Finnhub, {len(fmp_symbols)} from FMP)")
        except Exception as e:
            logger.error(f"Error loading biomedical symbols during NewsProcessor init: {e}")
            self._biomed_symbols_set = set() # Ensure it's initialized even on error

    def _is_relevant_research_news(self, headline: str, summary: Optional[str] = None) -> bool:
        text_to_check = headline.lower()
        if summary:
            text_to_check += " " + summary.lower()
        
        for keyword in RESEARCH_NEWS_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_to_check, re.IGNORECASE):
                return True
        return False

    def _is_biomed_company(self, symbol: str) -> bool:
        if not symbol: return False
        
        # First check our combined cached set
        if symbol in self._biomed_symbols_set:
            return True
        
        # If not in cache, check both APIs for dynamic profile check
        # Try Finnhub first (usually faster)
        if self.finnhub and self.finnhub.is_biomedical_company_from_profile(symbol):
            self._biomed_symbols_set.add(symbol)  # Cache the result
            return True
        
        # Try FMP as fallback
        if self.fmp and self.fmp.is_biomedical_company_from_profile(symbol):
            self._biomed_symbols_set.add(symbol)  # Cache the result
            return True
        
        return False

    def _analyze_and_store_news(
            self, 
            news_item: Dict[str, Any], 
            symbol: str, 
            predictions_list: List[StockPrediction],
            source_tag: str = "general"
        ) -> None:
        headline = news_item.get('headline')
        news_id = news_item.get('id') 

        if not headline: 
            logger.debug(f"News item (ID: {news_id}, URL: {news_item.get('url')}) from {source_tag} has no headline. Skipping for {symbol}.")
            return

        dedupe_key_part1: Union[int, str] = news_id if isinstance(news_id, int) else headline
        if (dedupe_key_part1, symbol) in self._processed_news_symbol_pairs:
            return

        summary = news_item.get('summary')
        news_url = news_item.get('url')
        news_source_api = news_item.get('source')
        
        logger.info(f"[{source_tag.upper()}] Analyzing news for {symbol}: '{headline[:80]}...' (ID: {news_id})")
        time.sleep(0.33) # API courtesy for LLM (approx 3 req/sec)

        llm_analysis: Optional[LLMAnalysisResult] = self.llm.analyze_news_sentiment(
            headline=headline, summary=summary, ticker=symbol
        )

        if llm_analysis and llm_analysis.confidence >= MIN_CONFIDENCE_THRESHOLD:
            prediction = StockPrediction(
                ticker=symbol, confidence=llm_analysis.confidence,
                predicted_movement=llm_analysis.predicted_movement, reasoning=llm_analysis.reasoning,
                news_headline=headline, news_url=news_url, news_source=news_source_api
            )
            predictions_list.append(prediction)
            logger.info(f"  Prediction for {symbol}: {prediction.predicted_movement}, Conf: {prediction.confidence}")
        elif llm_analysis:
            logger.debug(f"  LLM analysis for {symbol} (ID: {news_id}) below confidence threshold ({llm_analysis.confidence}).")
        else:
            logger.warning(f"  LLM analysis failed for {symbol} and news ID {news_id}.")
        
        self._processed_news_symbol_pairs.add((dedupe_key_part1, symbol))

    def process_market_news(self) -> List[StockPrediction]:
        predictions: List[StockPrediction] = []
        
        # Get news from enabled sources
        finnhub_news = []
        fmp_news = []
        
        if self.finnhub:
            logger.info("Processing general market news from Finnhub...")
            finnhub_news = self.finnhub.get_general_market_news(category='general')
        
        if self.fmp:
            logger.info("Processing general market news from FMP...")
            fmp_news = self.fmp.get_general_market_news()
        
        # Combine news from both sources
        all_market_news = []
        all_market_news.extend(finnhub_news)
        all_market_news.extend(fmp_news)
        
        if not all_market_news:
            logger.info("No market news items from enabled sources to process.")
            return predictions

        logger.info(f"Processing {len(all_market_news)} news items from enabled sources (Finnhub: {len(finnhub_news)}, FMP: {len(fmp_news)})...")
        
        for news_item in all_market_news:
            headline = news_item.get('headline') 
            if not headline: continue

            identified_symbols = news_item.get('identified_symbols', [])
            if not identified_symbols: continue
            
            if not self._is_relevant_research_news(headline, news_item.get('summary')): continue
            
            for symbol_str in identified_symbols:
                 symbol = str(symbol_str) if symbol_str is not None else ""
                 if not symbol: continue
                 if self._is_biomed_company(symbol):
                    source_prefix = "finnhub" if news_item.get('source') != 'FMP' else "fmp"
                    self._analyze_and_store_news(news_item, symbol, predictions, source_tag=f"{source_prefix}_market")
        return predictions

    def _process_single_fda_event(
        self, 
        event: Dict[str, Any], 
        predictions_list: List[StockPrediction], 
        symbols_news_fetched_for_fda: Set[str]
    ) -> None:
        symbol_str = event.get('symbol') 
        symbol = str(symbol_str) if symbol_str is not None else ""
        if not symbol or symbol in symbols_news_fetched_for_fda: return

        if not self._is_biomed_company(symbol): return
        
        event_date_str = event.get('date')
        if not isinstance(event_date_str, str) or not event_date_str.strip():
            logger.debug(f"FDA Event for {symbol} has invalid/missing date: '{event_date_str}'.")
            return
        
        try:
            event_dt = datetime.strptime(event_date_str, '%Y-%m-%d').date()
        except ValueError: 
            logger.debug(f"Could not parse FDA event date '{event_date_str}' for {symbol}.")
            return

        today = date.today()
        event_time_window_start = today - timedelta(days=FDA_NEWS_WINDOW_PAST_DAYS + 30)
        event_time_window_end = today + timedelta(days=FDA_NEWS_WINDOW_FUTURE_DAYS + 60)

        if not (event_time_window_start <= event_dt <= event_time_window_end):
            return

        logger.info(f"[FDA_CALENDAR] Relevant event for {symbol}: {event.get('eventName', 'N/A')} on {event_date_str}")
        
        news_start_date = (event_dt - timedelta(days=FDA_NEWS_WINDOW_PAST_DAYS)).strftime('%Y-%m-%d')
        news_end_date = (event_dt + timedelta(days=FDA_NEWS_WINDOW_FUTURE_DAYS)).strftime('%Y-%m-%d')

        # Get company news from enabled sources
        finnhub_company_news = []
        fmp_company_news = []
        
        if self.finnhub:
            finnhub_company_news = self.finnhub.get_company_news(symbol, news_start_date, news_end_date)
        if self.fmp:
            fmp_company_news = self.fmp.get_company_news(symbol, news_start_date, news_end_date)
        
        all_company_news = []
        all_company_news.extend(finnhub_company_news)
        all_company_news.extend(fmp_company_news)
        
        symbols_news_fetched_for_fda.add(symbol)

        for news_item in all_company_news:
            headline = news_item.get('headline') 
            if not headline: continue
            if not self._is_relevant_research_news(headline, news_item.get('summary')): continue 
            source_prefix = "finnhub" if news_item.get('source') != 'FMP' else "fmp"
            self._analyze_and_store_news(news_item, symbol, predictions_list, source_tag=f"{source_prefix}_fda_calendar")

    def process_fda_calendar_triggered_news(self) -> List[StockPrediction]:
        predictions: List[StockPrediction] = []
        logger.info("Processing news triggered by FDA calendar events...")
        
        # FDA calendar is only available through Finnhub
        if not self.finnhub:
            logger.warning("FDA calendar processing requested but Finnhub client is not available.")
            return predictions
            
        fda_events = self.finnhub.get_fda_calendar_events()
        if not fda_events:
            return predictions
        
        symbols_news_fetched_for_fda: Set[str] = set()
        for event in fda_events:
            self._process_single_fda_event(event, predictions, symbols_news_fetched_for_fda)
        return predictions

    def process_news_for_specific_tickers(self, tickers: List[str]) -> List[StockPrediction]:
        predictions: List[StockPrediction] = []
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=max(NEWS_LOOKBACK_DAYS, 14))).strftime('%Y-%m-%d') 

        for ticker_input in tickers:
            ticker = ticker_input.strip().upper()
            if not ticker: continue

            logger.info(f"Processing news for user-specified ticker: {ticker}")
            
            if not self._is_biomed_company(ticker):
                logger.warning(f"User-specified ticker {ticker} may not be biomedical by classification. Still processing as requested.")

            # Get company news from enabled sources
            finnhub_company_news = []
            fmp_company_news = []
            
            if self.finnhub:
                finnhub_company_news = self.finnhub.get_company_news(ticker, start_date, end_date)
            if self.fmp:
                fmp_company_news = self.fmp.get_company_news(ticker, start_date, end_date)
            
            all_company_news = []
            all_company_news.extend(finnhub_company_news)
            all_company_news.extend(fmp_company_news)

            for news_item in all_company_news:
                headline = news_item.get('headline') 
                if not headline: continue
                if not self._is_relevant_research_news(headline, news_item.get('summary')): continue
                source_prefix = "finnhub" if news_item.get('source') != 'FMP' else "fmp"
                self._analyze_and_store_news(news_item, ticker, predictions, source_tag=f"{source_prefix}_specific_ticker")
        return predictions