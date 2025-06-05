"""
Aggregate news articles by ticker symbol
Python 3.13.3 compatible
"""
from typing import Dict, List, Any
from collections import defaultdict
import re
from utils.simple_logger import log_info, log_debug, log_warning


class TickerAggregator:
    """Aggregate and organize news articles by ticker symbol"""
    
    def __init__(self) -> None:
        """Initialize ticker aggregator"""
        self.ticker_pattern = re.compile(r'^[A-Z]{1,5}$')
        # Updated to allow market ETFs but still exclude VIX for trading decisions
        self.excluded_tickers = {'VIX'}  # Only exclude volatility index
    
    def aggregate_by_ticker(self, articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group articles by ticker symbol with detailed logging"""
        if not articles:
            return {}
        
        ticker_buckets = defaultdict(list)
        dropped_count = 0
        drop_reasons = defaultdict(int)
        
        for article in articles:
            original_ticker = article.get('symbol', '')
            ticker = self._normalize_ticker(original_ticker)
            
            if self._is_valid_ticker(ticker):
                ticker_buckets[ticker].append(article)
            else:
                dropped_count += 1
                if not original_ticker:
                    drop_reasons['empty_symbol'] += 1
                elif not ticker:
                    drop_reasons['normalization_failed'] += 1
                elif not self.ticker_pattern.match(ticker):
                    drop_reasons['invalid_format'] += 1
                elif ticker in self.excluded_tickers:
                    drop_reasons['excluded_ticker'] += 1
                else:
                    drop_reasons['other'] += 1
        
        # Convert to regular dict and log statistics
        result = dict(ticker_buckets)
        self._log_aggregation_stats(result, dropped_count, drop_reasons)
        
        return result
    
    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol"""
        if not ticker:
            return ''
        
        # Remove common suffixes and normalize
        normalized = str(ticker).upper().strip()
        
        # Remove common suffixes
        suffixes = ['.TO', '.L', '.PA', '.DE', '.HK', '-USD', '-CAD']
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        
        return normalized
    
    def _is_valid_ticker(self, ticker: str) -> bool:
        """Validate ticker symbol"""
        if not ticker:
            return False
        
        # Must match pattern (1-5 uppercase letters)
        if not self.ticker_pattern.match(ticker):
            return False
        
        # Only exclude VIX now - allow SPY, QQQ, IWM as they are tradeable
        if ticker in self.excluded_tickers:
            return False
        
        return True
    
    def _log_aggregation_stats(self, ticker_buckets: Dict[str, List[Dict[str, Any]]], 
                              dropped_count: int, drop_reasons: Dict[str, int]) -> None:
        """Log aggregation statistics with drop analysis"""
        if not ticker_buckets:
            log_info("No valid ticker buckets created")
            return
        
        total_articles = sum(len(articles) for articles in ticker_buckets.values())
        unique_tickers = len(ticker_buckets)
        
        log_info(f"Created {unique_tickers} ticker buckets with {total_articles} total articles")
        
        if dropped_count > 0:
            log_warning(f"⚠️ Dropped {dropped_count} articles during ticker aggregation:")
            for reason, count in drop_reasons.items():
                log_warning(f"   {reason}: {count} articles")
        
        # Find tickers with most articles
        top_tickers = sorted(
            ticker_buckets.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]
        
        # Log top tickers
        for ticker, articles in top_tickers:
            log_debug(f"  {ticker}: {len(articles)} articles")
    
    def get_ticker_summary(self, ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Get summary statistics for ticker buckets"""
        if not ticker_buckets:
            return {}
        
        article_counts = [len(articles) for articles in ticker_buckets.values()]
        
        return {
            'total_tickers': len(ticker_buckets),
            'total_articles': sum(article_counts),
            'avg_articles_per_ticker': sum(article_counts) / len(article_counts),
            'max_articles_per_ticker': max(article_counts),
            'min_articles_per_ticker': min(article_counts),
            'tickers_with_multiple_articles': sum(1 for count in article_counts if count > 1)
        }
    
    def filter_by_article_count(self, ticker_buckets: Dict[str, List[Dict[str, Any]]], 
                               min_articles: int = 1, 
                               max_articles: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """Filter ticker buckets by article count"""
        filtered_buckets = {}
        
        for ticker, articles in ticker_buckets.items():
            article_count = len(articles)
            
            if min_articles <= article_count <= max_articles:
                filtered_buckets[ticker] = articles
        
        removed_count = len(ticker_buckets) - len(filtered_buckets)
        if removed_count > 0:
            log_debug(f"Filtered out {removed_count} tickers due to article count constraints")
        
        return filtered_buckets
    
    def prioritize_tickers(self, ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Prioritize tickers based on article quality and quantity"""
        if not ticker_buckets:
            return []
        
        ticker_scores = []
        
        for ticker, articles in ticker_buckets.items():
            score = self._calculate_ticker_priority_score(ticker, articles)
            ticker_scores.append((ticker, score))
        
        # Sort by score (highest first)
        ticker_scores.sort(key=lambda x: x[1], reverse=True)
        
        prioritized_tickers = [ticker for ticker, _ in ticker_scores]
        
        log_debug(f"Prioritized {len(prioritized_tickers)} tickers for analysis")
        return prioritized_tickers
    
    def _calculate_ticker_priority_score(self, ticker: str, articles: List[Dict[str, Any]]) -> float:
        """Calculate priority score for ticker based on articles"""
        if not articles:
            return 0.0
        
        score = 0.0
        
        # Article quantity factor (diminishing returns)
        article_count = len(articles)
        if article_count == 1:
            score += 1.0
        elif article_count <= 3:
            score += 1.5
        elif article_count <= 5:
            score += 2.0
        else:
            score += 2.5
        
        # Content quality factors
        high_value_keywords = [
            'earnings', 'revenue', 'guidance', 'acquisition', 'merger',
            'fda', 'approval', 'breakthrough', 'partnership', 'deal',
            'upgrade', 'downgrade', 'target', 'beats', 'misses'
        ]
        
        for article in articles:
            title = str(article.get('title', '')).lower()
            text = str(article.get('text', '')).lower()
            content = f"{title} {text}"
            
            # Keyword scoring
            keyword_matches = sum(1 for keyword in high_value_keywords if keyword in content)
            score += keyword_matches * 0.5
            
            # Source quality
            source = article.get('source', '')
            if source in ['press_release', 'earnings']:
                score += 1.0
            elif source in ['stock_news']:
                score += 0.5
        
        return score / len(articles)  # Normalize by article count