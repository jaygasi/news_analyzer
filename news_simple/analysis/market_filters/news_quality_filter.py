"""
Enhanced news quality filtering with fixed type issues and reduced complexity
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime, timezone, timedelta
from typing import Set, List, Optional, Dict, Any, Pattern, Tuple
from config import CONFIG
from utils.simple_logger import log_info, log_warning, log_debug


class NewsQualityFilter:
    """Enhanced news quality filter with proper type safety and reduced complexity."""
    
    # Pre-compiled regex patterns
    _SPAM_PATTERN: Pattern[str] = re.compile(
        r'\b(?:click here|ad:|advertisement|sponsored|promo|free trial)\b', 
        re.IGNORECASE
    )
    
    _HIGH_VALUE_PATTERN: Pattern[str] = re.compile(
        r'\b(?:earnings|revenue|acquisition|merger|fda|approval|partnership|deal|'
        r'breakthrough|guidance|beats|misses|announces|reports|files|launches|'
        r'dividend|buyback|ipo|secondary offering|analyst|upgrade|downgrade)\b',
        re.IGNORECASE
    )
    
    # Information patterns
    _INFO_PATTERNS: List[Pattern[str]] = [
        re.compile(r'\$[\d,.]+', re.IGNORECASE),
        re.compile(r'\b\d+%\b', re.IGNORECASE),
        re.compile(r'\bQ[1-4]\b', re.IGNORECASE),
        re.compile(r'\b\d{4}\b', re.IGNORECASE),
        re.compile(r'\b(?:million|billion|shares|revenue|profit)\b', re.IGNORECASE)
    ]
    
    # Low quality patterns
    _LOW_QUALITY_PATTERNS: List[Pattern[str]] = [
        re.compile(r'\b(?:click here|visit our website|subscribe now)\b', re.IGNORECASE),
        re.compile(r'\b(?:advertisement|promotional|sponsored content)\b', re.IGNORECASE),
        re.compile(r'\$\$\$|\bfree money\b|\bget rich\b', re.IGNORECASE),
        re.compile(r'\b(?:this one trick|doctors hate)\b', re.IGNORECASE)
    ]
    
    def __init__(self) -> None:
        """Initialize with optimized filtering configuration."""
        self._processed_headlines: Set[str] = set()
        self._headline_cache_limit = 500
        
        # Pre-define keyword sets for performance
        self._official_keywords = frozenset([
            'announces', 'reports', 'declares', 'files', 'receives',
            'completes', 'signs', 'launches', 'enters into', 'appoints',
            'releases', 'publishes', 'confirms', 'approves', 'issues'
        ])
        
        self._catalyst_keywords = frozenset([
            'earnings', 'guidance', 'fda', 'approval', 'merger', 'acquisition',
            'breakthrough', 'partnership', 'contract', 'deal', 'revenue',
            'dividend', 'buyback', 'ipo', 'analyst', 'upgrade', 'downgrade'
        ])
        
        # Enhanced time windows
        self._time_windows = {
            'production': 12,
            'testing': 48,
            'premium': 18
        }
    
    def filter_news_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters with proper error handling."""
        if news_df is None or news_df.empty:
            return news_df
        
        initial_count = len(news_df)
        log_debug(f"Starting quality filtering: {initial_count} articles")
        
        try:
            df = news_df.copy()
            
            # Apply filters in sequence
            df = self._filter_essential_fields(df)
            if df.empty:
                return df
            
            df = self._filter_by_time_safe(df)
            if df.empty:
                return df
            
            df = self._filter_content_quality(df)
            if df.empty:
                return df
            
            df = self._add_scoring(df)
            df = self._filter_final_quality(df)
            
            final_count = len(df)
            filter_rate = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
            
            log_info(f"Quality filter: {initial_count} -> {final_count} articles ({filter_rate:.1f}% filtered)")
            
            return df
            
        except Exception as e:
            log_warning(f"Error in quality filtering: {e}")
            return news_df
    
    def _filter_essential_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter articles missing essential fields."""
        if df.empty:
            return df
        
        try:
            essential_mask = (
                df['symbol'].notna() & 
                (df['symbol'].str.strip() != '') &
                df['title'].notna() & 
                (df['title'].str.strip() != '') &
                (df['title'].str.len() >= 5)
            )
            
            filtered_df = df[essential_mask].copy()
            
            removed_count = len(df) - len(filtered_df)
            if removed_count > 0:
                log_debug(f"Essential fields filter removed {removed_count} articles")
            
            return filtered_df
            
        except Exception as e:
            log_warning(f"Error in essential fields filter: {e}")
            return df
    
    def _filter_by_time_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by time with proper type safety."""
        if 'publishedDate' not in df.columns or df.empty:
            return df
        
        try:
            now = pd.Timestamp.now(tz='UTC')  # Use pandas Timestamp instead of datetime
            base_hours = self._time_windows['testing' if CONFIG.testing_mode else 'production']
            
            # Convert to datetime
            df = df.copy()
            df['_parsed_date'] = pd.to_datetime(df['publishedDate'], utc=True, errors='coerce')
            
            # Create time filter using vectorized operations
            time_filter = self._create_time_filter(df, now, base_hours)
            
            filtered_df = df[time_filter].drop(columns=['_parsed_date'])
            
            removed_count = len(df) - len(filtered_df)
            if removed_count > 0:
                log_debug(f"Time filter removed {removed_count} articles")
            
            return filtered_df
            
        except Exception as e:
            log_warning(f"Error in time filtering: {e}")
            return df
    
    def _create_time_filter(self, df: pd.DataFrame, now: pd.Timestamp, base_hours: int) -> pd.Series:
        """Create time filter mask using vectorized operations."""
        time_mask = pd.Series(True, index=df.index)
        
        # Handle missing dates
        missing_dates = df['_parsed_date'].isna()
        high_value_content = df.apply(self._has_high_value_content_safe, axis=1)
        
        # Keep articles with missing dates only if they have high-value content
        time_mask = time_mask & (~missing_dates | high_value_content)
        
        # Calculate age for valid dates using proper pandas operations
        valid_dates_mask = ~missing_dates
        if valid_dates_mask.any():
            # Use pandas Series subtraction for proper type handling
            valid_timestamps = df.loc[valid_dates_mask, '_parsed_date']
            ages_timedelta = valid_timestamps.rsub(now)  # This is now Timestamp - Series[Timestamp]
            ages_hours = ages_timedelta.dt.total_seconds() / 3600
            
            # Apply time limits
            time_limits_series = df.loc[valid_dates_mask].apply(
                lambda row: self._get_time_limit(row, base_hours), axis=1
            )
            
            # Update mask for articles exceeding time limits
            exceeded_mask = ages_hours > time_limits_series
            exceeded_indices = exceeded_mask[exceeded_mask].index
            time_mask.loc[exceeded_indices] = False
        
        return time_mask
    
    def _get_time_limit(self, row: pd.Series, base_hours: int) -> int:
        """Get time limit for a single article."""
        if self._has_high_value_content_safe(row):
            return self._time_windows['premium']
        elif self._is_breaking_news_safe(row):
            return 4
        return base_hours
    
    def _filter_content_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter based on content quality with reduced complexity."""
        if df.empty:
            return df
        
        try:
            # Create quality scores for each article
            quality_scores = df.apply(self._calculate_article_quality, axis=1)
            
            # Filter based on minimum quality threshold
            quality_mask = quality_scores >= 2  # Need at least 2 quality points
            
            filtered_df = df[quality_mask].copy()
            
            removed_count = len(df) - len(filtered_df)
            if removed_count > 0:
                log_debug(f"Content quality filter removed {removed_count} articles")
            
            return filtered_df
            
        except Exception as e:
            log_warning(f"Error in content quality filtering: {e}")
            return df
    
    def _calculate_article_quality(self, row: pd.Series) -> int:
        """Calculate quality score for a single article."""
        score = 0
        
        try:
            # High-value content gets automatic pass
            if self._has_high_value_content_safe(row):
                score += 4
                return score
            
            # Check basic quality criteria
            text_length = len(str(row.get('text', '')))
            title_words = len(str(row.get('title', '')).split())
            
            if text_length >= 20:
                score += 1
            if title_words >= 2:
                score += 1
            
            title = str(row.get('title', ''))
            text = str(row.get('text', ''))
            
            if not self._is_low_quality_content_safe(title, text):
                score += 1
            if self._has_sufficient_information_safe(title, text):
                score += 1
            
        except Exception:
            pass  # Return 0 score for problematic articles
        
        return score
    
    def _add_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add scoring columns with proper error handling."""
        if df.empty:
            return df
        
        try:
            df = df.copy()
            
            # Calculate scores using vectorized operations where possible
            df['priority_score'] = df.apply(self._calc_priority_score, axis=1)
            df['quality_score'] = df.apply(self._calc_quality_score, axis=1)
            df['relevance_score'] = df.apply(self._calc_relevance_score, axis=1)
            
            # Calculate composite score
            df['composite_score'] = (
                df['priority_score'] * 0.4 +
                df['quality_score'] * 0.3 +
                df['relevance_score'] * 0.3
            )
            
            # Sort by composite score
            df = df.sort_values(['composite_score', 'publishedDate'], 
                               ascending=[False, False])
            
            return df
            
        except Exception as e:
            log_warning(f"Error in scoring: {e}")
            return df
    
    def _calc_priority_score(self, row: pd.Series) -> float:
        """Calculate priority score for a single row."""
        try:
            score = 0.5
            combined_text = f"{row.get('title', '')} {row.get('text', '')}".lower()
            
            if self._HIGH_VALUE_PATTERN.search(combined_text):
                score += 0.3
            
            official_count = sum(1 for keyword in self._official_keywords 
                               if keyword in combined_text)
            score += min(official_count * 0.1, 0.2)
            
            catalyst_count = sum(1 for keyword in self._catalyst_keywords 
                               if keyword in combined_text)
            score += min(catalyst_count * 0.15, 0.25)
            
            # Time bonus
            score += self._get_time_bonus(row)
            
            return min(score, 1.0)
        except Exception:
            return 0.5
    
    def _get_time_bonus(self, row: pd.Series) -> float:
        """Get time-based bonus for recent articles."""
        try:
            if 'publishedDate' not in row:
                return 0.0
            
            pub_date = pd.to_datetime(row['publishedDate'], utc=True)
            if pd.isna(pub_date):
                return 0.0
            
            now = pd.Timestamp.now(tz='UTC')
            age_timedelta = now - pub_date  # Both are now pandas Timestamps
            age_hours = age_timedelta.total_seconds() / 3600
            
            if age_hours <= 2:
                return 0.2
            elif age_hours <= 6:
                return 0.15
            elif age_hours <= 12:
                return 0.1
            elif age_hours <= 24:
                return 0.05
            return 0.0
        except Exception:
            return 0.0
    
    def _calc_quality_score(self, row: pd.Series) -> float:
        """Calculate quality score for a single row."""
        try:
            score = 0.5
            
            text_length = len(str(row.get('text', '')))
            if text_length > 150:
                score += 0.2
            elif text_length > 50:
                score += 0.1
            
            title = str(row.get('title', ''))
            text = str(row.get('text', ''))
            if self._has_sufficient_information_safe(title, text):
                score += 0.2
            
            source = str(row.get('source', '')).lower()
            credible_sources = ['reuters', 'bloomberg', 'cnbc', 'marketwatch', 'sec filing', 'press_release']
            if any(credible in source for credible in credible_sources):
                score += 0.3
            
            return min(score, 1.0)
        except Exception:
            return 0.5
    
    def _calc_relevance_score(self, row: pd.Series) -> float:
        """Calculate relevance score for a single row."""
        try:
            score = 0.5
            combined_text = f"{row.get('title', '')} {row.get('text', '')}".lower()
            
            market_terms = ['stock', 'share', 'trading', 'market', 'investor', 'price', 'value']
            market_count = sum(1 for term in market_terms if term in combined_text)
            score += min(market_count * 0.05, 0.2)
            
            financial_matches = sum(1 for pattern in self._INFO_PATTERNS 
                                  if pattern.search(combined_text))
            score += min(financial_matches * 0.1, 0.3)
            
            return min(score, 1.0)
        except Exception:
            return 0.5
    
    def _filter_final_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final quality filters."""
        if df.empty:
            return df
        
        try:
            # Filter out obvious spam
            spam_mask = ~df['title'].str.contains(self._SPAM_PATTERN, na=False)
            
            # More lenient score threshold
            score_mask = df.get('composite_score', 0.5) >= 0.2
            
            # Combine filters
            final_mask = spam_mask & score_mask
            
            filtered_df = df[final_mask].copy()
            
            removed_count = len(df) - len(filtered_df)
            if removed_count > 0:
                log_debug(f"Final filter removed {removed_count} articles")
            
            return filtered_df
            
        except Exception as e:
            log_warning(f"Error in final filtering: {e}")
            return df
    
    def _has_high_value_content_safe(self, row: pd.Series) -> bool:
        """Check if article has high-value content."""
        try:
            title = str(row.get('title', '')).lower()
            text = str(row.get('text', '')).lower()
            combined = f"{title} {text}"
            
            return bool(self._HIGH_VALUE_PATTERN.search(combined))
        except Exception:
            return False
    
    def _is_breaking_news_safe(self, row: pd.Series) -> bool:
        """Check if article is breaking news."""
        try:
            title = str(row.get('title', '')).lower()
            return any(indicator in title for indicator in ['breaking', 'just in', 'urgent', 'alert'])
        except Exception:
            return False
    
    def _is_low_quality_content_safe(self, title: str, text: str) -> bool:
        """Check for low-quality content indicators."""
        try:
            combined = f"{title} {text}".lower()
            return any(pattern.search(combined) for pattern in self._LOW_QUALITY_PATTERNS)
        except Exception:
            return False
    
    def _has_sufficient_information_safe(self, title: str, text: str) -> bool:
        """Check if content has sufficient information."""
        try:
            combined = f"{title} {text}".lower()
            info_count = sum(1 for pattern in self._INFO_PATTERNS if pattern.search(combined))
            return info_count >= 1
        except Exception:
            return False
    
    def get_filter_statistics(self, original_df: pd.DataFrame, 
                            filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive filtering statistics."""
        if original_df.empty:
            return {}
        
        try:
            stats = {
                'original_count': len(original_df),
                'filtered_count': len(filtered_df),
                'filter_rate': (len(original_df) - len(filtered_df)) / len(original_df),
                'pass_rate': len(filtered_df) / len(original_df)
            }
            
            if not filtered_df.empty:
                # Score statistics
                score_columns = ['priority_score', 'quality_score', 'relevance_score', 'composite_score']
                for score_type in score_columns:
                    if score_type in filtered_df.columns:
                        stats[f'avg_{score_type}'] = filtered_df[score_type].mean()
                        stats[f'high_{score_type}_count'] = len(filtered_df[filtered_df[score_type] > 0.7])
                
                # Time statistics
                stats.update(self._calculate_time_stats(filtered_df))
            
            return stats
        except Exception as e:
            log_warning(f"Error calculating filter statistics: {e}")
            return {}
    
    def _calculate_time_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate time-related statistics with proper datetime handling."""
        try:
            if 'publishedDate' not in df.columns:
                return {}
            
            now = pd.Timestamp.now(tz='UTC')
            pub_dates = pd.to_datetime(df['publishedDate'], utc=True, errors='coerce')
            valid_dates = pub_dates.dropna()
            
            if valid_dates.empty:
                return {}
            
            # Use pandas operations for datetime arithmetic
            ages_timedelta = valid_dates.rsub(now)
            ages_hours = ages_timedelta.dt.total_seconds() / 3600
            
            return {
                'avg_age_hours': float(ages_hours.mean()),
                'newest_age_hours': float(ages_hours.min()),
                'oldest_age_hours': float(ages_hours.max())
            }
        except Exception:
            return {}