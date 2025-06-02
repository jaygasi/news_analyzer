"""
News quality filtering with extremely relaxed filters for maximum coverage
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime, timezone, timedelta
from typing import Set, List, Optional, Dict, Any
from config import CONFIG
from utils.simple_logger import log_info, log_warning, log_debug


class NewsQualityFilter:
    """Filter news articles with extremely relaxed criteria for maximum coverage"""
    
    def __init__(self) -> None:
        """Initialize with very permissive filtering."""
        self._processed_headlines: Set[str] = set()
        self._headline_cache_limit = 400
        
        # Pre-compile regex patterns with fixed group issues
        self._spam_pattern = re.compile(
            r'\b(?:click here|ad:|advertisement|sponsored|promo)\b', 
            re.IGNORECASE
        )
        
        # Patterns that should DEFINITELY be preserved (high value content)
        self._high_value_pattern = re.compile(
            r'\b(?:earnings|revenue|acquisition|merger|fda|approval|partnership|deal|breakthrough|guidance|beats|misses|announces|reports|files|launches)\b',
            re.IGNORECASE
        )
        
        # Official keywords for priority scoring
        self._official_keywords = frozenset([
            'announces', 'reports', 'declares', 'files', 'receives',
            'completes', 'signs', 'launches', 'enters into', 'appoints',
            'releases', 'publishes', 'confirms', 'approves', 'issues'
        ])
        
        # High-value catalyst keywords
        self._catalyst_keywords = frozenset([
            'earnings', 'guidance', 'fda', 'approval', 'merger', 'acquisition',
            'breakthrough', 'partnership', 'contract', 'deal', 'revenue'
        ])
    
    def filter_news_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Apply extremely relaxed filters for maximum article coverage."""
        if news_df is None or news_df.empty:
            return news_df
        
        initial_count = len(news_df)
        log_debug(f"Starting news quality filtering with {initial_count} articles")
        
        try:
            # Apply only essential filters
            filtered_df = news_df.copy()
            
            # Stage 1: Only absolute essentials
            filtered_df = self._filter_absolute_essentials(filtered_df)
            if filtered_df.empty:
                log_warning("All articles filtered at essentials stage")
                self._debug_essentials_failures(news_df)
                return filtered_df
            
            # Stage 2: Very relaxed time filter
            filtered_df = self._filter_time_very_relaxed(filtered_df)
            if filtered_df.empty:
                log_warning("All articles filtered at time stage")
                self._debug_time_failures(news_df)
                return filtered_df
            
            # Stage 3: Minimal content filter
            filtered_df = self._filter_content_minimal(filtered_df)
            if filtered_df.empty:
                log_warning("All articles filtered at content stage")
                self._debug_content_failures(news_df)
                return filtered_df
            
            # Stage 4: Priority scoring (doesn't filter, just scores)
            filtered_df = self._add_priority_scoring(filtered_df)
            
            final_count = len(filtered_df)
            filter_rate = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
            
            log_info(f"News quality filter: {initial_count} -> {final_count} articles "
                    f"({filter_rate:.1f}% filtered out)")
            
            return filtered_df
            
        except Exception as e:
            log_error(f"Error in news filtering: {e}")
            return news_df
    
    def _filter_absolute_essentials(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter only the absolute essentials - must have symbol and title."""
        if news_df.empty:
            return news_df
        
        try:
            mask = pd.Series(True, index=news_df.index)
            
            # Must have symbol
            if 'symbol' in news_df.columns:
                symbol_mask = news_df['symbol'].notna() & (news_df['symbol'].str.strip() != '')
                mask &= symbol_mask
                log_debug(f"Symbol filter: {symbol_mask.sum()}/{len(news_df)} articles have valid symbols")
            
            # Must have title
            if 'title' in news_df.columns:
                title_mask = news_df['title'].notna() & (news_df['title'].str.strip() != '')
                mask &= title_mask
                log_debug(f"Title filter: {title_mask.sum()}/{len(news_df)} articles have valid titles")
            
            filtered_df = news_df[mask]
            
            removed = len(news_df) - len(filtered_df)
            if removed > 0:
                log_debug(f"Essentials filter removed {removed} articles")
            
            return filtered_df
            
        except Exception as e:
            log_warning(f"Error in essentials filter: {e}")
            return news_df
    
    def _filter_time_very_relaxed(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Apply very relaxed time filtering."""
        if 'publishedDate' not in news_df.columns or news_df.empty:
            return news_df
        
        try:
            now = datetime.now(timezone.utc)
            # VERY relaxed time windows
            cutoff_hours = 72 if CONFIG.testing_mode else 12.0  # 12 hours in production, 72 in testing
            cutoff_time = now - timedelta(hours=cutoff_hours)
            
            log_debug(f"Time filter: accepting articles newer than {cutoff_hours} hours")
            
            try:
                published_dates = pd.to_datetime(news_df['publishedDate'], utc=True, errors='coerce')
                
                # Check for articles with invalid dates
                valid_dates_mask = published_dates.notna()
                log_debug(f"Valid dates: {valid_dates_mask.sum()}/{len(news_df)} articles")
                
                # Time filter only applies to articles with valid dates
                time_mask = (published_dates.isna()) | (published_dates > cutoff_time)
                
                filtered_df = news_df[time_mask]
                
                removed = len(news_df) - len(filtered_df)
                if removed > 0:
                    log_debug(f"Time filter removed {removed} articles (older than {cutoff_hours}h)")
                
                return filtered_df
                
            except Exception as e:
                log_warning(f"Error in time comparison: {e}, keeping all articles")
                return news_df
            
        except Exception as e:
            log_warning(f"Error in time filtering: {e}")
            return news_df
    
    def _filter_content_minimal(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Apply minimal content filtering - preserve high-value content."""
        if news_df.empty:
            return news_df
        
        try:
            mask = pd.Series(True, index=news_df.index)
            
            # Check for high-value content first
            high_value_mask = pd.Series(False, index=news_df.index)
            
            if 'title' in news_df.columns:
                title_high_value = news_df['title'].str.contains(self._high_value_pattern, na=False, regex=True)
                high_value_mask |= title_high_value
                log_debug(f"High-value titles: {title_high_value.sum()}/{len(news_df)} articles")
            
            if 'text' in news_df.columns:
                text_high_value = news_df['text'].str.contains(self._high_value_pattern, na=False, regex=True)
                high_value_mask |= text_high_value
                log_debug(f"High-value text: {text_high_value.sum()}/{len(news_df)} articles")
            
            log_debug(f"Total high-value articles: {high_value_mask.sum()}/{len(news_df)}")
            
            # For non-high-value content, apply minimal filters
            if 'text' in news_df.columns:
                # Very minimal text length requirement
                basic_text_mask = news_df['text'].str.len() >= 20  # Very low minimum
                log_debug(f"Basic text length (>=20): {basic_text_mask.sum()}/{len(news_df)} articles")
                
                # Combine: high-value content OR meets basic requirements
                mask &= high_value_mask | basic_text_mask
            else:
                # If no text column, just use high-value filter
                mask &= high_value_mask
            
            # Only filter extreme cases
            if 'title' in news_df.columns:
                # Only filter extremely short titles
                title_length_mask = news_df['title'].str.len() >= 5
                mask &= title_length_mask
                log_debug(f"Title length (>=5): {title_length_mask.sum()}/{len(news_df)} articles")
            
            # Very minimal spam filtering - only obvious spam
            if 'title' in news_df.columns:
                obvious_spam_mask = ~news_df['title'].str.contains(r'\b(?:FREE|WIN NOW|CLICK HERE)\b', na=False, regex=True, flags=re.IGNORECASE)
                mask &= obvious_spam_mask
                log_debug(f"Non-obvious spam: {obvious_spam_mask.sum()}/{len(news_df)} articles")
            
            filtered_df = news_df[mask]
            
            removed = len(news_df) - len(filtered_df)
            if removed > 0:
                log_debug(f"Content filter removed {removed} articles")
            
            return filtered_df
            
        except Exception as e:
            log_warning(f"Error in content filtering: {e}")
            return news_df
    
    def _add_priority_scoring(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Add priority scores without filtering."""
        if news_df.empty or 'title' not in news_df.columns:
            return news_df
        
        try:
            news_df = news_df.copy()
            news_df['priority_score'] = 0.5
            
            # Combine title and text for scoring
            title_text = news_df['title'].str.lower()
            if 'text' in news_df.columns:
                text_content = news_df['text'].str.lower()
                combined_text = title_text + ' ' + text_content
            else:
                combined_text = title_text
            
            # High-value keywords get big bonus
            high_value_mask = combined_text.str.contains(self._high_value_pattern, na=False, regex=True)
            news_df.loc[high_value_mask, 'priority_score'] += 0.30
            
            # Official keywords
            for keyword in self._official_keywords:
                keyword_pattern = fr'\b{re.escape(keyword)}\b'
                keyword_mask = combined_text.str.contains(keyword_pattern, na=False, regex=True)
                news_df.loc[keyword_mask, 'priority_score'] += 0.10
            
            # Catalyst keywords
            for keyword in self._catalyst_keywords:
                keyword_pattern = fr'\b{re.escape(keyword)}\b'
                keyword_mask = combined_text.str.contains(keyword_pattern, na=False, regex=True)
                news_df.loc[keyword_mask, 'priority_score'] += 0.15
            
            # Recency bonus
            if 'publishedDate' in news_df.columns:
                try:
                    now_pd = pd.Timestamp.now(tz=timezone.utc)
                    published_dates = pd.to_datetime(news_df['publishedDate'], utc=True, errors='coerce')
                    
                    valid_dates = published_dates.notna()
                    if valid_dates.any():
                        age_timedelta = now_pd - published_dates
                        age_hours = age_timedelta.dt.total_seconds() / 3600
                        
                        # Recency bonuses
                        recent_1h_mask = (age_hours <= 1.0) & valid_dates
                        recent_3h_mask = (age_hours > 1.0) & (age_hours <= 3.0) & valid_dates
                        recent_6h_mask = (age_hours > 3.0) & (age_hours <= 6.0) & valid_dates
                        
                        news_df.loc[recent_1h_mask, 'priority_score'] += 0.25
                        news_df.loc[recent_3h_mask, 'priority_score'] += 0.15
                        news_df.loc[recent_6h_mask, 'priority_score'] += 0.10
                        
                except Exception as e:
                    log_debug(f"Error in recency scoring: {e}")
            
            # Clip scores
            news_df['priority_score'] = news_df['priority_score'].clip(upper=1.0)
            
            # Sort by priority
            sort_columns = ['priority_score']
            if 'publishedDate' in news_df.columns:
                sort_columns.append('publishedDate')
            
            sorted_df = news_df.sort_values(sort_columns, ascending=[False, False])
            
            return sorted_df
            
        except Exception as e:
            log_warning(f"Error in priority scoring: {e}")
            return news_df
    
    def _debug_essentials_failures(self, news_df: pd.DataFrame) -> None:
        """Debug why articles fail essentials filter."""
        log_warning("=== DEBUGGING ESSENTIALS FAILURES ===")
        
        if 'symbol' in news_df.columns:
            null_symbols = news_df['symbol'].isna().sum()
            empty_symbols = (news_df['symbol'].str.strip() == '').sum()
            log_warning(f"Symbol issues: {null_symbols} null, {empty_symbols} empty")
        
        if 'title' in news_df.columns:
            null_titles = news_df['title'].isna().sum()
            empty_titles = (news_df['title'].str.strip() == '').sum()
            log_warning(f"Title issues: {null_titles} null, {empty_titles} empty")
    
    def _debug_time_failures(self, news_df: pd.DataFrame) -> None:
        """Debug why articles fail time filter."""
        log_warning("=== DEBUGGING TIME FAILURES ===")
        
        if 'publishedDate' not in news_df.columns:
            log_warning("No publishedDate column")
            return
        
        try:
            now = datetime.now(timezone.utc)
            cutoff_hours = 72 if CONFIG.testing_mode else 12.0
            cutoff_time = now - timedelta(hours=cutoff_hours)
            
            published_dates = pd.to_datetime(news_df['publishedDate'], utc=True, errors='coerce')
            
            null_dates = published_dates.isna().sum()
            valid_dates = published_dates.notna().sum()
            
            if valid_dates > 0:
                ages = (now - published_dates).dt.total_seconds() / 3600
                too_old = (ages > cutoff_hours).sum()
                
                log_warning(f"Date analysis: {null_dates} invalid, {valid_dates} valid, {too_old} too old (>{cutoff_hours}h)")
                
                if too_old > 0:
                    oldest_age = ages.max()
                    newest_age = ages.min()
                    log_warning(f"Age range: {newest_age:.1f}h to {oldest_age:.1f}h")
            else:
                log_warning("No valid dates found")
                
        except Exception as e:
            log_warning(f"Error in time debug: {e}")
    
    def _debug_content_failures(self, news_df: pd.DataFrame) -> None:
        """Debug why articles fail content filter."""
        log_warning("=== DEBUGGING CONTENT FAILURES ===")
        
        # Check high-value content
        high_value_count = 0
        if 'title' in news_df.columns:
            title_high_value = news_df['title'].str.contains(self._high_value_pattern, na=False, regex=True).sum()
            high_value_count += title_high_value
            log_warning(f"High-value titles: {title_high_value}")
        
        if 'text' in news_df.columns:
            text_high_value = news_df['text'].str.contains(self._high_value_pattern, na=False, regex=True).sum()
            high_value_count += text_high_value
            log_warning(f"High-value text: {text_high_value}")
            
            # Text length analysis
            text_lengths = news_df['text'].str.len()
            too_short = (text_lengths < 20).sum()
            log_warning(f"Text length issues: {too_short} articles < 20 chars")
            
            if too_short > 0:
                min_length = text_lengths.min()
                max_length = text_lengths.max()
                avg_length = text_lengths.mean()
                log_warning(f"Text length range: {min_length} to {max_length}, avg {avg_length:.1f}")
        
        log_warning(f"Total high-value content: {high_value_count}")
    
    def get_filter_statistics(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed statistics about the filtering process."""
        if original_df.empty:
            return {}
        
        stats = {
            'original_count': len(original_df),
            'filtered_count': len(filtered_df),
            'filter_rate': (len(original_df) - len(filtered_df)) / len(original_df) if len(original_df) > 0 else 0,
            'pass_rate': len(filtered_df) / len(original_df) if len(original_df) > 0 else 0,
        }
        
        if len(filtered_df) > 0:
            if 'priority_score' in filtered_df.columns:
                stats['avg_priority_score'] = filtered_df['priority_score'].mean()
                stats['high_priority_count'] = len(filtered_df[filtered_df['priority_score'] > 0.7])
            
            if 'publishedDate' in filtered_df.columns:
                try:
                    now = pd.Timestamp.now(tz=timezone.utc)
                    published_dates = pd.to_datetime(filtered_df['publishedDate'], utc=True, errors='coerce')
                    valid_dates = published_dates.notna()
                    
                    if valid_dates.any():
                        ages = (now - published_dates[valid_dates]).dt.total_seconds() / 3600
                        stats['avg_age_hours'] = ages.mean()
                        stats['newest_age_hours'] = ages.min()
                        stats['oldest_age_hours'] = ages.max()
                except:
                    pass
        
        return stats