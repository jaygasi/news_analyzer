"""
News quality filtering and processing
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime, timezone, timedelta
from typing import Set, List
from config import CONFIG
from utils.simple_logger import log_info, log_warning, log_debug


class NewsQualityFilter:
    """Filter news articles based on quality criteria"""
    
    def __init__(self) -> None:
        """Initialize with optimized data structures"""
        self._processed_headlines: Set[str] = set()
        self._headline_cache_limit = 400
        
        # Pre-compile regex patterns
        self._spam_pattern = re.compile(
            r'\b(click here|ad:|advertisement|sponsored)\b', 
            re.IGNORECASE
        )
        
        # Official keywords for priority scoring
        self._official_keywords = {
            'announces', 'reports', 'declares', 'files', 'receives',
            'completes', 'signs', 'launches', 'enters into', 'appoints'
        }
    
    def filter_news_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive news quality filters"""
        if news_df is None or news_df.empty:
            return news_df
        
        filtered_df = news_df.copy()
        filtered_df = self._filter_stale_news(filtered_df)
        filtered_df = self._deduplicate_headlines(filtered_df)
        filtered_df = self._filter_content_quality(filtered_df)
        filtered_df = self._add_priority_scoring(filtered_df)
        
        return filtered_df
    
    def _filter_stale_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out stale news articles"""
        if 'publishedDate' not in news_df.columns:
            return news_df
        
        try:
            now = datetime.now(timezone.utc)
            cutoff_hours = 20 if CONFIG.testing_mode else 1.5
            cutoff_time = now - timedelta(hours=cutoff_hours)
            
            if CONFIG.testing_mode:
                log_debug("[TESTING MODE] Extended news freshness to 20 hours")
            
            time_mask = news_df['publishedDate'] > cutoff_time
            return news_df[time_mask]
            
        except Exception as e:
            log_warning(f"Error filtering stale news: {e}")
            return news_df
    
    def _deduplicate_headlines(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate headlines"""
        if 'title' not in news_df.columns or news_df.empty:
            return news_df
        
        try:
            # Simple exact deduplication first
            news_df = news_df.drop_duplicates(subset=['title'], keep='first')
            
            # For smaller datasets, do similarity check
            if len(news_df) <= 30:
                return self._similarity_deduplication(news_df)
            
            return news_df
            
        except Exception as e:
            log_warning(f"Error in headline deduplication: {e}")
            return news_df
    
    def _similarity_deduplication(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Remove similar headlines using word overlap"""
        if len(news_df) <= 1:
            return news_df
        
        filtered_indices = []
        seen_word_sets = []
        
        for idx, row in news_df.iterrows():
            title = str(row['title']).lower().strip()
            title_words = set(title.split())
            
            # Skip very short titles
            if len(title_words) < 3:
                continue
            
            if not self._is_similar_to_seen(title_words, seen_word_sets):
                filtered_indices.append(idx)
                seen_word_sets.append(title_words)
                
                # Limit memory usage
                if len(seen_word_sets) > self._headline_cache_limit:
                    seen_word_sets = seen_word_sets[-self._headline_cache_limit//2:]
        
        return news_df.loc[filtered_indices]
    
    def _is_similar_to_seen(self, title_words: set, seen_word_sets: List[set]) -> bool:
        """Check if title is similar to previously seen titles"""
        for seen_words in seen_word_sets:
            if title_words and seen_words:
                intersection_size = len(title_words & seen_words)
                if intersection_size > 0:
                    union_size = len(title_words | seen_words)
                    if union_size > 0 and intersection_size / union_size > 0.75:
                        return True
        return False
    
    def _filter_content_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter based on content quality criteria"""
        if news_df.empty:
            return news_df
        
        try:
            mask = pd.Series([True] * len(news_df), index=news_df.index)
            
            # Title length filter
            if 'title' in news_df.columns:
                mask &= news_df['title'].str.len() >= 15
                mask &= ~news_df['title'].str.contains(self._spam_pattern, na=False)
            
            # Text length filter
            if 'text' in news_df.columns:
                mask &= news_df['text'].str.len() >= 80
            
            return news_df[mask]
            
        except Exception as e:
            log_warning(f"Error in content quality filtering: {e}")
            return news_df
    
    def _add_priority_scoring(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Add priority scores to news articles"""
        if news_df.empty or 'title' not in news_df.columns:
            return news_df
        
        try:
            news_df = news_df.copy()
            news_df['priority_score'] = 0.5
            
            # Keyword scoring
            title_lower = news_df['title'].str.lower()
            
            for keyword in self._official_keywords:
                keyword_mask = title_lower.str.contains(keyword, na=False, regex=False)
                news_df.loc[keyword_mask, 'priority_score'] += 0.08
            
            # Clip priority scores
            news_df['priority_score'] = news_df['priority_score'].clip(upper=1.0)
            
            # Sort by priority and date
            sort_columns = ['priority_score']
            if 'publishedDate' in news_df.columns:
                sort_columns.append('publishedDate')
            
            return news_df.sort_values(sort_columns, ascending=[False, False])
            
        except Exception as e:
            log_warning(f"Error in priority scoring: {e}")
            return news_df