"""
Enhanced news quality filtering with fixed regex patterns and error handling
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime, timezone, timedelta
from typing import Set, List, Optional, Dict, Any, Pattern
from config import CONFIG
from utils.simple_logger import log_info, log_warning, log_debug


class NewsQualityFilter:
    """Enhanced news quality filter with corrected regex patterns and robust error handling."""
    
    # Fixed pre-compiled regex patterns
    _SPAM_PATTERN: Pattern = re.compile(
        r'\b(?:click here|ad:|advertisement|sponsored|promo|free trial)\b', 
        re.IGNORECASE
    )
    
    _HIGH_VALUE_PATTERN: Pattern = re.compile(
        r'\b(?:earnings|revenue|acquisition|merger|fda|approval|partnership|deal|'
        r'breakthrough|guidance|beats|misses|announces|reports|files|launches|'
        r'dividend|buyback|ipo|secondary offering|analyst|upgrade|downgrade)\b',
        re.IGNORECASE
    )
    
    # Fixed information patterns with proper escaping
    _INFO_PATTERNS = [
        re.compile(r'\$[\d,.]+', re.IGNORECASE),  # Dollar amounts
        re.compile(r'\b\d+%\b', re.IGNORECASE),   # Percentages  
        re.compile(r'\bQ[1-4]\b', re.IGNORECASE), # Quarters
        re.compile(r'\b\d{4}\b', re.IGNORECASE),  # Years
        re.compile(r'\b(?:million|billion|shares|revenue|profit)\b', re.IGNORECASE)
    ]
    
    # Fixed low quality patterns
    _LOW_QUALITY_PATTERNS = [
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
            'production': 8,  # 8 hours for production
            'testing': 48,    # 48 hours for testing
            'premium': 12     # 12 hours for premium content
        }
    
    def filter_news_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Apply enhanced quality filters with robust error handling."""
        if news_df is None or news_df.empty:
            return news_df
        
        initial_count = len(news_df)
        log_debug(f"Starting enhanced quality filtering: {initial_count} articles")
        
        try:
            # Create a copy to avoid modifying original
            df = news_df.copy()
            
            # Stage 1: Essential validation
            df = self._filter_essential_fields(df)
            if df.empty:
                log_warning("All articles filtered - missing essential fields")
                return df
            
            # Stage 2: Enhanced time filtering with proper None handling
            df = self._filter_by_time_enhanced_safe(df)
            if df.empty:
                log_warning("All articles filtered - time criteria")
                return df
            
            # Stage 3: Content quality assessment
            df = self._filter_content_quality_safe(df)
            if df.empty:
                log_warning("All articles filtered - content quality")
                return df
            
            # Stage 4: Add enhanced scoring with error handling
            df = self._add_enhanced_scoring_safe(df)
            
            # Stage 5: Remove obvious spam/low quality
            df = self._filter_spam_and_low_quality_safe(df)
            
            final_count = len(df)
            filter_rate = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
            
            log_info(f"Enhanced quality filter: {initial_count} -> {final_count} articles "
                    f"({filter_rate:.1f}% filtered)")
            
            return df
            
        except Exception as e:
            log_warning(f"Error in enhanced quality filtering: {e}")
            return news_df
    
    def _filter_essential_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter articles missing essential fields."""
        if df.empty:
            return df
        
        try:
            # Essential field requirements
            essential_mask = (
                df['symbol'].notna() & 
                (df['symbol'].str.strip() != '') &
                df['title'].notna() & 
                (df['title'].str.strip() != '') &
                (df['title'].str.len() >= 10)  # Minimum meaningful title
            )
            
            filtered_df = df[essential_mask]
            
            removed_count = len(df) - len(filtered_df)
            if removed_count > 0:
                log_debug(f"Essential fields filter removed {removed_count} articles")
            
            return filtered_df
            
        except Exception as e:
            log_warning(f"Error in essential fields filter: {e}")
            return df
    
    def _filter_by_time_enhanced_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced time filtering with safe None handling."""
        if 'publishedDate' not in df.columns or df.empty:
            return df
        
        try:
            now = datetime.now(timezone.utc)
            
            # Dynamic time window based on mode and content type
            base_hours = self._time_windows['testing' if CONFIG.testing_mode else 'production']
            
            # Convert to datetime with error handling
            published_dates = pd.to_datetime(df['publishedDate'], utc=True, errors='coerce')
            
            # Enhanced time filtering logic with safe operations
            time_mask = pd.Series(True, index=df.index)
            
            for idx, (row_idx, row) in enumerate(df.iterrows()):
                try:
                    pub_date = published_dates.iloc[idx]
                    
                    if pd.isna(pub_date) or pub_date is None:
                        # Allow articles with missing dates if they have high-value content
                        if self._has_high_value_content_safe(row):
                            continue
                        else:
                            time_mask.loc[row_idx] = False
                            continue
                    
                    # Safe datetime arithmetic
                    try:
                        age_hours = (now - pub_date).total_seconds() / 3600
                    except (TypeError, AttributeError):
                        # If datetime arithmetic fails, treat as missing date
                        if self._has_high_value_content_safe(row):
                            continue
                        else:
                            time_mask.loc[row_idx] = False
                            continue
                    
                    # Dynamic time window based on content value
                    time_limit = base_hours
                    if self._has_high_value_content_safe(row):
                        time_limit = self._time_windows['premium']  # Extended for valuable content
                    elif self._is_breaking_news_safe(row):
                        time_limit = 2  # Shorter window for breaking news
                    
                    if age_hours > time_limit:
                        time_mask.loc[row_idx] = False
                        
                except Exception as e:
                    log_debug(f"Error processing time for article {idx}: {e}")
                    # Default to keeping the article
                    continue
            
            filtered_df = df[time_mask]
            
            removed_count = len(df) - len(filtered_df)
            if removed_count > 0:
                log_debug(f"Enhanced time filter removed {removed_count} articles")
            
            return filtered_df
            
        except Exception as e:
            log_warning(f"Error in enhanced time filtering: {e}")
            return df
    
    def _filter_content_quality_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced content quality filtering with error handling."""
        if df.empty:
            return df
        
        try:
            quality_mask = pd.Series(True, index=df.index)
            
            for idx, (row_idx, row) in enumerate(df.iterrows()):
                try:
                    # High-value content always passes
                    if self._has_high_value_content_safe(row):
                        continue
                    
                    # Check text content quality
                    text = str(row.get('text', ''))
                    title = str(row.get('title', ''))
                    
                    # Quality criteria
                    quality_checks = [
                        len(text) >= 30,  # Minimum content length
                        len(title.split()) >= 3,  # Meaningful title
                        not self._is_low_quality_content_safe(title, text),
                        self._has_sufficient_information_safe(title, text)
                    ]
                    
                    if not any(quality_checks[:2]):  # Must pass basic length checks
                        quality_mask.loc[row_idx] = False
                    elif not any(quality_checks[2:]):  # Additional quality checks
                        quality_mask.loc[row_idx] = False
                        
                except Exception as e:
                    log_debug(f"Error processing quality for article {idx}: {e}")
                    # Default to keeping the article
                    continue
            
            filtered_df = df[quality_mask]
            
            removed_count = len(df) - len(filtered_df)
            if removed_count > 0:
                log_debug(f"Content quality filter removed {removed_count} articles")
            
            return filtered_df
            
        except Exception as e:
            log_warning(f"Error in content quality filtering: {e}")
            return df
    
    def _has_high_value_content_safe(self, row: pd.Series) -> bool:
        """Check if article has high-value content with error handling."""
        try:
            title = str(row.get('title', '')).lower()
            text = str(row.get('text', '')).lower()
            combined = f"{title} {text}"
            
            return bool(self._HIGH_VALUE_PATTERN.search(combined))
        except Exception:
            return False
    
    def _is_breaking_news_safe(self, row: pd.Series) -> bool:
        """Check if article is breaking news with error handling."""
        try:
            title = str(row.get('title', '')).lower()
            return any(indicator in title for indicator in ['breaking', 'just in', 'urgent', 'alert'])
        except Exception:
            return False
    
    def _is_low_quality_content_safe(self, title: str, text: str) -> bool:
        """Check for low-quality content indicators with error handling."""
        try:
            combined = f"{title} {text}".lower()
            
            # Use pre-compiled patterns
            return any(pattern.search(combined) for pattern in self._LOW_QUALITY_PATTERNS)
        except Exception:
            return False
    
    def _has_sufficient_information_safe(self, title: str, text: str) -> bool:
        """Check if content has sufficient information with error handling."""
        try:
            combined = f"{title} {text}".lower()
            
            # Use pre-compiled patterns
            return any(pattern.search(combined) for pattern in self._INFO_PATTERNS)
        except Exception:
            return False
    
    def _add_enhanced_scoring_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced priority scoring with error handling."""
        if df.empty:
            return df
        
        try:
            df = df.copy()
            df['priority_score'] = 0.5
            df['quality_score'] = 0.5
            df['relevance_score'] = 0.5
            
            for idx, (row_idx, row) in enumerate(df.iterrows()):
                try:
                    title = str(row.get('title', '')).lower()
                    text = str(row.get('text', '')).lower()
                    combined = f"{title} {text}"
                    
                    # Calculate component scores safely
                    priority = self._calculate_priority_score_safe(combined, row)
                    quality = self._calculate_quality_score_safe(combined, row)
                    relevance = self._calculate_relevance_score_safe(combined, row)
                    
                    df.at[row_idx, 'priority_score'] = priority
                    df.at[row_idx, 'quality_score'] = quality
                    df.at[row_idx, 'relevance_score'] = relevance
                    
                except Exception as e:
                    log_debug(f"Error calculating scores for article {idx}: {e}")
                    # Keep default scores
                    continue
            
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
            log_warning(f"Error in enhanced scoring: {e}")
            return df
    
    def _calculate_priority_score_safe(self, combined_text: str, row: pd.Series) -> float:
        """Calculate priority score with error handling."""
        try:
            score = 0.5
            
            # High-impact keywords
            if self._HIGH_VALUE_PATTERN.search(combined_text):
                score += 0.3
            
            # Official announcements
            official_count = sum(1 for keyword in self._official_keywords 
                               if keyword in combined_text)
            score += min(official_count * 0.1, 0.2)
            
            # Catalyst keywords
            catalyst_count = sum(1 for keyword in self._catalyst_keywords 
                               if keyword in combined_text)
            score += min(catalyst_count * 0.15, 0.25)
            
            # Recent publication bonus
            if 'publishedDate' in row:
                try:
                    pub_date = pd.to_datetime(row['publishedDate'], utc=True)
                    if pub_date is not None and not pd.isna(pub_date):
                        age_hours = (datetime.now(timezone.utc) - pub_date).total_seconds() / 3600
                        
                        if age_hours <= 1:
                            score += 0.2
                        elif age_hours <= 3:
                            score += 0.1
                        elif age_hours <= 6:
                            score += 0.05
                except Exception:
                    pass  # Skip time bonus if calculation fails
            
            return min(score, 1.0)
        except Exception:
            return 0.5
    
    def _calculate_quality_score_safe(self, combined_text: str, row: pd.Series) -> float:
        """Calculate quality score with error handling."""
        try:
            score = 0.5
            
            # Content length bonus
            text_length = len(str(row.get('text', '')))
            if text_length > 200:
                score += 0.2
            elif text_length > 100:
                score += 0.1
            
            # Information density
            if self._has_sufficient_information_safe(row.get('title', ''), row.get('text', '')):
                score += 0.2
            
            # Source credibility (if available)
            source = str(row.get('source', '')).lower()
            credible_sources = ['reuters', 'bloomberg', 'cnbc', 'marketwatch', 'sec filing']
            if any(credible in source for credible in credible_sources):
                score += 0.3
            
            return min(score, 1.0)
        except Exception:
            return 0.5
    
    def _calculate_relevance_score_safe(self, combined_text: str, row: pd.Series) -> float:
        """Calculate relevance score with error handling."""
        try:
            score = 0.5
            
            # Market-relevant terms
            market_terms = ['stock', 'share', 'trading', 'market', 'investor', 'price', 'value']
            market_count = sum(1 for term in market_terms if term in combined_text)
            score += min(market_count * 0.05, 0.2)
            
            # Financial metrics using pre-compiled patterns
            financial_matches = sum(1 for pattern in self._INFO_PATTERNS 
                                  if pattern.search(combined_text))
            score += min(financial_matches * 0.1, 0.3)
            
            return min(score, 1.0)
        except Exception:
            return 0.5
    
    def _filter_spam_and_low_quality_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final filter for spam and low quality content with error handling."""
        if df.empty:
            return df
        
        try:
            # Filter out obvious spam
            spam_mask = ~df['title'].str.contains(self._SPAM_PATTERN, na=False)
            
            # Filter out extremely low scores
            score_mask = df.get('composite_score', 0.5) >= 0.3
            
            # Combine filters
            final_mask = spam_mask & score_mask
            
            filtered_df = df[final_mask]
            
            removed_count = len(df) - len(filtered_df)
            if removed_count > 0:
                log_debug(f"Spam/low-quality filter removed {removed_count} articles")
            
            return filtered_df
            
        except Exception as e:
            log_warning(f"Error in spam/low-quality filtering: {e}")
            return df
    
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
                for score_type in ['priority_score', 'quality_score', 'relevance_score', 'composite_score']:
                    if score_type in filtered_df.columns:
                        stats[f'avg_{score_type}'] = filtered_df[score_type].mean()
                        stats[f'high_{score_type}_count'] = len(filtered_df[filtered_df[score_type] > 0.7])
                
                # Time statistics
                if 'publishedDate' in filtered_df.columns:
                    try:
                        now = datetime.now(timezone.utc)
                        ages = (now - pd.to_datetime(filtered_df['publishedDate'], utc=True, errors='coerce')).dt.total_seconds() / 3600
                        valid_ages = ages.dropna()
                        
                        if not valid_ages.empty:
                            stats.update({
                                'avg_age_hours': valid_ages.mean(),
                                'newest_age_hours': valid_ages.min(),
                                'oldest_age_hours': valid_ages.max()
                            })
                    except Exception:
                        pass  # Skip time statistics if calculation fails
            
            return stats
        except Exception as e:
            log_warning(f"Error calculating filter statistics: {e}")
            return {}