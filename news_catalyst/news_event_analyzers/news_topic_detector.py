"""
Enhanced news topic detection with advanced NLP and caching
"""
import re
import hashlib
from typing import Tuple, List, Dict, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta

from news_event_analyzers.news_topic_keyword_lists import *
from utils.log_utils import logd, logw
from utils.performance_monitor import register_component_performance


class ConfidenceLevel(Enum):
    """Topic detection confidence levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


@dataclass
class TopicMatch:
    """Detailed topic match information"""
    topic: NewsTopic
    confidence: ConfidenceLevel
    keyword_matches: List[str] = field(default_factory=list)
    score: float = 0.0
    title_matches: int = 0
    content_matches: int = 0


class KeywordProcessor:
    """Optimized keyword processing with caching"""
    
    def __init__(self):
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._cache: Dict[str, List[str]] = {}
        self._cache_lock = threading.RLock()
        self.max_cache_size = 10000
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _compile_pattern(self, keyword: str) -> re.Pattern:
        """Compile regex pattern for keyword with caching"""
        if keyword not in self._compiled_patterns:
            # Escape special regex characters and create word boundary pattern
            escaped = re.escape(keyword.lower())
            pattern = rf'\b{escaped}\b'
            self._compiled_patterns[keyword] = re.compile(pattern, re.IGNORECASE)
        return self._compiled_patterns[keyword]
    
    def extract_keywords(self, text: str, keyword_list: List[str]) -> List[str]:
        """Extract matching keywords from text with caching"""
        cache_key = self._get_cache_key(f"{text[:100]}:{':'.join(sorted(keyword_list))}")
        
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        text_lower = text.lower()
        matches = []
        
        for keyword in keyword_list:
            # Fast substring check first
            if keyword.lower() in text_lower:
                # More precise regex check for word boundaries
                pattern = self._compile_pattern(keyword)
                if pattern.search(text_lower):
                    matches.append(keyword)
        
        # Cache result
        with self._cache_lock:
            if len(self._cache) >= self.max_cache_size:
                # Simple LRU: remove oldest entries
                oldest_keys = list(self._cache.keys())[:self.max_cache_size // 4]
                for key in oldest_keys:
                    del self._cache[key]
            
            self._cache[cache_key] = matches
        
        return matches
    
    def clear_cache(self) -> None:
        """Clear keyword extraction cache"""
        with self._cache_lock:
            self._cache.clear()


class EnhancedNewsTopicDetector:
    """Advanced news topic detector with improved accuracy and performance"""
    
    def __init__(self):
        self.keyword_processor = KeywordProcessor()
        self.financial_keywords = self._build_financial_keywords()
        self.sector_keywords = self._build_sector_keywords()
        
        # Performance tracking
        self.detection_count = 0
        self.cache_hits = 0
        self.processing_times = []
        
        # Thread safety
        self._stats_lock = threading.RLock()
    
    def _build_financial_keywords(self) -> Set[str]:
        """Build comprehensive financial keyword set"""
        return {
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
            'quarter', 'quarterly', 'annual', 'results', 'beat', 'miss',
            'upgrade', 'downgrade', 'analyst', 'target', 'price target',
            'dividend', 'buyback', 'acquisition', 'merger', 'partnership',
            'contract', 'deal', 'agreement', 'expansion', 'growth',
            'sales', 'performance', 'forecast', 'projection', 'eps',
            'ebitda', 'margin', 'cash flow', 'debt', 'equity'
        }
    
    def _build_sector_keywords(self) -> Dict[str, Set[str]]:
        """Build sector-specific keyword sets"""
        return {
            'biotech': {
                'clinical', 'trial', 'phase', 'fda', 'drug', 'therapy',
                'treatment', 'patient', 'efficacy', 'safety', 'approval',
                'regulatory', 'protocol', 'endpoint', 'adverse'
            },
            'tech': {
                'software', 'platform', 'cloud', 'ai', 'artificial intelligence',
                'machine learning', 'data', 'algorithm', 'api', 'saas'
            },
            'energy': {
                'oil', 'gas', 'renewable', 'solar', 'wind', 'energy',
                'electricity', 'power', 'grid', 'battery', 'storage'
            }
        }
    
    def _calculate_confidence_level(self, 
                                   keyword_count: int, 
                                   title_matches: int,
                                   content_matches: int,
                                   text_length: int) -> ConfidenceLevel:
        """Calculate confidence level based on multiple factors"""
        # Base score from keyword count
        base_score = keyword_count
        
        # Boost for title matches (more important)
        title_boost = title_matches * 2
        
        # Adjust for text length (longer text needs more matches)
        length_factor = max(1, text_length / 1000)  # Normalize to ~1000 chars
        adjusted_score = (base_score + title_boost) / length_factor
        
        if adjusted_score >= 5:
            return ConfidenceLevel.VERY_HIGH
        elif adjusted_score >= 3:
            return ConfidenceLevel.HIGH
        elif adjusted_score >= 2:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _check_title_keywords(self, title: str) -> Tuple[NewsTopic, List[str]]:
        """Enhanced title keyword detection with fuzzy matching"""
        title_lower = title.lower().strip()
        
        if not title_lower:
            return NewsTopic.UNKNOWN, []
        
        best_match = NewsTopic.UNKNOWN
        best_matches = []
        max_match_count = 0
        
        # Check each topic's keywords
        for topic, keyword_list in combined_topic_title_keywords.items():
            matches = self.keyword_processor.extract_keywords(title_lower, keyword_list)
            
            if len(matches) > max_match_count:
                max_match_count = len(matches)
                best_match = topic
                best_matches = matches
        
        # If no specific topic found, check for general financial patterns
        if best_match == NewsTopic.UNKNOWN:
            financial_matches = [
                kw for kw in self.financial_keywords 
                if kw in title_lower
            ]
            
            if financial_matches:
                # Determine sentiment based on context
                positive_indicators = ['beat', 'exceed', 'raise', 'increase', 'growth', 'strong', 'better', 'up']
                negative_indicators = ['miss', 'lower', 'cut', 'decline', 'weak', 'disappointing', 'down', 'fall']
                
                positive_count = sum(1 for word in positive_indicators if word in title_lower)
                negative_count = sum(1 for word in negative_indicators if word in title_lower)
                
                if positive_count > negative_count:
                    best_match = NewsTopic.POSITIVE_FINANCIAL_PERFORMANCE
                elif negative_count > positive_count:
                    best_match = NewsTopic.NEGATIVE_FINANCIAL_PERFORMANCE
                else:
                    best_match = NewsTopic.POSITIVE_FINANCIAL_PERFORMANCE  # Default optimistic
                
                best_matches = financial_matches[:3]  # Limit to top 3 matches
        
        return best_match, best_matches
    
    def _check_content_keywords(self, topic: NewsTopic, content: str) -> Tuple[int, List[str]]:
        """Enhanced content keyword detection"""
        if topic not in combined_topic_content_keywords:
            return 0, []
        
        keyword_list = combined_topic_content_keywords[topic]
        matches = self.keyword_processor.extract_keywords(content, keyword_list)
        
        # Count total occurrences (some keywords might appear multiple times)
        content_lower = content.lower()
        total_count = sum(content_lower.count(match.lower()) for match in matches)
        
        return total_count, matches
    
    def _analyze_sector_context(self, content: str) -> Optional[str]:
        """Analyze sector context for better topic classification"""
        content_lower = content.lower()
        sector_scores = {}
        
        for sector, keywords in self.sector_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                sector_scores[sector] = score
        
        if sector_scores:
            return max(sector_scores, key=sector_scores.get)
        return None
    
    def detect_topics(self, title: str, content: str) -> Tuple[NewsTopic, int]:
        """Enhanced topic detection with comprehensive analysis"""
        start_time = datetime.now()
        
        try:
            # Input validation and cleaning
            title = (title or "").strip()
            content = (content or "").strip()
            
            if not title and not content:
                return NewsTopic.UNKNOWN, 0
            
            # Combine title and content for full analysis
            full_text = f"{title} {content}".lower()
            
            # Step 1: Analyze title for primary topic
            title_topic, title_matches = self._check_title_keywords(title)
            title_match_count = len(title_matches)
            
            # Step 2: Analyze content for supporting evidence
            content_match_count = 0
            content_matches = []
            
            if title_topic != NewsTopic.UNKNOWN:
                content_match_count, content_matches = self._check_content_keywords(title_topic, full_text)
            else:
                # If no title topic found, try content analysis for all topics
                best_topic = NewsTopic.UNKNOWN
                best_content_count = 0
                best_content_matches = []
                
                for topic in combined_topic_content_keywords:
                    count, matches = self._check_content_keywords(topic, full_text)
                    if count > best_content_count:
                        best_content_count = count
                        best_content_matches = matches
                        best_topic = topic
                
                if best_topic != NewsTopic.UNKNOWN:
                    title_topic = best_topic
                    content_match_count = best_content_count
                    content_matches = best_content_matches
            
            # Step 3: Calculate final score and confidence
            total_keyword_count = title_match_count + content_match_count
            
            # Apply minimum threshold
            if total_keyword_count == 0 or title_topic == NewsTopic.UNKNOWN:
                return NewsTopic.UNKNOWN, 0
            
            # Boost score for title matches (they're more important)
            weighted_score = title_match_count * 2 + content_match_count
            
            # Step 4: Sector context validation
            sector = self._analyze_sector_context(full_text)
            if sector == 'biotech' and title_topic in biotech_topic_list:
                weighted_score = int(weighted_score * 1.2)  # 20% boost for biotech context
            
            # Update performance stats
            with self._stats_lock:
                self.detection_count += 1
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.processing_times.append(processing_time)
                
                # Keep only recent processing times
                if len(self.processing_times) > 1000:
                    self.processing_times = self.processing_times[-500:]
                
                # Register performance metrics periodically
                if self.detection_count % 100 == 0:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    register_component_performance(
                        'topic_detector',
                        detections_processed=self.detection_count,
                        avg_processing_time_ms=avg_time,
                        cache_hit_rate=self.cache_hits / max(self.detection_count, 1)
                    )
            
            logd(f"Topic detection: {title_topic.value} (score: {weighted_score}, "
                 f"title: {title_match_count}, content: {content_match_count})")
            
            return title_topic, max(1, weighted_score)  # Ensure minimum score of 1
            
        except Exception as e:
            logw(f"Error in topic detection: {e}")
            return NewsTopic.UNKNOWN, 0
    
    def detect_topics_detailed(self, title: str, content: str) -> TopicMatch:
        """Detailed topic detection with full match information"""
        topic, score = self.detect_topics(title, content)
        
        if topic == NewsTopic.UNKNOWN:
            return TopicMatch(topic=topic, confidence=ConfidenceLevel.LOW, score=0.0)
        
        # Get detailed match information
        _, title_matches = self._check_title_keywords(title)
        content_matches_count, content_matches = self._check_content_keywords(topic, f"{title} {content}")
        
        # Calculate confidence
        confidence = self._calculate_confidence_level(
            keyword_count=score,
            title_matches=len(title_matches),
            content_matches=content_matches_count,
            text_length=len(title) + len(content)
        )
        
        return TopicMatch(
            topic=topic,
            confidence=confidence,
            keyword_matches=title_matches + content_matches,
            score=float(score),
            title_matches=len(title_matches),
            content_matches=content_matches_count
        )
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        with self._stats_lock:
            if not self.processing_times:
                return {}
            
            return {
                'total_detections': self.detection_count,
                'cache_hit_rate': self.cache_hits / max(self.detection_count, 1),
                'avg_processing_time_ms': sum(self.processing_times) / len(self.processing_times),
                'max_processing_time_ms': max(self.processing_times),
                'min_processing_time_ms': min(self.processing_times)
            }
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        self.keyword_processor.clear_cache()
        with self._stats_lock:
            self.processing_times.clear()


# Maintain backward compatibility
NewsTopicDetector = EnhancedNewsTopicDetector