"""
Earnings Call Transcript Fetcher and Analyzer
Fetches and processes actual earnings call transcripts from FMP API
Python 3.13.3 compatible
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_error, log_debug, log_warning
import json
from pathlib import Path
import time

@dataclass
class EarningsTranscriptSegment:
    """Structured segment of earnings call transcript"""
    speaker: str
    role: str  # 'executive', 'analyst', 'operator'
    content: str
    sentiment_keywords: List[str]
    timestamp: Optional[str] = None
    segment_type: str = 'general'  # 'opening', 'results', 'guidance', 'qa', 'closing'


@dataclass
class EarningsAnalysis:
    """Comprehensive earnings call analysis"""
    ticker: str
    date: str
    quarter: str
    year: str
    
    # Transcript segments
    executive_segments: List[EarningsTranscriptSegment]
    analyst_segments: List[EarningsTranscriptSegment]
    
    # Key metrics extracted
    guidance_mentions: List[str]
    financial_metrics: Dict[str, str]
    forward_looking_statements: List[str]
    risk_factors: List[str]
    
    # Sentiment analysis
    overall_sentiment: str  # 'positive', 'negative', 'neutral'
    sentiment_confidence: float
    management_tone: str  # 'confident', 'cautious', 'defensive'
    
    # Summary
    key_highlights: List[str]
    analyst_concerns: List[str]
    
    # Raw data
    full_transcript: str
    transcript_length: int
    @property
    def direction(self) -> str:
        """Convert overall_sentiment to trading direction"""
        sentiment_to_direction = {
            'positive': 'BUY',
            'negative': 'SELL', 
            'neutral': 'NEUTRAL'
        }
        return sentiment_to_direction.get(self.overall_sentiment, 'NEUTRAL')
    
    @property
    def confidence(self) -> float:
        """Return sentiment confidence as trading confidence"""
        return self.sentiment_confidence
    
    @property
    def overall_score(self) -> float:
        """Calculate normalized trading score (-1.0 to 1.0)"""
        if self.overall_sentiment == 'positive':
            return self.sentiment_confidence  # 0.0 to 1.0
        elif self.overall_sentiment == 'negative':
            return -self.sentiment_confidence  # -1.0 to 0.0
        else:  # neutral
            return 0.0
    
    def get_reasoning(self) -> str:
        """Generate reasoning string for trading decisions"""
        parts = []
        
        # Sentiment component
        parts.append(f"Sentiment: {self.overall_sentiment} ({self.sentiment_confidence:.2f} confidence)")
        
        # Management tone
        if self.management_tone != 'neutral':
            parts.append(f"Management tone: {self.management_tone}")
        
        # Key highlights
        if self.key_highlights:
            highlights_summary = f"{len(self.key_highlights)} key highlights"
            parts.append(f"Highlights: {highlights_summary}")
        
        # Guidance mentions  
        if self.guidance_mentions:
            guidance_summary = f"{len(self.guidance_mentions)} guidance items"
            parts.append(f"Guidance: {guidance_summary}")
        
        return " | ".join(parts)
    
    @classmethod
    def create_empty_analysis(cls, ticker: str, quarter: str, year: str) -> 'EarningsAnalysis':
        """Create an empty EarningsAnalysis with default values"""
        return cls(
            ticker=ticker,
            date=datetime.now().isoformat(),
            quarter=quarter,
            year=year,
            executive_segments=[],
            analyst_segments=[],
            guidance_mentions=[],
            financial_metrics={},
            forward_looking_statements=[],
            risk_factors=[],
            overall_sentiment='neutral',
            sentiment_confidence=0.0,
            management_tone='neutral',
            key_highlights=[],
            analyst_concerns=[],
            full_transcript='',
            transcript_length=0
        )


class EarningsTranscriptFetcher(BaseFMPLoader):
    """Fetch and analyze earnings call transcripts from FMP API"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        
        # Initialize financial keyword dictionaries
        self._init_financial_keywords()
        
        # Speaker role identification patterns
        self._init_speaker_patterns()
        self.transcript_cache = {}  # Cache for successful transcripts
        self.failed_transcript_cache = set()  # Cache for failed fetches
        self.cache_file = Path("data/transcript_cache.json")
        self.failed_cache_file = Path("data/failed_transcripts.json")
        self._load_caches()
    
    def _init_financial_keywords(self):
        """Initialize financial keyword categories for analysis"""
        
        self.sentiment_keywords = {
            'positive': [
                'growth', 'strong', 'excellent', 'outperform', 'beat', 'exceed',
                'robust', 'solid', 'impressive', 'momentum', 'expansion',
                'opportunity', 'optimistic', 'confident', 'record', 'milestone',
                'breakthrough', 'success', 'achievement', 'profitable', 'efficient'
            ],
            'negative': [
                'decline', 'weak', 'poor', 'underperform', 'miss', 'below',
                'challenging', 'difficult', 'pressure', 'concern', 'risk',
                'uncertainty', 'volatile', 'disappointing', 'struggle', 'setback',
                'headwind', 'obstacle', 'competitive', 'margin compression', 'softness'
            ],
            'neutral': [
                'stable', 'consistent', 'maintain', 'continue', 'expect',
                'forecast', 'guidance', 'outlook', 'plan', 'strategy'
            ]
        }
        
        self.guidance_keywords = [
            'guidance', 'outlook', 'forecast', 'expect', 'anticipate',
            'project', 'target', 'goal', 'estimate', 'range',
            'full year', 'next quarter', 'going forward', 'in the future'
        ]
        
        self.financial_metric_keywords = [
            'revenue', 'sales', 'earnings', 'profit', 'margin', 'ebitda',
            'cash flow', 'debt', 'equity', 'return on investment', 'roi',
            'market share', 'growth rate', 'expenses', 'costs'
        ]
        
        self.risk_keywords = [
            'risk', 'uncertainty', 'challenge', 'headwind', 'competitive',
            'regulatory', 'market conditions', 'economic environment',
            'supply chain', 'inflation', 'interest rates', 'geopolitical'
        ]
    
    def _init_speaker_patterns(self):
        """Initialize patterns for identifying speaker roles"""
        
        self.executive_titles = [
            'ceo', 'chief executive', 'president', 'cfo', 'chief financial',
            'coo', 'chief operating', 'cto', 'chief technology', 'chairman',
            'founder', 'managing director', 'general manager'
        ]
        
        self.analyst_patterns = [
            'analyst', 'research', 'bank', 'securities', 'capital markets',
            'investment', 'equity research', 'senior analyst'
        ]
    
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
            # Create EarningsAnalysis from cached data with ALL required fields
            analysis_data = cached_data['analysis']
            return EarningsAnalysis(
                ticker=analysis_data.get('ticker', ticker),
                date=analysis_data.get('date', ''),
                quarter=analysis_data.get('quarter', str(quarter)),
                year=analysis_data.get('year', str(year)),
                # Add the missing 13 required arguments with defaults
                executive_segments=[],
                analyst_segments=[],
                guidance_mentions=[],
                financial_metrics={},
                forward_looking_statements=[],
                risk_factors=[],
                overall_sentiment='neutral',
                sentiment_confidence=0.0,
                management_tone='neutral',
                key_highlights=[],
                analyst_concerns=[],
                full_transcript=analysis_data.get('content', ''),
                transcript_length=len(analysis_data.get('content', ''))
            )
        
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
                    'content': getattr(analysis, 'content', ''),  # Use content instead of transcript
                    'reasoning': getattr(analysis, 'reasoning', ''),
                    'decision': getattr(analysis, 'decision', 'NEUTRAL'),
                    'confidence': getattr(analysis, 'confidence', 0.5)
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
    
    def fetch_recent_transcripts(self, ticker: str, lookback_quarters: int = 4) -> List[EarningsAnalysis]:
        """
        Fetch recent earnings transcripts for a ticker
        
        Args:
            ticker: Stock ticker symbol
            lookback_quarters: Number of quarters to look back
            
        Returns:
            List of EarningsAnalysis objects
        """
        transcripts = []
        current_date = datetime.now()
        current_year = current_date.year
        current_quarter = ((current_date.month - 1) // 3) + 1
        
        year = current_year
        quarter = current_quarter
        
        for i in range(lookback_quarters):
            try:
                analysis = self.fetch_earnings_transcript(ticker, year, quarter)
                if analysis:
                    transcripts.append(analysis)
                    log_debug(f"Successfully fetched Q{quarter} {year} transcript for {ticker}")
                else:
                    log_debug(f"No transcript found for {ticker} Q{quarter} {year}")
                
                # Move to previous quarter
                quarter -= 1
                if quarter < 1:
                    quarter = 4
                    year -= 1
                    
            except Exception as e:
                log_error(f"Error fetching Q{quarter} {year} transcript for {ticker}: {e}")
                continue
        
        log_info(f"Fetched {len(transcripts)} earnings transcripts for {ticker}")
        return transcripts
    
    def _analyze_transcript(self, ticker: str, date: str, quarter: str, 
                          year: str, transcript: str) -> EarningsAnalysis:
        """Analyze earnings call transcript comprehensively"""
        
        # Segment the transcript
        segments = self._segment_transcript(transcript)
        
        # Categorize segments by speaker role
        executive_segments = [s for s in segments if s.role == 'executive']
        analyst_segments = [s for s in segments if s.role == 'analyst']
        
        # Extract key information
        guidance_mentions = self._extract_guidance(transcript)
        financial_metrics = self._extract_financial_metrics(transcript)
        forward_looking_statements = self._extract_forward_looking(transcript)
        risk_factors = self._extract_risks(transcript)
        
        # Perform sentiment analysis
        overall_sentiment, sentiment_confidence = self._analyze_sentiment(transcript)
        management_tone = self._analyze_management_tone(executive_segments)
        
        # Generate highlights and concerns
        key_highlights = self._extract_highlights(executive_segments)
        analyst_concerns = self._extract_analyst_concerns(analyst_segments)
        
        return EarningsAnalysis(
            ticker=ticker,
            date=date,
            quarter=quarter,
            year=year,
            executive_segments=executive_segments,
            analyst_segments=analyst_segments,
            guidance_mentions=guidance_mentions,
            financial_metrics=financial_metrics,
            forward_looking_statements=forward_looking_statements,
            risk_factors=risk_factors,
            overall_sentiment=overall_sentiment,
            sentiment_confidence=sentiment_confidence,
            management_tone=management_tone,
            key_highlights=key_highlights,
            analyst_concerns=analyst_concerns,
            full_transcript=transcript,
            transcript_length=len(transcript)
        )
    
    def _segment_transcript(self, transcript: str) -> List[EarningsTranscriptSegment]:
        """Segment transcript by speaker and analyze each segment"""
        segments = []
        
        # Split by common speaker introduction patterns
        speaker_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+)*)\s*[-–—]\s*([^:]+):',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\s*:',
            r'([A-Z\s]+)\s*[-–—]\s*([^:]+):'
        ]
        
        lines = transcript.split('\n')
        current_speaker = None
        current_role = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts with a speaker
            speaker_found = False
            for pattern in speaker_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous segment
                    if current_speaker and current_content:
                        segment = self._create_segment(
                            current_speaker, current_role, '\n'.join(current_content)
                        )
                        segments.append(segment)
                    
                    # Start new segment
                    current_speaker = match.group(1).strip()
                    current_role = self._identify_speaker_role(current_speaker, line)
                    current_content = [line[match.end():].strip()]
                    speaker_found = True
                    break
            
            if not speaker_found:
                # Continue current segment
                if current_content is not None:
                    current_content.append(line)
        
        # Add final segment
        if current_speaker and current_content:
            segment = self._create_segment(
                current_speaker, current_role, '\n'.join(current_content)
            )
            segments.append(segment)
        
        return segments
    
    def _create_segment(self, speaker: str, role: str, content: str) -> EarningsTranscriptSegment:
        """Create a structured transcript segment"""
        
        # Extract sentiment keywords from content
        sentiment_keywords = []
        content_lower = content.lower()
        
        for sentiment_type, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    sentiment_keywords.append(f"{sentiment_type}:{keyword}")
        
        # Determine segment type
        segment_type = self._classify_segment_type(content)
        
        return EarningsTranscriptSegment(
            speaker=speaker,
            role=role,
            content=content,
            sentiment_keywords=sentiment_keywords,
            segment_type=segment_type
        )
    
    def _identify_speaker_role(self, speaker_name: str, full_line: str) -> str:
        """Identify if speaker is executive, analyst, or operator"""
        speaker_lower = speaker_name.lower()
        line_lower = full_line.lower()
        
        # Check for executive titles
        for title in self.executive_titles:
            if title in speaker_lower or title in line_lower:
                return 'executive'
        
        # Check for analyst indicators
        for pattern in self.analyst_patterns:
            if pattern in speaker_lower or pattern in line_lower:
                return 'analyst'
        
        # Check for operator/moderator
        if any(word in speaker_lower for word in ['operator', 'moderator', 'coordinator']):
            return 'operator'
        
        # Default to analyst if uncertain (most non-executive speakers are analysts)
        return 'analyst'
    
    def _classify_segment_type(self, content: str) -> str:
        """Classify the type of transcript segment"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['welcome', 'thank you for joining', 'call to order']):
            return 'opening'
        elif any(word in content_lower for word in ['results', 'performance', 'quarter', 'revenue']):
            return 'results'
        elif any(word in content_lower for word in ['guidance', 'outlook', 'expect', 'forecast']):
            return 'guidance'
        elif any(word in content_lower for word in ['question', 'answer', 'ask', 'wonder']):
            return 'qa'
        elif any(word in content_lower for word in ['conclude', 'closing', 'thank you', 'end']):
            return 'closing'
        
        return 'general'
    
    def _extract_guidance(self, transcript: str) -> List[str]:
        """Extract guidance and forward-looking statements"""
        guidance_statements = []
        sentences = re.split(r'[.!?]+', transcript)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            
            # Check if sentence contains guidance keywords
            if any(keyword in sentence_lower for keyword in self.guidance_keywords):
                # Also check for financial metrics
                if any(metric in sentence_lower for metric in self.financial_metric_keywords):
                    guidance_statements.append(sentence.strip())
        
        return guidance_statements[:10]  # Limit to top 10 most relevant
    
    def _extract_financial_metrics(self, transcript: str) -> Dict[str, str]:
        """Extract financial metrics and numbers from transcript"""
        metrics = {}
        
        # Patterns for financial metrics with numbers
        metric_patterns = [
            r'(revenue|sales)\s+(?:of\s+)?[\$]?([0-9,.\s]+(?:billion|million|thousand)?)',
            r'(earnings?|profit)\s+(?:of\s+)?[\$]?([0-9,.\s]+(?:billion|million|thousand)?)',
            r'(margin)\s+(?:of\s+)?([0-9.]+%?)',
            r'(growth)\s+(?:of\s+)?([0-9.]+%?)',
            r'(ebitda)\s+(?:of\s+)?[\$]?([0-9,.\s]+(?:billion|million|thousand)?)'
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            for metric, value in matches:
                metrics[metric.lower()] = value.strip()
        
        return metrics
    
    def _extract_forward_looking(self, transcript: str) -> List[str]:
        """Extract forward-looking statements"""
        forward_statements = []
        sentences = re.split(r'[.!?]+', transcript)
        
        future_indicators = [
            'will', 'plan to', 'intend to', 'expect to', 'anticipate',
            'going forward', 'in the future', 'next year', 'upcoming'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
                
            sentence_lower = sentence.lower()
            
            if any(indicator in sentence_lower for indicator in future_indicators):
                forward_statements.append(sentence.strip())
        
        return forward_statements[:8]  # Limit to top 8
    
    def _extract_risks(self, transcript: str) -> List[str]:
        """Extract risk factors and concerns"""
        risk_statements = []
        sentences = re.split(r'[.!?]+', transcript)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
                
            sentence_lower = sentence.lower()
            
            if any(risk_word in sentence_lower for risk_word in self.risk_keywords):
                risk_statements.append(sentence.strip())
        
        return risk_statements[:6]  # Limit to top 6
    
    def _analyze_sentiment(self, transcript: str) -> Tuple[str, float]:
        """Analyze overall sentiment of the transcript"""
        positive_count = 0
        negative_count = 0
        total_words = 0
        
        words = transcript.lower().split()
        total_words = len(words)
        
        for word in words:
            if any(pos_word in word for pos_word in self.sentiment_keywords['positive']):
                positive_count += 1
            elif any(neg_word in word for neg_word in self.sentiment_keywords['negative']):
                negative_count += 1
        
        if total_words == 0:
            return 'neutral', 0.0
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        sentiment_score = positive_ratio - negative_ratio
        confidence = min(1.0, abs(sentiment_score) * 10)  # Scale confidence
        
        if sentiment_score > 0.001:
            return 'positive', confidence
        elif sentiment_score < -0.001:
            return 'negative', confidence
        else:
            return 'neutral', confidence
    
    def _analyze_management_tone(self, executive_segments: List[EarningsTranscriptSegment]) -> str:
        """Analyze management tone from executive segments"""
        if not executive_segments:
            return 'neutral'
        
        confident_indicators = ['confident', 'strong', 'optimistic', 'growth', 'opportunity']
        cautious_indicators = ['cautious', 'careful', 'monitor', 'watch', 'uncertain']
        defensive_indicators = ['defend', 'challenging', 'difficult', 'pressure', 'concern']
        
        confident_count = 0
        cautious_count = 0
        defensive_count = 0
        
        for segment in executive_segments:
            content_lower = segment.content.lower()
            
            confident_count += sum(1 for indicator in confident_indicators if indicator in content_lower)
            cautious_count += sum(1 for indicator in cautious_indicators if indicator in content_lower)
            defensive_count += sum(1 for indicator in defensive_indicators if indicator in content_lower)
        
        if confident_count > cautious_count and confident_count > defensive_count:
            return 'confident'
        elif cautious_count > defensive_count:
            return 'cautious'
        elif defensive_count > 0:
            return 'defensive'
        else:
            return 'neutral'
    
    def _extract_highlights(self, executive_segments: List[EarningsTranscriptSegment]) -> List[str]:
        """Extract key highlights from executive segments"""
        highlights = []
        
        for segment in executive_segments:
            sentences = re.split(r'[.!?]+', segment.content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 30:
                    continue
                
                sentence_lower = sentence.lower()
                
                # Look for achievement/success statements
                if any(word in sentence_lower for word in ['record', 'strong', 'growth', 'success', 'achievement']):
                    highlights.append(sentence.strip())
        
        return highlights[:5]  # Top 5 highlights
    
    def _extract_analyst_concerns(self, analyst_segments: List[EarningsTranscriptSegment]) -> List[str]:
        """Extract concerns raised by analysts"""
        concerns = []
        
        for segment in analyst_segments:
            content_lower = segment.content.lower()
            
            # Look for question/concern patterns
            if any(word in content_lower for word in ['concern', 'worry', 'question', 'challenge', 'risk']):
                # Extract the sentence containing the concern
                sentences = re.split(r'[.!?]+', segment.content)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:
                        sentence_lower = sentence.lower()
                        if any(word in sentence_lower for word in ['concern', 'worry', 'challenge', 'risk']):
                            concerns.append(sentence.strip())
        
        return concerns[:5]  # Top 5 concerns
    
    def create_earnings_articles(self, analysis: EarningsAnalysis) -> List[Dict[str, Any]]:
        """Convert earnings analysis to article format for integration with news system"""
        articles = []
        
        # Create main earnings article
        main_article = {
            'symbol': analysis.ticker,
            'title': f"Earnings Call Analysis: {analysis.ticker} Q{analysis.quarter} {analysis.year}",
            'text': self._create_earnings_summary(analysis),
            'url': f"earnings_transcript_{analysis.ticker}_Q{analysis.quarter}_{analysis.year}",
            'publishedDate': analysis.date or datetime.now(timezone.utc).isoformat(),
            'source': 'earnings_transcript_analysis'
        }
        articles.append(main_article)
        
        # Create guidance-focused article if guidance exists
        if analysis.guidance_mentions:
            guidance_article = {
                'symbol': analysis.ticker,
                'title': f"Earnings Guidance Update: {analysis.ticker} Q{analysis.quarter} {analysis.year}",
                'text': self._create_guidance_summary(analysis),
                'url': f"earnings_guidance_{analysis.ticker}_Q{analysis.quarter}_{analysis.year}",
                'publishedDate': analysis.date or datetime.now(timezone.utc).isoformat(),
                'source': 'earnings_guidance_analysis'
            }
            articles.append(guidance_article)
        
        return articles
    
    def _create_earnings_summary(self, analysis: EarningsAnalysis) -> str:
        """Create comprehensive earnings summary text"""
        summary_parts = [
            f"Earnings call analysis for {analysis.ticker} Q{analysis.quarter} {analysis.year}.",
            f"Overall sentiment: {analysis.overall_sentiment} (confidence: {analysis.sentiment_confidence:.2f}).",
            f"Management tone: {analysis.management_tone}."
        ]
        
        if analysis.key_highlights:
            summary_parts.append(f"Key highlights: {'; '.join(analysis.key_highlights[:3])}.")
        
        if analysis.financial_metrics:
            metrics_str = ', '.join([f"{k}: {v}" for k, v in list(analysis.financial_metrics.items())[:3]])
            summary_parts.append(f"Financial metrics mentioned: {metrics_str}.")
        
        if analysis.analyst_concerns:
            summary_parts.append(f"Analyst concerns: {'; '.join(analysis.analyst_concerns[:2])}.")
        
        if analysis.risk_factors:
            summary_parts.append(f"Risk factors discussed: {'; '.join(analysis.risk_factors[:2])}.")
        
        return ' '.join(summary_parts)
    
    def _create_guidance_summary(self, analysis: EarningsAnalysis) -> str:
        """Create guidance-focused summary"""
        summary_parts = [
            f"Forward guidance from {analysis.ticker} Q{analysis.quarter} {analysis.year} earnings call."
        ]
        
        if analysis.guidance_mentions:
            summary_parts.append(f"Guidance statements: {'; '.join(analysis.guidance_mentions[:3])}.")
        
        if analysis.forward_looking_statements:
            summary_parts.append(f"Forward-looking: {'; '.join(analysis.forward_looking_statements[:2])}.")
        
        return ' '.join(summary_parts)
    
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