"""
Enhanced news analyzer with technical analysis integration
"""
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning
from analysis.technical_analyzer import TechnicalAnalyzer, TechnicalAnalysis

@dataclass
class EnhancedNewsAnalysis:
    """Enhanced news analysis with technical data"""
    symbol: str
    title: str
    content: str
    sentiment_score: float  # -1 to 1
    confidence: float      # 0 to 1
    topic: str
    timestamp: pd.Timestamp
    finbert_score: float = 0.0
    keyword_score: float = 0.0
    technical_analysis: Optional[TechnicalAnalysis] = None
    combined_confidence: float = 0.0  # News + Technical combined

class EnhancedNewsAnalyzer:
    """Enhanced news analyzer with technical validation"""
    
    def __init__(self, fmp_loader):
        self.fmp_loader = fmp_loader
        self.technical_analyzer = TechnicalAnalyzer(fmp_loader)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.finbert_labels = ["positive", "negative", "neutral"]
        
        # Initialize FinBERT
        self._load_finbert()
        
        # Enhanced keyword lists
        self.positive_keywords = [
            'beats', 'beat', 'exceeds', 'exceed', 'raises', 'upgrade', 'approval', 'approved',
            'growth', 'strong', 'positive', 'success', 'breakthrough', 'innovation',
            'revenue increase', 'profit increase', 'buyback', 'dividend', 'expansion',
            'outperform', 'surge', 'rally', 'boost', 'gain', 'rise', 'soar', 'bullish'
        ]
        
        self.negative_keywords = [
            'misses', 'miss', 'falls short', 'disappointing', 'decline', 'drop',
            'downgrade', 'concern', 'loss', 'cut', 'reduce', 'weak', 'struggle',
            'investigation', 'lawsuit', 'recall', 'bankruptcy', 'layoffs',
            'plunge', 'crash', 'fall', 'slump', 'tumble', 'sink', 'bearish'
        ]
        
        self.biotech_keywords = [
            'fda', 'approval', 'phase', 'trial', 'clinical', 'drug', 'therapy',
            'treatment', 'efficacy', 'safety', 'regulatory', 'orphan drug',
            'breakthrough therapy', 'fast track', 'priority review'
        ]
    
    def _load_finbert(self):
        """Load FinBERT model"""
        try:
            log_info("Loading FinBERT model...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_model.to(self.device)
            self.finbert_model.eval()
            log_info(f"FinBERT loaded successfully on {self.device}")
        except Exception as e:
            log_error(f"Failed to load FinBERT: {e}")
            log_warning("Falling back to keyword-only analysis")
    
    def _get_finbert_sentiment(self, text: str) -> Tuple[float, float]:
        """Get FinBERT sentiment score"""
        if not self.finbert_model or not self.finbert_tokenizer:
            return 0.0, 0.0
        
        try:
            max_length = 512
            inputs = self.finbert_tokenizer(
                text, 
                return_tensors="pt", 
                max_length=max_length, 
                truncation=True, 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            probs = predictions.cpu().numpy()[0]
            
            pos_prob = probs[0]  # positive
            neg_prob = probs[1]  # negative
            neu_prob = probs[2]  # neutral
            
            sentiment = pos_prob - neg_prob
            confidence = max(probs)
            
            return float(sentiment), float(confidence)
            
        except Exception as e:
            log_error(f"FinBERT analysis error: {e}")
            return 0.0, 0.0
    
    def _calculate_keyword_score(self, text: str) -> Tuple[float, float]:
        """Calculate enhanced keyword-based sentiment score"""
        text_lower = text.lower()
        
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        biotech_count = sum(1 for kw in self.biotech_keywords if kw in text_lower)
        
        # Calculate raw sentiment
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            sentiment = 0.0
            confidence = 0.1
        else:
            sentiment = (positive_count - negative_count) / total_keywords
            confidence = min(total_keywords / 5.0, 1.0)
        
        # Boost confidence for biotech news (higher impact)
        if biotech_count > 0:
            confidence = min(confidence * 1.4, 1.0)
        
        return sentiment, confidence
    
    def _detect_topic(self, text: str) -> str:
        """Enhanced topic detection"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['earnings', 'revenue', 'profit', 'eps', 'quarterly', 'guidance']):
            return 'earnings'
        elif any(kw in text_lower for kw in self.biotech_keywords):
            return 'biotech'
        elif any(kw in text_lower for kw in ['merger', 'acquisition', 'deal', 'buyout', 'takeover']):
            return 'ma'
        elif any(kw in text_lower for kw in ['upgrade', 'downgrade', 'target', 'analyst', 'rating']):
            return 'analyst'
        elif any(kw in text_lower for kw in ['dividend', 'buyback', 'split', 'spinoff']):
            return 'corporate_action'
        elif any(kw in text_lower for kw in ['contract', 'partnership', 'agreement', 'collaboration']):
            return 'business_development'
        else:
            return 'general'
    
    def _ensemble_score(self, finbert_sentiment: float, finbert_conf: float,
                       keyword_sentiment: float, keyword_conf: float) -> Tuple[float, float]:
        """Enhanced ensemble scoring with dynamic weights"""
        
        if finbert_conf == 0.0:
            return keyword_sentiment, keyword_conf
        
        # Dynamic weighting based on confidence levels
        finbert_weight = finbert_conf * 0.7  # FinBERT gets higher base weight
        keyword_weight = keyword_conf * 0.3
        
        total_weight = finbert_weight + keyword_weight
        if total_weight == 0:
            return 0.0, 0.0
        
        # Weighted sentiment
        ensemble_sentiment = (
            finbert_sentiment * finbert_weight + 
            keyword_sentiment * keyword_weight
        ) / total_weight
        
        # Enhanced confidence calculation
        confidence_boost = 1.0
        if abs(finbert_sentiment - keyword_sentiment) < 0.3:  # Agreement boost
            confidence_boost = 1.2
        
        ensemble_confidence = min((finbert_conf * keyword_conf) ** 0.5 * confidence_boost, 1.0)
        
        return ensemble_sentiment, ensemble_confidence
    
    def _combine_news_technical_confidence(self, news_confidence: float, 
                                         technical_analysis: Optional[TechnicalAnalysis],
                                         sentiment_score: float) -> float:
        """Combine news and technical confidence scores"""
        if technical_analysis is None:
            return news_confidence * 0.7  # Penalize missing technical data
        
        tech_confidence = technical_analysis.technical_confidence
        
        # Momentum alignment bonus
        momentum_alignment = 1.0
        if sentiment_score > 0 and technical_analysis.momentum_score > 0:
            momentum_alignment = 1.2  # Bullish news + bullish technicals
        elif sentiment_score < 0 and technical_analysis.momentum_score < 0:
            momentum_alignment = 1.2  # Bearish news + bearish technicals
        elif abs(sentiment_score) > 0.5 and abs(technical_analysis.momentum_score) > 0.3:
            if (sentiment_score > 0) != (technical_analysis.momentum_score > 0):
                momentum_alignment = 0.7  # Contradictory signals
        
        # Volume confirmation bonus
        volume_bonus = 1.0
        if technical_analysis.volume_score > 0.7:  # High volume
            volume_bonus = 1.1
        
        # Liquidity requirement
        liquidity_factor = max(technical_analysis.liquidity_score, 0.3)
        
        # Combined confidence calculation
        combined = (
            news_confidence * 0.6 +           # News confidence (primary)
            tech_confidence * 0.4             # Technical confidence (secondary)
        ) * momentum_alignment * volume_bonus * liquidity_factor
        
        return min(combined, 1.0)
    
    def analyze_news_with_technical(self, news_df: pd.DataFrame, 
                                  current_prices: pd.DataFrame) -> List[EnhancedNewsAnalysis]:
        """Analyze news with technical validation"""
        results = []
        
        if news_df is None or news_df.empty:
            return results
        
        log_info(f"Analyzing {len(news_df)} news articles with FinBERT + Technical Analysis")
        
        for idx, row in news_df.iterrows():
            try:
                symbol = row.get('symbol', '').strip()
                title = row.get('title', '').strip()
                content = row.get('content', '').strip()
                
                if not all([symbol, title, content]):
                    continue
                
                # Get current price data for technical analysis
                price_data = None
                if current_prices is not None and not current_prices.empty:
                    price_row = current_prices[current_prices['symbol'] == symbol]
                    if not price_row.empty:
                        price_data = price_row.iloc[0]
                
                # News analysis
                full_text = f"{title} {content}"
                finbert_sentiment, finbert_conf = self._get_finbert_sentiment(full_text)
                keyword_sentiment, keyword_conf = self._calculate_keyword_score(full_text)
                final_sentiment, news_confidence = self._ensemble_score(
                    finbert_sentiment, finbert_conf, keyword_sentiment, keyword_conf
                )
                
                # Technical analysis
                technical_analysis = None
                if price_data is not None:
                    technical_analysis = self.technical_analyzer.analyze_symbol(symbol, price_data)
                
                # Combined confidence
                combined_confidence = self._combine_news_technical_confidence(
                    news_confidence, technical_analysis, final_sentiment
                )
                
                # Topic detection
                topic = self._detect_topic(full_text)
                
                # Create enhanced analysis result
                analysis = EnhancedNewsAnalysis(
                    symbol=symbol,
                    title=title,
                    content=content,
                    sentiment_score=final_sentiment,
                    confidence=news_confidence,
                    topic=topic,
                    timestamp=pd.Timestamp.now(),
                    finbert_score=finbert_sentiment,
                    keyword_score=keyword_sentiment,
                    technical_analysis=technical_analysis,
                    combined_confidence=combined_confidence
                )
                
                results.append(analysis)
                
                # Log high confidence results with technical details
                if combined_confidence >= CONFIG.min_confidence_score:
                    tech_info = ""
                    if technical_analysis:
                        tech_info = (f"tech_conf={technical_analysis.technical_confidence:.2f} "
                                   f"momentum={technical_analysis.momentum_score:.2f} "
                                   f"liquidity={technical_analysis.liquidity_score:.2f}")
                    
                    log_info(f"HIGH CONFIDENCE: {symbol} "
                            f"sentiment={final_sentiment:.3f} "
                            f"news_conf={news_confidence:.3f} "
                            f"combined_conf={combined_confidence:.3f} "
                            f"{tech_info} topic={topic}")
                
            except Exception as e:
                log_error(f"Error analyzing news for {symbol}: {e}")
                continue
        
        return results
    
    def filter_high_confidence(self, analyses: List[EnhancedNewsAnalysis], 
                             use_combined_confidence: bool = True) -> List[EnhancedNewsAnalysis]:
        """Filter for high confidence analyses"""
        confidence_field = 'combined_confidence' if use_combined_confidence else 'confidence'
        
        high_conf = [
            analysis for analysis in analyses
            if getattr(analysis, confidence_field) >= CONFIG.min_confidence_score
        ]
        
        if high_conf:
            log_info(f"Filtered to {len(high_conf)} high-confidence signals using {confidence_field} "
                    f"(threshold: {CONFIG.min_confidence_score})")
        
        return high_conf