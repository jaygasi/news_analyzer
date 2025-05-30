from typing import Dict, Optional
from config.config import Config
from src.analyzers.llm_interface import LLMManager
import logging
import time

# Import traditional sentiment analysis libraries with error handling
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install with: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER Sentiment not available. Install with: pip install vaderSentiment")

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Multi-method sentiment analysis using LLMs, TextBlob, and VADER"""
    
    def __init__(self):
        # Initialize LLM manager for multiple AI providers
        try:
            self.llm_manager = LLMManager()
            logger.info(f"✅ LLM Manager initialized with providers: {self.llm_manager.get_available_providers()}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM Manager: {e}")
            self.llm_manager = None
        
        # Initialize traditional sentiment analyzers
        self.textblob_available = TEXTBLOB_AVAILABLE and Config.USE_TEXTBLOB_ANALYSIS
        self.vader_available = VADER_AVAILABLE and Config.USE_VADER_ANALYSIS
        
        if self.vader_available:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                logger.info("✅ VADER sentiment analyzer initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize VADER: {e}")
                self.vader_available = False
                self.vader_analyzer = None
        else:
            self.vader_analyzer = None
        
        # Log configuration
        self._log_configuration()
        
    def _log_configuration(self):
        """Log the current sentiment analysis configuration"""
        methods = []
        
        if self.llm_manager and self.llm_manager.get_available_providers():
            methods.append(f"LLM ({', '.join(self.llm_manager.get_available_providers())})")
        
        if self.textblob_available:
            methods.append("TextBlob")
        
        if self.vader_available:
            methods.append("VADER")
        
        if methods:
            logger.info(f"✅ Sentiment analysis methods available: {', '.join(methods)}")
        else:
            logger.warning("⚠️ No sentiment analysis methods available!")
        
    def analyze_with_textblob(self, text: str) -> float:
        """Analyze sentiment using TextBlob"""
        if not self.textblob_available:
            return 0.0
            
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            logger.debug(f"TextBlob sentiment: {polarity:.3f}")
            return polarity
        except Exception as e:
            logger.error(f"TextBlob analysis error: {e}")
            return 0.0
    
    def analyze_with_vader(self, text: str) -> float:
        """Analyze sentiment using VADER"""
        if not self.vader_available or not self.vader_analyzer:
            return 0.0
            
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound_score = scores['compound']
            logger.debug(f"VADER sentiment: {compound_score:.3f}")
            return compound_score
        except Exception as e:
            logger.error(f"VADER analysis error: {e}")
            return 0.0
    
    def analyze_with_llm(self, title: str, content: str, ticker: str) -> Dict:
        """Analyze sentiment using configured LLM providers with fallback"""
        
        if not self.llm_manager:
            logger.warning("LLM Manager not available for sentiment analysis")
            return self._create_fallback_result("LLM Manager not initialized")
        
        try:
            # Use the LLM manager's fallback system
            result = self.llm_manager.analyze_with_fallback(title, content, ticker)
            
            # Get which provider was actually used
            primary_provider = self.llm_manager.get_primary_provider()
            if primary_provider:
                result["llm_provider_used"] = primary_provider.__class__.__name__.replace('Provider', '')
            
            logger.info(f"✅ LLM analysis for {ticker}: {result['trade_signal']} "
                       f"(confidence: {result['confidence_score']:.2f}, "
                       f"provider: {result.get('llm_provider_used', 'Unknown')})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM analysis failed for {ticker}: {e}")
            return self._create_fallback_result(f"LLM analysis failed: {str(e)}")
    
    def _create_fallback_result(self, reason: str) -> Dict:
        """Create a fallback result when LLM analysis fails"""
        return {
            "sentiment_score": 0.0,
            "confidence_score": 0.0,
            "trade_signal": "NEUTRAL",
            "reasoning": f"Analysis unavailable: {reason}",
            "key_factors": ["analysis_unavailable"],
            "llm_provider_used": "None"
        }
    
    def comprehensive_analysis(self, title: str, content: str, ticker: str) -> Dict:
        """Combine multiple sentiment analysis methods with smart weighting"""
        
        logger.info(f"🔍 Starting comprehensive analysis for {ticker}")
        
        # Initialize results dictionary
        results = {
            "textblob_score": 0.0,
            "vader_score": 0.0,
            "llm_score": 0.0,
            "sentiment_score": 0.0,
            "confidence_score": 0.0,
            "trade_signal": "NEUTRAL",
            "reasoning": "",
            "key_factors": [],
            "analysis_errors": [],
            "llm_provider_used": "None"
        }
        
        # Validate inputs
        if not title and not content:
            logger.warning(f"Empty title and content for {ticker}")
            results["reasoning"] = "No content available for analysis"
            return results
        
        # TextBlob analysis
        if self.textblob_available:
            try:
                results["textblob_score"] = self.analyze_with_textblob(f"{title} {content}")
                logger.debug(f"TextBlob analysis complete for {ticker}: {results['textblob_score']:.3f}")
            except Exception as e:
                error_msg = f"TextBlob error: {str(e)}"
                results["analysis_errors"].append(error_msg)
                logger.error(f"TextBlob analysis failed for {ticker}: {e}")
        
        # VADER analysis
        if self.vader_available:
            try:
                results["vader_score"] = self.analyze_with_vader(f"{title} {content}")
                logger.debug(f"VADER analysis complete for {ticker}: {results['vader_score']:.3f}")
            except Exception as e:
                error_msg = f"VADER error: {str(e)}"
                results["analysis_errors"].append(error_msg)
                logger.error(f"VADER analysis failed for {ticker}: {e}")
        
        # LLM analysis (most important)
        llm_result = None
        if self.llm_manager:
            try:
                llm_result = self.analyze_with_llm(title, content, ticker)
                results["llm_score"] = llm_result["sentiment_score"]
                results["reasoning"] = llm_result["reasoning"]
                results["key_factors"] = llm_result.get("key_factors", [])
                results["llm_provider_used"] = llm_result.get("llm_provider_used", "None")
                logger.debug(f"LLM analysis complete for {ticker}: {results['llm_score']:.3f}")
            except Exception as e:
                error_msg = f"LLM error: {str(e)}"
                results["analysis_errors"].append(error_msg)
                logger.error(f"LLM analysis failed for {ticker}: {e}")
        else:
            results["analysis_errors"].append("LLM Manager not available")
        
        # Combine scores with weighted average
        results = self._combine_sentiment_scores(results, llm_result)
        
        # Determine final trade signal and confidence
        results = self._determine_trade_signal(results, llm_result)
        
        # Final validation and cleanup
        results = self._validate_final_results(results)
        
        # Log final results
        logger.info(f"✅ Analysis complete for {ticker}: "
                   f"{results['trade_signal']} "
                   f"(sentiment: {results['sentiment_score']:.2f}, "
                   f"confidence: {results['confidence_score']:.2f}, "
                   f"provider: {results['llm_provider_used']})")
        
        if results["analysis_errors"]:
            logger.warning(f"Analysis errors for {ticker}: {'; '.join(results['analysis_errors'])}")
        
        return results
    
    def _combine_sentiment_scores(self, results: Dict, llm_result: Optional[Dict]) -> Dict:
        """Combine sentiment scores from different methods with weighted averaging"""
        
        weights = []
        scores = []
        
        # TextBlob (lower weight for traditional methods)
        if results["textblob_score"] != 0.0:
            weights.append(0.15)
            scores.append(results["textblob_score"])
        
        # VADER (medium weight)
        if results["vader_score"] != 0.0:
            weights.append(0.25)
            scores.append(results["vader_score"])
        
        # LLM (highest weight as it's most contextually aware)
        if results["llm_score"] != 0.0:
            weights.append(0.6)
            scores.append(results["llm_score"])
        
        # Calculate weighted average
        if weights and scores:
            total_weight = sum(weights)
            results["sentiment_score"] = sum(w * s for w, s in zip(weights, scores)) / total_weight
            logger.debug(f"Combined sentiment score: {results['sentiment_score']:.3f} "
                        f"(weights: {weights}, scores: {[f'{s:.3f}' for s in scores]})")
        else:
            logger.warning("No valid sentiment scores to combine")
            results["sentiment_score"] = 0.0
        
        return results
    
    def _determine_trade_signal(self, results: Dict, llm_result: Optional[Dict]) -> Dict:
        """Determine final trade signal and confidence score"""
        
        if llm_result and llm_result.get("confidence_score", 0) > 0:
            # Use LLM's confidence and signal as primary
            results["confidence_score"] = llm_result["confidence_score"]
            results["trade_signal"] = llm_result["trade_signal"]
            
            # Check for disagreement between LLM and traditional methods
            traditional_scores = []
            if results["textblob_score"] != 0.0:
                traditional_scores.append(results["textblob_score"])
            if results["vader_score"] != 0.0:
                traditional_scores.append(results["vader_score"])
            
            if traditional_scores:
                traditional_avg = sum(traditional_scores) / len(traditional_scores)
                
                # Reduce confidence if there's strong disagreement
                disagreement = abs(results["llm_score"] - traditional_avg)
                if disagreement > 0.5:
                    confidence_reduction = min(0.3, disagreement * 0.2)
                    results["confidence_score"] *= (1 - confidence_reduction)
                    results["reasoning"] += f" [Confidence reduced by {confidence_reduction:.1%} due to mixed traditional signals]"
                    logger.debug(f"Confidence reduced due to disagreement: LLM={results['llm_score']:.3f}, Traditional={traditional_avg:.3f}")
        
        else:
            # Fallback to rule-based signal generation using combined sentiment
            combined_sentiment = results["sentiment_score"]
            
            if combined_sentiment > 0.4:
                results["trade_signal"] = "LONG"
                results["confidence_score"] = min(0.6, abs(combined_sentiment))
            elif combined_sentiment < -0.4:
                results["trade_signal"] = "SHORT"
                results["confidence_score"] = min(0.6, abs(combined_sentiment))
            else:
                results["trade_signal"] = "NEUTRAL"
                results["confidence_score"] = 0.3
            
            if not results["reasoning"]:
                results["reasoning"] = f"Traditional analysis: combined sentiment {combined_sentiment:.2f}"
            
            logger.debug(f"Using fallback signal generation: {results['trade_signal']} (confidence: {results['confidence_score']:.3f})")
        
        return results
    
    def _validate_final_results(self, results: Dict) -> Dict:
        """Validate and clean final results"""
        
        # Ensure confidence score is within valid range
        results["confidence_score"] = max(0.0, min(1.0, results["confidence_score"]))
        
        # Ensure sentiment score is within valid range
        results["sentiment_score"] = max(-1.0, min(1.0, results["sentiment_score"]))
        
        # Validate trade signal
        valid_signals = ["LONG", "SHORT", "NEUTRAL"]
        if results["trade_signal"] not in valid_signals:
            logger.warning(f"Invalid trade signal '{results['trade_signal']}', defaulting to NEUTRAL")
            results["trade_signal"] = "NEUTRAL"
        
        # Ensure reasoning exists
        if not results["reasoning"]:
            results["reasoning"] = f"Analysis complete with {results['trade_signal']} signal"
        
        # Ensure key_factors is a list
        if not isinstance(results["key_factors"], list):
            results["key_factors"] = ["standard_analysis"]
        
        return results
    
    def switch_llm_provider(self, provider_name: str) -> bool:
        """Switch to a different LLM provider"""
        if not self.llm_manager:
            logger.error("LLM Manager not available for provider switching")
            return False
        
        try:
            success = self.llm_manager.switch_primary_provider(provider_name)
            if success:
                logger.info(f"✅ Successfully switched LLM provider to: {provider_name}")
            else:
                logger.error(f"❌ Failed to switch to LLM provider: {provider_name}")
            return success
        except Exception as e:
            logger.error(f"❌ Error switching LLM provider to {provider_name}: {e}")
            return False
    
    def get_available_llm_providers(self) -> list:
        """Get list of available LLM providers"""
        if not self.llm_manager:
            return []
        
        try:
            return self.llm_manager.get_available_providers()
        except Exception as e:
            logger.error(f"Error getting available LLM providers: {e}")
            return []
    
    def get_current_llm_provider(self) -> str:
        """Get current primary LLM provider"""
        if not self.llm_manager:
            return "None"
        
        try:
            provider = self.llm_manager.get_primary_provider()
            if provider:
                return provider.__class__.__name__.replace('Provider', '')
            else:
                return "None"
        except Exception as e:
            logger.error(f"Error getting current LLM provider: {e}")
            return "Error"
    
    def get_analysis_capabilities(self) -> Dict:
        """Get information about available analysis capabilities"""
        return {
            "llm_providers_available": self.get_available_llm_providers(),
            "current_llm_provider": self.get_current_llm_provider(),
            "textblob_available": self.textblob_available,
            "vader_available": self.vader_available,
            "total_methods": len([x for x in [
                bool(self.get_available_llm_providers()),
                self.textblob_available,
                self.vader_available
            ] if x])
        }
    
    def test_analysis(self, test_title: str = "Test earnings beat", 
                     test_content: str = "Company reported strong quarterly results exceeding analyst expectations",
                     test_ticker: str = "TEST") -> Dict:
        """Test the sentiment analysis system with sample content"""
        logger.info(f"🧪 Testing sentiment analysis with sample content for {test_ticker}")
        
        try:
            result = self.comprehensive_analysis(test_title, test_content, test_ticker)
            
            logger.info("✅ Sentiment analysis test completed successfully")
            logger.info(f"Test results: {result['trade_signal']} "
                       f"(sentiment: {result['sentiment_score']:.2f}, "
                       f"confidence: {result['confidence_score']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis test failed: {e}")
            return {
                "error": str(e),
                "sentiment_score": 0.0,
                "confidence_score": 0.0,
                "trade_signal": "NEUTRAL",
                "reasoning": f"Test failed: {str(e)}"
            }
