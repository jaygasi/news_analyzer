import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy import and_, func
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import SessionLocal
from src.database.models import NewsArticle, TradingSignal, Company
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from config.config import Config

# Import yfinance with error handling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available. Install with: pip install yfinance")

logger = logging.getLogger(__name__)

class TradingSignalGenerator:
    """Generate trading signals from analyzed news articles"""
    
    def __init__(self):
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("✅ Sentiment analyzer initialized for signal generation")
        except Exception as e:
            logger.error(f"❌ Failed to initialize sentiment analyzer: {e}")
            raise
    
    def get_stock_context(self, ticker: str) -> Dict:
        """Get current stock context using yfinance"""
        if not YFINANCE_AVAILABLE:
            logger.warning(f"yfinance not available, returning empty context for {ticker}")
            return {}
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get stock info with timeout protection
            try:
                info = stock.info
            except Exception as e:
                logger.warning(f"Could not get stock info for {ticker}: {e}")
                info = {}
            
            # Get recent price history
            try:
                hist = stock.history(period="30d")
            except Exception as e:
                logger.warning(f"Could not get price history for {ticker}: {e}")
                hist = None
            
            if hist is None or hist.empty:
                logger.warning(f"No price data available for {ticker}")
                return self._create_minimal_context(ticker, info)
            
            # Calculate price metrics
            try:
                current_price = float(hist['Close'].iloc[-1])
                price_30d_ago = float(hist['Close'].iloc[0])
                price_change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
                
                # Calculate volume metrics
                avg_volume = hist['Volume'].mean()
                recent_volume = hist['Volume'].iloc[-1]
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Error calculating metrics for {ticker}: {e}")
                return self._create_minimal_context(ticker, info)
            
            # Build context dictionary
            context = {
                "current_price": round(current_price, 2),
                "30d_change": round(price_change_30d, 2),
                "avg_volume": int(avg_volume) if avg_volume else 0,
                "recent_volume": int(recent_volume) if recent_volume else 0,
                "volume_ratio": round(volume_ratio, 2),
                "market_cap": info.get('marketCap', 0) or 0,
                "pe_ratio": info.get('trailingPE', 0) or 0,
                "sector": info.get('sector', 'Unknown') or 'Unknown',
                "industry": info.get('industry', 'Unknown') or 'Unknown',
                "beta": info.get('beta', 1.0) or 1.0
            }
            
            logger.debug(f"Retrieved stock context for {ticker}: ${current_price:.2f}, 30d: {price_change_30d:.1f}%")
            return context
            
        except Exception as e:
            logger.error(f"Error getting stock context for {ticker}: {e}")
            return {}
    
    def _create_minimal_context(self, ticker: str, info: Dict) -> Dict:
        """Create minimal context when full data is unavailable"""
        return {
            "current_price": 0.0,
            "30d_change": 0.0,
            "avg_volume": 0,
            "recent_volume": 0,
            "volume_ratio": 1.0,
            "market_cap": info.get('marketCap', 0) or 0,
            "pe_ratio": info.get('trailingPE', 0) or 0,
            "sector": info.get('sector', 'Unknown') or 'Unknown',
            "industry": info.get('industry', 'Unknown') or 'Unknown',
            "beta": info.get('beta', 1.0) or 1.0
        }
    
    def calculate_position_size(self, confidence: float, volatility: float = 1.0) -> float:
        """Calculate suggested position size based on confidence and volatility"""
        try:
            # Base position size from config
            base_size = Config.MAX_POSITION_SIZE
            
            # Confidence multiplier (0.5 to 1.5)
            confidence_multiplier = 0.5 + confidence
            
            # Volatility adjustment (higher volatility -> smaller position)
            volatility_multiplier = 1.0 / max(volatility, 0.5)
            
            # Calculate final position size
            position_size = base_size * confidence_multiplier * volatility_multiplier
            
            # Cap at maximum
            max_position = Config.MAX_POSITION_SIZE * 1.5
            final_size = min(position_size, max_position)
            
            logger.debug(f"Position size calculation: base={base_size:.3f}, "
                        f"confidence={confidence:.3f}, volatility={volatility:.3f}, "
                        f"final={final_size:.3f}")
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return Config.MAX_POSITION_SIZE
    
    def calculate_stop_loss_take_profit(self, current_price: float, signal_type: str, 
                                      confidence: float) -> tuple:
        """Calculate stop loss and take profit levels"""
        
        try:
            if current_price <= 0:
                logger.warning("Invalid current price for stop loss calculation")
                return 0.0, 0.0
            
            # Base percentages
            if signal_type == "LONG":
                stop_loss_pct = -Config.DEFAULT_STOP_LOSS_PCT
                take_profit_pct = Config.DEFAULT_TAKE_PROFIT_PCT
            elif signal_type == "SHORT":
                stop_loss_pct = Config.DEFAULT_STOP_LOSS_PCT
                take_profit_pct = -Config.DEFAULT_TAKE_PROFIT_PCT
            else:  # NEUTRAL
                return current_price, current_price
            
            # Adjust based on confidence
            confidence_adjustment = (confidence - 0.5) * 0.5  # -0.25 to +0.25
            
            # More aggressive stops for higher confidence
            stop_loss_pct *= (1 - confidence_adjustment * 0.5)
            take_profit_pct *= (1 + confidence_adjustment)
            
            # Calculate actual prices
            stop_loss_price = current_price * (1 + stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)
            
            logger.debug(f"Stop/TP calculation for {signal_type}: "
                        f"price=${current_price:.2f}, stop=${stop_loss_price:.2f}, "
                        f"target=${take_profit_price:.2f}")
            
            return round(stop_loss_price, 2), round(take_profit_price, 2)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss/take profit: {e}")
            return current_price if current_price > 0 else 0.0, current_price if current_price > 0 else 0.0
    
    def process_unanalyzed_articles(self):
        """Process articles that haven't been analyzed yet"""
        db = SessionLocal()
        
        try:
            # Get unprocessed articles with company information
            articles = db.query(NewsArticle).join(Company).filter(
                NewsArticle.processed == False,
                NewsArticle.content.isnot(None),
                NewsArticle.content != ""
            ).limit(Config.MAX_ARTICLES_PER_RUN).all()
            
            if not articles:
                logger.info("No unprocessed articles found")
                return
            
            logger.info(f"Processing {len(articles)} unanalyzed articles")
            
            processed_count = 0
            signal_count = 0
            
            for article in articles:
                try:
                    if self._process_single_article(article, db):
                        signal_count += 1
                    processed_count += 1
                    
                    # Commit periodically to avoid long transactions
                    if processed_count % 10 == 0:
                        db.commit()
                        logger.debug(f"Committed batch of articles (processed: {processed_count})")
                        
                except Exception as e:
                    logger.error(f"Error processing article {article.id}: {e}")
                    # Mark as processed with error to avoid retry loops
                    article.processing_error = str(e)
                    article.processed = True
            
            # Final commit
            db.commit()
            logger.info(f"✅ Article processing completed: {processed_count} processed, {signal_count} signals generated")
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in process_unanalyzed_articles: {e}")
            db.rollback()
        except Exception as e:
            logger.error(f"Unexpected error in process_unanalyzed_articles: {e}")
            db.rollback()
        finally:
            db.close()
    
    def _process_single_article(self, article: NewsArticle, db_session) -> bool:
        """Process a single article and generate signals if warranted"""
        
        if not article.company:
            logger.warning(f"Article {article.id} has no associated company")
            article.processed = True
            return False
        
        ticker = article.company.ticker
        logger.debug(f"Processing article for {ticker}: {article.title[:50]}...")
        
        try:
            # Perform comprehensive sentiment analysis
            analysis = self.sentiment_analyzer.comprehensive_analysis(
                title=article.title or "",
                content=article.content or "",
                ticker=ticker
            )
            
            # Update article with analysis results
            article.sentiment_score = analysis['sentiment_score']
            article.textblob_score = analysis.get('textblob_score', 0.0)
            article.vader_score = analysis.get('vader_score', 0.0)
            article.llm_score = analysis.get('llm_score', 0.0)
            article.confidence_score = analysis['confidence_score']
            article.llm_provider_used = analysis.get('llm_provider_used', 'None')
            article.processed = True
            
            # Check if we should generate a trading signal
            should_generate_signal = (
                analysis['confidence_score'] >= Config.MIN_CONFIDENCE_SCORE and 
                analysis['trade_signal'] in ['LONG', 'SHORT'] and
                analysis['sentiment_score'] != 0.0  # Ensure non-zero sentiment
            )
            
            if should_generate_signal:
                signal_created = self._create_trading_signal(article, analysis, db_session)
                if signal_created:
                    logger.info(f"✅ Generated {analysis['trade_signal']} signal for {ticker} "
                               f"(confidence: {analysis['confidence_score']:.2f})")
                    return True
                else:
                    logger.warning(f"Failed to create trading signal for {ticker}")
            else:
                logger.debug(f"No signal generated for {ticker} "
                           f"(confidence: {analysis['confidence_score']:.2f}, "
                           f"signal: {analysis['trade_signal']})")
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing article {article.id} for {ticker}: {e}")
            article.processing_error = str(e)
            article.processed = True
            return False
    
    def _create_trading_signal(self, article: NewsArticle, analysis: Dict, db_session) -> bool:
        """Create a trading signal from analysis results"""
        
        try:
            ticker = article.company.ticker
            
            # Get current stock context
            stock_context = self.get_stock_context(ticker)
            current_price = stock_context.get('current_price', 0)
            
            # Calculate risk management parameters
            beta = abs(stock_context.get('beta', 1.0))
            position_size = self.calculate_position_size(
                analysis['confidence_score'], 
                beta
            )
            
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                current_price,
                analysis['trade_signal'],
                analysis['confidence_score']
            )
            
            # Create trading signal record
            trading_signal = TradingSignal(
                company_id=article.company_id,
                news_article_id=article.id,
                signal_type=analysis['trade_signal'],
                confidence_score=analysis['confidence_score'],
                reasoning=analysis['reasoning'],
                llm_provider_used=analysis.get('llm_provider_used', 'None'),
                
                # Market data
                stock_price=current_price,
                volume=stock_context.get('recent_volume', 0),
                market_cap=stock_context.get('market_cap', 0),
                pe_ratio=stock_context.get('pe_ratio', 0),
                price_change_30d=stock_context.get('30d_change', 0),
                sector=stock_context.get('sector', 'Unknown'),
                industry=stock_context.get('industry', 'Unknown'),
                
                # Risk management
                suggested_position_size=position_size,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                
                # Analysis breakdown
                textblob_score=analysis.get('textblob_score', 0.0),
                vader_score=analysis.get('vader_score', 0.0),
                llm_score=analysis.get('llm_score', 0.0),
                combined_sentiment=analysis['sentiment_score']
            )
            
            db_session.add(trading_signal)
            return True
            
        except Exception as e:
            logger.error(f"Error creating trading signal: {e}")
            return False
    
    def get_daily_signals(self, date: Optional[datetime] = None) -> List[Dict]:
        """Get trading signals for a specific date (default: today)"""
        if date is None:
            date = datetime.now().date()
        
        db = SessionLocal()
        
        try:
            # Query signals for the specified date
            start_time = datetime.combine(date, datetime.min.time())
            end_time = start_time + timedelta(days=1)
            
            signals = db.query(TradingSignal).join(Company).filter(
                TradingSignal.created_at >= start_time,
                TradingSignal.created_at < end_time
            ).order_by(TradingSignal.confidence_score.desc()).all()
            
            signal_data = []
            
            for signal in signals:
                try:
                    # Get current stock context (optional, may fail)
                    stock_context = {}
                    try:
                        stock_context = self.get_stock_context(signal.company.ticker)
                    except Exception as e:
                        logger.debug(f"Could not get current stock context for {signal.company.ticker}: {e}")
                    
                    signal_info = {
                        "id": signal.id,
                        "ticker": signal.company.ticker,
                        "company_name": signal.company.company_name,
                        "signal_type": signal.signal_type,
                        "confidence_score": signal.confidence_score,
                        "reasoning": signal.reasoning,
                        "llm_provider_used": signal.llm_provider_used,
                        "suggested_position_size": signal.suggested_position_size,
                        "stop_loss_price": signal.stop_loss_price,
                        "take_profit_price": signal.take_profit_price,
                        "created_at": signal.created_at,
                        "stock_context": stock_context,
                        "news_title": signal.news_article.title if signal.news_article else "",
                        
                        # Analysis breakdown
                        "textblob_score": signal.textblob_score,
                        "vader_score": signal.vader_score,
                        "llm_score": signal.llm_score,
                        "combined_sentiment": signal.combined_sentiment,
                        
                        # Market data from signal creation
                        "signal_stock_price": signal.stock_price,
                        "signal_volume": signal.volume,
                        "market_cap": signal.market_cap,
                        "pe_ratio": signal.pe_ratio,
                        "price_change_30d": signal.price_change_30d,
                        "sector": signal.sector,
                        "industry": signal.industry
                    }
                    
                    signal_data.append(signal_info)
                    
                except Exception as e:
                    logger.error(f"Error processing signal {signal.id}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(signal_data)} signals for {date}")
            return signal_data
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting daily signals: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting daily signals: {e}")
            return []
        finally:
            db.close()
    
    def get_urgent_signals(self, min_confidence: float = 0.85) -> List[Dict]:
        """Get urgent high-confidence signals from the last hour"""
        db = SessionLocal()
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            signals = db.query(TradingSignal).join(Company).filter(
                TradingSignal.created_at >= cutoff_time,
                TradingSignal.confidence_score >= min_confidence,
                TradingSignal.executed == False
            ).order_by(TradingSignal.confidence_score.desc()).all()
            
            urgent_signals = []
            
            for signal in signals:
                try:
                    signal_info = {
                        "id": signal.id,
                        "ticker": signal.company.ticker,
                        "company_name": signal.company.company_name,
                        "signal_type": signal.signal_type,
                        "confidence_score": signal.confidence_score,
                        "reasoning": signal.reasoning,
                        "llm_provider_used": signal.llm_provider_used,
                        "created_at": signal.created_at,
                        "suggested_position_size": signal.suggested_position_size,
                        "stop_loss_price": signal.stop_loss_price,
                        "take_profit_price": signal.take_profit_price
                    }
                    urgent_signals.append(signal_info)
                    
                except Exception as e:
                    logger.error(f"Error processing urgent signal {signal.id}: {e}")
                    continue
            
            if urgent_signals:
                logger.info(f"Found {len(urgent_signals)} urgent signals")
            else:
                logger.debug("No urgent signals found")
            
            return urgent_signals
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting urgent signals: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting urgent signals: {e}")
            return []
        finally:
            db.close()
    
    def get_signal_stats(self) -> Dict:
        """Get statistics about generated signals"""
        db = SessionLocal()
        
        try:
            today = datetime.now().date()
            today_start = datetime.combine(today, datetime.min.time())
            
            # Basic counts
            total_signals = db.query(TradingSignal).count()
            today_signals = db.query(TradingSignal).filter(
                TradingSignal.created_at >= today_start
            ).count()
            
            # High confidence signals
            high_confidence = db.query(TradingSignal).filter(
                TradingSignal.confidence_score >= 0.8
            ).count()
            
            # Today's signal distribution
            long_signals = db.query(TradingSignal).filter(
                TradingSignal.signal_type == 'LONG',
                TradingSignal.created_at >= today_start
            ).count()
            
            short_signals = db.query(TradingSignal).filter(
                TradingSignal.signal_type == 'SHORT',
                TradingSignal.created_at >= today_start
            ).count()
            
            # LLM provider usage (today)
            try:
                provider_usage = db.query(
                    TradingSignal.llm_provider_used,
                    func.count(TradingSignal.id).label('count')
                ).filter(
                    TradingSignal.created_at >= today_start
                ).group_by(TradingSignal.llm_provider_used).all()
                
                provider_dict = {provider: count for provider, count in provider_usage if provider}
            except Exception as e:
                logger.error(f"Error getting provider usage stats: {e}")
                provider_dict = {}
            
            return {
                'total_signals': total_signals,
                'today_signals': today_signals,
                'high_confidence': high_confidence,
                'today_long_signals': long_signals,
                'today_short_signals': short_signals,
                'provider_usage': provider_dict,
                'last_update': datetime.now().isoformat()
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting signal stats: {e}")
            return {
                'error': f"Database error: {str(e)}",
                'total_signals': -1,
                'last_update': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Unexpected error getting signal stats: {e}")
            return {
                'error': f"Unexpected error: {str(e)}",
                'total_signals': -1,
                'last_update': datetime.now().isoformat()
            }
        finally:
            db.close()
    
    def test_signal_generation(self) -> Dict:
        """Test signal generation with sample data"""
        logger.info("🧪 Testing signal generation system...")
        
        try:
            # Test sentiment analyzer
            analyzer_test = self.sentiment_analyzer.test_analysis()
            
            # Test stock context retrieval
            test_ticker = "AAPL"
            stock_context = self.get_stock_context(test_ticker)
            
            # Test position sizing
            test_position = self.calculate_position_size(0.85, 1.2)
            
            # Test stop loss calculation
            test_stop, test_tp = self.calculate_stop_loss_take_profit(150.0, "LONG", 0.85)
            
            return {
                "sentiment_analyzer_test": analyzer_test,
                "stock_context_test": {
                    "ticker": test_ticker,
                    "context_retrieved": bool(stock_context),
                    "context_keys": list(stock_context.keys()) if stock_context else []
                },
                "position_sizing_test": {
                    "confidence": 0.85,
                    "volatility": 1.2,
                    "calculated_size": test_position
                },
                "risk_management_test": {
                    "price": 150.0,
                    "signal": "LONG",
                    "stop_loss": test_stop,
                    "take_profit": test_tp
                },
                "test_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Signal generation test failed: {e}")
            return {
                "test_status": "failed",
                "error": str(e)
            }
