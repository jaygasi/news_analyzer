"""
Enhanced Configuration for Profit-Maximizing Trading System
Python 3.13.3 compatible
"""
import os
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Enhanced configuration class with profit-focused parameters"""
    
    # ================================================================
    # 🏗️ BASIC SYSTEM CONFIGURATION
    # ================================================================
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.absolute()
    DATA_DIR: Path = PROJECT_ROOT / "data"
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    # Basic timing
    ANALYSIS_INTERVAL_MINUTES: int = int(os.getenv('ANALYSIS_INTERVAL_MINUTES', '30'))
    LEARNING_UPDATE_FREQUENCY: int = int(os.getenv('LEARNING_UPDATE_FREQUENCY', '4'))  # Every 4 cycles
    
    # ================================================================
    # 💰 PROFIT-FOCUSED CORE PARAMETERS
    # ================================================================
    
    # Profit prediction thresholds
    MIN_EXPECTED_PROFIT: float = float(os.getenv('MIN_EXPECTED_PROFIT', '0.015'))  # 1.5% minimum expected profit
    MIN_PREDICTION_CONFIDENCE: float = float(os.getenv('MIN_PREDICTION_CONFIDENCE', '0.7'))  # 70% confidence minimum
    MIN_SHARPE_RATIO: float = float(os.getenv('MIN_SHARPE_RATIO', '1.5'))  # Minimum expected Sharpe ratio
    
    # Multi-horizon profit weights (how much to weight each time horizon)
    PROFIT_WEIGHT_15MIN: float = float(os.getenv('PROFIT_WEIGHT_15MIN', '0.1'))    # 10% weight
    PROFIT_WEIGHT_1H: float = float(os.getenv('PROFIT_WEIGHT_1H', '0.3'))          # 30% weight  
    PROFIT_WEIGHT_4H: float = float(os.getenv('PROFIT_WEIGHT_4H', '0.4'))          # 40% weight
    PROFIT_WEIGHT_EOD: float = float(os.getenv('PROFIT_WEIGHT_EOD', '0.2'))        # 20% weight
    
    # Risk tolerance parameters
    MAX_POSITION_SIZE: float = float(os.getenv('MAX_POSITION_SIZE', '0.1'))        # 10% max position
    MAX_PORTFOLIO_RISK: float = float(os.getenv('MAX_PORTFOLIO_RISK', '0.15'))     # 15% max portfolio risk
    DEFAULT_STOP_LOSS: float = float(os.getenv('DEFAULT_STOP_LOSS', '0.02'))       # 2% stop loss
    DEFAULT_TAKE_PROFIT: float = float(os.getenv('DEFAULT_TAKE_PROFIT', '0.04'))   # 4% take profit
    
    # Kelly criterion parameters
    KELLY_MULTIPLIER: float = float(os.getenv('KELLY_MULTIPLIER', '0.5'))          # Conservative Kelly
    MAX_KELLY_FRACTION: float = float(os.getenv('MAX_KELLY_FRACTION', '0.2'))      # Max 20% Kelly
    MIN_KELLY_FRACTION: float = float(os.getenv('MIN_KELLY_FRACTION', '0.01'))     # Min 1% Kelly
    
    # ================================================================
    # 🧠 UNCERTAINTY AND CONFIDENCE PARAMETERS
    # ================================================================
    
    # Uncertainty thresholds
    MAX_MODEL_UNCERTAINTY: float = float(os.getenv('MAX_MODEL_UNCERTAINTY', '0.3'))         # Skip if uncertainty > 30%
    MIN_HISTORICAL_CASES: int = int(os.getenv('MIN_HISTORICAL_CASES', '5'))                 # Min similar historical cases
    EPISTEMIC_UNCERTAINTY_THRESHOLD: float = float(os.getenv('EPISTEMIC_UNCERTAINTY_THRESHOLD', '0.4'))
    ALEATORIC_UNCERTAINTY_THRESHOLD: float = float(os.getenv('ALEATORIC_UNCERTAINTY_THRESHOLD', '0.3'))
    
    # Confidence boosting
    HIGH_CONFIDENCE_BONUS: float = float(os.getenv('HIGH_CONFIDENCE_BONUS', '0.2'))         # Boost for high confidence
    CONSENSUS_BONUS: float = float(os.getenv('CONSENSUS_BONUS', '0.15'))                    # Boost for model consensus
    MARKET_REGIME_CONFIDENCE_MIN: float = float(os.getenv('MARKET_REGIME_CONFIDENCE_MIN', '0.6'))
    
    # ================================================================
    # 📊 MARKET REGIME DETECTION PARAMETERS
    # ================================================================
    
    # Regime detection settings
    REGIME_DETECTION_LOOKBACK_DAYS: int = int(os.getenv('REGIME_DETECTION_LOOKBACK_DAYS', '30'))
    VOLATILITY_HIGH_THRESHOLD: float = float(os.getenv('VOLATILITY_HIGH_THRESHOLD', '0.25'))     # 25% annual vol
    VOLATILITY_LOW_THRESHOLD: float = float(os.getenv('VOLATILITY_LOW_THRESHOLD', '0.12'))       # 12% annual vol
    TREND_STRENGTH_THRESHOLD: float = float(os.getenv('TREND_STRENGTH_THRESHOLD', '0.6'))
    
    # Market regime adaptations
    BULL_MARKET_PROFIT_MULTIPLIER: float = float(os.getenv('BULL_MARKET_PROFIT_MULTIPLIER', '1.2'))
    BEAR_MARKET_PROFIT_MULTIPLIER: float = float(os.getenv('BEAR_MARKET_PROFIT_MULTIPLIER', '0.8'))
    HIGH_VOL_RISK_MULTIPLIER: float = float(os.getenv('HIGH_VOL_RISK_MULTIPLIER', '1.5'))
    LOW_VOL_RISK_MULTIPLIER: float = float(os.getenv('LOW_VOL_RISK_MULTIPLIER', '0.7'))
    
    # ================================================================
    # 🎓 PROFIT-BASED LEARNING SYSTEM
    # ================================================================
    
    # Learning system configuration
    ENABLE_PROFIT_LEARNING: bool = os.getenv('ENABLE_PROFIT_LEARNING', 'true').lower() == 'true'
    LEARNING_BATCH_SIZE: int = int(os.getenv('LEARNING_BATCH_SIZE', '32'))
    LEARNING_RATE: float = float(os.getenv('LEARNING_RATE', '1e-4'))
    LEARNING_EPOCHS: int = int(os.getenv('LEARNING_EPOCHS', '5'))
    
    # Profit-based labeling thresholds
    EXCELLENT_TRADE_THRESHOLD: float = float(os.getenv('EXCELLENT_TRADE_THRESHOLD', '0.03'))     # 3%+ profit
    GOOD_TRADE_THRESHOLD: float = float(os.getenv('GOOD_TRADE_THRESHOLD', '0.01'))               # 1%+ profit
    POOR_TRADE_THRESHOLD: float = float(os.getenv('POOR_TRADE_THRESHOLD', '-0.01'))              # -1% loss
    TERRIBLE_TRADE_THRESHOLD: float = float(os.getenv('TERRIBLE_TRADE_THRESHOLD', '-0.03'))      # -3% loss
    
    # Learning data requirements
    MIN_TRADES_FOR_LEARNING: int = int(os.getenv('MIN_TRADES_FOR_LEARNING', '50'))
    MIN_PROFITABLE_TRADES: int = int(os.getenv('MIN_PROFITABLE_TRADES', '15'))
    MIN_UNPROFITABLE_TRADES: int = int(os.getenv('MIN_UNPROFITABLE_TRADES', '10'))
    
    # Continuous learning parameters
    REAL_TIME_LEARNING_ENABLED: bool = os.getenv('REAL_TIME_LEARNING_ENABLED', 'true').lower() == 'true'
    INCREMENTAL_LEARNING_BATCH: int = int(os.getenv('INCREMENTAL_LEARNING_BATCH', '10'))
    LEARNING_MEMORY_SIZE: int = int(os.getenv('LEARNING_MEMORY_SIZE', '1000'))
    
    # ================================================================
    # ⏰ EXIT TIMING OPTIMIZATION
    # ================================================================
    
    # Exit timing parameters
    ENABLE_EXIT_OPTIMIZATION: bool = os.getenv('ENABLE_EXIT_OPTIMIZATION', 'true').lower() == 'true'
    MAX_HOLD_TIME_MINUTES: int = int(os.getenv('MAX_HOLD_TIME_MINUTES', '480'))     # 8 hours max hold
    MIN_HOLD_TIME_MINUTES: int = int(os.getenv('MIN_HOLD_TIME_MINUTES', '15'))      # 15 min minimum
    
    # Dynamic exit thresholds
    PROFIT_ACCELERATION_THRESHOLD: float = float(os.getenv('PROFIT_ACCELERATION_THRESHOLD', '0.005'))  # 0.5%/hour
    PROFIT_DECELERATION_THRESHOLD: float = float(os.getenv('PROFIT_DECELERATION_THRESHOLD', '-0.002')) # -0.2%/hour
    VOLATILITY_EXIT_MULTIPLIER: float = float(os.getenv('VOLATILITY_EXIT_MULTIPLIER', '2.0'))
    
    # Time-based exit rules
    EOD_EXIT_ENABLED: bool = os.getenv('EOD_EXIT_ENABLED', 'true').lower() == 'true'
    EOD_EXIT_TIME_MINUTES: int = int(os.getenv('EOD_EXIT_TIME_MINUTES', '30'))      # Exit 30min before close
    WEEKEND_EXIT_ENABLED: bool = os.getenv('WEEKEND_EXIT_ENABLED', 'true').lower() == 'true'
    
    # ================================================================
    # 📈 ENHANCED PRICE TRACKING
    # ================================================================
    
    # Multi-horizon tracking intervals
    PRICE_CHECK_INTERVALS: List[int] = [
        int(x) for x in os.getenv('PRICE_CHECK_INTERVALS', '15,60,240').split(',')
    ]  # 15min, 1h, 4h
    
    # Price tracking configuration
    PRICE_UPDATE_FREQUENCY_SECONDS: int = int(os.getenv('PRICE_UPDATE_FREQUENCY_SECONDS', '60'))
    PRICE_HISTORY_DAYS: int = int(os.getenv('PRICE_HISTORY_DAYS', '7'))
    ENABLE_AFTER_HOURS_TRACKING: bool = os.getenv('ENABLE_AFTER_HOURS_TRACKING', 'true').lower() == 'true'
    
    # Market hours (EST)
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 30
    MARKET_CLOSE_HOUR: int = 16
    MARKET_CLOSE_MINUTE: int = 0
    
    # ================================================================
    # 🤖 AI MODEL CONFIGURATION
    # ================================================================
    
    # Multi-horizon predictor
    PREDICTOR_MODEL_TYPE: str = os.getenv('PREDICTOR_MODEL_TYPE', 'ensemble')  # ensemble, transformer, lstm
    PREDICTOR_ENSEMBLE_SIZE: int = int(os.getenv('PREDICTOR_ENSEMBLE_SIZE', '5'))
    PREDICTOR_DROPOUT_RATE: float = float(os.getenv('PREDICTOR_DROPOUT_RATE', '0.1'))
    
    # Model ensemble weights
    FINBERT_WEIGHT: float = float(os.getenv('FINBERT_WEIGHT', '0.3'))
    ROBERTA_WEIGHT: float = float(os.getenv('ROBERTA_WEIGHT', '0.3'))
    TECHNICAL_WEIGHT: float = float(os.getenv('TECHNICAL_WEIGHT', '0.2'))
    SENTIMENT_WEIGHT: float = float(os.getenv('SENTIMENT_WEIGHT', '0.2'))
    
    # Model performance thresholds
    MIN_MODEL_ACCURACY: float = float(os.getenv('MIN_MODEL_ACCURACY', '0.65'))
    MODEL_CONFIDENCE_THRESHOLD: float = float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', '0.7'))
    ENSEMBLE_AGREEMENT_THRESHOLD: float = float(os.getenv('ENSEMBLE_AGREEMENT_THRESHOLD', '0.6'))
    
    # ================================================================
    # 📊 FUNDAMENTAL FILTERING (EXISTING)
    # ================================================================
    
    # Enable/disable fundamental filtering
    ENABLE_FUNDAMENTAL_FILTERING: bool = os.getenv('ENABLE_FUNDAMENTAL_FILTERING', 'true').lower() == 'true'
    
    # Stock filtering criteria
    MIN_STOCK_PRICE: float = float(os.getenv('MIN_STOCK_PRICE', '5.0'))
    MAX_STOCK_PRICE: float = float(os.getenv('MAX_STOCK_PRICE', '1000.0'))
    MIN_MARKET_CAP: int = int(os.getenv('MIN_MARKET_CAP', '500000000'))  # 500M minimum
    MIN_VOLUME: int = int(os.getenv('MIN_VOLUME', '500000'))  # 500K shares
    MAX_BETA: float = float(os.getenv('MAX_BETA', '3.0'))
    
    # Allowed exchanges
    ALLOWED_EXCHANGES: List[str] = os.getenv('ALLOWED_EXCHANGES', 'NASDAQ,NYSE,AMEX').split(',')
    
    # ================================================================
    # 📰 NEWS AND DATA SOURCES (EXISTING)
    # ================================================================
    
    # Enable/disable news sources
    ENABLE_STOCK_NEWS: bool = os.getenv('ENABLE_STOCK_NEWS', 'true').lower() == 'true'
    ENABLE_PRESS_RELEASES: bool = os.getenv('ENABLE_PRESS_RELEASES', 'true').lower() == 'true'
    ENABLE_EARNINGS_EVENTS: bool = os.getenv('ENABLE_EARNINGS_EVENTS', 'true').lower() == 'true'
    ENABLE_TECHNICAL_ANALYSIS: bool = os.getenv('ENABLE_TECHNICAL_ANALYSIS', 'true').lower() == 'true'
    
    # Article limits
    MAX_ARTICLES_PER_SOURCE: int = int(os.getenv('MAX_ARTICLES_PER_SOURCE', '1000'))
    MAX_EARNINGS_EVENTS_PER_CYCLE: int = int(os.getenv('MAX_EARNINGS_EVENTS_PER_CYCLE', '10'))
    
    # ================================================================
    # 💾 DATA STORAGE AND LOGGING
    # ================================================================
    
    # Database paths
    ARTICLE_TRACKER_DB: Path = DATA_DIR / 'article_tracker.db'
    PROFIT_TRACKING_DB: Path = DATA_DIR / 'profit_tracking.db'
    LEARNING_DATA_DB: Path = DATA_DIR / 'learning_data.db'
    
    # CSV output
    CSV_OUTPUT_PATH: Path = DATA_DIR / 'profit_trading_decisions.csv'
    PERFORMANCE_LOG_PATH: Path = DATA_DIR / 'performance_log.csv'
    LEARNING_LOG_PATH: Path = DATA_DIR / 'learning_progress.csv'
    
    # Model checkpoints
    MODEL_CHECKPOINT_DIR: Path = DATA_DIR / 'model_checkpoints'
    MODEL_CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # ================================================================
    # 🔑 API CONFIGURATION (EXISTING)
    # ================================================================
    
    # API Keys (from environment)
    FMP_API_KEY: str = os.getenv('FMP_API_KEY', '')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    
    # API rate limits
    FMP_REQUESTS_PER_MINUTE: int = int(os.getenv('FMP_REQUESTS_PER_MINUTE', '300'))
    OPENAI_REQUESTS_PER_MINUTE: int = int(os.getenv('OPENAI_REQUESTS_PER_MINUTE', '50'))
    
    # ================================================================
    # 🔧 HELPER METHODS
    # ================================================================
    
    @classmethod
    def get_profit_weights(cls) -> Dict[str, float]:
        """Get profit prediction weights for different time horizons"""
        return {
            '15min': cls.PROFIT_WEIGHT_15MIN,
            '1h': cls.PROFIT_WEIGHT_1H,
            '4h': cls.PROFIT_WEIGHT_4H,
            'eod': cls.PROFIT_WEIGHT_EOD
        }
    
    @classmethod
    def get_risk_parameters(cls) -> Dict[str, float]:
        """Get risk management parameters"""
        return {
            'max_position_size': cls.MAX_POSITION_SIZE,
            'max_portfolio_risk': cls.MAX_PORTFOLIO_RISK,
            'default_stop_loss': cls.DEFAULT_STOP_LOSS,
            'default_take_profit': cls.DEFAULT_TAKE_PROFIT,
            'kelly_multiplier': cls.KELLY_MULTIPLIER,
            'max_kelly_fraction': cls.MAX_KELLY_FRACTION,
            'min_kelly_fraction': cls.MIN_KELLY_FRACTION
        }
    
    @classmethod
    def get_market_regime_config(cls) -> Dict[str, Any]:
        """Get market regime detection configuration"""
        return {
            'lookback_days': cls.REGIME_DETECTION_LOOKBACK_DAYS,
            'volatility_high_threshold': cls.VOLATILITY_HIGH_THRESHOLD,
            'volatility_low_threshold': cls.VOLATILITY_LOW_THRESHOLD,
            'trend_strength_threshold': cls.TREND_STRENGTH_THRESHOLD,
            'bull_market_multiplier': cls.BULL_MARKET_PROFIT_MULTIPLIER,
            'bear_market_multiplier': cls.BEAR_MARKET_PROFIT_MULTIPLIER,
            'high_vol_risk_multiplier': cls.HIGH_VOL_RISK_MULTIPLIER,
            'low_vol_risk_multiplier': cls.LOW_VOL_RISK_MULTIPLIER
        }
    
    @classmethod
    def get_learning_config(cls) -> Dict[str, Any]:
        """Get learning system configuration"""
        return {
            'enabled': cls.ENABLE_PROFIT_LEARNING,
            'batch_size': cls.LEARNING_BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'epochs': cls.LEARNING_EPOCHS,
            'excellent_threshold': cls.EXCELLENT_TRADE_THRESHOLD,
            'good_threshold': cls.GOOD_TRADE_THRESHOLD,
            'poor_threshold': cls.POOR_TRADE_THRESHOLD,
            'terrible_threshold': cls.TERRIBLE_TRADE_THRESHOLD,
            'min_trades': cls.MIN_TRADES_FOR_LEARNING,
            'min_profitable': cls.MIN_PROFITABLE_TRADES,
            'min_unprofitable': cls.MIN_UNPROFITABLE_TRADES,
            'real_time_enabled': cls.REAL_TIME_LEARNING_ENABLED,
            'incremental_batch': cls.INCREMENTAL_LEARNING_BATCH,
            'memory_size': cls.LEARNING_MEMORY_SIZE
        }
    
    @classmethod
    def get_exit_timing_config(cls) -> Dict[str, Any]:
        """Get exit timing optimization configuration"""
        return {
            'enabled': cls.ENABLE_EXIT_OPTIMIZATION,
            'max_hold_time': cls.MAX_HOLD_TIME_MINUTES,
            'min_hold_time': cls.MIN_HOLD_TIME_MINUTES,
            'profit_acceleration_threshold': cls.PROFIT_ACCELERATION_THRESHOLD,
            'profit_deceleration_threshold': cls.PROFIT_DECELERATION_THRESHOLD,
            'volatility_exit_multiplier': cls.VOLATILITY_EXIT_MULTIPLIER,
            'eod_exit_enabled': cls.EOD_EXIT_ENABLED,
            'eod_exit_time': cls.EOD_EXIT_TIME_MINUTES,
            'weekend_exit_enabled': cls.WEEKEND_EXIT_ENABLED
        }
    
    @classmethod
    def get_model_ensemble_weights(cls) -> Dict[str, float]:
        """Get AI model ensemble weights"""
        return {
            'finbert': cls.FINBERT_WEIGHT,
            'roberta': cls.ROBERTA_WEIGHT,
            'technical': cls.TECHNICAL_WEIGHT,
            'sentiment': cls.SENTIMENT_WEIGHT
        }
    
    @classmethod
    def validate_configuration(cls) -> List[str]:
        """Validate configuration and return any warnings"""
        warnings = []
        
        # Check profit weights sum to 1
        profit_weights = cls.get_profit_weights()
        if abs(sum(profit_weights.values()) - 1.0) > 0.01:
            warnings.append(f"Profit weights don't sum to 1.0: {sum(profit_weights.values())}")
        
        # Check model weights sum to 1
        model_weights = cls.get_model_ensemble_weights()
        if abs(sum(model_weights.values()) - 1.0) > 0.01:
            warnings.append(f"Model weights don't sum to 1.0: {sum(model_weights.values())}")
        
        # Check risk parameters
        if cls.MAX_POSITION_SIZE > 0.5:
            warnings.append("MAX_POSITION_SIZE > 50% is very risky")
        
        if cls.MAX_KELLY_FRACTION > 0.3:
            warnings.append("MAX_KELLY_FRACTION > 30% is aggressive")
        
        # Check thresholds
        if cls.MIN_EXPECTED_PROFIT <= 0:
            warnings.append("MIN_EXPECTED_PROFIT should be positive")
        
        if cls.MIN_PREDICTION_CONFIDENCE < 0.5:
            warnings.append("MIN_PREDICTION_CONFIDENCE < 50% may lead to poor trades")
        
        # Check API keys
        if not cls.FMP_API_KEY:
            warnings.append("FMP_API_KEY not set - data fetching will fail")
        
        return warnings
    
    @classmethod
    def get_trading_thresholds(cls) -> Dict[str, float]:
        """Get all trading decision thresholds"""
        return {
            'min_expected_profit': cls.MIN_EXPECTED_PROFIT,
            'min_prediction_confidence': cls.MIN_PREDICTION_CONFIDENCE,
            'min_sharpe_ratio': cls.MIN_SHARPE_RATIO,
            'max_model_uncertainty': cls.MAX_MODEL_UNCERTAINTY,
            'min_historical_cases': cls.MIN_HISTORICAL_CASES,
            'market_regime_confidence_min': cls.MARKET_REGIME_CONFIDENCE_MIN
        }
    
    @classmethod
    def get_price_tracking_config(cls) -> Dict[str, Any]:
        """Get price tracking configuration"""
        return {
            'intervals': cls.PRICE_CHECK_INTERVALS,
            'update_frequency': cls.PRICE_UPDATE_FREQUENCY_SECONDS,
            'history_days': cls.PRICE_HISTORY_DAYS,
            'after_hours_enabled': cls.ENABLE_AFTER_HOURS_TRACKING,
            'market_open': f"{cls.MARKET_OPEN_HOUR:02d}:{cls.MARKET_OPEN_MINUTE:02d}",
            'market_close': f"{cls.MARKET_CLOSE_HOUR:02d}:{cls.MARKET_CLOSE_MINUTE:02d}"
        }