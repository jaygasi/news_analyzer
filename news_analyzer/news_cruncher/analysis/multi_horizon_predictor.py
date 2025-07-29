"""
Multi-Horizon Predictor - Predicts profits at multiple time horizons (15min, 1h, 4h, EOD)
Python 3.13.3 compatible
"""
import asyncio
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import joblib
from config import Config
from utils.simple_logger import log_info, log_error, log_debug, log_warning


@dataclass
class ProfitPrediction:
    """Multi-horizon profit prediction with uncertainty quantification"""
    ticker: str
    
    # Profit predictions at multiple time horizons (in percentage points)
    profit_15min: float = 0.0
    profit_1h: float = 0.0  
    profit_4h: float = 0.0
    profit_eod: float = 0.0
    
    # Risk assessment
    downside_risk: float = 0.0      # Maximum expected loss
    upside_potential: float = 0.0   # Maximum expected gain
    volatility_forecast: float = 0.0 # Expected price volatility
    
    # Timing optimization
    optimal_entry_delay: int = 0     # Minutes to wait before entering
    optimal_exit_time: int = 0       # Minutes to optimal exit
    stop_loss_level: float = 0.0     # Exit if loss exceeds this %
    take_profit_level: float = 0.0   # Exit if profit exceeds this %
    
    # Confidence and uncertainty
    prediction_confidence: float = 0.0        # How sure we are (0-1)
    model_uncertainty: float = 0.0            # Model's uncertainty estimate
    similar_historical_cases: int = 0         # Number of similar past cases
    epistemic_uncertainty: float = 0.0       # Knowledge uncertainty
    aleatoric_uncertainty: float = 0.0       # Data uncertainty
    
    # Market context
    market_regime: str = "unknown"            # Current market regime
    regime_confidence: float = 0.0            # Confidence in regime detection
    
    # Supporting analysis
    news_impact_score: float = 0.0
    technical_momentum: float = 0.0
    earnings_catalyst: bool = False
    
    # Metadata
    analysis_timestamp: datetime = None
    model_version: str = "v1.0"


class ProfitPredictionNetwork(nn.Module):
    """Neural network for multi-horizon profit prediction"""
    
    def __init__(self, text_dim: int = 768, numerical_dim: int = 50, hidden_dim: int = 256):
        super().__init__()
        
        self.text_dim = text_dim
        self.numerical_dim = numerical_dim
        self.hidden_dim = hidden_dim
        
        # Text processing (from transformer embeddings)
        self.text_processor = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Numerical features processing
        self.numerical_processor = nn.Sequential(
            nn.Linear(numerical_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Combined features
        combined_dim = hidden_dim // 2 + hidden_dim // 4
        
        # Multi-horizon profit heads
        self.profit_heads = nn.ModuleDict({
            'profit_15min': nn.Linear(combined_dim, 1),
            'profit_1h': nn.Linear(combined_dim, 1),
            'profit_4h': nn.Linear(combined_dim, 1),
            'profit_eod': nn.Linear(combined_dim, 1)
        })
        
        # Risk prediction heads
        self.risk_heads = nn.ModuleDict({
            'downside_risk': nn.Linear(combined_dim, 1),
            'upside_potential': nn.Linear(combined_dim, 1),
            'volatility': nn.Linear(combined_dim, 1)
        })
        
        # Uncertainty estimation heads
        self.uncertainty_heads = nn.ModuleDict({
            'model_uncertainty': nn.Linear(combined_dim, 1),
            'epistemic_uncertainty': nn.Linear(combined_dim, 1),
            'aleatoric_uncertainty': nn.Linear(combined_dim, 1)
        })
        
        # Confidence and timing heads
        self.meta_heads = nn.ModuleDict({
            'confidence': nn.Linear(combined_dim, 1),
            'optimal_exit_time': nn.Linear(combined_dim, 1),
            'entry_delay': nn.Linear(combined_dim, 1)
        })
        
    def forward(self, text_features: torch.Tensor, numerical_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning all predictions"""
        
        # Process inputs
        text_processed = self.text_processor(text_features)
        numerical_processed = self.numerical_processor(numerical_features)
        
        # Combine features
        combined = torch.cat([text_processed, numerical_processed], dim=1)
        
        # Get all predictions
        outputs = {}
        
        # Profit predictions
        for horizon, head in self.profit_heads.items():
            outputs[horizon] = torch.tanh(head(combined)) * 0.1  # Scale to ±10%
            
        # Risk predictions
        for risk_type, head in self.risk_heads.items():
            outputs[risk_type] = torch.sigmoid(head(combined)) * 0.2  # Scale to 0-20%
            
        # Uncertainty predictions
        for uncertainty_type, head in self.uncertainty_heads.items():
            outputs[uncertainty_type] = torch.sigmoid(head(combined))  # Scale to 0-1
            
        # Meta predictions
        outputs['confidence'] = torch.sigmoid(self.meta_heads['confidence'](combined))
        outputs['optimal_exit_time'] = torch.sigmoid(self.meta_heads['optimal_exit_time'](combined)) * 480  # 0-480 minutes
        outputs['entry_delay'] = torch.sigmoid(self.meta_heads['entry_delay'](combined)) * 30  # 0-30 minutes
        
        return outputs


class MultiHorizonPredictor:
    """Main class for multi-horizon profit prediction"""
    
    def __init__(self):
        """Initialize the multi-horizon predictor"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.tokenizer = None
        self.text_model = None
        self.profit_model = None
        self.scaler = None
        
        # Model state
        self.model_loaded = False
        self.model_version = "1.0"
        
        # Performance tracking
        self.predictions_made = 0
        self.total_accuracy = 0.0
        
        # Initialize models
        self._initialize_models()
        
        log_info("🔮 Multi-Horizon Predictor initialized")

    def _initialize_models(self) -> None:
        """Initialize all required models"""
        try:
            # Initialize tokenizer and text model
            log_debug("Loading text processing models...")
            self.tokenizer = AutoTokenizer.from_pretrained('nlpaueb/sec-bert-base')
            self.text_model = AutoModel.from_pretrained('nlpaueb/sec-bert-base')
            self.text_model.to(self.device)
            self.text_model.eval()
            
            # Initialize profit prediction network
            log_debug("Initializing profit prediction network...")
            self.profit_model = ProfitPredictionNetwork()
            self.profit_model.to(self.device)
            
            # Initialize scaler for numerical features
            self.scaler = StandardScaler()
            
            # Load pre-trained weights if available
            self._load_pretrained_weights()
            
            self.model_loaded = True
            log_info("✅ All models initialized successfully")
            
        except Exception as e:
            log_error(f"Model initialization failed: {e}")
            self.model_loaded = False

    def _load_pretrained_weights(self) -> None:
        """Load pre-trained model weights if available"""
        try:
            model_path = Config.MODEL_CHECKPOINT_DIR / 'multi_horizon_predictor.pth'
            scaler_path = Config.MODEL_CHECKPOINT_DIR / 'profit_scaler.joblib'
            
            if model_path.exists():
                log_debug("Loading pre-trained profit model...")
                checkpoint = torch.load(model_path, map_location=self.device)
                self.profit_model.load_state_dict(checkpoint['model_state_dict'])
                log_info("✅ Pre-trained profit model loaded")
                
            if scaler_path.exists():
                log_debug("Loading feature scaler...")
                self.scaler = joblib.load(scaler_path)
                log_info("✅ Feature scaler loaded")
                
        except Exception as e:
            log_warning(f"Could not load pre-trained weights: {e}")

    async def predict_profit(self, 
                           ticker: str,
                           articles: List[Dict[str, Any]],
                           market_regime: Dict[str, Any]) -> ProfitPrediction:
        """
        Predict profit across multiple time horizons for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            articles: List of news articles related to the ticker
            market_regime: Current market regime information
            
        Returns:
            ProfitPrediction object with multi-horizon predictions
        """
        try:
            if not self.model_loaded:
                log_error("Models not loaded, cannot make predictions")
                return self._create_default_prediction(ticker)
            
            log_debug(f"Predicting profit for {ticker} with {len(articles)} articles")
            
            # Step 1: Extract and process text features
            text_features = await self._extract_text_features(articles)
            
            # Step 2: Extract numerical features
            numerical_features = await self._extract_numerical_features(ticker, articles, market_regime)
            
            # Step 3: Make model predictions
            model_outputs = await self._make_model_predictions(text_features, numerical_features)
            
            # Step 4: Calculate uncertainty estimates
            uncertainty_estimates = await self._estimate_uncertainty(text_features, numerical_features)
            
            # Step 5: Find similar historical cases
            similar_cases = await self._find_similar_cases(ticker, text_features, numerical_features)
            
            # Step 6: Create comprehensive prediction
            prediction = self._create_profit_prediction(
                ticker=ticker,
                model_outputs=model_outputs,
                uncertainty_estimates=uncertainty_estimates,
                similar_cases=similar_cases,
                market_regime=market_regime,
                articles=articles
            )
            
            self.predictions_made += 1
            
            log_debug(f"Profit prediction for {ticker}: "
                     f"1h={prediction.profit_1h:+.2%}, 4h={prediction.profit_4h:+.2%}, "
                     f"confidence={prediction.prediction_confidence:.2f}")
            
            return prediction
            
        except Exception as e:
            log_error(f"Error predicting profit for {ticker}: {e}")
            return self._create_default_prediction(ticker)

    async def _extract_text_features(self, articles: List[Dict[str, Any]]) -> torch.Tensor:
        """Extract text features using transformer model"""
        try:
            # Combine all article texts
            combined_text = ""
            for article in articles:
                title = article.get('title', '')
                content = article.get('content', '')
                combined_text += f"{title} {content} "
            
            # Limit text length
            max_length = 512
            if len(combined_text) > max_length * 4:  # Rough token estimate
                combined_text = combined_text[:max_length * 4]
            
            # Tokenize
            inputs = self.tokenizer(
                combined_text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                text_features = outputs.last_hidden_state.mean(dim=1)  # Average pooling
            
            return text_features
            
        except Exception as e:
            log_error(f"Error extracting text features: {e}")
            # Return default features
            return torch.zeros(1, 768, device=self.device)

    async def _extract_numerical_features(self, 
                                        ticker: str,
                                        articles: List[Dict[str, Any]],
                                        market_regime: Dict[str, Any]) -> torch.Tensor:
        """Extract numerical features for prediction"""
        try:
            features = []
            
            # Article-based features
            features.append(len(articles))  # Number of articles
            
            # Calculate sentiment scores
            positive_count = sum(1 for a in articles if a.get('sentiment', 0) > 0)
            negative_count = sum(1 for a in articles if a.get('sentiment', 0) < 0)
            features.extend([positive_count, negative_count])
            
            # Time-based features
            now = datetime.now()
            features.extend([
                now.hour,  # Hour of day
                now.weekday(),  # Day of week
                (now.hour >= 9.5 and now.hour < 16),  # Market hours
            ])
            
            # Market regime features
            regime_map = {'bull_market': 1, 'bear_market': -1, 'sideways': 0, 'unknown': 0}
            features.append(regime_map.get(market_regime.get('regime', 'unknown'), 0))
            features.append(market_regime.get('confidence', 0.5))
            features.append(market_regime.get('volatility', 0.15))
            
            # Article recency features
            recent_articles = sum(1 for a in articles if self._is_recent_article(a, hours=4))
            features.append(recent_articles)
            
            # Earnings-related features
            earnings_mentioned = sum(1 for a in articles if 'earnings' in a.get('title', '').lower())
            features.append(earnings_mentioned)
            
            # Add more features to reach target dimension
            while len(features) < 50:
                features.append(0.0)  # Padding features
            
            # Convert to tensor
            features_array = np.array(features[:50], dtype=np.float32)
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'scale_'):
                features_array = self.scaler.transform(features_array.reshape(1, -1)).flatten()
            
            return torch.tensor(features_array, dtype=torch.float32, device=self.device).unsqueeze(0)
            
        except Exception as e:
            log_error(f"Error extracting numerical features: {e}")
            return torch.zeros(1, 50, device=self.device)

    def _is_recent_article(self, article: Dict[str, Any], hours: int) -> bool:
        """Check if article is recent within specified hours"""
        try:
            published_at = article.get('publishedAt')
            if published_at:
                if isinstance(published_at, str):
                    published_dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                else:
                    published_dt = published_at
                
                cutoff = datetime.now() - timedelta(hours=hours)
                return published_dt >= cutoff
        except:
            pass
        return False

    async def _make_model_predictions(self, 
                                    text_features: torch.Tensor,
                                    numerical_features: torch.Tensor) -> Dict[str, float]:
        """Make predictions using the neural network"""
        try:
            with torch.no_grad():
                outputs = self.profit_model(text_features, numerical_features)
                
                # Convert tensors to floats
                predictions = {}
                for key, tensor in outputs.items():
                    predictions[key] = float(tensor.squeeze().cpu().numpy())
                
                return predictions
                
        except Exception as e:
            log_error(f"Error making model predictions: {e}")
            return self._get_default_predictions()

    def _get_default_predictions(self) -> Dict[str, float]:
        """Get default predictions when model fails"""
        return {
            'profit_15min': 0.0,
            'profit_1h': 0.0,
            'profit_4h': 0.0,
            'profit_eod': 0.0,
            'downside_risk': 0.02,
            'upside_potential': 0.02,
            'volatility': 0.15,
            'model_uncertainty': 0.5,
            'epistemic_uncertainty': 0.5,
            'aleatoric_uncertainty': 0.3,
            'confidence': 0.5,
            'optimal_exit_time': 240,
            'entry_delay': 5
        }

    async def _estimate_uncertainty(self, 
                                  text_features: torch.Tensor,
                                  numerical_features: torch.Tensor) -> Dict[str, float]:
        """Estimate prediction uncertainty using Monte Carlo dropout"""
        try:
            if not self.model_loaded:
                return {'epistemic': 0.5, 'aleatoric': 0.3, 'total': 0.5}
            
            # Enable dropout for uncertainty estimation
            self.profit_model.train()
            
            # Make multiple predictions with dropout
            num_samples = 20
            predictions = []
            
            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = self.profit_model(text_features, numerical_features)
                    # Focus on main profit predictions
                    profit_pred = {
                        'profit_1h': float(outputs['profit_1h'].squeeze().cpu().numpy()),
                        'profit_4h': float(outputs['profit_4h'].squeeze().cpu().numpy())
                    }
                    predictions.append(profit_pred)
            
            # Calculate uncertainty
            profits_1h = [p['profit_1h'] for p in predictions]
            profits_4h = [p['profit_4h'] for p in predictions]
            
            epistemic_uncertainty = (np.std(profits_1h) + np.std(profits_4h)) / 2
            aleatoric_uncertainty = np.mean([abs(p['profit_1h'] - p['profit_4h']) for p in predictions])
            total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
            
            # Set model back to eval mode
            self.profit_model.eval()
            
            return {
                'epistemic': float(epistemic_uncertainty),
                'aleatoric': float(aleatoric_uncertainty),
                'total': float(total_uncertainty)
            }
            
        except Exception as e:
            log_error(f"Error estimating uncertainty: {e}")
            return {'epistemic': 0.5, 'aleatoric': 0.3, 'total': 0.5}

    async def _find_similar_cases(self, 
                                ticker: str,
                                text_features: torch.Tensor,
                                numerical_features: torch.Tensor) -> int:
        """Find number of similar historical cases"""
        try:
            # This would be implemented to search through historical data
            # For now, return a reasonable estimate based on features
            
            # Simple heuristic: more articles = more similar cases
            num_features = float(numerical_features[0, 0].cpu().numpy())  # Number of articles
            similar_cases = max(1, int(num_features * 2))  # Rough estimate
            
            return min(similar_cases, 50)  # Cap at 50
            
        except Exception as e:
            log_error(f"Error finding similar cases: {e}")
            return 5  # Default

    def _create_profit_prediction(self,
                                ticker: str,
                                model_outputs: Dict[str, float],
                                uncertainty_estimates: Dict[str, float],
                                similar_cases: int,
                                market_regime: Dict[str, Any],
                                articles: List[Dict[str, Any]]) -> ProfitPrediction:
        """Create comprehensive profit prediction object"""
        
        # Calculate additional metrics
        news_impact_score = min(1.0, len(articles) / 10.0)  # Scale based on article count
        earnings_catalyst = any('earnings' in a.get('title', '').lower() for a in articles)
        
        # Technical momentum (placeholder - would integrate with technical analysis)
        technical_momentum = 0.5  # Neutral default
        
        prediction = ProfitPrediction(
            ticker=ticker,
            
            # Multi-horizon profits
            profit_15min=model_outputs.get('profit_15min', 0.0),
            profit_1h=model_outputs.get('profit_1h', 0.0),
            profit_4h=model_outputs.get('profit_4h', 0.0),
            profit_eod=model_outputs.get('profit_eod', 0.0),
            
            # Risk metrics
            downside_risk=model_outputs.get('downside_risk', 0.02),
            upside_potential=model_outputs.get('upside_potential', 0.02),
            volatility_forecast=model_outputs.get('volatility', 0.15),
            
            # Timing
            optimal_entry_delay=int(model_outputs.get('entry_delay', 5)),
            optimal_exit_time=int(model_outputs.get('optimal_exit_time', 240)),
            stop_loss_level=model_outputs.get('downside_risk', 0.02) * 0.8,
            take_profit_level=model_outputs.get('upside_potential', 0.02) * 0.9,
            
            # Uncertainty and confidence
            prediction_confidence=model_outputs.get('confidence', 0.5),
            model_uncertainty=uncertainty_estimates.get('total', 0.5),
            epistemic_uncertainty=uncertainty_estimates.get('epistemic', 0.5),
            aleatoric_uncertainty=uncertainty_estimates.get('aleatoric', 0.3),
            similar_historical_cases=similar_cases,
            
            # Market context
            market_regime=market_regime.get('regime', 'unknown'),
            regime_confidence=market_regime.get('confidence', 0.5),
            
            # Supporting metrics
            news_impact_score=news_impact_score,
            technical_momentum=technical_momentum,
            earnings_catalyst=earnings_catalyst,
            
            # Metadata
            analysis_timestamp=datetime.now(),
            model_version=self.model_version
        )
        
        return prediction

    def _create_default_prediction(self, ticker: str) -> ProfitPrediction:
        """Create default prediction when models fail"""
        return ProfitPrediction(
            ticker=ticker,
            profit_15min=0.0,
            profit_1h=0.0,
            profit_4h=0.0,
            profit_eod=0.0,
            downside_risk=0.02,
            upside_potential=0.02,
            volatility_forecast=0.15,
            optimal_entry_delay=5,
            optimal_exit_time=240,
            stop_loss_level=0.015,
            take_profit_level=0.015,
            prediction_confidence=0.3,
            model_uncertainty=0.7,
            similar_historical_cases=1,
            epistemic_uncertainty=0.6,
            aleatoric_uncertainty=0.4,
            market_regime="unknown",
            regime_confidence=0.3,
            news_impact_score=0.3,
            technical_momentum=0.5,
            earnings_catalyst=False,
            analysis_timestamp=datetime.now(),
            model_version=self.model_version
        )

    async def reload_models(self) -> None:
        """Reload models after learning updates"""
        try:
            log_info("🔄 Reloading models after learning update...")
            self._load_pretrained_weights()
            log_info("✅ Models reloaded successfully")
        except Exception as e:
            log_error(f"Error reloading models: {e}")

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and information"""
        return {
            'available': self.model_loaded,
            'model_version': self.model_version,
            'device': str(self.device),
            'predictions_made': self.predictions_made,
            'architecture': 'Multi-Horizon Transformer + Neural Network',
            'horizons': ['15min', '1h', '4h', 'eod'],
            'uncertainty_estimation': 'Monte Carlo Dropout',
            'features': [
                'Multi-horizon profit prediction',
                'Uncertainty quantification',
                'Risk assessment',
                'Optimal timing prediction',
                'Similar case matching'
            ]
        }

    def save_model_checkpoint(self, performance_metrics: Dict[str, float]) -> None:
        """Save model checkpoint with performance metrics"""
        try:
            if not self.model_loaded:
                return
                
            checkpoint_path = Config.MODEL_CHECKPOINT_DIR / 'multi_horizon_predictor.pth'
            
            checkpoint = {
                'model_state_dict': self.profit_model.state_dict(),
                'model_version': self.model_version,
                'performance_metrics': performance_metrics,
                'save_timestamp': datetime.now().isoformat(),
                'predictions_made': self.predictions_made
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Save scaler
            scaler_path = Config.MODEL_CHECKPOINT_DIR / 'profit_scaler.joblib'
            joblib.dump(self.scaler, scaler_path)
            
            log_info(f"💾 Model checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            log_error(f"Error saving model checkpoint: {e}")
