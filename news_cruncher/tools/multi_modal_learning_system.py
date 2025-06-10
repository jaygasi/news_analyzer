"""
Multi-Modal Learning System - Train FinBERT and RoBERTa on CSV Data
Eliminates training-inference gap and enables true adaptive learning
"""
import sys
from pathlib import Path

# Fix Python path - add the project root directory  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import pickle
from config import Config
from utils.simple_logger import log_info, log_error, log_debug
from analysis.enhanced_neural_analyzer import EnhancedNeuralAnalyzer

class TradingDataset(Dataset):
    """Dataset that combines text, numerical, and categorical features like CSV"""
    
    def __init__(self, texts: List[str], numerical_features: np.ndarray, 
                 categorical_features: Dict[str, np.ndarray], labels: np.ndarray, 
                 tokenizer, max_length: int = 512):
        self.texts = texts
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical_features': torch.FloatTensor(self.numerical_features[idx]),
            'categorical_features': {k: torch.LongTensor([v[idx]]) for k, v in self.categorical_features.items()},
            'labels': torch.LongTensor([self.labels[idx]])
        }

class MultiModalFinBERT(nn.Module):
    """FinBERT enhanced with numerical and categorical features from CSV"""
    
    def __init__(self, finbert_model_name: str = "ProsusAI/finbert", 
                 num_numerical_features: int = 10, categorical_vocab_sizes: Dict[str, int] = None,
                 hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        
        # Load FinBERT
        self.finbert = AutoModelForSequenceClassification.from_pretrained(
            finbert_model_name, 
            num_labels=num_classes,
            attn_implementation="eager"
        )
        
        # Freeze FinBERT initially (we'll fine-tune later)
        for param in self.finbert.parameters():
            param.requires_grad = False
        
        # Enable fine-tuning of last layer
        for param in self.finbert.classifier.parameters():
            param.requires_grad = True
        
        # Numerical features processor
        self.numerical_processor = nn.Sequential(
            nn.Linear(num_numerical_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, 16) 
            for name, vocab_size in (categorical_vocab_sizes or {}).items()
        })
        
        # Final fusion layer
        categorical_dim = len(categorical_vocab_sizes or {}) * 16
        fusion_input_dim = 768 + (hidden_dim // 2) + categorical_dim  # FinBERT + numerical + categorical
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, numerical_features, categorical_features):
        # FinBERT text processing
        finbert_outputs = self.finbert.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = finbert_outputs.pooler_output  # [batch_size, 768]
        
        # Process numerical features
        numerical_processed = self.numerical_processor(numerical_features)  # [batch_size, hidden_dim//2]
        
        # Process categorical features
        categorical_embeddings = []
        for name, values in categorical_features.items():
            if name in self.categorical_embeddings:
                emb = self.categorical_embeddings[name](values.squeeze())
                categorical_embeddings.append(emb)
        
        categorical_features_concat = torch.cat(categorical_embeddings, dim=1) if categorical_embeddings else torch.empty(text_features.size(0), 0).to(text_features.device)
        
        # Fuse all features
        combined_features = torch.cat([text_features, numerical_processed, categorical_features_concat], dim=1)
        
        # Final prediction
        logits = self.fusion_layer(combined_features)
        return logits

class MultiModalLearningSystem:
    """Complete system for training models on CSV data"""
    
    def __init__(self):
        self.csv_path = Config.CSV_OUTPUT_PATH
        self.model_save_dir = Config.DATA_DIR
        self.model_save_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.finbert_tokenizer = None
        self.enhanced_analyzer = None
        self.numerical_scaler = StandardScaler()
        self.categorical_encoders = {}
        self.label_encoder = LabelEncoder()
        
        # Training configuration
        self.batch_size = Config.ADAPTIVE_LEARNING_BATCH_SIZE if hasattr(Config, 'ADAPTIVE_LEARNING_BATCH_SIZE') else 8
        self.learning_rate = Config.ADAPTIVE_LEARNING_LEARNING_RATE if hasattr(Config, 'ADAPTIVE_LEARNING_LEARNING_RATE') else 2e-5
        self.epochs = Config.ADAPTIVE_LEARNING_EPOCHS if hasattr(Config, 'ADAPTIVE_LEARNING_EPOCHS') else 3
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, bool]:
        """Load CSV data and prepare for training"""
        if not self.csv_path.exists():
            log_error(f"CSV file not found: {self.csv_path}")
            return None, False
        
        try:
            df = pd.read_csv(self.csv_path)
            log_info(f"Loaded {len(df)} trading records from CSV")
            
            # Filter completed trades only
            completed_trades = df[df['tracking_status'] == 'completed'].copy()
            log_info(f"Found {len(completed_trades)} completed trades for training")
            
            if len(completed_trades) < 10:
                log_error("Need at least 10 completed trades for training")
                return None, False
            
            return completed_trades, True
            
        except Exception as e:
            log_error(f"Error loading CSV data: {e}")
            return None, False
    
    def create_performance_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Convert price changes to training labels"""
        labels = []
        
        for _, row in df.iterrows():
            # Use close price change as primary performance metric
            close_change = row.get('price_close_change_pct', 0)
            
            # Define thresholds from config
            profit_threshold = getattr(Config, 'ADAPTIVE_PROFIT_THRESHOLD', 0.0)
            good_threshold = getattr(Config, 'ADAPTIVE_GOOD_TRADE_THRESHOLD', 2.0)
            
            if close_change >= good_threshold:
                labels.append(2)  # Excellent trade
            elif close_change >= profit_threshold:
                labels.append(1)  # Good trade
            else:
                labels.append(0)  # Poor trade
        
        return np.array(labels)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
        """Extract features matching CSV structure"""
        
        # Text features (reasoning)
        texts = df['reasoning'].fillna('').astype(str).tolist()
        
        # Numerical features (matching CSV columns)
        numerical_columns = [
            'confidence', 'news_score', 'technical_score', 'combined_score',
            'news_confidence', 'technical_strength', 'article_count',
            'source_count', 'agreement_score'
        ]
        
        numerical_data = []
        for col in numerical_columns:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').fillna(0).values
                numerical_data.append(values)
            else:
                numerical_data.append(np.zeros(len(df)))
        
        numerical_features = np.column_stack(numerical_data)
        
        # Categorical features
        categorical_columns = ['news_direction', 'technical_direction', 'news_source', 'analysis_method']
        categorical_features = {}
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.categorical_encoders:
                    self.categorical_encoders[col] = LabelEncoder()
                    encoded_values = self.categorical_encoders[col].fit_transform(df[col].fillna('unknown').astype(str))
                else:
                    # Transform with existing encoder, handle unknown values
                    values = df[col].fillna('unknown').astype(str)
                    known_values = []
                    for val in values:
                        if val in self.categorical_encoders[col].classes_:
                            known_values.append(val)
                        else:
                            known_values.append('unknown') # Add unknown to classes if not present
                            if 'unknown' not in self.categorical_encoders[col].classes_:
                                self.categorical_encoders[col].classes_ = np.append(self.categorical_encoders[col].classes_, 'unknown')
                    encoded_values = self.categorical_encoders[col].transform(known_values)
                
                categorical_features[col] = encoded_values
        
        return texts, numerical_features, categorical_features
    
    def train_finbert_multimodal(self, texts: List[str], numerical_features: np.ndarray, 
                                categorical_features: Dict[str, np.ndarray], labels: np.ndarray) -> bool:
        """Train multi-modal FinBERT on CSV data"""
        try:
            from transformers import AutoTokenizer
            
            # Initialize tokenizer
            self.finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            
            # Get categorical vocabulary sizes
            categorical_vocab_sizes = {}
            for name, values in categorical_features.items():
                unique_values = np.unique(values)
                self.categorical_encoders[name].classes_ = unique_values # Ensure encoder knows all unique values
                categorical_vocab_sizes[name] = len(unique_values) +1 # +1 for potential unknown during inference if not handled by fit_transform

            
            # Initialize model
            model = MultiModalFinBERT(
                num_numerical_features=numerical_features.shape[1],
                categorical_vocab_sizes=categorical_vocab_sizes
            )
            
            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Prepare data
            X_train_indices, X_test_indices, y_train, y_test = train_test_split(
                list(range(len(texts))), labels, test_size=0.2, random_state=42, stratify=labels if len(np.unique(labels)) > 1 else None
            )
            
            # Scale numerical features
            numerical_features_scaled = self.numerical_scaler.fit_transform(numerical_features)
            
            # Create datasets
            train_dataset = TradingDataset(
                [texts[i] for i in X_train_indices],
                numerical_features_scaled[X_train_indices],
                {k: v[X_train_indices] for k, v in categorical_features.items()},
                y_train,
                self.finbert_tokenizer
            )
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            log_info(f"🚀 Starting FinBERT multi-modal training: {len(train_dataset)} samples, {self.epochs} epochs")
            
            # Training loop
            model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                num_batches = len(train_loader)
                
                for batch_idx, batch in enumerate(train_loader):
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    numerical_features_batch = batch['numerical_features'].to(device)
                    categorical_features_batch = {k: v.to(device) for k, v in batch['categorical_features'].items()}
                    labels_batch = batch['labels'].squeeze().to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    logits = model(input_ids, attention_mask, numerical_features_batch, categorical_features_batch)
                    loss = criterion(logits, labels_batch)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if batch_idx % 10 == 0:
                        log_debug(f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                log_info(f"✅ Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save model to proper adaptive checkpoint path
            save_path = self.model_save_dir / "finbert_multimodal_adaptive.pth"
            # Save scalers and encoders
            with open(self.model_save_dir / "data_processors.pkl", "wb") as f:
                pickle.dump({
                    'numerical_scaler': self.numerical_scaler,
                    'categorical_encoders': self.categorical_encoders,
                    'categorical_vocab_sizes': categorical_vocab_sizes
                }, f)
            log_info(f"💾 Data processors saved to: {self.model_save_dir / 'data_processors.pkl'}")

            # Save complete model state for proper learning continuity
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_name': 'ProsusAI/finbert',
                'training_timestamp': datetime.now().isoformat(),
                'num_samples': len(texts),
                'performance': {'final_loss': avg_loss},
                'categorical_vocab_sizes': categorical_vocab_sizes,  # IMPORTANT: Save vocab sizes for inference
                'num_numerical_features': numerical_features.shape[1],  # IMPORTANT: Save feature count
                'training_completed': True  # Mark as fully trained checkpoint
            }, save_path)
            
            log_info(f"💾 Multi-modal FinBERT saved to: {save_path}")
            return True
            
        except Exception as e:
            log_error(f"FinBERT multi-modal training failed: {e}")
            return False
    
    def train_enhanced_neural_multimodal(self, texts: List[str], numerical_features: np.ndarray, categorical_features: Dict[str, np.ndarray], labels: np.ndarray) -> bool:
        """Train enhanced neural analyzer on CSV data with proper model updating"""
        try:
            # Initialize enhanced analyzer if not already done
            if self.enhanced_analyzer is None:
                self.enhanced_analyzer = EnhancedNeuralAnalyzer()
            
            if not self.enhanced_analyzer.is_available:
                log_error("Enhanced Neural Analyzer not available")
                return False
            
            # Create rich text features by incorporating numerical data
            enriched_texts = []
            for i, text in enumerate(texts):
                # Add numerical context to text for richer training
                numerical_context = ""
                if len(numerical_features[i]) > 0:
                    numerical_context += f" [CONFIDENCE: {numerical_features[i][0]:.2f}]"
                if len(numerical_features[i]) > 1:
                    numerical_context += f" [NEWS_SCORE: {numerical_features[i][1]:.2f}]"
                if len(numerical_features[i]) > 2:
                    numerical_context += f" [TECH_SCORE: {numerical_features[i][2]:.2f}]"
                
                enriched_texts.append(text + numerical_context)
            
            # Prepare for training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = self.enhanced_analyzer.model
            model.train()
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Simple batch training on enriched text
            log_info(f"🧠 Enhanced Neural training on {len(enriched_texts)} enriched samples")
            
            # Convert texts to training batches
            tokenizer = self.enhanced_analyzer.tokenizer
            batch_size = min(self.batch_size, len(enriched_texts))
            
            total_loss = 0
            num_batches = 0
            
            for epoch in range(self.epochs):
                epoch_loss = 0
                
                for i in range(0, len(enriched_texts), batch_size):
                    batch_texts = enriched_texts[i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]
                    
                    # Tokenize batch
                    inputs = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(device)
                    
                    labels_tensor = torch.LongTensor(batch_labels).to(device)
                    
                    # Forward pass through enhanced model
                    outputs = model(inputs['input_ids'], inputs['attention_mask'])
                    
                    # Get sentiment logits for loss calculation
                    if isinstance(outputs, dict) and 'sentiment_logits' in outputs:
                        logits = outputs['sentiment_logits']
                    else:
                        # Fallback if model returns different format
                        logits = outputs
                    
                    loss = criterion(logits, labels_tensor)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_epoch_loss = epoch_loss / max(1, (len(enriched_texts) // batch_size))
                log_info(f"✅ Enhanced Neural Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.4f}")
                total_loss += avg_epoch_loss
            
            # Save trained model
            save_path = self.model_save_dir / "enhanced_neural_multimodal_adaptive.pth"

            # Save actual model state for learning continuity
            if hasattr(self.enhanced_analyzer, 'model') and self.enhanced_analyzer.model is not None:
                torch.save({
                    'model_state_dict': self.enhanced_analyzer.model.state_dict(),
                    'enriched_training': True,
                    'training_timestamp': datetime.now().isoformat(),
                    'num_samples': len(texts),
                    'text_enrichment_applied': True,
                    'training_completed': True
                }, save_path)
                log_info(f"💾 Enhanced Neural model state saved to: {save_path}")
            else:
                # Fallback metadata only
                torch.save({
                    'enriched_training': True,
                    'training_timestamp': datetime.now().isoformat(),
                    'num_samples': len(texts),
                    'training_completed': True
                }, save_path)
                log_info(f"💾 Enhanced Neural training metadata saved to: {save_path}")
            
            log_info(f"💾 Enhanced Neural model trained and saved to: {save_path}")
            
            # Return model to eval mode
            model.eval()
            
            return True
            
        except Exception as e:
            log_error(f"Enhanced Neural multimodal training failed: {e}", exc_info=True)
            return False
    
    def check_training_trigger(self) -> bool:
        """Check if we should trigger training based on config settings"""
        if not self.csv_path.exists():
            return False
        
        try:
            df = pd.read_csv(self.csv_path)
            completed_trades = df[df['tracking_status'] == 'completed']
            
            min_trades = getattr(Config, 'ADAPTIVE_LEARNING_MIN_TRADES', 20)
            
            if len(completed_trades) >= min_trades:
                # Check if enough time has passed since last training
                last_training_file = self.model_save_dir / "last_training.json"
                
                if last_training_file.exists():
                    with open(last_training_file, 'r') as f:
                        last_training = json.load(f)
                    
                    last_timestamp = datetime.fromisoformat(last_training['timestamp'])
                    hours_since = (datetime.now() - last_timestamp).total_seconds() / 3600
                    
                    check_hours = getattr(Config, 'ADAPTIVE_LEARNING_CHECK_HOURS', 6)
                    
                    if hours_since >= check_hours:
                        return True
                else:
                    return True # No previous training, so train if min_trades met
            
            return False
            
        except Exception as e:
            log_error(f"Error checking training trigger: {e}")
            return False
    
    def run_adaptive_learning(self) -> bool:
        """Main method to run adaptive learning if conditions are met"""
        # This check is now done by the scheduler, but good for direct calls
        # if not self.check_training_trigger():
        #     log_debug("Adaptive learning trigger conditions not met for direct run_adaptive_learning call")
        #     return False
        
        log_info("🎯 Starting adaptive learning cycle...")
        
        # Load and prepare data
        df, success = self.load_and_prepare_data()
        if not success or df is None:
            return False
        
        # Create labels and features
        labels = self.create_performance_labels(df)
        texts, numerical_features, categorical_features = self.prepare_features(df)
        
        if len(texts) == 0:
            log_error("No data available for training after feature preparation.")
            return False

        log_info(f"📊 Training data prepared: {len(texts)} samples, {len(np.unique(labels))} classes")
        
        # Train models
        finbert_success = self.train_finbert_multimodal(texts, numerical_features, categorical_features, labels)
        enhanced_success = self.train_enhanced_neural_multimodal(texts, numerical_features, categorical_features, labels)
        
        # Update last training timestamp
        if finbert_success or enhanced_success: # Consider success if at least one model trained
            last_training_file = self.model_save_dir / "last_training.json"
            with open(last_training_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'samples_trained': len(texts),
                    'finbert_success': finbert_success,
                    'enhanced_success': enhanced_success
                }, f)
            
            log_info("✅ Adaptive learning cycle completed successfully!")
            return True
        
        log_error("Adaptive learning cycle failed as no model could be trained.")
        return False
    
    def save_training_log(self, model_type: str, num_samples: int, performance_metrics: Dict[str, float] = None) -> None:
        """Save training log for tracking learning progress"""
        try:
            log_path = Config.DATA_DIR / 'last_training.json'
            
            training_info = {
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'num_samples': num_samples,
                'performance_metrics': performance_metrics or {},
                'learning_enabled': True
            }
            
            # Load existing log if it exists
            existing_log = {}
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        existing_log = json.load(f)
                except:
                    pass
            
            # Update with new training info
            existing_log[model_type] = training_info
            
            # Save updated log
            with open(log_path, 'w') as f:
                json.dump(existing_log, f, indent=2)
                
            log_info(f"📝 Training log updated: {log_path}")
            
        except Exception as e:
            log_error(f"Error saving training log: {e}")

def main():
    """Main function for running adaptive learning"""
    if not getattr(Config, 'ENABLE_ADAPTIVE_LEARNING', False):
        log_info("Adaptive learning is disabled in config")
        return
    
    learning_system = MultiModalLearningSystem()
    
    # This main function is for direct execution.
    # The scheduler will call run_adaptive_learning directly.
    # For testing, you might want to call check_training_trigger first
    if learning_system.check_training_trigger():
        log_info("Triggering adaptive learning from main execution...")
        success = learning_system.run_adaptive_learning()
        if success:
            log_info("🎓 Models have been trained on your trading history!")
            log_info("🚀 Next prediction cycles will use improved models")
        else:
            log_info("❌ Adaptive learning process failed during main execution.")
    else:
        log_info("⏳ Waiting for more trading data or time before training (from main execution).")

if __name__ == "__main__":
    # This allows running the training script directly
    # Ensure your Config is loaded correctly if you run this standalone
    # from config import Config # Make sure Config is initialized
    # Config.load_config() # Or however your config is loaded
    main()