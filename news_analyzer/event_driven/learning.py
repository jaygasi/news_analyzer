import database
import logging
from ml_models import ProfitMaximizingModel
import train_model
from datetime import datetime

class ContinuousLearningSystem:
    def __init__(self, profit_model: ProfitMaximizingModel):
        self.profit_model = profit_model
        logging.info("ContinuousLearningSystem initialized. Focus on triggering retraining.")

    def review_and_trigger_retraining(
        self,
        min_trades_for_retrain: int = 100,
        performance_threshold: float = 0.7, # e.g., if validation accuracy drops below 70%
        force_retrain: bool = False
    ):
        """
        Reviews recent trade performance and model metrics to trigger retraining if conditions are met.
        Conditions:
        1. A sufficient number of new performance-tracked trades are available.
        2. Model performance metrics (e.g., validation accuracy) fall below a threshold.
        3. Manual force retraining.
        """
        logging.info("Reviewing conditions for model retraining...")

        # Check for untracked trades first
        untracked_trades_for_perf = database.get_untracked_trades(status='pending')
        if len(untracked_trades_for_perf) > 0:
            logging.info(f"Skipping retraining: {len(untracked_trades_for_perf)} trades still need performance tracking (run price_tracker.py).")
            return

        completed_trades_for_training = database.get_training_data()
        if not completed_trades_for_training:
            logging.warning("No completed trades with performance data available for retraining.")
            return

        current_data_size = len(completed_trades_for_training)
        latest_training_info = database.get_latest_training_history()

        last_trained_on_samples = 0
        last_val_accuracy = 0.0
        if latest_training_info:
            import re
            match = re.search(r'Trained on (\d+) samples', latest_training_info.get('notes', ''))
            if match:
                last_trained_on_samples = int(match.group(1))
            last_val_accuracy = latest_training_info.get('val_accuracy', 0.0)
            logging.info(f"Last model trained on {last_trained_on_samples} samples with validation accuracy: {last_val_accuracy:.2f}. Currently have {current_data_size} completed trades.")
        else:
            logging.info("No previous training history found. Considering initial training.")

        # Condition 1: Sufficient new data
        enough_new_data = (current_data_size - last_trained_on_samples) >= min_trades_for_retrain
        if enough_new_data:
            logging.info(f"Condition met: Enough new data for retraining ({current_data_size - last_trained_on_samples} new trades, threshold {min_trades_for_retrain}).")

        # Condition 2: Performance degradation (only if there's a history)
        performance_degraded = False
        if latest_training_info and last_val_accuracy < performance_threshold:
            performance_degraded = True
            logging.warning(f"Condition met: Model performance degraded (accuracy {last_val_accuracy:.2f} < threshold {performance_threshold:.2f}).")

        # Trigger retraining if any condition is met
        if force_retrain or enough_new_data or performance_degraded:
            logging.info("Triggering model retraining...")
            try:
                # Pass current_data_size to train_model for logging purposes
                train_model.train_model(notes=f"Retrained with {current_data_size} samples. Triggered by: " +
                                            (f"Force Retrain, " if force_retrain else "") +
                                            (f"New Data ({current_data_size - last_trained_on_samples} new), " if enough_new_data else "") +
                                            (f"Performance Degradation (Acc: {last_val_accuracy:.2f}), " if performance_degraded else ""))
                logging.info("Model retraining initiated successfully.")
                # After successful retraining, update the profit_model instance
                self.profit_model._initialize_sync() # Re-load the newly trained model
            except Exception as e:
                logging.error(f"Error during model retraining: {e}", exc_info=True)
        else:
            logging.info("No conditions met for retraining. Model remains current.")

    def update_from_trade_result(self, trade_decision: dict, actual_outcome: dict):
        """
        Placeholder for online learning or logging, this method is called per trade.
        The actual data for batch retraining is collected by database.log_trade
        and database.update_trade_performance.
        """
        actual_profit_per_share = actual_outcome.get('exit_price', 0) - actual_outcome.get('entry_price', 0)
        if trade_decision.get('direction') == 'SHORT':
            actual_profit_per_share = actual_outcome.get('entry_price', 0) - actual_outcome.get('exit_price', 0)

        logging.info(f"Learning from completed trade {trade_decision.get('ticker')}. Actual Profit/Share: {actual_profit_per_share:.4f}")


