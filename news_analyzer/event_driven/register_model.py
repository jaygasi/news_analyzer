# --- register_model.py ---
import sqlite3
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    import database
    from logger_config import setup_logging
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)


def register_model_in_db(model_path: str, scaler_path: Optional[str]):
    """
    Manually registers a pre-existing model file in the database.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        logging.error(f"Model file not found at '{model_path}'. Please check the path.")
        return

    if scaler_path:
        scaler_file = Path(scaler_path)
        if not scaler_file.exists():
            logging.warning(f"Scaler file not found at '{scaler_path}'. It will be registered as NULL.")
            scaler_path = None

    model_version_str = model_file.name

    # Check if this version is already registered
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM training_history WHERE model_version = ?", (model_version_str,)
    )
    if cursor.fetchone():
        logging.warning(
            f"Model version '{model_version_str}' is already registered in the database."
        )
        conn.close()
        return
    conn.close()

    # Create placeholder data for the log entry
    log_data = {
        "training_timestamp": datetime.now().isoformat(),
        "model_version": model_version_str,
        "scaler_path": scaler_path,
        "avg_training_loss": 0.0,
        "val_accuracy": 0.0,
        "val_f1_macro": 0.0,
        "val_confidence_mse": 0.0,
        "val_volatility_mse": 0.0,
        "notes": "Manually registered existing model.",
    }

    try:
        database.log_training_run(log_data)
        logging.info(
            f"Successfully registered model '{model_version_str}' in the database."
        )
        print(f"\n✅ Success! Model '{model_version_str}' has been registered.")
        print("You can now run 'main.py', and it should find and load this model.")
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during model registration: {e}",
            exc_info=True,
        )


if __name__ == "__main__":
    setup_logging()

    # Ensure the database and tables exist before proceeding.
    database.init_db()

    print("--- Manual Model Registration Utility ---")
    model_filename = input(
        "Enter the filename of the model to register (e.g., roberta_hybrid_... .pth): "
    ).strip()
    scaler_filename = input(
        "Enter the filename of the associated scaler (e.g., feature_scaler_... .pkl) [optional]: "
    ).strip()

    if not model_filename:
        print("No model filename provided. Exiting.")
    else:
        register_model_in_db(model_filename, scaler_filename or None)
