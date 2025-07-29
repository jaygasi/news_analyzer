# --- logger_config.py ---
import logging
import sys
import os

def setup_logging():
    """
    Configures the root logger for the application.
    This provides structured, leveled logging instead of using print().
    The logging level can be set via the LOG_LEVEL environment variable
    (e.g., 'DEBUG', 'INFO', 'WARNING'). Defaults to 'INFO'.
    """
    # Get log level from environment variable, default to INFO
    log_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(module)s:%(lineno)d] - %(message)s",
        handlers=[
            logging.FileHandler("system.log", mode='w'), # Log to a file, overwrite each run
            logging.StreamHandler(sys.stdout) # Also log to the console
        ]
    )
    logging.info(f"Logging initialized with level {log_level_name}")