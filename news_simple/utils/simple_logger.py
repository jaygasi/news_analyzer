"""
Simplified logging utility
"""
import logging
import sys
from pathlib import Path
from config import CONFIG

def setup_logger(name: str = "trading_system") -> logging.Logger:
    """Setup simple logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_file = CONFIG.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# Global logger
logger = setup_logger()

def log_info(msg: str):
    logger.info(msg)

def log_error(msg: str):
    logger.error(msg)

def log_debug(msg: str):
    logger.debug(msg)

def log_warning(msg: str):
    logger.warning(msg)