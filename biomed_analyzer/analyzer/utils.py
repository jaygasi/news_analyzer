# biomed_analyzer/analyzer/utils.py
import logging
import sys

# Define logger at the module level so it's configured once
logger = logging.getLogger("biomed_analyzer")

def setup_logger(name="biomed_analyzer", level=logging.INFO):
    """Sets up or reconfigures the logger."""
    global logger # Declare that we are using the global logger instance
    logger = logging.getLogger(name) # Get the logger instance

    # Remove existing handlers to avoid duplicate logs if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Ensure sub-loggers (if any) also respect this level by setting it on the main logger
    logger.propagate = False # Optional: prevent passing to root logger if root is configured elsewhere
    
    return logger


def clean_text_for_llm(text: str) -> str:
    """Basic cleaning of text before sending to LLM."""
    if not text:
        return ""
    # Replace multiple newlines/spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    max_len = 8000 # Gemini 1.5 Flash has a large context, but keep it reasonable for prompt + response
    if len(text) > max_len:
        # logger.warning(f"Text truncated for LLM input. Original length: {len(text)}")
        # Avoid using logger directly here if this util is used very early.
        # Rely on LLM service to log truncation if it happens there.
        return text[:max_len] + " [TRUNCATED]"
    return text

import re # For clean_text_for_llm