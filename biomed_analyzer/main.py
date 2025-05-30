# biomed_analyzer/main.py
import json
import argparse
import logging
import sys
from typing import List, Dict, Any, Optional, Tuple

# Import our modules
from analyzer.config import config, ConfigError
from analyzer.utils import logger, setup_logger
from analyzer.models import StockPrediction

class ApplicationError(Exception):
    """Custom application error"""
    pass

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with enhanced help"""
    parser = argparse.ArgumentParser(
        description="Production Biomedical News Analyzer for Stock Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with all available APIs
  python main.py --disable-finnhub                 # Use only FMP API
  python main.py --tickers "PFE,MRNA"              # Analyze specific tickers
  python main.py --mode general                     # Only general market news
  python main.py --log-level DEBUG                 # Verbose logging
  python diagnose.py                                # Run diagnostics first

For help with setup: python diagnose.py
        """
    )
    
    parser.add_argument(
        "--llm", type=str, default=config.DEFAULT_LLM,
        help=f"LLM service to use (default: {config.DEFAULT_LLM})"
    )
    parser.add_argument(
        "--output", type=str, default="predictions.json",
        help="Output file path (default: predictions.json)"
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated tickers to analyze (e.g., 'PFE,MRNA,GILD')"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "general", "fda", "tickers"],
        help="Processing mode (default: all)"
    )
    parser.add_argument(
        "--disable-finnhub", action="store_true",
        help="Disable Finnhub API (use only FMP)"
    )
    parser.add_argument(
        "--disable-fmp", action="store_true",
        help="Disable FMP API (use only Finnhub)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Test configuration without making API calls"
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Run diagnostic checks and exit"
    )
    
    return parser.parse_args()

def configure_logging(log_level_str: str):
    """Configure application logging"""
    log_level_enum = getattr(logging, log_level_str.upper(), logging.INFO)
    setup_logger(level=log_level_enum)
    logger.info(f"Logging configured to {log_level_str.upper()} level")

def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    if args.disable_finnhub and args.disable_fmp:
        raise ApplicationError("Cannot disable both Finnhub and FMP APIs - at least one must be enabled")
    
    if args.disable_finnhub and not config.has_fmp():
        raise ApplicationError("Cannot disable Finnhub - FMP API key not available")
    
    if args.disable_fmp and not config.has_finnhub():
        raise ApplicationError("Cannot disable FMP - Finnhub API key not available")
    
    if args.mode == "fda" and args.disable_finnhub:
        raise ApplicationError("FDA calendar mode requires Finnhub API (cannot disable)")
    
    if args.mode == "tickers" and not args.tickers:
        raise ApplicationError("Tickers mode requires --tickers argument")

def initialize_services(args: argparse.Namespace) -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """Initialize services with robust error handling"""
    logger.info("Initializing services...")
    
    # Determine which APIs to use
    use_finnhub = not args.disable_finnhub and config.has_finnhub()
    use_fmp = not args.disable_fmp and config.has_fmp()
    
    if not use_finnhub and not use_fmp:
        raise ApplicationError("No news APIs available - check your API keys")
    
    logger.info(f"Using APIs: {'Finnhub' if use_finnhub else ''}{' + ' if use_finnhub and use_fmp else ''}{'FMP' if use_fmp else ''}")
    
    # Initialize clients
    finnhub_client = None
    fmp_client = None
    
    if use_finnhub:
        try:
            from analyzer.finnhub_client import FinnhubClient
            finnhub_client = FinnhubClient()
            logger.info("✓ Finnhub client initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Finnhub client: {e}")
            if not use_fmp:
                raise ApplicationError(f"Finnhub initialization failed and no FMP fallback: {e}")
    
    if use_fmp:
        try:
            from analyzer.fmp_client import FMPClient  
            fmp_client = FMPClient()
            logger.info("✓ FMP client initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize FMP client: {e}")
            if not use_finnhub:
                raise ApplicationError(f"FMP initialization failed and no Finnhub fallback: {e}")
    
    # Initialize news processor
    try:
        from analyzer.news_processor import NewsProcessor
        processor = NewsProcessor(
            finnhub_client=finnhub_client,
            fmp_client=fmp_client,
            llm_service_name=args.llm
        )
        logger.info("✓ News processor initialized successfully")
        return finnhub_client, fmp_client, processor
    except Exception as e:
        raise ApplicationError(f"Failed to initialize news processor: {e}")

def run_analysis(processor: object, args: argparse.Namespace) -> List[StockPrediction]:
    """Run news analysis with comprehensive error handling"""
    all_predictions: List[StockPrediction] = []
    
    # Determine what to run
    run_general = args.mode in ["all", "general"]
    run_fda = args.mode in ["all", "fda"]
    run_specific_tickers = args.mode == "tickers" or args.tickers
    
    logger.info(f"Analysis plan: General={run_general}, FDA={run_fda}, Specific Tickers={run_specific_tickers}")
    
    try:
        # Process specific tickers first if provided
        if args.tickers:
            ticker_list = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
            if ticker_list:
                logger.info(f"Processing {len(ticker_list)} user-specified tickers: {ticker_list}")
                try:
                    specific_preds = processor.process_news_for_specific_tickers(ticker_list)
                    all_predictions.extend(specific_preds)
                    logger.info(f"✓ Found {len(specific_preds)} predictions from specific tickers")
                except Exception as e:
                    logger.error(f"✗ Error processing specific tickers: {e}")
                
                # If mode is tickers-only, return here
                if args.mode == "tickers":
                    return all_predictions
        
        # Process general market news
        if run_general:
            logger.info("Processing general market news...")
            try:
                general_preds = processor.process_market_news()
                all_predictions.extend(general_preds)
                logger.info(f"✓ Found {len(general_preds)} predictions from general market news")
            except Exception as e:
                logger.error(f"✗ Error processing general market news: {e}")
        
        # Process FDA calendar events
        if run_fda:
            logger.info("Processing FDA calendar-triggered news...")
            try:
                fda_preds = processor.process_fda_calendar_triggered_news()
                all_predictions.extend(fda_preds)
                logger.info(f"✓ Found {len(fda_preds)} predictions from FDA calendar events")
            except Exception as e:
                logger.error(f"✗ Error processing FDA calendar news: {e}")
    
    except Exception as e:
        logger.error(f"Critical error during analysis: {e}")
        raise ApplicationError(f"Analysis failed: {e}")
    
    return all_predictions

def save_predictions(predictions: List[StockPrediction], output_file: str) -> None:
    """Save predictions with deduplication and validation"""
    if not predictions:
        logger.info("No predictions to save")
        # Create empty output file
        try:
            with open(output_file, 'w') as f:
                json.dump([], f, indent=2)
            logger.info(f"Created empty predictions file: {output_file}")
        except Exception as e:
            logger.error(f"Failed to create empty output file: {e}")
        return
    
    # Deduplication logic
    unique_predictions = {}
    for pred in predictions:
        # Create unique key based on ticker and news
        key = (pred.ticker, pred.news_headline, pred.news_url or "")
        if key not in unique_predictions:
            unique_predictions[key] = pred
    
    final_predictions = list(unique_predictions.values())
    
    # Sort by confidence (highest first)
    final_predictions.sort(key=lambda x: x.confidence, reverse=True)
    
    try:
        # Convert to JSON-serializable format
        output_data = [pred.model_dump() for pred in final_predictions]
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"✓ Saved {len(final_predictions)} unique predictions to {output_file}")
        
        # Print summary
        if final_predictions:
            logger.info("=== PREDICTION SUMMARY ===")
            for pred in final_predictions[:5]:  # Show top 5
                logger.info(f"{pred.ticker}: {pred.predicted_movement} (confidence: {pred.confidence})")
            if len(final_predictions) > 5:
                logger.info(f"... and {len(final_predictions) - 5} more")
        
    except Exception as e:
        logger.error(f"Failed to save predictions to {output_file}: {e}")
        raise ApplicationError(f"Failed to save results: {e}")

def main():
    """Main application entry point"""
    try:
        args = parse_arguments()
        
        # Handle special modes
        if args.diagnose:
            print("Running diagnostics...")
            import subprocess
            return subprocess.call([sys.executable, "diagnose.py"])
        
        # Configure logging
        configure_logging(args.log_level)
        
        # Print startup banner
        logger.info("=" * 60)
        logger.info("🏥 BIOMEDICAL NEWS ANALYZER - PRODUCTION VERSION")
        logger.info("=" * 60)
        
        # Show configuration status
        config.print_config_status()
        
        # Validate arguments
        validate_arguments(args)
        
        # Initialize services
        if args.dry_run:
            logger.info("DRY RUN MODE - Testing configuration only")
            _finnhub, _fmp, processor = initialize_services(args)
            logger.info("✓ Configuration test successful - all services initialized")
            return 0
        
        # Run the analysis
        logger.info(f"Starting analysis (mode: {args.mode})...")
        _finnhub, _fmp, processor = initialize_services(args)
        predictions = run_analysis(processor, args)
        
        # Save results
        save_predictions(predictions, args.output)
        
        logger.info("✓ Analysis completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except ConfigError as e:
        logger.critical(f"Configuration error: {e}")
        logger.info("Run 'python diagnose.py' to check your setup")
        return 1
    except ApplicationError as e:
        logger.critical(f"Application error: {e}")
        return 1
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        if args.log_level == "DEBUG":
            import traceback
            logger.critical(traceback.format_exc())
        logger.info("Run with --log-level DEBUG for more details")
        return 1

if __name__ == "__main__":
    sys.exit(main())