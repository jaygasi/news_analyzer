"""
Enhanced CSV logger for trading decisions with configurable price tracking intervals
Python 3.13.3 compatible
"""
import csv
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from core.enhanced_decision_engine import TradingDecision
from config import Config
from utils.simple_logger import log_info, log_error, log_debug, log_warning


class CSVLogger:
    """Enhanced CSV logger with dynamic headers based on configurable price intervals"""

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        """Initialize CSV logger with dynamic headers"""
        self.csv_path: Path = csv_path or Config.CSV_OUTPUT_PATH

        # Generate dynamic headers based on current configuration
        self.headers = self._generate_dynamic_headers()

        self._ensure_csv_exists()
        
        # Log the configuration for debugging
        intervals = Config.get_price_check_labels()
        log_info(f"CSV Logger initialized with dynamic intervals: {', '.join(intervals)}")

    def _generate_dynamic_headers(self) -> List[str]:
        """Generate CSV headers with dynamic price tracking columns"""
        # Base headers (always the same)
        base_headers = [
            # Basic decision info
            'timestamp',
            'ticker',
            'decision',
            'confidence',
            'reasoning',
            
            # Combined scores
            'news_score',
            'technical_score',
            'combined_score',
            'article_count',
            
            # News analysis
            'news_direction',
            'news_confidence',
            'news_reasoning',
            'news_source',
            
            # Technical analysis
            'technical_direction',
            'technical_strength',
            'technical_reasoning',
            
            # Analysis metadata
            'analysis_method',
            'sources_used',
            'analysis_timestamp'
        ]
        
        # Dynamic price tracking headers based on configuration
        price_headers = Config.get_csv_price_headers()
        
        return base_headers + price_headers

    def _ensure_csv_exists(self) -> None:
        """Ensure CSV file exists with proper headers"""
        try:
            if not self.csv_path.exists():
                self._create_csv_with_headers()
            else:
                self._validate_csv_headers()

        except Exception as e:
            log_error(f"Error ensuring CSV exists: {e}")
            raise

    def _create_csv_with_headers(self) -> None:
        """Create new CSV file with dynamic headers"""
        try:
            # Ensure output directory exists
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

            log_info(f"Created new CSV file with dynamic headers: {self.csv_path}")
            log_debug(f"Dynamic price headers: {Config.get_csv_price_headers()}")

        except Exception as e:
            log_error(f"Error creating CSV file: {e}")
            raise

    def _validate_csv_headers(self) -> None:
        """Validate existing CSV has correct headers"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                existing_headers = next(reader, [])

                if existing_headers != self.headers:
                    log_warning("CSV headers don't match current configuration, backing up and recreating")
                    log_debug(f"Expected: {self.headers}")
                    log_debug(f"Found: {existing_headers}")
                    self._backup_existing_csv()
                    self._create_csv_with_headers()
                else:
                    log_debug("CSV headers validated successfully")

        except Exception as e:
            log_error(f"Error validating CSV headers: {e}")
            # If we can't read the file, recreate it
            self._create_csv_with_headers()

    def _backup_existing_csv(self) -> None:
        """Backup existing CSV file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.csv_path.parent / f"{self.csv_path.stem}_backup_{timestamp}.csv"

            if self.csv_path.exists():
                import shutil
                shutil.copy2(self.csv_path, backup_path)
                log_info(f"Backed up existing CSV to: {backup_path}")

        except Exception as e:
            log_error(f"Error backing up CSV: {e}")

    def log_decision(self, decision: TradingDecision) -> bool:
        """Log a single trading decision to CSV"""
        try:
            # CONFIGURABLE FILTER: Skip NONE decisions if config is set
            if Config.ONLY_LOG_TRADING_DECISIONS and decision.decision not in ['LONG', 'SHORT']:
                log_debug(f"Skipping {decision.ticker} - NONE decision (filtered by config)")
                return False
            
            row_data = self._decision_to_row(decision)

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)

            # Log decision with price info for debugging
            price_info = ""
            if hasattr(decision, 'recommendation_price') and decision.recommendation_price:
                price_info = f" @ ${decision.recommendation_price:.2f}"
            
            log_debug(f"✅ Logged decision: {decision.ticker} - {decision.decision}{price_info}")
            return True

        except Exception as e:
            log_error(f"Error logging decision to CSV: {e}")
            return False

    def log_decisions_batch(self, decisions: List[TradingDecision]) -> int:
        """Log multiple trading decisions in batch with enhanced logging - ONLY LONG/SHORT"""
        
        if not decisions:
            return 0

        try:
            logged_count = 0
            decisions_with_prices = 0
            skipped_none_decisions = 0

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                for decision in decisions:
                    try:
                        # CONFIGURABLE FILTER: Check config setting for logging behavior
                        should_skip_none = Config.ONLY_LOG_TRADING_DECISIONS and decision.decision not in ['LONG', 'SHORT']

                        if should_skip_none:
                            skipped_none_decisions += 1
                            log_debug(f"Skipping {decision.ticker} - NONE decision (confidence: {decision.confidence:.3f})")
                            continue
                        
                        row_data = self._decision_to_row(decision)
                        writer.writerow(row_data)
                        logged_count += 1
                        
                        # Count how many have prices
                        if hasattr(decision, 'recommendation_price') and decision.recommendation_price:
                            decisions_with_prices += 1
                            
                    except Exception as e:
                        log_error(f"Error logging decision for {decision.ticker}: {e}")

            # FIXED: Accurate logging messages
            # FIXED: Accurate logging messages that match configuration
            if logged_count > 0:
                decision_type_desc = "LONG/SHORT only" if Config.ONLY_LOG_TRADING_DECISIONS else "all decisions"
                log_info(f"✅ Logged {logged_count} trading decisions to CSV ({decision_type_desc})")
                log_info(f"💰 {decisions_with_prices}/{logged_count} logged decisions have entry prices")
                
                if decisions_with_prices < logged_count:
                    log_warning(f"⚠️ {logged_count - decisions_with_prices} decisions logged without entry prices")
            else:
                if Config.ONLY_LOG_TRADING_DECISIONS:
                    log_info(f"📝 No LONG/SHORT decisions to log to CSV")
                else:
                    log_info(f"📝 No decisions to log to CSV (unexpected)")
                
            if skipped_none_decisions > 0:
                log_info(f"⏭️ Skipped {skipped_none_decisions} NONE decisions (ONLY_LOG_TRADING_DECISIONS = {Config.ONLY_LOG_TRADING_DECISIONS})")
            
            return logged_count

        except Exception as e:
            log_error(f"Error in batch logging: {e}")
            return 0

    def update_decision_prices(self, decision: TradingDecision) -> bool:
        """Update price data for existing decision with dynamic field support"""
        try:
            # For now, we'll append a new row with updated price data
            # This creates an audit trail of price updates
            
            row_data = self._decision_to_row(decision)

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)

            # Log the price update with dynamic labels
            updates = []
            checkpoint_info = Config.get_checkpoint_info()
            
            for i, checkpoint in enumerate(checkpoint_info):
                price = decision.get_checkpoint_price(i)
                change_pct = decision.get_checkpoint_change_pct(i)
                
                if price is not None and change_pct is not None:
                    label = checkpoint['short_label']
                    updates.append(f"{label}: {change_pct:+.2f}%")
            
            update_info = " | ".join(updates) if updates else "baseline only"
            log_debug(f"📊 Updated price data for {decision.ticker}: {update_info}")
            
            return True

        except Exception as e:
            log_error(f"Error updating decision prices: {e}")
            return False

    def _decision_to_row(self, decision: TradingDecision) -> List[str]:
        """Convert TradingDecision to CSV row with dynamic price tracking data"""

        # Extract sources used if available (for multi-source analysis)
        sources_used_str = ""
        if hasattr(decision, 'sources_used') and decision.sources_used:
            sources_used_str = ",".join(decision.sources_used)

        # Helper function to safely format floats
        def format_float(value: Optional[float], decimals: int = 4) -> str:
            if value is not None and str(value).lower() not in ['none', 'nan', '']:
                try:
                    return f"{float(value):.{decimals}f}"
                except (ValueError, TypeError):
                    return ''
            return ''
        
        # Helper function to safely format timestamps
        def format_timestamp(value: Optional[datetime]) -> str:
            if value is not None:
                try:
                    if isinstance(value, str):
                        return value
                    return value.isoformat()
                except (AttributeError, TypeError):
                    return ''
            return ''

        # Helper function to safely format percentage changes
        def format_percentage(value: Optional[float]) -> str:
            if value is not None and str(value).lower() not in ['none', 'nan', '']:
                try:
                    return f"{float(value):.2f}"
                except (ValueError, TypeError):
                    return ''
            return ''

        # Base row data (always the same structure)
        base_data = [
            # Basic decision info
            decision.analysis_timestamp or datetime.now().isoformat(),
            decision.ticker,
            decision.decision,
            format_float(decision.confidence),
            decision.reasoning[:300] if decision.reasoning else '',
            
            # Combined scores
            format_float(decision.news_score),
            format_float(decision.technical_score),
            format_float(decision.combined_score),
            str(decision.article_count),
            
            # News analysis
            decision.news_prediction.direction if decision.news_prediction else '',
            format_float(decision.news_prediction.confidence) if decision.news_prediction else '',
            decision.news_prediction.reasoning[:200] if decision.news_prediction and decision.news_prediction.reasoning else '',
            decision.news_prediction.source if decision.news_prediction else '',
            
            # Technical analysis
            decision.technical_signal.direction if decision.technical_signal else '',
            format_float(decision.technical_signal.strength) if decision.technical_signal else '',
            decision.technical_signal.reasoning[:200] if decision.technical_signal and decision.technical_signal.reasoning else '',
            
            # Analysis metadata
            "standard_analysis",
            sources_used_str,
            decision.analysis_timestamp or datetime.now().isoformat()
        ]

        # Dynamic price tracking data
        price_data = []
        
        # Entry price and timestamp
        price_data.extend([
            format_float(getattr(decision, 'recommendation_price', None), 2),
            format_timestamp(getattr(decision, 'recommendation_timestamp', None))
        ])
        
        # Dynamic checkpoint data based on configuration
        checkpoint_info = Config.get_checkpoint_info()
        
        for checkpoint in checkpoint_info:
            field_prefix = checkpoint['field_prefix']
            
            # Get values from the dynamic price tracking data
            price = decision.price_tracking_data.get(field_prefix)
            timestamp = decision.price_tracking_data.get(f"{field_prefix}_timestamp")
            change_pct = decision.price_tracking_data.get(f"{field_prefix}_change_pct")
            
            price_data.extend([
                format_float(price, 2),
                format_timestamp(timestamp),
                format_percentage(change_pct)
            ])
        
        # Tracking status
        price_data.append(getattr(decision, 'tracking_status', 'pending'))
        
        return base_data + price_data

    def get_csv_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics about the CSV file"""
        try:
            if not self.csv_path.exists():
                return {'exists': False}

            file_size = self.csv_path.stat().st_size

            # Count rows and analyze by categories
            row_count = 0
            decisions_by_type = {'LONG': 0, 'SHORT': 0, 'NONE': 0}
            tracking_stats = {'pending': 0, 'completed': 0, 'partial': 0, 'no_tracking': 0}
            price_stats = {'with_entry_price': 0, 'without_entry_price': 0}

            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                for row in reader:
                    row_count += 1

                    # Decision type
                    decision_type = row.get('decision', 'NONE')
                    decisions_by_type[decision_type] = decisions_by_type.get(decision_type, 0) + 1
                    
                    # Tracking status
                    tracking_status = row.get('tracking_status', 'no_tracking')
                    if tracking_status in tracking_stats:
                        tracking_stats[tracking_status] += 1
                    else:
                        tracking_stats['no_tracking'] += 1
                    
                    # Price availability
                    if row.get('recommendation_price'):
                        price_stats['with_entry_price'] += 1
                    else:
                        price_stats['without_entry_price'] += 1

            # Dynamic interval statistics
            interval_info = {
                'configured_intervals': Config.get_price_check_labels(),
                'check1_minutes': Config.PRICE_CHECK_1_MINUTES,
                'check2_minutes': Config.PRICE_CHECK_2_MINUTES,
                'close_time': f"{Config.CLOSE_PRICE_HOUR:02d}:{Config.CLOSE_PRICE_MINUTE:02d} EST"
            }

            return {
                'exists': True,
                'file_size_bytes': file_size,
                'total_decisions': row_count,
                'decisions_by_type': decisions_by_type,
                'tracking_statistics': tracking_stats,
                'price_statistics': price_stats,
                'interval_configuration': interval_info,
                'file_path': str(self.csv_path),
                'headers_count': len(self.headers)
            }

        except Exception as e:
            log_error(f"Error getting CSV statistics: {e}")
            return {'exists': False, 'error': str(e)}

    def read_recent_decisions(self, limit: int = 100) -> List[Dict[str, str]]:
        """Read recent decisions from CSV with enhanced filtering"""
        try:
            if not self.csv_path.exists():
                return []

            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                # Read all rows into memory first
                all_rows = list(reader)

                # Take the last 'limit' rows
                recent_rows = all_rows[-limit:] if len(all_rows) > limit else all_rows

                return recent_rows

        except Exception as e:
            log_error(f"Error reading recent decisions: {e}")
            return []

    def debug_csv_content(self, num_rows: int = 5) -> None:
        """Debug method to inspect CSV content with dynamic field names"""
        try:
            if not self.csv_path.exists():
                log_warning("CSV file does not exist")
                return

            log_info(f"🔍 CSV Debug - Last {num_rows} entries (dynamic intervals: {', '.join(Config.get_price_check_labels())}):")
            
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                all_rows = list(reader)
                
                if not all_rows:
                    log_warning("CSV file is empty (no data rows)")
                    return
                
                recent_rows = all_rows[-num_rows:] if len(all_rows) > num_rows else all_rows
                
                for i, row in enumerate(recent_rows, 1):
                    ticker = row.get('ticker', 'UNKNOWN')
                    decision = row.get('decision', 'UNKNOWN')
                    entry_price = row.get('recommendation_price', 'MISSING')
                    tracking_status = row.get('tracking_status', 'MISSING')
                    
                    # Show dynamic checkpoint prices
                    checkpoint_prices = []
                    checkpoint_info = Config.get_checkpoint_info()
                    for checkpoint in checkpoint_info:
                        field_prefix = checkpoint['field_prefix']
                        price = row.get(field_prefix, '')
                        if price:
                            checkpoint_prices.append(f"{checkpoint['short_label']}:${price}")
                    
                    price_info = f" [{', '.join(checkpoint_prices)}]" if checkpoint_prices else ""
                    
                    log_info(f"  {i}. {ticker} {decision} @ ${entry_price} (status: {tracking_status}){price_info}")

        except Exception as e:
            log_error(f"Error debugging CSV content: {e}")

    def validate_price_data_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of price data in CSV with dynamic field support"""
        try:
            if not self.csv_path.exists():
                return {'error': 'CSV file does not exist'}

            validation_results = {
                'total_rows': 0,
                'trading_decisions': 0,  # LONG/SHORT only
                'decisions_with_entry_price': 0,
                'decisions_with_tracking': 0,
                'completed_tracking': 0,
                'issues': [],
                'dynamic_configuration': {
                    'intervals': Config.get_price_check_labels(),
                    'check1_minutes': Config.PRICE_CHECK_1_MINUTES,
                    'check2_minutes': Config.PRICE_CHECK_2_MINUTES
                }
            }

            checkpoint_info = Config.get_checkpoint_info()

            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                for row_num, row in enumerate(reader, 1):
                    validation_results['total_rows'] += 1
                    
                    decision = row.get('decision', '')
                    ticker = row.get('ticker', '')
                    
                    if decision in ['LONG', 'SHORT']:
                        validation_results['trading_decisions'] += 1
                        
                        # Check entry price
                        entry_price = row.get('recommendation_price', '')
                        if entry_price:
                            validation_results['decisions_with_entry_price'] += 1
                        else:
                            validation_results['issues'].append(f"Row {row_num}: {ticker} {decision} missing entry price")
                        
                        # Check tracking status
                        tracking_status = row.get('tracking_status', '')
                        if tracking_status == 'completed':
                            validation_results['completed_tracking'] += 1
                        elif tracking_status in ['pending', 'partial']:
                            validation_results['decisions_with_tracking'] += 1
                        
                        # Validate dynamic checkpoint prices
                        if entry_price:
                            try:
                                entry = float(entry_price)
                                
                                for checkpoint in checkpoint_info:
                                    field_prefix = checkpoint['field_prefix']
                                    price_field = field_prefix
                                    change_field = f"{field_prefix}_change_pct"
                                    
                                    price_value = row.get(price_field)
                                    change_value = row.get(change_field)
                                    
                                    if price_value and change_value:
                                        try:
                                            price = float(price_value)
                                            change_pct = float(change_value)
                                            
                                            calculated_change = ((price - entry) / entry) * 100
                                            if abs(calculated_change - change_pct) > 0.1:  # Allow 0.1% tolerance
                                                validation_results['issues'].append(
                                                    f"Row {row_num}: {ticker} {checkpoint['short_label']} price change mismatch "
                                                    f"(calculated: {calculated_change:.2f}%, stored: {change_pct:.2f}%)"
                                                )
                                        except (ValueError, ZeroDivisionError):
                                            validation_results['issues'].append(
                                                f"Row {row_num}: {ticker} invalid {checkpoint['short_label']} price data"
                                            )
                            except (ValueError, ZeroDivisionError):
                                validation_results['issues'].append(f"Row {row_num}: {ticker} invalid entry price")

            return validation_results

        except Exception as e:
            log_error(f"Error validating price data integrity: {e}")
            return {'error': str(e)}

    def get_dynamic_field_mapping(self) -> Dict[str, str]:
        """Get mapping of dynamic field names to their configured intervals"""
        mapping = {
            'recommendation_price': 'Entry price',
            'recommendation_timestamp': 'Entry timestamp'
        }
        
        checkpoint_info = Config.get_checkpoint_info()
        for checkpoint in checkpoint_info:
            field_prefix = checkpoint['field_prefix']
            label = checkpoint['short_label']
            
            mapping[field_prefix] = f"{label} price"
            mapping[f"{field_prefix}_timestamp"] = f"{label} timestamp"
            mapping[f"{field_prefix}_change_pct"] = f"{label} change %"
        
        mapping['tracking_status'] = 'Tracking status'
        
        return mapping