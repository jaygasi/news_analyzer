"""
CSV logger for trading decisions
Python 3.13.3 compatible
"""
import csv
import os
from typing import List, Dict, Any, Optional # Added Optional
from datetime import datetime
from pathlib import Path
from core.decision_engine import TradingDecision
from config import Config
from utils.simple_logger import log_info, log_error, log_debug


class CSVLogger:
    """Log trading decisions to CSV file"""

    def __init__(self, csv_path: Optional[Path] = None) -> None: # Changed type hint
        """Initialize CSV logger"""
        self.csv_path: Path = csv_path or Config.CSV_OUTPUT_PATH # Added explicit type hint for instance variable

        # Enhanced headers to include new emergency services and features
        self.headers = [
            'timestamp',
            'ticker',
            'decision',
            'confidence',
            'news_score',
            'technical_score',
            'combined_score',
            'article_count',
            'news_direction',
            'news_confidence',
            'news_source',  # Now includes: finbert, gemini, openai, claude, alpha_vantage, polygon, tiingo, enhanced_keyword_analysis
            'news_reasoning',
            'news_raw_score',  # New: raw score from analysis
            'technical_direction',
            'technical_strength',
            'technical_reasoning',
            'final_reasoning',
            'analysis_method',  # New: primary/emergency/keyword
            'market_context'    # New: for ETF assignments (TLT, QQQ, etc.)
        ]

        self._ensure_csv_exists()

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
        """Create new CSV file with headers"""
        try:
            # Ensure output directory exists
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

            # SonarLint S3457 fix: Changed f-string to .format()
            log_info("Created new CSV file with enhanced headers: {}".format(self.csv_path))

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
                    log_info("CSV headers don't match enhanced version, backing up and recreating")
                    self._backup_existing_csv()
                    self._create_csv_with_headers()

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
            row_data = self._decision_to_row(decision)

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)

            log_debug(f"Logged decision: {decision.ticker} - {decision.decision}")
            return True

        except Exception as e:
            log_error(f"Error logging decision to CSV: {e}")
            return False

    def log_decisions_batch(self, decisions: List[TradingDecision]) -> int:
        """Log multiple trading decisions in batch"""
        if not decisions:
            return 0

        try:
            logged_count = 0

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                for decision in decisions:
                    try:
                        row_data = self._decision_to_row(decision)
                        writer.writerow(row_data)
                        logged_count += 1
                    except Exception as e:
                        log_error(f"Error logging decision for {decision.ticker}: {e}")

            log_info(f"Logged {logged_count} trading decisions to CSV")
            return logged_count

        except Exception as e:
            log_error(f"Error in batch logging: {e}")
            return 0

    def _decision_to_row(self, decision: TradingDecision) -> List[str]:
        """Convert TradingDecision to CSV row with enhanced columns"""

        # Determine analysis method
        analysis_method = "unknown"
        if decision.news_prediction:
            source = decision.news_prediction.source
            if source in ['finbert', 'gemini', 'openai', 'claude']:
                analysis_method = "primary_llm"
            elif source in ['alpha_vantage', 'polygon', 'tiingo']:
                analysis_method = "emergency_api"
            elif source == 'enhanced_keyword_analysis':
                analysis_method = "keyword_fallback"

        # Determine market context (for ETF assignments)
        market_context = ""
        if decision.ticker in ['SPY', 'QQQ', 'IWM', 'TLT', 'XLE', 'GLD']:
            market_context = f"etf_market_news"

        return [
            decision.analysis_timestamp or datetime.now().isoformat(),
            decision.ticker,
            decision.decision,
            f"{decision.confidence:.4f}",
            f"{decision.news_score:.4f}",
            f"{decision.combined_score:.4f}",
            str(decision.article_count),
            decision.news_prediction.direction if decision.news_prediction else '',
            f"{decision.news_prediction.confidence:.4f}" if decision.news_prediction else '',
            decision.news_prediction.source if decision.news_prediction else '',
            (decision.news_prediction.reasoning[:200] if decision.news_prediction and decision.news_prediction.reasoning else ''),
            f"{decision.news_prediction.raw_score:.4f}" if decision.news_prediction and hasattr(decision.news_prediction, 'raw_score') else '',
            decision.technical_signal.direction if decision.technical_signal else '',
            f"{decision.technical_signal.strength:.4f}" if decision.technical_signal else '',
            (decision.technical_signal.reasoning[:200] if decision.technical_signal and decision.technical_signal.reasoning else ''),
            decision.reasoning[:300] if decision.reasoning else '',
            analysis_method,
            market_context
        ]

    def get_csv_statistics(self) -> Dict[str, Any]:
        """Get statistics about the CSV file"""
        try:
            if not self.csv_path.exists():
                return {'exists': False}

            file_size = self.csv_path.stat().st_size

            # Count rows and analyze by new categories
            row_count = 0
            decisions_by_type = {'LONG': 0, 'SHORT': 0, 'NONE': 0}
            analysis_methods = {}
            services_used = {}

            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                for row in reader:
                    row_count += 1

                    # Decision type
                    decision_type = row.get('decision', 'NONE')
                    decisions_by_type[decision_type] = decisions_by_type.get(decision_type, 0) + 1

                    # Analysis method
                    method = row.get('analysis_method', 'unknown')
                    analysis_methods[method] = analysis_methods.get(method, 0) + 1

                    # Services used
                    service = row.get('news_source', 'unknown')
                    services_used[service] = services_used.get(service, 0) + 1

            return {
                'exists': True,
                'file_size_bytes': file_size,
                'total_decisions': row_count,
                'decisions_by_type': decisions_by_type,
                'analysis_methods': analysis_methods,
                'services_used': services_used,
                'file_path': str(self.csv_path)
            }

        except Exception as e:
            log_error(f"Error getting CSV statistics: {e}")
            return {'exists': False, 'error': str(e)}

    def read_recent_decisions(self, limit: int = 100) -> List[Dict[str, str]]:
        """Read recent decisions from CSV"""
        try:
            if not self.csv_path.exists():
                return []

            # SonarLint S1481 fix: Removed unused variable 'decisions'
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

    def export_filtered_decisions(self, output_path: Path,
                                 decision_type: Optional[str] = None, # Pylance fix
                                 min_confidence: Optional[float] = None, # Pylance fix
                                 start_date: Optional[str] = None, # Pylance fix
                                 end_date: Optional[str] = None, # Pylance fix
                                 analysis_method: Optional[str] = None) -> int: # Pylance fix
        """Export filtered decisions to new CSV file with enhanced filtering"""
        try:
            if not self.csv_path.exists():
                return 0

            exported_count = 0

            with open(self.csv_path, 'r', encoding='utf-8') as input_file:
                with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
                    reader = csv.DictReader(input_file)
                    writer = csv.DictWriter(output_file, fieldnames=self.headers)

                    writer.writeheader()

                    for row in reader:
                        if self._should_include_row(row, decision_type, min_confidence, start_date, end_date, analysis_method):
                            writer.writerow(row)
                            exported_count += 1

            log_info(f"Exported {exported_count} filtered decisions to {output_path}")
            return exported_count

        except Exception as e:
            log_error(f"Error exporting filtered decisions: {e}")
            return 0

    def _should_include_row(self, row: Dict[str, str],
                           decision_type: Optional[str] = None, # Pylance fix
                           min_confidence: Optional[float] = None, # Pylance fix
                           start_date: Optional[str] = None, # Pylance fix
                           end_date: Optional[str] = None, # Pylance fix
                           analysis_method: Optional[str] = None) -> bool: # Pylance fix
        """Check if row should be included in filtered export"""
        try: # Outer try-except for general safety, as in original code
            conditions = []

            # Filter by decision type
            if decision_type is not None:
                conditions.append(row.get('decision') == decision_type)

            # Filter by confidence
            if min_confidence is not None:
                try:
                    confidence_str = row.get('confidence')
                    # Ensure confidence_str is not None or empty before conversion
                    confidence = float(confidence_str) if confidence_str else 0.0
                    conditions.append(confidence >= min_confidence)
                except ValueError:
                    log_error(f"Invalid confidence value '{row.get('confidence')}' in row. Skipping row for filtering.")
                    return False # Exclude row if confidence is unparseable

            # Filter by analysis method
            if analysis_method is not None:
                conditions.append(row.get('analysis_method') == analysis_method)

            # Filter by date range
            row_timestamp = row.get('timestamp', '') # Get timestamp, default to empty string if missing

            if start_date is not None:
                conditions.append(row_timestamp >= start_date)
            if end_date is not None:
                conditions.append(row_timestamp <= end_date)

            return all(conditions)

        except Exception as e:
            log_error(f"Error checking row for inclusion in filtered export: {e}. Row: {row}")
            return False