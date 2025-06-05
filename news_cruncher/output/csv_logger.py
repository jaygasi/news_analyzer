"""
Enhanced CSV logger for trading decisions - FIXED IMPORTS
Python 3.13.3 compatible
"""
import csv
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from core.decision_engine import TradingDecision  # FIXED: Import from correct location
from config import Config
from utils.simple_logger import log_info, log_error, log_debug


class CSVLogger:
    """CSV logger with fixed imports"""

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        """Initialize CSV logger"""
        self.csv_path: Path = csv_path or Config.CSV_OUTPUT_PATH

        # Headers for regular trading decisions
        self.headers = [
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
            'sources_used',  # If available from multi-source
            'analysis_timestamp'
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

            log_info(f"Created new CSV file: {self.csv_path}")

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
                    log_info("CSV headers don't match, backing up and recreating")
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
        """Convert TradingDecision to CSV row"""

        # Extract sources used if available (for multi-source analysis)
        sources_used_str = ""
        if hasattr(decision, 'sources_used') and decision.sources_used:
            sources_used_str = ",".join(decision.sources_used)

        return [
            # Basic decision info
            decision.analysis_timestamp or datetime.now().isoformat(),
            decision.ticker,
            decision.decision,
            f"{decision.confidence:.4f}",
            decision.reasoning[:300] if decision.reasoning else '',
            
            # Combined scores
            f"{decision.news_score:.4f}",
            f"{decision.technical_score:.4f}",
            f"{decision.combined_score:.4f}",
            str(decision.article_count),
            
            # News analysis
            decision.news_prediction.direction if decision.news_prediction else '',
            f"{decision.news_prediction.confidence:.4f}" if decision.news_prediction else '',
            decision.news_prediction.reasoning[:200] if decision.news_prediction and decision.news_prediction.reasoning else '',
            decision.news_prediction.source if decision.news_prediction else '',
            
            # Technical analysis
            decision.technical_signal.direction if decision.technical_signal else '',
            f"{decision.technical_signal.strength:.4f}" if decision.technical_signal else '',
            decision.technical_signal.reasoning[:200] if decision.technical_signal and decision.technical_signal.reasoning else '',
            
            # Analysis metadata
            "standard_analysis",
            sources_used_str,
            decision.analysis_timestamp or datetime.now().isoformat()
        ]

    def get_csv_statistics(self) -> Dict[str, Any]:
        """Get statistics about the CSV file"""
        try:
            if not self.csv_path.exists():
                return {'exists': False}

            file_size = self.csv_path.stat().st_size

            # Count rows and analyze by categories
            row_count = 0
            decisions_by_type = {'LONG': 0, 'SHORT': 0, 'NONE': 0}

            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                for row in reader:
                    row_count += 1

                    # Decision type
                    decision_type = row.get('decision', 'NONE')
                    decisions_by_type[decision_type] = decisions_by_type.get(decision_type, 0) + 1

            return {
                'exists': True,
                'file_size_bytes': file_size,
                'total_decisions': row_count,
                'decisions_by_type': decisions_by_type,
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