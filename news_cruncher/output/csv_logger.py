"""
Enhanced CSV logger for multi-source trading decisions
Python 3.13.3 compatible
"""
import csv
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from core.enhanced_decision_engine import EnhancedTradingDecision
from config import Config
from utils.simple_logger import log_info, log_error, log_debug


class CSVLogger:
    """Enhanced CSV logger with detailed multi-source analysis tracking"""

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        """Initialize enhanced CSV logger"""
        self.csv_path: Path = csv_path or Config.CSV_OUTPUT_PATH

        # Comprehensive headers for multi-source analysis
        self.headers = [
            # Basic decision info
            'timestamp',
            'ticker',
            'decision',
            'confidence',
            'final_reasoning',
            
            # Combined scores
            'news_score',
            'technical_score',
            'combined_score',
            'article_count',
            
            # Multi-source news analysis
            'news_direction',
            'news_confidence',
            'news_reasoning',
            'sources_count',
            'sources_used',  # Comma-separated list
            
            # Individual service scores
            'finbert_prediction',    # direction:confidence:score
            'gemini_prediction',     # direction:confidence:score
            'openai_prediction',     # direction:confidence:score
            'claude_prediction',     # direction:confidence:score
            'alpha_vantage_prediction',  # direction:confidence:score
            'polygon_prediction',    # direction:confidence:score
            'tiingo_prediction',     # direction:confidence:score
            'keyword_prediction',    # direction:confidence:score
            
            # Weighted scores for each service
            'finbert_weighted_score',
            'gemini_weighted_score',
            'openai_weighted_score',
            'claude_weighted_score',
            'alpha_vantage_weighted_score',
            'polygon_weighted_score',
            'tiingo_weighted_score',
            'keyword_weighted_score',
            
            # Service weights used
            'service_weights',  # JSON string of weights
            
            # Technical analysis
            'technical_direction',
            'technical_strength',
            'technical_reasoning',
            
            # Analysis metadata
            'analysis_method',  # "multi_source"
            'market_context'    # ETF context if applicable
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
        """Create new CSV file with enhanced headers"""
        try:
            # Ensure output directory exists
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

            log_info(f"Created new enhanced CSV file: {self.csv_path}")

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
                    log_info("CSV headers don't match enhanced multi-source version, backing up and recreating")
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

    def log_decision(self, decision: EnhancedTradingDecision) -> bool:
        """Log a single enhanced trading decision to CSV"""
        try:
            row_data = self._decision_to_row(decision)

            with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)

            log_debug(f"Logged enhanced decision: {decision.ticker} - {decision.decision}")
            return True

        except Exception as e:
            log_error(f"Error logging decision to CSV: {e}")
            return False

    def log_decisions_batch(self, decisions: List[EnhancedTradingDecision]) -> int:
        """Log multiple enhanced trading decisions in batch"""
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

            log_info(f"Logged {logged_count} enhanced trading decisions to CSV")
            return logged_count

        except Exception as e:
            log_error(f"Error in batch logging: {e}")
            return 0

    def _decision_to_row(self, decision: EnhancedTradingDecision) -> List[str]:
        """Convert EnhancedTradingDecision to CSV row with comprehensive data"""

        # Prepare individual service predictions dictionary
        service_predictions = {}
        service_weighted_scores = {}
        
        if decision.individual_predictions:
            for pred in decision.individual_predictions:
                source = pred['source']
                service_predictions[source] = f"{pred['direction']}:{pred['confidence']:.3f}:{pred['raw_score']:.3f}"
        
        if decision.weighted_scores:
            service_weighted_scores = decision.weighted_scores

        # All possible services
        all_services = ['finbert', 'gemini', 'openai', 'claude', 'alpha_vantage', 'polygon', 'tiingo', 'enhanced_keyword_analysis']

        # Determine market context
        market_context = ""
        if decision.ticker in ['SPY', 'QQQ', 'IWM', 'TLT', 'XLE', 'GLD', 'VIX']:
            market_context = "etf_market_news"

        # Prepare sources used string
        sources_used_str = ",".join(decision.sources_used) if decision.sources_used else ""
        
        # Prepare service weights JSON
        service_weights_json = json.dumps(decision.source_weights) if decision.source_weights else "{}"

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
            
            # Multi-source news analysis
            decision.news_prediction.direction if decision.news_prediction else '',
            f"{decision.news_prediction.confidence:.4f}" if decision.news_prediction else '',
            decision.news_prediction.reasoning[:200] if decision.news_prediction and decision.news_prediction.reasoning else '',
            str(len(decision.sources_used)) if decision.sources_used else '0',
            sources_used_str,
            
            # Individual service predictions (direction:confidence:raw_score)
            service_predictions.get('finbert', ''),
            service_predictions.get('gemini', ''),
            service_predictions.get('openai', ''),
            service_predictions.get('claude', ''),
            service_predictions.get('alpha_vantage', ''),
            service_predictions.get('polygon', ''),
            service_predictions.get('tiingo', ''),
            service_predictions.get('enhanced_keyword_analysis', ''),
            
            # Weighted scores for each service
            f"{service_weighted_scores.get('finbert', 0.0):.4f}",
            f"{service_weighted_scores.get('gemini', 0.0):.4f}",
            f"{service_weighted_scores.get('openai', 0.0):.4f}",
            f"{service_weighted_scores.get('claude', 0.0):.4f}",
            f"{service_weighted_scores.get('alpha_vantage', 0.0):.4f}",
            f"{service_weighted_scores.get('polygon', 0.0):.4f}",
            f"{service_weighted_scores.get('tiingo', 0.0):.4f}",
            f"{service_weighted_scores.get('enhanced_keyword_analysis', 0.0):.4f}",
            
            # Service weights used
            service_weights_json,
            
            # Technical analysis
            decision.technical_signal.direction if decision.technical_signal else '',
            f"{decision.technical_signal.strength:.4f}" if decision.technical_signal else '',
            decision.technical_signal.reasoning[:200] if decision.technical_signal and decision.technical_signal.reasoning else '',
            
            # Analysis metadata
            "multi_source",
            market_context
        ]

    def get_csv_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics about the CSV file"""
        try:
            if not self.csv_path.exists():
                return {'exists': False}

            file_size = self.csv_path.stat().st_size

            # Count rows and analyze by categories
            row_count = 0
            decisions_by_type = {'LONG': 0, 'SHORT': 0, 'NONE': 0}
            source_usage_stats = {}
            multi_source_stats = {'single_source': 0, 'multi_source': 0}

            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                for row in reader:
                    row_count += 1

                    # Decision type
                    decision_type = row.get('decision', 'NONE')
                    decisions_by_type[decision_type] = decisions_by_type.get(decision_type, 0) + 1

                    # Sources used analysis
                    sources_used = row.get('sources_used', '')
                    if sources_used:
                        source_list = sources_used.split(',')
                        source_count = len(source_list)
                        
                        if source_count == 1:
                            multi_source_stats['single_source'] += 1
                        else:
                            multi_source_stats['multi_source'] += 1
                        
                        # Track individual source usage
                        for source in source_list:
                            source = source.strip()
                            source_usage_stats[source] = source_usage_stats.get(source, 0) + 1

            return {
                'exists': True,
                'file_size_bytes': file_size,
                'total_decisions': row_count,
                'decisions_by_type': decisions_by_type,
                'source_usage_stats': source_usage_stats,
                'multi_source_stats': multi_source_stats,
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

    def export_multi_source_analysis(self, output_path: Path, 
                                    min_sources: int = 2) -> int:
        """Export decisions that used multiple sources for analysis"""
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
                        sources_count = int(row.get('sources_count', '0'))
                        if sources_count >= min_sources:
                            writer.writerow(row)
                            exported_count += 1

            log_info(f"Exported {exported_count} multi-source decisions to {output_path}")
            return exported_count

        except Exception as e:
            log_error(f"Error exporting multi-source analysis: {e}")
            return 0

    def get_service_performance_report(self) -> Dict[str, Any]:
        """Generate a report on individual service performance"""
        try:
            if not self.csv_path.exists():
                return {}

            service_stats = {}
            total_decisions = 0

            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                for row in reader:
                    total_decisions += 1
                    sources_used = row.get('sources_used', '').split(',')
                    
                    for source in sources_used:
                        source = source.strip()
                        if source:
                            if source not in service_stats:
                                service_stats[source] = {
                                    'usage_count': 0,
                                    'decisions_contributed': 0,
                                    'avg_weighted_score': 0.0,
                                    'total_weighted_score': 0.0
                                }
                            
                            service_stats[source]['usage_count'] += 1
                            service_stats[source]['decisions_contributed'] += 1
                            
                            # Get weighted score for this service
                            weighted_score_col = f"{source}_weighted_score"
                            if weighted_score_col in row:
                                try:
                                    weighted_score = float(row[weighted_score_col])
                                    service_stats[source]['total_weighted_score'] += weighted_score
                                except (ValueError, TypeError):
                                    pass

            # Calculate averages
            for source, stats in service_stats.items():
                if stats['decisions_contributed'] > 0:
                    stats['avg_weighted_score'] = stats['total_weighted_score'] / stats['decisions_contributed']
                    stats['usage_percentage'] = (stats['decisions_contributed'] / total_decisions) * 100

            return {
                'total_decisions_analyzed': total_decisions,
                'service_performance': service_stats
            }

        except Exception as e:
            log_error(f"Error generating service performance report: {e}")
            return {}