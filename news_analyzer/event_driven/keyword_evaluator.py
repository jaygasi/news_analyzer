import json
import logging
from collections import defaultdict
import pandas as pd
import numpy as np

import database
from config import (
    KEYWORD_WEIGHTS,
    KEYWORD_EVAL_PERFORMANCE_COLUMN,
    KEYWORD_EVAL_MIN_TRADES_FOR_STATS,
    KEYWORD_EVAL_POSITIVE_PERF_THRESHOLD,
    KEYWORD_EVAL_NEGATIVE_PERF_THRESHOLD,
    KEYWORD_EVAL_AI_EXPLANATION_THRESHOLD,
    KEYWORD_WEIGHT_ADJUSTMENT_FACTOR,
)
from logger_config import setup_logging


def fetch_data_for_evaluation():
    """Fetches trades with performance and keyword matches from the database."""
    logging.info("Fetching data for keyword evaluation...")
    trades_query = f"""
        SELECT id, {KEYWORD_EVAL_PERFORMANCE_COLUMN}, feature_model_explanation
        FROM trades
        WHERE tracking_status = 'completed'
    """
    conn = database.get_db_connection()
    trades_df = pd.read_sql_query(trades_query, conn)
    conn.close()

    if trades_df.empty:
        logging.warning("No trades with performance data found. Cannot evaluate keywords.")
        return None, None

    keyword_matches_query = "SELECT trade_id, keyword, weight FROM keyword_matches"
    conn = database.get_db_connection()
    keyword_matches_df = pd.read_sql_query(keyword_matches_query, conn)
    conn.close()

    if keyword_matches_df.empty:
        logging.warning("No keyword matches found. Cannot evaluate keywords.")
        return None, None

    logging.info(f"Fetched {len(trades_df)} trades and {len(keyword_matches_df)} keyword matches.")
    return trades_df, keyword_matches_df


def _populate_keyword_base_stats(trades_df: pd.DataFrame, keyword_matches_df: pd.DataFrame) -> dict:
    """Populates initial statistics for each keyword based on trade performance and AI mentions."""
    keyword_stats = defaultdict(lambda: {
        'trade_ids': [],
        'performances': [],
        'ai_explanation_mentions': 0,
        'current_weight': 0.0
    })
    for _, row in keyword_matches_df.iterrows():
        trade_id = row['trade_id']
        keyword = row['keyword']

        trade_info = trades_df[trades_df['id'] == trade_id]
        if not trade_info.empty:
            performance = trade_info.iloc[0][KEYWORD_EVAL_PERFORMANCE_COLUMN]
            ai_explanation_str = trade_info.iloc[0]['feature_model_explanation']

            keyword_stats[keyword]['trade_ids'].append(trade_id)
            keyword_stats[keyword]['performances'].append(performance)
            keyword_stats[keyword]['current_weight'] = KEYWORD_WEIGHTS.get(keyword, 0.0)

            if ai_explanation_str:
                try:
                    ai_explanation_list = json.loads(ai_explanation_str)
                    if any(keyword.lower() in phrase.lower() for phrase in ai_explanation_list):
                        keyword_stats[keyword]['ai_explanation_mentions'] += 1
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse AI explanation for trade_id {trade_id}")
    return keyword_stats


def _calculate_suggested_weight(current_weight: float, avg_perf: float, std_dev_perf: float, num_trades: int) -> float:
    """
    Calculates a suggested new weight for a keyword based on its performance.
    This is a simplified heuristic and can be made more sophisticated.
    """
    if num_trades < KEYWORD_EVAL_MIN_TRADES_FOR_STATS:
        # If not enough data, suggest keeping current weight or a small default
        return current_weight

    # Basic proportional adjustment
    # Factor in how far the average performance is from zero
    adjustment = avg_perf * KEYWORD_WEIGHT_ADJUSTMENT_FACTOR

    # Dampen adjustment if performance is highly volatile
    if std_dev_perf > abs(avg_perf) * 1.5: # If std dev is much larger than avg perf
        adjustment *= 0.5 # Halve the adjustment

    suggested_weight = current_weight + adjustment

    # Clamp weights to a reasonable range, e.g., -5.0 to 5.0
    suggested_weight = max(-5.0, min(5.0, suggested_weight))

    # If performance is very neutral, nudge weight towards zero
    if abs(avg_perf) < 0.05 and abs(suggested_weight) > 0.1:
        suggested_weight *= 0.8 # Nudge towards zero

    return round(suggested_weight, 2)


def _generate_suggestion_text(stats: dict, avg_perf: float, std_dev_perf: float, ai_mention_ratio: float, num_trades: int) -> str:
    """
    Generates a human-readable suggestion for a keyword.
    """
    current_weight = stats['current_weight']

    if num_trades < KEYWORD_EVAL_MIN_TRADES_FOR_STATS:
        if num_trades > 0 and avg_perf <= (KEYWORD_EVAL_POSITIVE_PERF_THRESHOLD / 2):
            return "Very few trades, poor/neutral perf. Consider REMOVING or reducing weight drastically."
        return "Observe (insufficient data)."

    suggestion_core = ""
    if avg_perf > KEYWORD_EVAL_POSITIVE_PERF_THRESHOLD:
        suggestion_core = "Strong positive perf."
    elif avg_perf < KEYWORD_EVAL_NEGATIVE_PERF_THRESHOLD:
        suggestion_core = "Significant negative perf."
    else:
        suggestion_core = "Neutral performance."

    ai_favor_note = ""
    is_significant_perf = (avg_perf > KEYWORD_EVAL_POSITIVE_PERF_THRESHOLD or
                           avg_perf < KEYWORD_EVAL_NEGATIVE_PERF_THRESHOLD)
    if is_significant_perf and ai_mention_ratio > KEYWORD_EVAL_AI_EXPLANATION_THRESHOLD:
        ai_favor_note = " (AI also favors)" if avg_perf > KEYWORD_EVAL_POSITIVE_PERF_THRESHOLD else " (AI also correlates with these outcomes)"

    variance_note = ""
    if std_dev_perf > (abs(avg_perf) * 0.75) and abs(avg_perf) > 0.1:
        variance_note = " (High Variance)"

    return f"{suggestion_core}{ai_favor_note}{variance_note}"


def _print_evaluation_report(keyword_stats: dict) -> Tuple[set, Dict[str, float]]:
    """Prints the main keyword evaluation report table and returns suggested weights."""
    print("\n--- Keyword Evaluation Report ---")
    header = f"{'Keyword':<30} | {'Cur.Wgt':>7} | {'Sug.Wgt':>7} | {'Trades':>6} | {'Avg.Perf':>9} | {'Std.Dev':>8} | {'AI Ment.':>8} | Suggestion"
    print(header)
    print("-" * len(header))

    evaluated_keywords = set()
    suggested_weights = {}

    for keyword, stats in sorted(keyword_stats.items(), key=lambda item: len(item[1]['trade_ids']), reverse=True):
        evaluated_keywords.add(keyword)
        num_trades = len(stats['trade_ids'])
        if num_trades == 0:
            continue

        performances = np.array(stats['performances'])
        avg_perf = np.mean(performances) if num_trades > 0 else 0.0
        std_dev_perf = np.std(performances) if num_trades > 1 else 0.0
        ai_mention_ratio = stats['ai_explanation_mentions'] / num_trades if num_trades > 0 else 0.0
        current_weight = stats['current_weight']

        suggested_weight = _calculate_suggested_weight(current_weight, avg_perf, std_dev_perf, num_trades)
        suggested_weights[keyword] = suggested_weight
        suggestion_text = _generate_suggestion_text(stats, avg_perf, std_dev_perf, ai_mention_ratio, num_trades)

        print(f"{keyword:<30} | {current_weight:>7.2f} | {suggested_weight:>7.2f} | {num_trades:>6} | {avg_perf:>8.2f}% | {std_dev_perf:>8.2f} | {stats['ai_explanation_mentions']:>8} | {suggestion_text}")

    print("-" * len(header))
    return evaluated_keywords, suggested_weights


def _print_unused_keywords_report(evaluated_keywords: set):
    """Prints a report of keywords in config but not found in evaluated trades."""
    print("\n--- Unused Keywords in Config ---")
    unused_keywords = set(KEYWORD_WEIGHTS.keys()) - evaluated_keywords
    if unused_keywords:
        for keyword in sorted(list(unused_keywords)):
            print(f"- '{keyword}' (Weight: {KEYWORD_WEIGHTS[keyword]:.2f}) was not found in any processed trades with performance data.")
    else:
        print("All keywords in config.py were found in processed trades.")


def analyze_keywords(trades_df: pd.DataFrame, keyword_matches_df: pd.DataFrame):
    """Analyzes keyword performance and AI explanation alignment."""
    logging.info(f"Starting keyword analysis using '{KEYWORD_EVAL_PERFORMANCE_COLUMN}'...")

    keyword_stats = _populate_keyword_base_stats(trades_df, keyword_matches_df)
    evaluated_keywords, suggested_weights = _print_evaluation_report(keyword_stats)
    _print_unused_keywords_report(evaluated_keywords)

    print("\n--- Suggested KEYWORD_WEIGHTS for config.py ---")
    print("KEYWORD_WEIGHTS = {\n")
    for keyword, weight in suggested_weights.items():
        print(f"    '{keyword}': {weight:.2f},\n")
    print("}\n")
    print("Review suggestions and manually update KEYWORD_WEIGHTS in config.py if desired.")


def main():
    setup_logging()
    trades_df, keyword_matches_df = fetch_data_for_evaluation()
    if trades_df is not None and keyword_matches_df is not None:
        analyze_keywords(trades_df, keyword_matches_df)
    else:
        logging.info("Exiting due to missing data for evaluation.")


if __name__ == "__main__":
    main()


