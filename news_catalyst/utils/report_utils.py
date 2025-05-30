"""
Optimized report utilities with improved performance and vectorized operations
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Union, Dict, Any
import warnings

# Suppress pandas performance warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def convert_list_to_dataframe(data_list: List[Dict[str, Any]], 
                             column_list: List[str]) -> pd.DataFrame:
    """
    Convert list of dictionaries to DataFrame with specified columns.
    
    Args:
        data_list: List of dictionaries containing data
        column_list: List of column names to include
        
    Returns:
        DataFrame with specified columns
    """
    if not data_list:
        return pd.DataFrame(columns=column_list)
    
    df = pd.DataFrame(data_list)
    
    # Use intersection to get available columns, maintaining order
    available_columns = [col for col in column_list if col in df.columns]
    
    if available_columns:
        return df[available_columns].copy()
    else:
        return pd.DataFrame(columns=column_list)


def normalize_columns(df: pd.DataFrame, 
                     columns: List[str], 
                     feature_range: tuple = (0, 1)) -> pd.DataFrame:
    """
    Normalize specified columns using MinMaxScaler with improved error handling.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
        feature_range: Range for normalization (default: 0-1)
        
    Returns:
        DataFrame with normalized columns
    """
    if df.empty or not columns:
        return df.copy()
    
    # Check which columns actually exist
    existing_columns = [col for col in columns if col in df.columns]
    
    if not existing_columns:
        return df.copy()
    
    df_copy = df.copy()
    
    # Handle missing values before normalization
    for col in existing_columns:
        if df_copy[col].isna().any():
            # Fill NaN with median for better normalization
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # Vectorized normalization
    scaler = MinMaxScaler(feature_range=feature_range)
    df_copy[existing_columns] = scaler.fit_transform(df_copy[existing_columns])
    
    return df_copy


def calculate_ratios_score(in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ratios score with optimized computation.
    
    Args:
        in_df: Input DataFrame with financial ratios
        
    Returns:
        DataFrame with ratios_score column added
    """
    if in_df.empty:
        return in_df.copy()
    
    df = in_df.copy()
    required_columns = ['grossProfitMarginTTM', 'currentRatioTTM', 'debtEquityRatioTTM']
    
    # Check if required columns exist
    if not all(col in df.columns for col in required_columns):
        df['ratios_score'] = 0.0
        return df
    
    # Normalize relevant columns
    df_normalized = normalize_columns(df, required_columns)
    
    # Vectorized score calculation
    df['ratios_score'] = np.round(
        df_normalized['grossProfitMarginTTM'] * 0.5 +
        df_normalized['currentRatioTTM'] * 0.3 -
        df_normalized['debtEquityRatioTTM'] * 0.2,
        2
    )
    
    return df


def calculate_quarterly_income_score(in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate quarterly income score with optimized computation.
    
    Args:
        in_df: Input DataFrame with income data
        
    Returns:
        DataFrame with quarterly_income_score column added
    """
    if in_df.empty:
        return in_df.copy()
    
    df = in_df.copy()
    required_columns = [
        'last_revenue', 'last_net_income', 'revenue_trend', 
        'net_income_trend', 'cost_expenses_trend'
    ]
    
    # Check if required columns exist
    if not all(col in df.columns for col in required_columns):
        df['quarterly_income_score'] = 0.0
        return df
    
    # Normalize relevant columns
    df_normalized = normalize_columns(df, required_columns)
    
    # Vectorized score calculation
    df['quarterly_income_score'] = np.round(
        df_normalized['last_revenue'] * 0.2 +
        df_normalized['last_net_income'] * 0.15 +
        df_normalized['revenue_trend'] * 0.25 +
        df_normalized['net_income_trend'] * 0.2 -
        df_normalized['cost_expenses_trend'] * 0.1,
        2
    )
    
    return df


def calculate_annual_income_score(in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate annual income score with optimized computation.
    
    Args:
        in_df: Input DataFrame with income data
        
    Returns:
        DataFrame with annual_income_score column added
    """
    if in_df.empty:
        return in_df.copy()
    
    df = in_df.copy()
    required_columns = [
        'last_revenue', 'last_net_income', 'revenue_trend',
        'net_income_trend', 'cost_expenses_trend'
    ]
    
    # Check if required columns exist
    if not all(col in df.columns for col in required_columns):
        df['annual_income_score'] = 0.0
        return df
    
    # Normalize relevant columns
    df_normalized = normalize_columns(df, required_columns)
    
    # Vectorized score calculation
    df['annual_income_score'] = np.round(
        df_normalized['last_revenue'] * 0.2 +
        df_normalized['last_net_income'] * 0.15 +
        df_normalized['revenue_trend'] * 0.25 +
        df_normalized['net_income_trend'] * 0.2 -
        df_normalized['cost_expenses_trend'] * 0.1,
        2
    )
    
    return df


def calculate_balance_sheet_score(in_df: pd.DataFrame, 
                                 score_column: str) -> pd.DataFrame:
    """
    Generic balance sheet score calculation to reduce code duplication.
    
    Args:
        in_df: Input DataFrame with balance sheet data
        score_column: Name of the score column to create
        
    Returns:
        DataFrame with score column added
    """
    if in_df.empty:
        return in_df.copy()
    
    df = in_df.copy()
    required_columns = [
        'last_total_assets', 'last_cash_short_term_investments',
        'last_total_debt', 'total_assets_trend', 'total_shareholders_equity_trend'
    ]
    
    # Check if required columns exist
    if not all(col in df.columns for col in required_columns):
        df[score_column] = 0.0
        return df
    
    # Normalize relevant columns
    df_normalized = normalize_columns(df, required_columns)
    
    # Vectorized score calculation
    df[score_column] = np.round(
        df_normalized['last_total_assets'] * 0.2 +
        df_normalized['last_cash_short_term_investments'] * 0.15 -
        df_normalized['last_total_debt'] * 0.2 +
        df_normalized['total_assets_trend'] * 0.25 +
        df_normalized['total_shareholders_equity_trend'] * 0.2,
        2
    )
    
    return df


def calculate_quarterly_balance_sheet_score(in_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate quarterly balance sheet score."""
    return calculate_balance_sheet_score(in_df, 'quarterly_balance_sheet_score')


def calculate_annual_balance_sheet_score(in_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate annual balance sheet score."""
    return calculate_balance_sheet_score(in_df, 'annual_balance_sheet_score')


def calculate_cashflow_score(in_df: pd.DataFrame, 
                           score_column: str,
                           weights: Dict[str, float]) -> pd.DataFrame:
    """
    Generic cashflow score calculation to reduce code duplication.
    
    Args:
        in_df: Input DataFrame with cashflow data
        score_column: Name of the score column to create
        weights: Dictionary of column weights
        
    Returns:
        DataFrame with score column added
    """
    if in_df.empty:
        return in_df.copy()
    
    df = in_df.copy()
    required_columns = [
        'last_operating_cashflow', 'last_free_cashflow',
        'operating_cashflow_trend', 'free_cashflow_trend', 'capital_expenditure_trend'
    ]
    
    # Check if required columns exist
    if not all(col in df.columns for col in required_columns):
        df[score_column] = 0.0
        return df
    
    # Normalize relevant columns
    df_normalized = normalize_columns(df, required_columns)
    
    # Vectorized score calculation with custom weights
    df[score_column] = np.round(
        df_normalized['last_operating_cashflow'] * weights['operating'] +
        df_normalized['last_free_cashflow'] * weights['free'] +
        df_normalized['operating_cashflow_trend'] * weights['operating_trend'] +
        df_normalized['free_cashflow_trend'] * weights['free_trend'] -
        df_normalized['capital_expenditure_trend'] * weights['capex'],
        2
    )
    
    return df


def calculate_quarterly_cashflow_score(in_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate quarterly cashflow score."""
    weights = {
        'operating': 0.25,
        'free': 0.2,
        'operating_trend': 0.25,
        'free_trend': 0.2,
        'capex': 0.1
    }
    return calculate_cashflow_score(in_df, 'quarterly_cashflow_score', weights)


def calculate_annual_cashflow_score(in_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate annual cashflow score."""
    weights = {
        'operating': 0.2,
        'free': 0.2,
        'operating_trend': 0.2,
        'free_trend': 0.2,
        'capex': 0.2
    }
    return calculate_cashflow_score(in_df, 'annual_cashflow_score', weights)


def calculate_price_target_score(in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price target score with optimized computation.
    
    Args:
        in_df: Input DataFrame with price target data
        
    Returns:
        DataFrame with price_target_score column added
    """
    if in_df.empty:
        return in_df.copy()
    
    df = in_df.copy()
    required_columns = [
        'avg_price_target_change_percent',
        'price_target_coefficient_variation',
        'num_price_target_analysts'
    ]
    
    # Check if required columns exist
    if not all(col in df.columns for col in required_columns):
        df['price_target_score'] = 0.0
        return df
    
    # Normalize relevant columns
    df_normalized = normalize_columns(df, required_columns)
    
    # Vectorized score calculation
    df['price_target_score'] = np.round(
        df_normalized['avg_price_target_change_percent'] * 0.6 -
        df_normalized['price_target_coefficient_variation'] * 0.2 +
        df_normalized['num_price_target_analysts'] * 0.2,
        2
    )
    
    return df


def calculate_inst_own_score(in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate institutional ownership score with optimized computation.
    
    Args:
        in_df: Input DataFrame with institutional ownership data
        
    Returns:
        DataFrame with inst_own_score column added
    """
    if in_df.empty:
        return in_df.copy()
    
    df = in_df.copy()
    required_columns = [
        'investors_holding', 'investors_holding_change', 'total_invested',
        'total_invested_change', 'investors_put_call_ratio'
    ]
    
    # Check if required columns exist
    if not all(col in df.columns for col in required_columns):
        df['inst_own_score'] = 0.0
        return df
    
    # Normalize relevant columns
    df_normalized = normalize_columns(df, required_columns)
    
    # Vectorized score calculation
    df['inst_own_score'] = np.round(
        df_normalized['investors_holding'] * 0.4 +
        df_normalized['investors_holding_change'] * 0.2 +
        df_normalized['total_invested'] * 0.4 +
        df_normalized['total_invested_change'] * 0.2 -
        df_normalized['investors_put_call_ratio'] * 0.2,
        2
    )
    
    return df


def calculate_final_score(*score_dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate final score by merging multiple score DataFrames.
    
    Args:
        *score_dfs: Variable number of DataFrames with scores
        
    Returns:
        DataFrame with final_score column
    """
    if not score_dfs or all(df.empty for df in score_dfs):
        return pd.DataFrame()
    
    # Filter out empty DataFrames
    valid_dfs = [df for df in score_dfs if not df.empty and 'symbol' in df.columns]
    
    if not valid_dfs:
        return pd.DataFrame()
    
    # Start with the first valid DataFrame
    result_df = valid_dfs[0].copy()
    
    # Merge all other DataFrames
    for df in valid_dfs[1:]:
        # Get score columns (exclude 'symbol')
        score_cols = [col for col in df.columns if col != 'symbol' and 'score' in col]
        if score_cols:
            merge_cols = ['symbol'] + score_cols
            result_df = result_df.merge(
                df[merge_cols], 
                on='symbol', 
                how='outer',
                suffixes=('', '_y')
            )
    
    # Fill NaN values with 0
    score_columns = [col for col in result_df.columns if 'score' in col]
    result_df[score_columns] = result_df[score_columns].fillna(0)
    
    # Calculate final score as sum of all score columns
    if score_columns:
        result_df['final_score'] = result_df[score_columns].sum(axis=1).round(2)
    else:
        result_df['final_score'] = 0.0
    
    # Sort by final score descending
    result_df = result_df.sort_values('final_score', ascending=False)
    
    return result_df


def align_section_order(scores_df: pd.DataFrame, 
                       section_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align section DataFrame order to match scores DataFrame with optimization.
    
    Args:
        scores_df: Reference DataFrame with symbol order
        section_df: DataFrame to reorder
        
    Returns:
        Reordered section DataFrame
    """
    if scores_df.empty or section_df.empty:
        return section_df.copy()
    
    if 'symbol' not in scores_df.columns or 'symbol' not in section_df.columns:
        return section_df.copy()
    
    # More efficient: use merge with left join to maintain order
    aligned_df = scores_df[['symbol']].merge(
        section_df, 
        on='symbol', 
        how='left'
    )
    
    return aligned_df.dropna()