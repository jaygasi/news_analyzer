import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def convert_list_to_dataframe(data_list: list, column_list: list):
    df = pd.DataFrame(data_list)
    df = df[column_list] if not df.empty else pd.DataFrame(columns=column_list)
    return df


# Helper function for normalizing columns
def normalize_columns(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def calculate_ratios_score(in_df: pd.DataFrame):
    df = in_df.copy()
    # Normalize relevant columns
    df = normalize_columns(df, ['grossProfitMarginTTM', 'currentRatioTTM', 'debtEquityRatioTTM'])

    # Calculate the ratios score with weights, penalizing high debt
    in_df['ratios_score'] = round((
        df['grossProfitMarginTTM'] * 0.5 +  # High weight on profitability
        df['currentRatioTTM'] * 0.3 -  # Medium weight on liquidity
        df['debtEquityRatioTTM'] * 0.2  # Penalty for high leverage
    ), 2)
    return in_df


def calculate_quarterly_income_score(in_df: pd.DataFrame):
    df = in_df.copy()
    # Normalize relevant columns
    df = normalize_columns(df, ['last_revenue', 'last_net_income', 'revenue_trend', 'net_income_trend', 'cost_expenses_trend'])

    # Calculate score, penalizing high cost/expenses trend
    in_df['quarterly_income_score'] = round((
        df['last_revenue'] * 0.2 +
        df['last_net_income'] * 0.15 +
        df['revenue_trend'] * 0.25 +
        df['net_income_trend'] * 0.2 -
        df['cost_expenses_trend'] * 0.1
    ), 2)
    return in_df


def calculate_annual_income_score(in_df: pd.DataFrame):
    df = in_df.copy()
    # Normalize relevant columns
    df = normalize_columns(df, ['last_revenue', 'last_net_income', 'revenue_trend', 'net_income_trend', 'cost_expenses_trend'])

    # Calculate score
    in_df['annual_income_score'] = round((
        df['last_revenue'] * 0.2 +
        df['last_net_income'] * 0.15 +
        df['revenue_trend'] * 0.25 +
        df['net_income_trend'] * 0.2 -
        df['cost_expenses_trend'] * 0.1
    ), 2)
    return in_df


def calculate_quarterly_balance_sheet_score(in_df: pd.DataFrame):
    df = in_df.copy()
    # Normalize relevant columns
    df = normalize_columns(df, ['last_total_assets', 'last_cash_short_term_investments', 'last_total_debt', 'total_assets_trend', 'total_shareholders_equity_trend'])

    # Calculate score, penalizing high debt
    in_df['quarterly_balance_sheet_score'] = round((
        df['last_total_assets'] * 0.2 +
        df['last_cash_short_term_investments'] * 0.15 -
        df['last_total_debt'] * 0.2 +
        df['total_assets_trend'] * 0.25 +
        df['total_shareholders_equity_trend'] * 0.2
    ), 2)
    return in_df


def calculate_annual_balance_sheet_score(in_df: pd.DataFrame):
    df = in_df.copy()
    # Normalize relevant columns
    df = normalize_columns(df, ['last_total_assets', 'last_cash_short_term_investments', 'last_total_debt', 'total_assets_trend', 'total_shareholders_equity_trend'])

    # Calculate score, penalizing high debt
    in_df['annual_balance_sheet_score'] = round((
        df['last_total_assets'] * 0.2 +
        df['last_cash_short_term_investments'] * 0.15 -
        df['last_total_debt'] * 0.2 +
        df['total_assets_trend'] * 0.25 +
        df['total_shareholders_equity_trend'] * 0.2
    ), 2)
    return in_df


def calculate_quarterly_cashflow_score(in_df: pd.DataFrame):
    df = in_df.copy()
    # Normalize relevant columns
    df = normalize_columns(df, ['last_operating_cashflow', 'last_free_cashflow', 'operating_cashflow_trend', 'free_cashflow_trend', 'capital_expenditure_trend'])

    # Calculate score, penalizing high capital expenditures
    in_df['quarterly_cashflow_score'] = round((
        df['last_operating_cashflow'] * 0.25 +
        df['last_free_cashflow'] * 0.2 +
        df['operating_cashflow_trend'] * 0.25 +
        df['free_cashflow_trend'] * 0.2 -
        df['capital_expenditure_trend'] * 0.1
    ), 2)
    return in_df


def calculate_annual_cashflow_score(in_df: pd.DataFrame):
    df = in_df.copy()
    # Normalize relevant columns
    df = normalize_columns(df, ['last_operating_cashflow', 'last_free_cashflow', 'operating_cashflow_trend', 'free_cashflow_trend', 'capital_expenditure_trend'])

    # Calculate score, penalizing high capital expenditures
    in_df['annual_cashflow_score'] = round((
        df['last_operating_cashflow'] * 0.2 +
        df['last_free_cashflow'] * 0.2 +
        df['operating_cashflow_trend'] * 0.2 +
        df['free_cashflow_trend'] * 0.2 -
        df['capital_expenditure_trend'] * 0.2
    ), 2)
    return in_df


def calculate_price_target_score(in_df: pd.DataFrame):
    df = in_df.copy()
    # Normalize relevant columns
    df = normalize_columns(df, ['avg_price_target_change_percent',
                                'price_target_coefficient_variation',
                                'num_price_target_analysts'])

    # Calculate score, penalizing high price target coefficient variation
    in_df['price_target_score'] = round((
        df['avg_price_target_change_percent'] * 0.6 -
        df['price_target_coefficient_variation'] * 0.2 +
        df['num_price_target_analysts'] * 0.2
    ), 2)
    return in_df


def calculate_inst_own_score(in_df: pd.DataFrame):
    if len(in_df) == 0:
        return in_df

    df = in_df.copy()
    # Normalize relevant columns
    df = normalize_columns(df, ['investors_holding', 'investors_holding_change', 'total_invested',
                                'total_invested_change', 'investors_put_call_ratio'])

    # Calculate score, penalizing high put/call ratio
    in_df['inst_own_score'] = round((
        df['investors_holding'] * 0.4 +
        df['investors_holding_change'] * 0.2 +
        df['total_invested'] * 0.4 +
        df['total_invested_change'] * 0.2 -
        df['investors_put_call_ratio'] * 0.2
    ), 2)
    return in_df


def calculate_final_score(ratios_df, quarterly_income_df, annual_income_df,
                          quarterly_balance_sheet_df, annual_balance_sheet_df,
                          quarterly_cashflow_df, annual_cashflow_df,
                          price_target_df, inst_own_df=None):
    # Merge all score columns from each section by symbol
    scores_df = ratios_df[['symbol', 'ratios_score']].merge(
        quarterly_income_df[['symbol', 'quarterly_income_score']], on='symbol', how='outer'
    )

    scores_df = scores_df.merge(
        annual_income_df[['symbol', 'annual_income_score']], on='symbol', how='outer'
    )
    scores_df = scores_df.merge(
        quarterly_balance_sheet_df[['symbol', 'quarterly_balance_sheet_score']], on='symbol', how='outer'
    )
    scores_df = scores_df.merge(
        annual_balance_sheet_df[['symbol', 'annual_balance_sheet_score']], on='symbol', how='outer'
    )
    scores_df = scores_df.merge(
        quarterly_cashflow_df[['symbol', 'quarterly_cashflow_score']], on='symbol', how='outer'
    )
    scores_df = scores_df.merge(
        annual_cashflow_df[['symbol', 'annual_cashflow_score']], on='symbol', how='outer'
    )
    scores_df = scores_df.merge(
        price_target_df[['symbol', 'price_target_score']], on='symbol', how='outer'
    )
    if inst_own_df is not None:
        scores_df = scores_df.merge(
            inst_own_df[['symbol', 'inst_own_score']], on='symbol', how='outer'
        )
    if 'inst_own_score' not in scores_df.columns:
        scores_df['inst_own_score'] = 0

    # Replace nans
    scores_df = scores_df.fillna(0)

    # Calculate final_score as a sum of individual section scores, handling any missing values
    scores_df['final_score'] = round(scores_df[['ratios_score', 'quarterly_income_score',
                                          'annual_income_score', 'quarterly_balance_sheet_score',
                                          'annual_balance_sheet_score', 'quarterly_cashflow_score',
                                          'annual_cashflow_score', 'price_target_score',
                                          'inst_own_score']].sum(axis=1), 2)

    scores_df.sort_values(by='final_score', ascending=False, inplace=True)
    return scores_df


def align_section_order(scores_df, section_df):
    # Filter section_df to only include rows where the symbol exists in scores_df
    aligned_section_df = section_df[section_df['symbol'].isin(scores_df['symbol'])]
    # Sort aligned_section_df to match the order of symbols in scores_df
    aligned_section_df = aligned_section_df.set_index('symbol').reindex(scores_df['symbol']).reset_index()
    return aligned_section_df