import numpy as np
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
import numpy as np
from scipy.stats import linregress


def calculate_trend_linear_regression(values: list):
    x = np.arange(len(values))
    slope, intercept, _, _, _ = linregress(x, values)
    trend = intercept + slope * x
    return trend


def calculate_trend(df: pd.DataFrame,
                    target_column: str='close',
                    out_column: str = ['close_smoothed_slope'],
                    bandwidth: int = 9,
                    slope_window: int = 3):
    # Add smoothed line
    df = add_kernel_reg_smoothed_line(df,
                                      column_list=[target_column],
                                      output_cols=[f"{target_column}_smoothed"],
                                      bandwidth=bandwidth,
                                      var_type='c')

    # Calculate slope
    df = compute_slope(df,
                       target_col=f"{target_column}_smoothed",
                       slope_col=out_column,
                       window_size=slope_window)

    return df[out_column]


def compute_slope(df: pd.DataFrame, target_col: str, slope_col: str, window_size: int) -> pd.DataFrame:
    """
    Computes the slope of a time series for the specified target column and adds it as a new column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        target_col (str): The name of the column containing the target price data.
        slope_col (str): The name of the column to store the computed slopes.
        window_size (int): The rolling window size to compute the slopes.

    Returns:
        pd.DataFrame: The DataFrame with the computed slopes added as a new column.
    """

    # Ensure the target column exists in the DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' does not exist in the DataFrame.")

    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Compute the rolling slope and add it as a new column
    df.loc[:, slope_col] = df[target_col].rolling(window=window_size).apply(compute_slope_internal, raw=True)

    return df


def compute_slope_internal(y_values):
    # Remove non-finite values (NaN, inf) from y_values
    y_values = y_values[np.isfinite(y_values)]

    # Check if y_values is empty, all zeros, or contains fewer than 2 points
    if len(y_values) < 2 or np.all(y_values == 0):
        return 0

    # Recreate x_values after filtering y_values
    x_values = np.arange(len(y_values))

    # Calculate the slope
    m, _ = np.polyfit(x_values, y_values, 1)
    return m


def calculate_trend_numpy(values):
    if len(values) == 0:
        return 0, 0

    x = np.arange(1, len(values) + 1, 1)
    y = np.array(values)

    #  Handle nan values
    x_new = x[~np.isnan(y)]
    y_new = y[~np.isnan(y)]

    m, c = np.polyfit(x_new, y_new, 1)
    return m, c


def add_kernel_reg_smoothed_line(df, column_list=['close'], output_cols=None, bandwidth=2, var_type='c'):
    """
    Adds smoothed lines to the dataframe using kernel regression for multiple columns.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data.
    column_list (list of str): A list of column names containing the data to be smoothed. Default is ['Close'].
    output_cols (list of str, optional): A list of output column names where the smoothed data will be stored.
                                         If None, the output column names will be the input column names with '_Smoothed' appended.
    bandwidth (float or list of floats): The bandwidth parameter for kernel regression. Default is 10.
    var_type (str): A string of length equal to the number of variables in exog, containing a code for each variable.
                    Default is 'c' for continuous variables.

    Returns:
    pd.DataFrame: The dataframe with additional columns containing the smoothed values.
    """

    if output_cols is None:
        output_cols = [f"{col}_smoothed" for col in column_list]
    elif len(column_list) != len(output_cols):
        raise "ERROR: Number of input columns have to equal the number of output columns"

    if not isinstance(bandwidth, list):
        bandwidth = [bandwidth] * len(column_list)

    for input_col, output_col, bw in zip(column_list, output_cols, bandwidth):
        data_list = df[input_col].values
        index_list = np.arange(0, len(data_list))

        kernel_regression = KernelReg(endog=np.array(data_list), exog=index_list, var_type=var_type, bw=[bw])
        smoothed_values, _ = kernel_regression.fit(index_list)

        df[output_col] = smoothed_values

    return df

def compute_slope(df: pd.DataFrame, target_col: str, slope_col: str, window_size: int) -> pd.DataFrame:
    """
    Computes the slope of a time series for the specified target column and adds it as a new column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        target_col (str): The name of the column containing the target price data.
        slope_col (str): The name of the column to store the computed slopes.
        window_size (int): The rolling window size to compute the slopes.

    Returns:
        pd.DataFrame: The DataFrame with the computed slopes added as a new column.
    """

    def compute_slope_internal(y_values):
        x_values = np.arange(len(y_values))
        m, _ = np.polyfit(x_values, y_values, 1)
        return m

    # Ensure the target column exists in the DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' does not exist in the DataFrame.")

    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Compute the rolling slope and add it as a new column
    df.loc[:, slope_col] = df[target_col].rolling(window=window_size).apply(compute_slope_internal, raw=True)

    return df
