"""
Optimized plotting utilities with improved error handling and performance
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

from utils.log_utils import logd, logw, loge
from utils.indicator_utils import add_kernel_reg_smoothed_line


# Color palette configuration
LIGHT_PALETTE = {
    "bg_color": "#ffffff",
    "plot_bg_color": "#ffffff", 
    "grid_color": "#e6e6e6",
    "text_color": "#2e2e2e",
    "dark_candle": "#4d98c4",
    "light_candle": "#cccccc",
    "volume_color": "#f5f5f5",
    "border_color": "#2e2e2e",
    "color_1": "#5c285b",
    "color_2": "#802c62",
    "color_3": "#a33262",
    "color_4": "#c43d5c",
    "color_5": "#de4f51",
    "color_6": "#f26841",
    "color_7": "#fd862b",
    "color_8": "#ffa600",
    "color_9": "#3366d6"
}


def validate_plot_data(df: pd.DataFrame, 
                      required_columns: list) -> Tuple[bool, str]:
    """
    Validate DataFrame for plotting requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if len(df) < 2:
        return False, "DataFrame must have at least 2 rows for plotting"
    
    return True, ""


def safe_index_access(df: pd.DataFrame, 
                     index: Optional[int], 
                     default_value: Any = None) -> Any:
    """
    Safely access DataFrame by index with bounds checking.
    
    Args:
        df: DataFrame to access
        index: Index to access (None = skip)
        default_value: Default value if access fails
        
    Returns:
        Value at index or default_value
    """
    if index is None or df.empty:
        return default_value
    
    try:
        if 0 <= index < len(df):
            return index
        else:
            logw(f"Index {index} out of bounds for DataFrame with {len(df)} rows")
            return default_value
    except (IndexError, TypeError):
        return default_value


def create_volume_colors(df: pd.DataFrame) -> list:
    """
    Create volume bar colors based on price movement.
    
    Args:
        df: DataFrame with 'open' and 'close' columns
        
    Returns:
        List of colors for volume bars
    """
    if 'open' not in df.columns or 'close' not in df.columns:
        return [LIGHT_PALETTE["volume_color"]] * len(df)
    
    return [
        LIGHT_PALETTE["light_candle"] if close > open else LIGHT_PALETTE["dark_candle"]
        for open, close in zip(df['open'], df['close'])
    ]


def add_price_markers(fig: 'go.Figure', 
                     df: pd.DataFrame,
                     entry_index: Optional[int],
                     exit_index: Optional[int]) -> None:
    """
    Add buy/sell markers to the plot with validation.
    
    Args:
        fig: Plotly figure object
        df: DataFrame with price data
        entry_index: Index for entry marker
        exit_index: Index for exit marker
    """
    if not PLOTLY_AVAILABLE:
        return
    
    # Add buy (entry) marker
    safe_entry = safe_index_access(df, entry_index)
    if safe_entry is not None:
        try:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[safe_entry]], 
                    y=[df['close'].iloc[safe_entry]], 
                    mode='markers+text',
                    marker=dict(color='green', size=12, symbol="triangle-up"),
                    text="BUY", 
                    textposition="top center", 
                    name="Entry",
                    showlegend=True
                ), 
                row=1, col=1
            )
        except (IndexError, KeyError) as e:
            logw(f"Error adding entry marker: {e}")
    
    # Add sell (exit) marker
    safe_exit = safe_index_access(df, exit_index)
    if safe_exit is not None:
        try:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[safe_exit]], 
                    y=[df['close'].iloc[safe_exit]], 
                    mode='markers+text',
                    marker=dict(color='red', size=12, symbol="triangle-down"),
                    text="SELL", 
                    textposition="bottom center", 
                    name="Exit",
                    showlegend=True
                ), 
                row=1, col=1
            )
        except (IndexError, KeyError) as e:
            logw(f"Error adding exit marker: {e}")


def add_smoothed_line(fig: 'go.Figure', df: pd.DataFrame) -> pd.DataFrame:
    """
    Add smoothed price line to the plot.
    
    Args:
        fig: Plotly figure object
        df: DataFrame with price data
        
    Returns:
        DataFrame with smoothed data added
    """
    if not PLOTLY_AVAILABLE:
        return df
    
    try:
        # Add smoothed close line
        df_with_smooth = add_kernel_reg_smoothed_line(
            df.copy(), 
            column_list=['close'], 
            output_cols=['smoothed_close'], 
            bandwidth=3
        )
        
        if 'smoothed_close' in df_with_smooth.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_smooth.index, 
                    y=df_with_smooth['smoothed_close'], 
                    mode='lines', 
                    line=dict(color='purple', width=2), 
                    name='Smoothed Close',
                    opacity=0.8
                ), 
                row=1, col=1
            )
        
        return df_with_smooth
    except Exception as e:
        logw(f"Error adding smoothed line: {e}")
        return df


def plot_chart(symbol: str,
               df: pd.DataFrame,
               entry_index: Optional[int] = None,
               exit_index: Optional[int] = None,
               title: Optional[str] = None,
               plots_dir: Union[str, Path] = "plots",
               file_name: str = "chart.png",
               palette: Optional[Dict[str, str]] = None) -> Optional['go.Figure']:
    """
    Create comprehensive trading chart with candlesticks, volume, and markers.
    
    Args:
        symbol: Stock symbol
        df: DataFrame with OHLCV data
        entry_index: Index for buy marker
        exit_index: Index for sell marker  
        title: Chart title (auto-generated if None)
        plots_dir: Directory to save plot
        file_name: Output file name
        palette: Color palette (uses default if None)
        
    Returns:
        Plotly figure object or None if creation failed
    """
    if not PLOTLY_AVAILABLE:
        loge("Plotly not available for chart creation")
        return None
    
    # Validate input data
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    is_valid, error_msg = validate_plot_data(df, required_columns)
    if not is_valid:
        loge(f"Invalid plot data for {symbol}: {error_msg}")
        return None
    
    logd(f"Creating chart for {symbol}")
    
    # Use provided palette or default
    if palette is None:
        palette = LIGHT_PALETTE
    
    # Generate title if not provided
    if title is None:
        title = f"{symbol} - Trading Chart"
    
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            row_heights=[0.8, 0.2], 
            vertical_spacing=0.05,
            subplot_titles=["Price Action", "Volume"]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing_line_color=palette['light_candle'],
                decreasing_line_color=palette['dark_candle'],
                name='Price'
            ), 
            row=1, col=1
        )
        
        # Add smoothed line
        df_with_smooth = add_smoothed_line(fig, df)
        
        # Add volume bars with color coding
        volume_colors = create_volume_colors(df)
        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df['volume'], 
                marker_color=volume_colors, 
                name='Volume',
                opacity=0.7
            ), 
            row=2, col=1
        )
        
        # Add buy/sell markers
        add_price_markers(fig, df, entry_index, exit_index)
        
        # Update layout
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'font': {'size': 16}},
            font=dict(family="Arial", size=12, color=palette["text_color"]),
            autosize=True, 
            width=1280, 
            height=720,
            plot_bgcolor=palette["plot_bg_color"],
            paper_bgcolor=palette["bg_color"],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right", 
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor=palette["grid_color"], 
            gridcolor=palette["grid_color"], 
            rangeslider_visible=False
        )
        
        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor=palette["grid_color"], 
            gridcolor=palette["grid_color"]
        )
        
        # Update volume y-axis
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Save plot
        plots_path = Path(plots_dir)
        plots_path.mkdir(parents=True, exist_ok=True)
        
        output_path = plots_path / file_name
        try:
            fig.write_image(str(output_path), format="png", engine="kaleido")
            logd(f"Chart saved to {output_path}")
        except Exception as e:
            logw(f"Failed to save chart: {e}")
        
        return fig
        
    except Exception as e:
        loge(f"Error creating chart for {symbol}: {e}")
        return None


def create_performance_chart(performance_data: Dict[str, Any],
                           output_path: Union[str, Path] = "performance_chart.png") -> Optional['go.Figure']:
    """
    Create performance summary chart.
    
    Args:
        performance_data: Dictionary with performance metrics
        output_path: Path to save chart
        
    Returns:
        Plotly figure object or None if creation failed
    """
    if not PLOTLY_AVAILABLE:
        loge("Plotly not available for performance chart creation")
        return None
    
    try:
        fig = go.Figure()
        
        # Add metrics as bar chart
        metrics = list(performance_data.keys())
        values = list(performance_data.values())
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color=LIGHT_PALETTE["color_1"],
            name="Performance Metrics"
        ))
        
        fig.update_layout(
            title="System Performance Summary",
            xaxis_title="Metrics",
            yaxis_title="Values",
            showlegend=False
        )
        
        # Save chart
        fig.write_image(str(output_path), format="png")
        logd(f"Performance chart saved to {output_path}")
        
        return fig
        
    except Exception as e:
        loge(f"Error creating performance chart: {e}")
        return None


def validate_plotly_installation() -> bool:
    """
    Validate that Plotly and required dependencies are available.
    
    Returns:
        True if Plotly is properly installed
    """
    if not PLOTLY_AVAILABLE:
        return False
    
    try:
        # Test basic functionality
        test_fig = go.Figure()
        test_fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))
        return True
    except Exception:
        return False