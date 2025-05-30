import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.log_utils import *
from utils.indicator_utils import *


light_palette = {}
light_palette["bg_color"] = "#ffffff"
light_palette["plot_bg_color"] = "#ffffff"
light_palette["grid_color"] = "#e6e6e6"
light_palette["text_color"] = "#2e2e2e"
light_palette["dark_candle"] = "#4d98c4"
light_palette["light_candle"] = "#cccccc"
light_palette["volume_color"] = "#f5f5f5"
light_palette["border_color"] = "#2e2e2e"
light_palette["color_1"] = "#5c285b"
light_palette["color_2"] = "#802c62"
light_palette["color_3"] = "#a33262"
light_palette["color_4"] = "#c43d5c"
light_palette["color_5"] = "#de4f51"
light_palette["color_6"] = "#f26841"
light_palette["color_7"] = "#fd862b"
light_palette["color_8"] = "#ffa600"
light_palette["color_9"] = "#3366d6"


def plot_chart(symbol,
               df,
               entry_index,
               exit_index,
               title,
               plots_dir="plots",
               file_name="channels.png"):
    logd(f"Plotting chart for {symbol}")
    palette = light_palette

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.8, 0.2], vertical_spacing=0.05,
        subplot_titles=(["Price", "Volume"])
    )

    # Plot candlesticks
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'],
                                 close=df['close'], increasing_line_color=palette['light_candle'],
                                 decreasing_line_color=palette['dark_candle'], name='Price'), row=1, col=1)

    # Add smoothed close
    df = add_kernel_reg_smoothed_line(df, column_list=['close'], output_cols=['smoothed_close'], bandwidth=3)
    fig.add_trace(go.Scatter(x=df.index, y=df['smoothed_close'], mode='lines', line=dict(color='purple', width=1), name='close smoothed'), row=1, col=1)

    # Add volume with color coding based on whether close > open (increasing)
    volume_colors = [light_palette["light_candle"] if df['close'].iloc[i] > df['open'].iloc[i] else light_palette["dark_candle"] for i in range(len(df))]

    # Plot volume
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=volume_colors, name='Volume'), row=2, col=1)

    # Add buy (entry) marker
    if entry_index is not None and 0 <= entry_index < len(df):
        fig.add_trace(
            go.Scatter(x=[df.index[entry_index]], y=[df['close'].iloc[entry_index]], mode='markers+text',
                       marker=dict(color='green', size=10, symbol="triangle-up"),
                       text="Buy", textposition="top center", name="Entry"), row=1, col=1
        )

    # Add sell (exit) marker
    if exit_index is not None and 0 <= exit_index < len(df):
        fig.add_trace(
            go.Scatter(x=[df.index[exit_index]], y=[df['close'].iloc[exit_index]], mode='markers+text',
                       marker=dict(color='red', size=10, symbol="triangle-down"),
                       text="Sell", textposition="bottom center", name="Exit"), row=1, col=1
        )

    fig.update_layout(title={'text': title, 'x': 0.5},
                      font=dict(family="Verdana", size=12, color=palette["text_color"]),
                      autosize=True, width=1280, height=720,
                      plot_bgcolor=palette["plot_bg_color"],
                      paper_bgcolor=palette["bg_color"])

    fig.update_xaxes(showline=True, linewidth=1, linecolor=palette["grid_color"], gridcolor=palette["grid_color"], rangeslider_visible=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=palette["grid_color"], gridcolor=palette["grid_color"])

    # Update y-axis for the volume chart
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    os.makedirs(plots_dir, exist_ok=True)
    fig.write_image(os.path.join(plots_dir, file_name), format="png")

    return fig
