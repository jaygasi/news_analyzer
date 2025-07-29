import finnhub
import pandas as pd
from datetime import datetime, timedelta, time
import os
from scipy.stats import pearsonr
import config  # Import the new configuration file
import requests  # Needed for FMP requests

# --- Path Configuration ---
# For Google Colab, set this to your Google Drive path. e.g., '/content/drive/My Drive/MarginAnalysis/'
GDRIVE_PATH = ""
MARGIN_DATA_CSV = os.path.join(GDRIVE_PATH, "margin_rates_history.csv")

# Setup Finnhub client using the key from the config file
# This will be used only if 'finnhub' is the selected provider.
try:
    finnhub_client = finnhub.Client(api_key=config.FINNHUB_API_KEY)
except AttributeError:
    print(
        "WARNING: FINNHUB_API_KEY not found in config.py. Finnhub provider will not work."
    )
    finnhub_client = None


def get_price_finnhub(symbol, target_datetime):
    """Fetches price data from Finnhub."""
    if not finnhub_client or config.FINNHUB_API_KEY == "YOUR_API_KEY_HERE":
        print(
            "ERROR: Finnhub is selected, but the API key is missing or invalid in config.py."
        )
        return None

    from_timestamp = int(target_datetime.timestamp())
    to_timestamp = int((target_datetime + timedelta(minutes=30)).timestamp())

    print(f"DEBUG (Finnhub): Fetching 1-minute price data for {symbol}...")

    try:
        res = finnhub_client.stock_candles(symbol, "1", from_timestamp, to_timestamp)
        if res.get("s") != "ok" or not res.get("o"):
            print(f"  -> No data returned from Finnhub for {symbol}.")
            return None

        price_at_target = res["o"][0]
        actual_time_found = datetime.fromtimestamp(res["t"][0])
        print(
            f"  -> Found price at {actual_time_found.strftime('%H:%M:%S')}: ${price_at_target:.2f}"
        )
        return price_at_target

    except Exception as e:
        print(f"  -> A Finnhub API error occurred for {symbol}: {e}")
        return None


def get_price_fmp(symbol, target_datetime):
    """Fetches price data from Financial Modeling Prep (FMP)."""
    try:
        if config.FMP_API_KEY == "YOUR_API_KEY_HERE":
            print("ERROR: FMP is selected, but the API key is missing in config.py.")
            return None
    except AttributeError:
        print("ERROR: FMP is selected, but FMP_API_KEY was not found in config.py.")
        return None

    from_date = target_datetime.strftime("%Y-%m-%d")
    to_date = (target_datetime + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"DEBUG (FMP): Fetching 1-minute price data for {symbol}...")

    try:
        # Corrected URL construction to match API_URLs.md and use query parameters
        base_url = "https://financialmodelingprep.com/stable/historical-chart/1min"
        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "apikey": config.FMP_API_KEY,
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        if not data:
            print(f"  -> No data returned from FMP for {symbol}.")
            return None

        # Find the first data point at or after our target time
        for candle in data:
            candle_time_str = candle.get("date")
            candle_time = datetime.strptime(candle_time_str, "%Y-%m-%d %H:%M:%S")

            if candle_time >= target_datetime:
                price_at_target = candle.get("open")
                print(
                    f"  -> Found price at {candle_time.strftime('%H:%M:%S')}: ${price_at_target:.2f}"
                )
                return price_at_target

        print(
            f"  -> Could not find a price candle after the target time from FMP for {symbol}."
        )
        return None

    except Exception as e:
        print(f"  -> An FMP API error occurred for {symbol}: {e}")
        return None


def get_price_after_timestamp(symbol, event_timestamp, provider):
    """
    Determines the correct target time and calls the appropriate API provider.
    """
    market_open = time(9, 30)
    market_close = time(16, 0)

    event_time = event_timestamp.time()
    event_date = event_timestamp.date()

    if market_open <= event_time <= market_close:
        target_datetime = event_timestamp + timedelta(minutes=5)
        print(
            f"INFO: Event for {symbol} was during market hours. Targeting time around {target_datetime.strftime('%H:%M:%S')}."
        )
    else:
        print(
            f"INFO: Event for {symbol} was outside market hours. Targeting next market open."
        )
        target_date = (
            event_date + pd.tseries.offsets.BDay(1)
            if event_time >= market_close
            else event_date
        )
        target_datetime = datetime.combine(target_date, market_open) + timedelta(
            minutes=5
        )

    # Call the provider specified in the config file
    if provider == "finnhub":
        return get_price_finnhub(symbol, target_datetime)
    elif provider == "fmp":
        return get_price_fmp(symbol, target_datetime)

    return None


def populate_missing_prices(provider):
    """
    Opens the master data file and fills in any blank prices.
    """
    if not os.path.exists(MARGIN_DATA_CSV):
        print(
            f"Data file not found at '{MARGIN_DATA_CSV}'. Please run the data collector first."
        )
        return

    df = pd.read_csv(MARGIN_DATA_CSV, parse_dates=["timestamp"])
    rows_to_populate = df[df["price_5min_after"].isnull()]

    if rows_to_populate.empty:
        print("No new rows need price population.")
        return

    print(
        f"Found {len(rows_to_populate)} rows to populate with prices using '{provider.upper()}'."
    )

    for index, row in rows_to_populate.iterrows():
        price = get_price_after_timestamp(row["SYM"], row["timestamp"], provider)
        df.loc[index, "price_5min_after"] = price

    df.to_csv(MARGIN_DATA_CSV, index=False)
    print("Successfully updated the data file with new prices.")


def analyze_correlation():
    """
    Performs a correlation analysis on the master data file.
    """
    if not os.path.exists(MARGIN_DATA_CSV):
        print("Correlation data file not found.")
        return

    df = pd.read_csv(MARGIN_DATA_CSV)
    df.dropna(subset=["FEERATE", "price_5min_after"], inplace=True)

    print("\n--- Correlation Analysis Report ---")

    if len(df) < 10:
        print(
            "Not enough complete data points for a significant correlation analysis yet."
        )
        return

    corr, p_value = pearsonr(df["FEERATE"], df["price_5min_after"])
    print(f"\nOverall Analysis (all stocks with complete data):")
    print(f"  - Data points analyzed: {len(df)}")
    print(f"  - Correlation (fee rate vs price): {corr:.4f}")
    print(f"  - P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  -> The overall correlation is statistically significant.")
    else:
        print("  -> The overall correlation is not statistically significant.")


if __name__ == "__main__":
    # --- FIX: Check for PRICE_API_PROVIDER at the start and exit if not found ---
    try:
        provider = config.PRICE_API_PROVIDER.lower()
        if provider not in ["finnhub", "fmp"]:
            raise ValueError(
                f"Provider must be 'finnhub' or 'fmp', but got '{provider}'"
            )
    except AttributeError:
        print(
            "FATAL ERROR: The 'PRICE_API_PROVIDER' variable is missing from your config.py file."
        )
        print(
            "Please add 'PRICE_API_PROVIDER = \"finnhub\"' or 'PRICE_API_PROVIDER = \"fmp\"' to config.py."
        )
        exit()  # Exit the script if the provider isn't set
    except ValueError as e:
        print(f"FATAL ERROR in config.py: {e}")
        exit()

    populate_missing_prices(provider)
    analyze_correlation()
