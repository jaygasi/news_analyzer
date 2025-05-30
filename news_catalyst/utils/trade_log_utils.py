import pandas as pd
from utils.log_utils import *


def calculate_bid_ask_spread(ask_price: float, bid_price: float):
    # Ensure prices are valid
    if pd.isna(ask_price) or pd.isna(bid_price):
        logw("Invalid ask or bid price found during bid-ask spread calculation.")
        return None

    # Calculate the midpoint price
    midpoint_price = (ask_price + bid_price) / 2

    # Ensure the midpoint price is valid
    if midpoint_price <= 0:
        logw("Midpoint price is invalid or zero during bid-ask spread calculation.")
        return None

    # Calculate the bid-ask spread as a percentage of the midpoint price
    bid_ask_spread = ask_price - bid_price
    bid_ask_spread_percentage = (bid_ask_spread / midpoint_price)

    return bid_ask_spread_percentage
