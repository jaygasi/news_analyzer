import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from enum import Enum
import hashlib
from utils.string_utils import *


def create_md5_hash(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()



class TiingoIntradayInterval(Enum):
    """
    Enum for different Tiingo intraday resample intervals.
    """
    MIN_1 = '1min'
    MIN_5 = '5min'
    MIN_15 = '15min'
    MIN_30 = '30min'
    HOUR_1 = '1hour'
    HOUR_4 = '4hour'
    DAY_1 = '1day'

class TiingoDailyInterval(Enum):
    """
    Enum for different daily Tiingo resample intervals.
    """
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTYLY = 'monthly'
    ANNUALLY = 'annually'


class TiingoDataLoader:
    """
    TiingoDataLoader provides methods to interact with Tiingo APIs to fetch data such as stock prices, crypto prices, news, and forex data.

    Attributes:
        api_key (str): Tiingo API key.
    """

    def __init__(self, api_key: str):
        """
        Initializes the TiingoDataLoader with the given API key.

        Parameters:
            api_key (str): Tiingo API key.
        """
        self.api_key = api_key

    def fetch_intraday_prices(self, symbol: str, start_date_str: str, end_date_str: str, interval: TiingoIntradayInterval, cache_data=False, cache_dir="cache") -> pd.DataFrame:
        """
        Fetches stock prices from Tiingo API.

        Parameters:
            symbol (str): Stock symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            interval (TiingoIntradayInterval): Data interval.
            cache_data (bool): Whether to cache the data.
            cache_dir (str): Directory to save the cached data.

        Returns:
            pd.DataFrame: DataFrame with stock prices.
        """
        try:
            file_name = f"{symbol}_{interval.value}_{start_date_str}_{end_date_str}.csv"
            path = os.path.join(cache_dir, file_name)

            if cache_data and os.path.exists(path):
                prices_df = pd.read_csv(path, parse_dates=['date'])
                prices_df.set_index('date', inplace=True)
                prices_df.index.name = 'date'
                return prices_df
            else:
                fetch_url = f"https://api.tiingo.com/iex/{symbol}/prices?startDate={start_date_str}&endDate={end_date_str}&resampleFreq={interval.value}&columns=date,open,high,low,close,volume&token={self.api_key}"
                headers = {'Accept': 'application/json'}

                response = requests.get(fetch_url, headers=headers)
                response.raise_for_status()  # Raise an error for bad status codes

                data = response.json()

                if not data:
                    print("No data returned from Tiingo API.")
                    return None

                prices_df = pd.DataFrame()
                for row in data:
                    row_df = pd.DataFrame({
                        'date': [row['date']],
                        'open': [row['open']],
                        'high': [row['high']],
                        'low': [row['low']],
                        'close': [row['close']],
                        'volume': [row['volume']]
                    })
                    prices_df = pd.concat([prices_df, row_df], axis=0, ignore_index=True)

                prices_df['date'] = pd.to_datetime(prices_df['date'])

                if cache_data:
                    os.makedirs(cache_dir, exist_ok=True)
                    prices_df.to_csv(path, index=False)

                prices_df.set_index('date', inplace=True)
                prices_df.index.name = 'date'

                return prices_df
        except Exception as ex:
            print(f"Failed to fetch Tiingo prices: {ex}")
            return None

    def fetch_multiple_intraday_prices(self, symbol_list: List[str], start_date_str: str, end_date_str: str, interval: TiingoIntradayInterval, cache_data=False, cache_dir="cache") -> pd.DataFrame:
        prices_dict = {}
        for symbol in symbol_list:
            print(f"Fetching prices for {symbol}")
            # fetch prices
            prices_df = self.fetch_intraday_prices(symbol, start_date_str, end_date_str,
                                                                 interval, cache_data=cache_data,
                                                                 cache_dir=cache_dir)
            prices_dict[symbol] = prices_df
        return prices_dict

    def fetch_end_of_day_prices(self, symbol: str, start_date: str, end_date: str, interval: TiingoDailyInterval, cache_data = False, cache_dir: str = "cache") -> pd.DataFrame:
        """
        Fetches daily stock prices from Tiingo API.

        Parameters:
            symbol (str): Stock symbol.
            interval (TiingoDailyInterval): Data interval.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            data_dir (str): Directory to save the data.

        Returns:
            pd.DataFrame: DataFrame with stock prices.
        """
        try:
            file_name = f"{symbol}_{interval.value}_{start_date}_{end_date}.csv"
            path = os.path.join(cache_dir, file_name)

            if cache_data is True and os.path.exists(path):
                prices_df = pd.read_csv(path, parse_dates=['date'])
                if 'date' in prices_df.columns:
                    prices_df.set_index('date', inplace=True)
                    prices_df.index.name = 'date'
                return prices_df
            else:
                fetch_url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}&endDate={end_date}&resampleFreq={interval.value}&columns=date,open,high,low,close,volume&token={self.api_key}"
                headers = {'Accept': 'application/json'}

                response = requests.get(fetch_url, headers=headers)
                data = response.json()

                prices_df = pd.DataFrame(data)
                prices_df.rename(columns={"adjClose": "adj_close"}, inplace=True)
                """
                for row in data:
                    row_df = pd.DataFrame({
                        'date': [row['date']],
                        'open': [row['open']],
                        'high': [row['high']],
                        'low': [row['low']],
                        'close': [row['close']],
                        'volume': [row['volume']],
                        'adj_close': [row.get('adjClose')]
                    })
                    prices_df = pd.concat([prices_df, row_df], axis=0, ignore_index=True)
                """

                if cache_data is True:
                    os.makedirs(cache_dir, exist_ok=True)
                    prices_df.to_csv(path, index=False)

                prices_df.set_index('date', inplace=True)
                prices_df.index.name = 'date'

                return prices_df
        except Exception as ex:
            print(f"Failed to fetch Tiingo prices: {ex}")
            return None

    def fetch_multiple_end_of_day_prices(self, symbol_list: List[str], start_date_str: str, end_date_str: str, interval=TiingoDailyInterval.DAILY, cache_data=False, cache_dir="cache") -> pd.DataFrame:
        """
         Fetches daily prices for multiple symbols.

         Parameters:
         symbol_list (List[str]): List of stock symbols.
         start_date_str (str): Start date in 'YYYY-MM-DD' format.
         end_date_str (str): End date in 'YYYY-MM-DD' format.
         interval (TiingoDailyInterval): The interval, e.g. daily, weekly, monthly
         cache_data (bool): Flag to specify if data should be cached. Default is False.
         cache_dir (str): Directory to cache the data. Default is "cache".

         Returns:
         pd.DataFrame: DataFrame containing the daily prices for multiple symbols.
         """
        prices_dict = {}
        for symbol in symbol_list:
            print(f"Fetching prices for {symbol}")
            # fetch prices
            prices_df = self.fetch_end_of_day_prices(symbol, start_date_str, end_date_str, interval, cache_data=cache_data, cache_dir=cache_dir)
            prices_dict[symbol] = prices_df
        return prices_dict

    def fetch_latest_news_articles(self,
                            news_article_limit=50,
                            cache_data: bool = False,
                            cache_dir='cache'):

        start_date = datetime.today()
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date = datetime.today() + timedelta(days=1)
        end_date_str = end_date.strftime("%Y-%m-%d")

        path = os.path.join(cache_dir, f"news_{start_date_str}_{end_date_str}.csv")
        if cache_data is True:
            if os.path.exists(path):
                news_df = pd.read_csv(path)
                if 'title' in news_df.columns:
                    news_df['title'] = str(news_df['title'])
                if 'description' in news_df.columns:
                    news_df['description'] = str(news_df['description'])
                return news_df

        base_url = 'https://api.tiingo.com/tiingo/news'

        try:
            url = f"{base_url}?startDate={start_date_str}&endDate={end_date_str}&limit={news_article_limit}&token={self.api_key}"
            response = requests.get(url)
            if response.status_code == 200:
                news_json = response.json()

                # Convert to DataFrame
                news_df = pd.DataFrame(news_json)
                if not news_df.empty:
                    news_df['title'] = news_df['title'].apply(clean_string)
                    news_df['description'] = news_df['description'].apply(clean_string)
                    news_df.rename(columns={'tickers': 'symbols', 'id': 'news_id', 'description': 'text'}, inplace=True)
                    news_df['content'] = news_df['title'] + news_df['text']

                    # Pick primary symbol
                    if 'symbols' in news_df.columns:
                        news_df['symbol'] = news_df['symbols'].apply(
                            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)

                    # Drop duplicate articles
                    news_df = news_df.drop_duplicates(subset='title')
                    #news_df['id'] = news_df['id'].astype(str)

                    # Cache news for review
                    if cache_data is True:
                        os.makedirs(cache_dir, exist_ok=True)
                        if len(news_df) > 0:
                            news_df.to_csv(path, index=False)

                # Convert dates to pd dates
                if len(news_df) > 0 and 'publishedDate' in news_df.columns:
                    news_df['publishedDate'] = pd.to_datetime(news_df['publishedDate'], errors='coerce')

                return news_df
            else:
                print(f"Error: Tiingo News API returned error HTTP status code: {response.status_code}")
                return None
        except Exception as ex:
            print(f"Error: Fetch news stories API failed with exception: {str(ex)}")
            return None

    def fetch_news_article_by_symbol(self, symbol: str,
                                     start_date_str: str,
                                     end_date_str: str,
                                     news_article_limit=50,
                                     cache_data: bool = False,
                                     cache_dir='cache'):
        path = os.path.join(cache_dir, f"{symbol}_{start_date_str}_{end_date_str}_news.csv")
        if cache_data is True:
            if os.path.exists(path):
                news_df = pd.read_csv(path)
                if 'title' in news_df.columns:
                    news_df['title'] = str(news_df['title'])
                if 'description' in news_df.columns:
                    news_df['description'] = str(news_df['description'])
                return news_df

        base_url = 'https://api.tiingo.com/tiingo/news'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Token ' + self.api_key
        }

        params = {
            'startDate': start_date_str,
            'endDate': end_date_str,
            'limit': news_article_limit,
            'tickers': symbol
        }

        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                news_json = response.json()

                # Convert to DataFrame
                news_df = pd.DataFrame(news_json)
                if not news_df.empty:
                    news_df['title'] = news_df['title'].apply(clean_string)
                    news_df['description'] = news_df['description'].apply(clean_string)
                    news_df.rename(columns={'tickers': 'symbols'}, inplace=True)
                    news_df['symbols'] = news_df['symbols'].apply(join_items)

                    # Drop duplicate articles
                    news_df = news_df.drop_duplicates(subset='title')
                    news_df['id'] = news_df['id'].astype(str)

                    # Cache news for review
                    if cache_data is True:
                        os.makedirs(cache_dir, exist_ok=True)
                        if len(news_df) > 0:
                            news_df.to_csv(path)

                # Convert dates to pd dates
                if len(news_df) > 0 and 'publishedDate' in news_df.columns:
                    news_df['publishedDate'] = pd.to_datetime(news_df['publishedDate'], errors='coerce')

                return news_df
            else:
                print(f"Error: Tiingo News API returned error HTTP status code: {response.status_code}")
                return None
        except Exception as ex:
            print(f"Error: Fetch news stories API failed with exception: {str(ex)}")
            return None

    def fetch_multiple_news_articles(self, symbol_list: list[str],
                                     start_date_str,
                                     end_date_str,
                                     limit=50,
                                     cache_data: bool=False,
                                     cache_dir: str = 'cache'):
        print(f"Fetching Tiingo news data....")
        i = 1
        news_dict = {}
        for symbol in symbol_list:
            #logd(f"Fetching growth for {symbol}... ({i}/{len(symbol_list)})")

            # Fetch tiingo news
            news_df = self.fetch_news_article_by_symbol(symbol,
                                                        start_date_str,
                                                        end_date_str,
                                                        limit,
                                                        cache_data,
                                                        cache_dir)
            if news_df is None or len(news_df) == 0:
                continue

            news_dict[symbol] = news_df

        return news_dict

    def fetch_news_articles_by_tags(self,
                                    tags: str,
                                    start_date_str: str,
                                    end_date_str: str,
                                    news_article_limit=50,
                                    cache_data: bool = False,
                                    cache_dir='cache'):
        tag_hash = create_md5_hash(tags)
        path = os.path.join(cache_dir, f"news_tags_{tag_hash}_{start_date_str}_{end_date_str}.csv")
        if cache_data is True:
            if os.path.exists(path):
                news_df = pd.read_csv(path)
                if 'title' in news_df.columns:
                    news_df['title'] = str(news_df['title'])
                if 'description' in news_df.columns:
                    news_df['description'] = str(news_df['description'])
                return news_df

        base_url = 'https://api.tiingo.com/tiingo/news'

        try:
            url = f"{base_url}?tags={tags}&startDate={start_date_str}&endDate={end_date_str}&limit={news_article_limit}&token={self.api_key}"
            response = requests.get(url)
            if response.status_code == 200:
                news_json = response.json()

                # Convert to DataFrame
                news_df = pd.DataFrame(news_json)
                if not news_df.empty:
                    news_df['title'] = news_df['title'].apply(clean_string)
                    news_df['description'] = news_df['description'].apply(clean_string)
                    news_df.rename(columns={'tickers': 'symbols'}, inplace=True)
                    news_df['symbols'] = news_df['symbols'].apply(join_items)
                    # Drop duplicate articles
                    news_df = news_df.drop_duplicates(subset='title')
                    news_df['id'] = news_df['id'].astype(str)

                    # Cache news for review
                    if cache_data is True:
                        os.makedirs(cache_dir, exist_ok=True)
                        if len(news_df) > 0:
                            news_df.to_csv(path)

                # Convert dates to pd dates
                if len(news_df) > 0 and 'publishedDate' in news_df.columns:
                    news_df['publishedDate'] = pd.to_datetime(news_df['publishedDate'], errors='coerce')

                return news_df
            else:
                print(f"Error: Tiingo News API returned error HTTP status code: {response.status_code}")
                return None
        except Exception as ex:
            print(f"Error: Fetch news stories API failed with exception: {str(ex)}")
            return None

    def fetch_forex_intraday_prices(self, symbol: str, start_date_str: str, end_date_str: str, interval: TiingoIntradayInterval, cache_data=False, cache_dir="cache") -> pd.DataFrame:
        """
        Fetches FOREX stock prices from Tiingo API.

        Parameters:
            symbol (str): FOREX symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            interval (TiingoIntradayInterval): Data interval.
            cache_data (bool): Whether to cache the data.
            cache_dir (str): Directory to save the cached data.

        Returns:
            pd.DataFrame: DataFrame with stock prices.
        """
        try:
            file_name = f"{symbol}_{interval.value}_{start_date_str}_{end_date_str}.csv"
            path = os.path.join(cache_dir, file_name)

            if cache_data and os.path.exists(path):
                prices_df = pd.read_csv(path, parse_dates=['date'])
                prices_df.set_index('date', inplace=True)
                prices_df.index.name = 'date'
                return prices_df
            else:
                fetch_url = f"https://api.tiingo.com/tiingo/fx/{symbol}/prices?startDate={start_date_str}&resampleFreq={interval.value}&token={self.api_key}"
                headers = {'Accept': 'application/json'}

                response = requests.get(fetch_url, headers=headers)
                response.raise_for_status()  # Raise an error for bad status codes

                data = response.json()

                if not data:
                    print("No data returned from Tiingo API.")
                    return None

                prices_df = pd.DataFrame()
                for row in data:
                    row_df = pd.DataFrame({
                        'date': [row['date']],
                        'open': [row['open']],
                        'high': [row['high']],
                        'low': [row['low']],
                        'close': [row['close']]
                    })
                    prices_df = pd.concat([prices_df, row_df], axis=0, ignore_index=True)

                prices_df['date'] = pd.to_datetime(prices_df['date'])

                if cache_data:
                    os.makedirs(cache_dir, exist_ok=True)
                    prices_df.to_csv(path, index=False)

                prices_df.set_index('date', inplace=True)
                prices_df.index.name = 'date'

                return prices_df
        except Exception as ex:
            print(f"Failed to fetch FOREX Tiingo prices: {ex}")
            return None

    def fetch_multiple_forex_intraday_prices(self, symbol_list: List[str], start_date_str: str, end_date_str: str, interval: TiingoIntradayInterval, cache_data=False, cache_dir="cache") -> pd.DataFrame:
        prices_dict = {}
        for symbol in symbol_list:
            print(f"Fetching FOREX prices for {symbol}")
            # fetch prices
            prices_df = self.fetch_forex_intraday_prices(symbol, start_date_str, end_date_str,
                                                         interval, cache_data=cache_data, cache_dir=cache_dir)
            prices_dict[symbol] = prices_df
        return prices_dict
