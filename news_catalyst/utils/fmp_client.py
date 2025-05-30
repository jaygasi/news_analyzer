from utils.log_utils import *
import requests
from utils.string_utils import *
import pandas as pd


class FmpClient:
    """
    Configure FMP client with api key
    """
    def __init__(self, fmp_api_key):
        self._api_key = fmp_api_key

    def fetch_stock_screener_results(self, exchange_list="nyse,nasdaq,amex&limit", market_cap_more_than=2000000000, priceMoreThan=10, volume_more_than=100000, beta_lower_than=1, country='US', limit=1000):
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock-screener?exchange={exchange_list}&limit={limit}&marketCapMoreThan={market_cap_more_than}&betaLowerThan={beta_lower_than}&volumeMoreThan={volume_more_than}&country={country}&priceMoreThan={priceMoreThan}&isActivelyTrading=true&isFund=false&isEtf=false&apikey={self._api_key}"
            logd(url)
            response = requests.get(url)

            if response.status_code == 200:
                securities_data = response.json()
                if securities_data:
                    securities_df = pd.DataFrame(securities_data)

                    return securities_df
                return None
            else:
                return None
        except Exception as ex:
            print(ex)
            return None

    def fetch_daily_prices(self, symbol, start_date_str, end_date_str):
        try:
            file_name = f"{symbol}-{start_date_str}-{end_date_str}-prices.csv"
            path = os.path.join(CACHE_DIR, file_name)
            if os.path.exists(path):
                prices_df = pd.read_csv(path)
                prices_df['date'] = pd.to_datetime(prices_df['date'])
                prices_df.set_index('date', inplace=True)
                return prices_df
            else:
                # Load remotely
                url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date_str}&to={end_date_str}&apikey={self._api_key}&serietype=line"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    historical_data = data.get('historical', [])
                    if historical_data:
                        prices_df = pd.DataFrame(historical_data)
                        prices_df['date'] = pd.to_datetime(prices_df['date'])
                        prices_df.set_index('date', inplace=True)
                        prices_df.sort_index(ascending=True, inplace=True)

                        # Cache price file
                        prices_df.to_csv(path)

                        return prices_df
                    else:
                        return None
                else:
                    return None
        except Exception as ex:
            print(ex)
            return None

    def fetch_tradable_list(self):
        try:
            url = f"https://financialmodelingprep.com/api/v3/available-traded/list?apikey={self._api_key}"
            response = requests.get(url)

            if response.status_code == 200:
                securities_data = response.json()
                if securities_data:
                    securities_df = pd.DataFrame(securities_data)

                    return securities_df
                return None
            else:
                return None
        except Exception as ex:
            print(ex)
            return None

    def get_analyst_ratings(self, symbol):
        try:
            url = f"https://financialmodelingprep.com/api/v3/grade/{symbol}?apikey={self._api_key}"
            response = requests.get(url)

            if response.status_code == 200:
                grades_data = response.json()
                if grades_data:
                    grades_df = pd.DataFrame(grades_data)
                    grades_df['date'] = pd.to_datetime(grades_df['date'], errors='coerce')
                    # Filter out invalid dates (NaT values after conversion)
                    grades_df = grades_df.dropna(subset=['date'])

                    return grades_df
                return None
            else:
                return None
        except Exception as ex:
            loge(ex)
            return None

    def get_income_growth(self, symbol, period='annual'):
        try:
            url = f"https://financialmodelingprep.com/api/v3/income-statement-growth/{symbol}?period={period}&apikey={self._api_key}"
            response = requests.get(url)

            if response.status_code == 200:
                growth_data = response.json()
                if growth_data:
                    growth_df = pd.DataFrame(growth_data)

                    return growth_df
                return None
            else:
                return None
        except Exception as ex:
            print(ex)
            return None

    def get_financial_ratios(self, symbol, period):
        try:
            url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?period={period}&apikey={self._api_key}"
            response = requests.get(url)

            if response.status_code == 200:
                ratios_data = response.json()
                if ratios_data:
                    ratios_df = pd.DataFrame(ratios_data)

                    return ratios_df
                return None
            else:
                return None
        except Exception as ex:
            print(ex)
            return None

    def get_social_sentiment(self, symbol):
        try:
            url = f"https://financialmodelingprep.com/api/v4/historical/social-sentiment?symbol={symbol}&apikey={self._api_key}"
            response = requests.get(url)

            if response.status_code == 200:
                social_sentiment_data = response.json()
                if social_sentiment_data:
                    social_sentiment_df = pd.DataFrame(social_sentiment_data)
                    social_sentiment_df['date'] = pd.to_datetime(social_sentiment_df['date'], errors='coerce')
                    # Filter out invalid dates (NaT values after conversion)
                    social_sentiment_df = social_sentiment_df.dropna(subset=['date'])

                    return social_sentiment_df
                return None
            else:
                return None
        except Exception as ex:
            loge(ex)
            return None

    def get_stock_news(self, symbol, limit):
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={symbol}&limit={limit}&apikey={self._api_key}"
            response = requests.get(url)

            if response.status_code == 200:
                news_data = response.json()
                if news_data:
                    news_df = pd.DataFrame(news_data)
                    news_df['publishedDate'] = pd.to_datetime(news_df['publishedDate'], errors='coerce')
                    # Filter out invalid dates (NaT values after conversion)
                    news_df = news_df.dropna(subset=['publishedDate'])

                    return news_df
                return None
            else:
                return None
        except Exception as ex:
            loge(ex)
            return None

    def fetch_daily_prices(self, symbol, start_date_str, end_date_str):
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date_str}&to={end_date_str}&apikey={self._api_key}&serietype=line"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                historical_data = data.get('historical', [])
                if historical_data:
                    prices_df = pd.DataFrame(historical_data)
                    prices_df['date'] = pd.to_datetime(prices_df['date'])
                    prices_df.set_index('date', inplace=True)
                    prices_df.sort_index(ascending=True, inplace=True)
                    return prices_df
                else:
                    return None
            else:
                return None
        except Exception as ex:
            print(ex)
            return None

    def fetch_all_prices(self):
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock/full/real-time-price?apikey={self._api_key}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data:
                    all_prices_df = pd.DataFrame(data)
                    return all_prices_df
                else:
                    return None
            else:
                return None
        except Exception as ex:
            print(ex)
            return None

    def fetch_dividends(self, symbol):
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{symbol}?apikey={self._api_key}"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                historical_data = data.get('historical', [])
                if historical_data:
                    dividends_df = pd.DataFrame(historical_data)
                    dividends_df['paymentDate'] = pd.to_datetime(dividends_df['paymentDate'])
                    dividends_df['declarationDate'] = pd.to_datetime(dividends_df['declarationDate'])
                    dividends_df.set_index('paymentDate', inplace=True)
                    return dividends_df
                else:
                    return None
            else:
                return None
        except Exception as ex:
            print(ex)
            return None

  