import pandas as pd
from data_loaders.fmp_data_loader import FmpDataLoader
from datetime import datetime, timedelta
from universe_selection.universe_selector import UniverseSelector
from collections import Counter
import os
from sklearn.feature_extraction.text import CountVectorizer
from utils.log_utils import logw, logi
from news_event_analyzers.news_topic_keyword_lists import combined_topic_title_keywords
from utils.log_utils import *

# Ensure RESULTS_DIR is defined
RESULTS_DIR = "results"  # Replace with your actual results directory


class TopicKeywordAnalyzer:
    def __init__(self, fmp_api_key: str):
        self.data_loader = FmpDataLoader(fmp_api_key)
        self.universe_selector = UniverseSelector(fmp_api_key)

    def fetch_news(self, symbol_list: list) -> pd.DataFrame:
        """
        Fetches news articles for a list of symbols over the last 5 years.
        """
        start_date = datetime.today() - timedelta(days=365 * 5)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")

        combined_news_df = pd.DataFrame()
        for symbol in symbol_list:
            logd(f"Fetching news for {symbol}")
            try:
                news_df = self.data_loader.fetch_stock_news_by_date_range(
                    symbol, start_date_str, end_date, limit=500
                )
                if news_df is not None and not news_df.empty:
                    combined_news_df = pd.concat([combined_news_df, news_df], ignore_index=True)
            except Exception as e:
                logw(f"Error fetching news for {symbol}: {e}")

        return combined_news_df

    def analyze_topic_keywords(self, combined_news_df: pd.DataFrame):
        """
        Analyzes keywords in news articles grouped by topics.
        """
        for topic, keyword_list in combined_topic_title_keywords.items():
            topic_content = ""

            # Gather relevant news content for the topic
            for _, row in combined_news_df.iterrows():
                title = row.get('title', "").lower()
                text = row.get('text', "").lower()

                if any(keyword.lower() in title for keyword in keyword_list):
                    topic_content += f"{title} {text} "

            if not topic_content.strip():
                logi(f"No relevant news articles found for topic: {topic}")
                continue

            # Extract 1-3 word phrases using CountVectorizer
            vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english')
            ngrams = vectorizer.fit_transform([topic_content])
            ngram_counts = Counter(dict(zip(vectorizer.get_feature_names_out(), ngrams.toarray()[0])))

            # Convert to DataFrame for easier analysis
            keyword_df = pd.DataFrame(ngram_counts.items(), columns=['keyword', 'count'])
            keyword_df = keyword_df.sort_values(by='count', ascending=False)

            # Save keyword statistics to CSV
            try:
                os.makedirs(RESULTS_DIR, exist_ok=True)
                file_name = f"{topic.value}_keyword_stats.csv"
                path = os.path.join(RESULTS_DIR, file_name)
                keyword_df.to_csv(path, index=False)
                logi(f"Keyword stats saved for topic {topic} at {path}")
            except Exception as e:
                logw(f"Error saving keyword stats for topic {topic}: {e}")

    def perform_analysis(self):
        """
        Performs the complete analysis: fetches news and analyzes topic keywords.
        """
        # Fetch symbols
        symbol_list = self.universe_selector.get_symbol_list()
        if len(symbol_list) == 0:
            logw("Symbol list is empty.")
            return

        # Fetch news
        combined_news_df = self.fetch_news(symbol_list)
        if combined_news_df.empty:
            logw("No news articles were fetched.")
            return

        # Analyze topic keywords
        self.analyze_topic_keywords(combined_news_df)

