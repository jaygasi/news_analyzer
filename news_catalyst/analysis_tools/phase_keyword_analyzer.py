import pandas as pd
from data_loaders.fmp_data_loader import FmpDataLoader
from datetime import datetime, timedelta
from universe_selection.universe_selector import UniverseSelector
from collections import Counter
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from utils.log_utils import *


class PhaseKeywordAnalyzer:
    def __init__(self, fmp_api_key: str):
        self.data_loader = FmpDataLoader(fmp_api_key)
        self.universe_selector = UniverseSelector(fmp_api_key)

    def fetch_news(self, symbol_list: list):
        # Set start- and end dates
        start_date = datetime.today() - timedelta(days=365 * 5)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date = datetime.today()
        end_date_str = end_date.strftime("%Y-%m-%d")

        combined_news_df = pd.DataFrame()
        for symbol in symbol_list:
            news_df = self.data_loader.fetch_stock_news_by_date_range(symbol,
                                                                      start_date_str,
                                                                      end_date_str,
                                                                      limit=500)
            combined_news_df = pd.concat([combined_news_df, news_df], ignore_index=True, axis=0)
        return combined_news_df

    def extract_keywords(self, topic: str, news_df: pd.DataFrame):
        # Filter news articles based on the topic
        topic_news_df = news_df[news_df['title'].str.contains(topic, case=False, na=False)]
        if topic_news_df.empty:
            print(f"No news articles found for topic '{topic}'")
            return pd.DataFrame()

        # Store news for review
        file_name = f"{topic.replace(' ', '_')}_news_df.csv"
        path = os.path.join(RESULTS_DIR, file_name)
        topic_news_df.to_csv(path)

        # Combine all titles for keyword extraction
        combined_titles = " ".join(topic_news_df['title'])

        # Preprocess the text
        combined_titles = re.sub(r'[^\w\s]', '', combined_titles)  # Remove punctuation
        combined_titles = combined_titles.lower()  # Convert to lowercase

        # Extract 1-3 word phrases using CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english')
        ngrams = vectorizer.fit_transform([combined_titles])
        ngram_counts = Counter(dict(zip(vectorizer.get_feature_names_out(), ngrams.toarray()[0])))

        # Convert to a DataFrame for easier analysis
        keyword_df = pd.DataFrame(ngram_counts.items(), columns=['keyword', 'count'])
        keyword_df = keyword_df.sort_values(by='count', ascending=False)

        return keyword_df

    def perform_analysis(self):
        # Fetch symbols
        symbol_list = self.universe_selector.get_symbol_list()

        # Fetch news
        combined_news_df = self.fetch_news(symbol_list)
        if combined_news_df is None or len(combined_news_df) == 0:
            logw(f"No news returned from FMP")
            return

        # Extract keyword phrases for topics
        topic_list = ["phase 3", "fda approved", "fda rejected", "fda approval", "fda rejection"]
        for topic in topic_list:
            logd(f"Analyzing topic: {topic}")
            keyword_list_df = self.extract_keywords(topic, combined_news_df)

            if not keyword_list_df.empty:
                # Optional: Get the top x keywords for each topic
                # keyword_list_df = keyword_list_df.head(50)

                # Save the output
                output_path = os.path.join(RESULTS_DIR, f"{topic.replace(' ', '_')}_keywords.csv")
                keyword_list_df.to_csv(output_path, index=False)
                logd(f"Top keywords for '{topic}' saved to {output_path}")

