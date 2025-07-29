import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime
from newspaper import Article, Config

def parse_finviz_date(date_str):
    """
    Intelligently parses the various date/time formats from Finviz.

    Args:
        date_str (str): The date string from Finviz (e.g., "10:30AM", "Jul-23-25 10:30AM", "Jul-23").

    Returns:
        datetime: A datetime object representing the parsed date and time.
    """
    try:
        # Case 1: Full date and time (e.g., "Jul-23-25 10:30AM")
        if ' ' in date_str:
            return datetime.strptime(date_str, '%b-%d-%y %I:%M%p')
        
        # Case 2: Time only, which means today (e.g., "10:30AM")
        else:
            try:
                article_time = datetime.strptime(date_str, '%I:%M%p').time()
                return datetime.combine(datetime.now().date(), article_time)
            # Case 3: Date only, which means current year (e.g., "Jul-23")
            except ValueError:
                current_year = datetime.now().year
                article_date = datetime.strptime(f"{date_str}-{current_year}", '%b-%d-%Y')
                return article_date.replace(hour=23, minute=59, second=59)

    except (ValueError, TypeError):
        return datetime.now()

def get_article_summary(url):
    """
    Fetches an article from a URL and returns its summary.

    Args:
        url (str): The URL of the news article.

    Returns:
        str: A summary of the article, or an empty string if it fails.
    """
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
        config.request_timeout = 10

        article = Article(url, config=config)
        article.download()
        article.parse()
        return article.text[:400].replace('\n', ' ') + '...'
    except Exception as e:
        print(f"  - Could not fetch summary for {url}. Using title as fallback. Error: {e}")
        return ""

def fetch_and_process_news():
    """
    Fetches all current news articles directly from the Finviz HTML,
    processes them, and returns a list of dictionaries.
    """
    try:
        # Use a session and more comprehensive headers to appear like a real browser
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        url = 'https://finviz.com/news.ashx'
        
        response = session.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the two main news tables within the 'news' div
        news_div = soup.find('div', id='news')
        if not news_div:
            print("Could not find the main news container div on the page.")
            with open('finviz_error_page.html', 'w', encoding='utf-8') as f:
                f.write(str(soup))
            print("The HTML content received has been saved to 'finviz_error_page.html'.")
            return []

        # The page has two tables, one for news and one for blogs. We need to find all article rows.
        all_rows = news_div.find_all('tr', class_='news_table-row')
        
        news_list = []
        print(f"\nFound {len(all_rows)} articles. Processing and fetching summaries...")
        
        for row in all_rows:
            # The structure for news with tickers is different from blogs.
            # Stock news has 4 cells, blogs have 3.
            cells = row.find_all('td')
            
            if len(cells) == 4: # This is a stock news row
                date_str = cells[1].text.strip()
                title_cell = cells[2]
                tickers = cells[3].text.strip()
            elif len(cells) == 3: # This is a blog row
                date_str = cells[1].text.strip()
                title_cell = cells[2]
                tickers = 'N/A' # Blogs don't have tickers
            else:
                continue # Skip malformed rows or ads

            # *** FIX: Check if the title cell actually contains a link before proceeding ***
            if not title_cell.a:
                continue

            title = title_cell.a.text.strip()
            link = title_cell.a['href']
            
            print(f"  - Processing: {title[:50]}... (Tickers: {tickers})")

            parsed_date = parse_finviz_date(date_str)
            source = urlparse(link).netloc
            summary = get_article_summary(link)

            article = {
                'date': parsed_date.strftime('%Y-%m-%d %H:%M:%S'),
                'title': title,
                'summary': summary if summary else title,
                'link': link,
                'ticker': tickers,
                'source': source
            }
            news_list.append(article)

        return news_list

    except Exception as e:
        print(f"An error occurred while fetching news: {e}")
        return []

def main():
    """
    Main function to run the news scraper and save the output to a JSON file.
    """
    print("--- Finviz News Scraper ---")
    print("Fetching news directly from Finviz page...")

    news_data = fetch_and_process_news()

    if news_data:
        output_filename = 'finviz_news.json'
        try:
            with open(output_filename, 'w') as f:
                json.dump(news_data, f, indent=4)
            print(f"\nSuccessfully saved {len(news_data)} news articles to '{output_filename}'")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")
    else:
        print("No news was fetched. The output file was not created.")

if __name__ == "__main__":
    main()
