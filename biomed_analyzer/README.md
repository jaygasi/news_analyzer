# Biomedical News Analyzer

This application scans financial news for biomedical companies, analyzes relevant articles (especially those related to clinical trials, research findings, FDA announcements, and FDA calendar events) using an LLM (e.g., Google Gemini), and provides potential stock movement predictions with a confidence score.

## Features

### News Sources
-   **Dual API Support**: Fetches news from both Finnhub and FMP (Financial Modeling Prep) APIs for comprehensive coverage:
    -   **Finnhub**: General market news, company-specific news, and FDA calendar events
    -   **FMP**: General market news and company-specific news with different perspectives
    -   **Flexible Configuration**: Can disable either API if needed (use `--disable-finnhub` or `--disable-fmp`)

### Company Identification
-   Dynamically identifies biomedical companies by:
    -   Caching biomedical symbols from both APIs based on GICS industry classifications
    -   Performing on-demand profile checks for symbols not in cache
    -   Combining symbol sets from both sources for maximum coverage

### News Processing
-   Identifies news relevant to research findings, clinical trials, etc., using configurable keywords
-   Utilizes a Large Language Model (LLM) for sentiment analysis and prediction:
    -   Currently supports Google Gemini
    -   Designed for easy extension to other LLMs
-   Deduplicates news across sources to avoid duplicate analysis
-   Outputs predictions in a structured JSON format

### Processing Modes
-   **all** (default): Process general market news, FDA calendar events, and any specified tickers
-   **general**: Process only general market news from both APIs
-   **fda**: Process only FDA calendar-triggered news (Finnhub only)
-   **tickers**: Process only user-specified tickers

## API Requirements

### Required API Keys
You'll need API keys from the following services:

1. **Finnhub** (https://finnhub.io/)
   - Free tier available
   - Provides general market news, company news, and FDA calendar
   - Required for FDA calendar functionality

2. **FMP - Financial Modeling Prep** (https://financialmodelingprep.com/)
   - Free tier available (250 requests/day)
   - Provides additional news coverage and company profiles
   - Can be disabled if not available

3. **Google Gemini** (https://ai.google.dev/)
   - Required for LLM analysis
   - Free tier available with rate limits

### API Usage Notes
- **Minimum Requirement**: At least one of Finnhub or FMP must be enabled
- **FDA Calendar**: Only available through Finnhub API
- **Rate Limiting**: Both APIs have rate limits; the application includes appropriate delays
- **Cost Optimization**: You can disable either API to reduce API usage

## Installation

1. Run the setup script:
   ```bash
   setup.bat
   ```

2. Update the `.env` file with your API keys:
   ```
   FINNHUB_API_KEY=your_finnhub_key_here
   FMP_API_KEY=your_fmp_key_here
   GEMINI_API_KEY=your_gemini_key_here
   ```

3. Activate the virtual environment:
   ```bash
   venv\Scripts\activate.bat
   ```

## Usage

### Basic Usage
```bash
# Process all news sources (default)
python main.py

# Process only general market news
python main.py --mode general

# Process specific tickers
python main.py --tickers "PFE,MRNA,GILD"

# Use only Finnhub (disable FMP)
python main.py --disable-fmp

# Use only FMP (disable Finnhub, note: no FDA calendar)
python main.py --disable-finnhub
```

### Command Line Options
```bash
python main.py --help
```

Options include:
- `--mode`: Processing mode (all, general, fda, tickers)
- `--tickers`: Comma-separated list of specific tickers
- `--output`: Output file path (default: predictions.json)
- `--llm`: LLM service to use (default: gemini)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--disable-finnhub`: Disable Finnhub API
- `--disable-fmp`: Disable FMP API

## Configuration

Key configuration options in `analyzer/config.py`:
- `RESEARCH_NEWS_KEYWORDS`: Keywords to identify relevant biomedical news
- `MIN_CONFIDENCE_THRESHOLD`: Minimum LLM confidence score to include predictions
- `NEWS_LOOKBACK_DAYS`: How many days back to search for general news
- `FMP_NEWS_COUNT`: Maximum news items to fetch from FMP
- `FINNHUB_NEWS_COUNT`: Maximum news items to fetch from Finnhub

## Output

The application generates a JSON file with predictions containing:
- Ticker symbol
- Predicted movement (UP, DOWN, NEUTRAL)
- Confidence score (1-10)
- Reasoning from the LLM
- News headline and source information

## Project Structure

```
biomed_analyzer/
├── analyzer/
│   ├── __init__.py
│   ├── config.py              # Configuration and API keys
│   ├── finnhub_client.py      # Finnhub API client
│   ├── fmp_client.py          # FMP API client
│   ├── news_processor.py      # Main news processing logic
│   ├── models.py              # Data models
│   ├── utils.py               # Utility functions
│   └── llm_services/          # LLM service implementations
│       ├── __init__.py
│       ├── base_llm.py
│       └── gemini_llm.py
├── main.py                    # Entry point
├── requirements.txt
├── setup.bat
└── README.md
```

## Troubleshooting

### Common Issues
1. **Missing API Keys**: Ensure all required API keys are set in the `.env` file
2. **Rate Limiting**: If you hit rate limits, the application will log warnings and continue
3. **No Predictions**: Check that your tickers are biomedical companies or adjust the confidence threshold
4. **API Disabled**: Some features (like FDA calendar) require specific APIs to be enabled

### Logs
Check the console output for detailed logging. Increase log level to `DEBUG` for more verbose output:
```bash
python main.py --log-level DEBUG
```

## Contributing

The application is designed for easy extension:
- Add new LLM services by implementing the `BaseLLM` interface
- Add new news sources by creating new client classes
- Modify news filtering keywords in the configuration
- Extend the prediction models as needed