# 🔧 Complete Troubleshooting Guide

## 🚨 Installation Issues

### Problem: "pip install fails with dependency conflicts"

**Solution 1 - Clean Virtual Environment:**
```bash
# Remove existing virtual environment
rm -rf venv/

# Create fresh virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Solution 2 - Install Core Dependencies Only:**
```bash
# Install minimal dependencies first
pip install requests python-dotenv sqlalchemy pandas numpy beautifulsoup4 feedparser schedule

# Then add AI libraries one by one
pip install google-generativeai  # For Gemini
pip install textblob vaderSentiment  # For traditional sentiment
```

### Problem: "ModuleNotFoundError: No module named 'src'"

**Solution:**
```bash
# Make sure you're in the project root directory
cd earnings_analyzer

# Check that src/ directory exists with __init__.py files
ls -la src/
ls -la src/*/

# If missing, create them:
touch src/__init__.py
touch src/analyzers/__init__.py
touch src/collectors/__init__.py
touch src/database/__init__.py
touch src/notifications/__init__.py
```

## 🔑 API Key Issues

### Problem: "No LLM providers available"

**Symptoms:**
- System starts but shows "⚠️ No LLM providers available"
- Sentiment analysis returns neutral results only

**Solution:**
1. **Check your .env file:**
```bash
# Make sure .env exists and has content
cat .env | grep API_KEY

# Should show something like:
# GOOGLE_API_KEY=your_actual_key_here
# ANTHROPIC_API_KEY=your_actual_key_here
```

2. **Test API keys individually:**
```python
# Test Gemini API
import google.generativeai as genai
genai.configure(api_key="your_google_api_key")
model = genai.GenerativeModel('gemini-1.0-pro')
response = model.generate_content("Hello world")
print(response.text)
```

3. **Common API key mistakes:**
- Using example keys (`your_api_key_here`) instead of real ones
- Extra spaces around the key
- Missing quotes in .env file
- Wrong variable names

### Problem: "Gmail authentication failed"

**Symptoms:**
- Daily reports not being sent
- "Gmail connection test failed" error

**Solution:**
1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate App Password:**
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
   - Use this 16-character password, not your regular Gmail password

3. **Test Gmail connection:**
```python
import smtplib
import ssl

sender_email = "your_email@gmail.com"
password = "your_16_char_app_password"

try:
    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls(context=context)
        server.login(sender_email, password)
    print("✅ Gmail connection successful")
except Exception as e:
    print(f"❌ Gmail error: {e}")
```

## 🗄️ Database Issues

### Problem: "Database connection test failed"

**Solution:**
```bash
# Check if database file exists and is writable
ls -la earnings_trader.db

# If using SQLite, delete and recreate
rm earnings_trader.db
python main.py test

# Check permissions
chmod 666 earnings_trader.db  # If needed
```

### Problem: "Column length too long" or "String data truncated"

**Symptoms:**
- Database errors when storing articles
- "Data too long for column" errors

**Solution:**
This should not happen with the fixed models.py, but if it does:
```bash
# Delete database and recreate with fixed schema
rm earnings_trader.db
python -c "from src.database.connection import create_tables; create_tables()"
```

## 🌐 Network and API Issues

### Problem: "RSS feeds not working"

**Symptoms:**
- "No working RSS feeds found"
- Timeout errors during news collection

**Solution:**
1. **Test network connectivity:**
```bash
# Test basic connectivity
curl -I https://finance.yahoo.com/rss/topstories

# Should return HTTP 200 or 301/302
```

2. **Check firewall/proxy settings**
3. **Test individual feeds:**
```python
import requests
import feedparser

url = "https://finance.yahoo.com/rss/topstories"
response = requests.get(url, timeout=10)
print(f"Status: {response.status_code}")

feed = feedparser.parse(response.content)
print(f"Entries: {len(feed.entries)}")
```

### Problem: "API rate limits exceeded"

**Symptoms:**
- "429 Too Many Requests" errors
- Intermittent API failures

**Solution:**
1. **Increase delays in .env:**
```bash
API_CALL_DELAY_SECONDS=2.0
RSS_INTER_FEED_DELAY_SECONDS=1.0
```

2. **Check API quotas:**
- Finnhub: 60 calls/minute (free tier)
- Google Gemini: Varies by usage
- Check your usage in respective dashboards

## 📊 Data Collection Issues

### Problem: "No earnings events found"

**Solution:**
1. **Check if APIs are working:**
```bash
python main.py test-earnings
```

2. **Verify API keys for earnings data:**
- Finnhub (recommended, free tier)
- Financial Modeling Prep
- Polygon.io

3. **Check date range:**
```python
# In config.py, adjust:
OPERATIONAL_EARNINGS_DAYS_AHEAD = 14  # Look further ahead
```

### Problem: "No articles collected"

**Solution:**
1. **Test news collection:**
```bash
python main.py test-news
```

2. **Check RSS feed status:**
- Some feeds may be temporarily down
- Check if your IP is blocked (try different network)

3. **Lower filtering thresholds:**
```python
# In config.py:
MIN_EARNINGS_CONFIDENCE_TO_COLLECT = 0.1  # Lower threshold
MIN_EARNINGS_RELEVANCE_SCORE_THRESHOLD = 1.0  # Lower threshold
```

## 🤖 AI Analysis Issues

### Problem: "LLM analysis keeps failing"

**Symptoms:**
- All sentiment scores are 0.0
- "All LLM providers failed" messages

**Solution:**
1. **Test each provider individually:**
```bash
python main.py test
# Look for specific provider errors in logs
```

2. **Check API quotas and billing:**
- Ensure your API keys have remaining quota
- Check if billing is set up (for paid APIs)

3. **Use fallback configuration:**
```bash
# In .env, enable traditional methods:
USE_TEXTBLOB_ANALYSIS=true
USE_VADER_ANALYSIS=true
```

### Problem: "TextBlob or VADER not working"

**Solution:**
```bash
# Install NLTK data for TextBlob
python -c "import nltk; nltk.download('punkt'); nltk.download('brown'); nltk.download('vader_lexicon')"

# Test individually
python -c "from textblob import TextBlob; print(TextBlob('good news').sentiment)"
python -c "from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; analyzer = SentimentIntensityAnalyzer(); print(analyzer.polarity_scores('good news'))"
```

## 🔄 Runtime Issues

### Problem: "Scheduler stops unexpectedly"

**Solution:**
1. **Check logs for errors:**
```bash
tail -f logs/earnings_trader.log
```

2. **Run in debug mode:**
```bash
LOG_LEVEL=DEBUG python main.py schedule
```

3. **Use process monitoring:**
```bash
# Run with automatic restart
while true; do
    python main.py schedule
    echo "Restarting in 10 seconds..."
    sleep 10
done
```

### Problem: "Memory usage keeps growing"

**Solution:**
1. **Enable database cleanup:**
```python
# Add to your cron or scheduler
from src.database.connection import cleanup_old_data
cleanup_old_data(days_to_keep=30)
```

2. **Limit article processing:**
```bash
# In .env:
MAX_ARTICLES_PER_RUN=20
MAX_ARTICLES_PER_RSS_FEED=5
```

## 🧪 Testing Commands

Use these commands to isolate issues:

```bash
# Test basic system
python main.py test

# Test individual components
python main.py test-earnings
python main.py test-news

# Check system status
python main.py status

# Run single cycle (debug)
LOG_LEVEL=DEBUG python main.py run

# Test specific functionality
python -c "from src.analyzers.sentiment_analyzer import SentimentAnalyzer; sa = SentimentAnalyzer(); print(sa.test_analysis())"
```

## 📝 Logging and Debugging

### Enable Detailed Logging:
```bash
# In .env:
LOG_LEVEL=DEBUG

# Or run with debug:
LOG_LEVEL=DEBUG python main.py run
```

### Check Log Files:
```bash
# Monitor logs in real-time
tail -f logs/earnings_trader.log

# Search for specific errors
grep -i error logs/earnings_trader.log
grep -i "api key" logs/earnings_trader.log
```

### Common Log Messages and Solutions:

| Log Message | Problem | Solution |
|-------------|---------|----------|
| "No LLM providers available" | Missing API keys | Add keys to .env |
| "Database connection failed" | DB issues | Check file permissions, recreate DB |
| "Gmail connection test failed" | Email setup | Use App Password, not regular password |
| "RSS feed timeout" | Network issues | Check connectivity, increase timeouts |
| "Rate limit exceeded" | API limits | Increase delays, check quotas |

## 🆘 Getting Help

If you're still having issues:

1. **Check the logs first**: `tail -f logs/earnings_trader.log`
2. **Run diagnostics**: `python main.py status`
3. **Test components individually**: Use the test commands above
4. **Check your .env configuration**: Ensure all required keys are present
5. **Verify network connectivity**: Test API endpoints manually

Remember: The system is designed to be resilient, so most errors should be recoverable. Check the logs for specific error messages and use the solutions above.