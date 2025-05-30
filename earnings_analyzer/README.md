# 📊 Earnings News Trader

An intelligent trading signal generator that analyzes earnings-related news using AI sentiment analysis and multiple data sources.

## ✨ Features

- 🤖 **Multi-LLM Sentiment Analysis**: Claude, OpenAI GPT, Google Gemini, and Grok support
- 📰 **Comprehensive News Collection**: RSS feeds + Finnhub API integration
- 📅 **Earnings Calendar Integration**: Multiple API sources with fallback support
- 📧 **Smart Notifications**: Daily reports and urgent alerts via Gmail
- 🎯 **Risk Management**: Automated position sizing and stop-loss calculations
- 📊 **Performance Tracking**: Signal accuracy and performance metrics
- 🔄 **Automated Scheduling**: Continuous monitoring during market hours

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd earnings_analyzer

# Run the quick setup script
python quick_setup.py
```

### 2. Configure API Keys

Edit the `.env` file and add your API keys:

```bash
# Minimum required for basic functionality:
GOOGLE_API_KEY=your_google_gemini_key_here
GMAIL_EMAIL=your_email@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password
FINNHUB_API_KEY=your_finnhub_key_here
```

### 3. Test the System

```bash
# Test earnings data collection
python main.py test-earnings

# Test news collection
python main.py test-news

# Validate configuration
python main.py test
```

### 4. Run the System

```bash
# Single collection cycle
python main.py run

# Continuous monitoring (production mode)
python main.py schedule
```

## 📋 Requirements

- Python 3.8+
- Internet connection
- Gmail account (for notifications)
- At least one LLM API key
- At least one earnings data API key

## 🔑 API Keys Setup

### Required APIs

| Service | Purpose | Free Tier | Get API Key |
|---------|---------|-----------|-------------|
| **Google Gemini** | AI sentiment analysis | ✅ Yes | [Get Key](https://makersuite.google.com/app/apikey) |
| **Gmail** | Email notifications | ✅ Yes | [Setup Guide](https://support.google.com/accounts/answer/185833) |
| **Finnhub** | Earnings data & news | ✅ Yes | [Get Key](https://finnhub.io/register) |

### Optional APIs (Enhanced Features)

| Service | Purpose | Free Tier | Get API Key |
|---------|---------|-----------|-------------|
| **Anthropic Claude** | Premium AI analysis | ❌ Paid | [Get Key](https://console.anthropic.com/) |
| **OpenAI GPT** | Premium AI analysis | ❌ Paid | [Get Key](https://platform.openai.com/api-keys) |
| **Financial Modeling Prep** | Enhanced earnings data | ❌ Paid | [Get Key](https://financialmodelingprep.com/developer/docs) |
| **Polygon.io** | Enhanced market data | ❌ Paid | [Get Key](https://polygon.io/) |

## 🛠️ Installation (Manual)

If the quick setup script doesn't work, follow these manual steps:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env

# Edit .env with your API keys
nano .env  # or use your preferred editor

# Initialize database
python main.py test
```

## 📖 Usage Guide

### Commands

```bash
# Production Commands
python main.py schedule         # Start continuous monitoring
python main.py run             # Single collection cycle
python main.py earnings        # Earnings-focused scan
python main.py scan            # Quick urgent signals scan

# Testing Commands
python main.py test            # Validate system setup
python main.py test-earnings   # Test earnings APIs
python main.py test-news       # Test news collection
python main.py status          # Show system status

# Configuration Commands
python main.py switch-llm=gemini  # Switch primary LLM provider
```

### Configuration Options

Key settings in `.env`:

```bash
# Trading Sensitivity
MIN_CONFIDENCE_SCORE=0.7          # Minimum confidence for signals (0.0-1.0)
MAX_POSITION_SIZE=0.05            # Max position size (5% of portfolio)
URGENT_SIGNAL_CONFIDENCE_THRESHOLD=0.8  # Urgent alert threshold

# System Behavior  
OPERATIONAL_EARNINGS_DAYS_AHEAD=7  # How far ahead to look for earnings
MAX_ARTICLES_PER_RUN=50           # Articles to process per cycle
LOG_LEVEL=INFO                    # Logging detail level
```

## 📊 How It Works

### 1. Data Collection
- **Earnings Calendar**: Fetches upcoming earnings from multiple APIs
- **News Collection**: Monitors RSS feeds and company-specific news
- **Market Data**: Real-time stock prices and volume data

### 2. AI Analysis
- **Multi-LLM Processing**: Uses Claude, GPT, or Gemini for sentiment analysis
- **Traditional Methods**: TextBlob and VADER for validation
- **Confidence Scoring**: Combines multiple analysis methods

### 3. Signal Generation
- **Risk Assessment**: Automated position sizing based on volatility
- **Stop-Loss Calculation**: Dynamic risk management levels
- **Quality Filtering**: Only high-confidence signals are generated

### 4. Notifications
- **Daily Reports**: Comprehensive morning briefings
- **Urgent Alerts**: Immediate notifications for high-confidence signals
- **Performance Tracking**: Signal accuracy and portfolio impact

## 📁 Project Structure

```
earnings_analyzer/
├── config/
│   └── config.py              # Configuration management
├── src/
│   ├── analyzers/
│   │   ├── llm_interface.py   # AI provider integrations
│   │   ├── sentiment_analyzer.py  # Multi-method analysis
│   │   └── signal_generator.py    # Trading signal logic
│   ├── collectors/
│   │   └── news_collector.py  # News and earnings data collection
│   ├── database/
│   │   ├── models.py          # Database schema
│   │   └── connection.py      # Database management
│   ├── notifications/
│   │   └── gmail_notifier.py  # Email notification system
│   └── utils/
├── logs/                      # Application logs
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── .env.template             # Configuration template
└── README.md                 # This file
```

## 🔒 Security & Privacy

- **API Keys**: Stored locally in `.env` file (never committed to version control)
- **Data Storage**: All data stored locally in SQLite database
- **No Cloud Dependencies**: Runs entirely on your machine
- **Secure Email**: Uses Gmail App Passwords, not your main password

## ⚠️ Important Disclaimers

- **Educational Purpose**: This system is for educational and research purposes only
- **Not Financial Advice**: All signals and analysis should be verified independently
- **Risk Management**: Always use proper position sizing and stop-losses
- **Paper Trading**: Recommended to test with paper trading first
- **No Guarantees**: Past performance does not guarantee future results

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**2. Database Errors**
```bash
# Delete and recreate database
rm earnings_trader.db
python main.py test
```

**3. API Connection Issues**
```bash
# Check your .env file has valid API keys
# Test individual components
python main.py test-earnings
python main.py test-news
```

**4. Gmail Authentication Issues**
- Use Gmail App Password, not regular password
- Enable 2-factor authentication first
- Follow [Google's setup guide](https://support.google.com/accounts/answer/185833)

### Getting Help

1. Check the logs: `tail -f logs/earnings_trader.log`
2. Run system status: `python main.py status`
3. Test individual components: `python main.py test-*`
4. Review configuration: Check `.env` file values

## 🔄 Updates & Maintenance

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Clean old data (optional)
python -c "from src.database.connection import cleanup_old_data; cleanup_old_data(30)"

# Check system status
python main.py status
```

## 📈 Performance Tips

1. **Start Small**: Begin with lower confidence thresholds to see more signals
2. **Monitor Logs**: Check `logs/earnings_trader.log` regularly
3. **API Limits**: Respect API rate limits to avoid blocks
4. **System Resources**: Monitor CPU/memory usage during market hours
5. **Database Maintenance**: Clean old data periodically

## 🤝 Contributing

This is an educational project. Feel free to:
- Report bugs and issues
- Suggest improvements
- Share your modifications
- Add new data sources or LLM providers

## 📄 License

This project is for educational purposes. Please ensure compliance with all API terms of service and local regulations regarding automated trading systems.

---

**Remember**: Always do your own research and never risk more than you can afford to lose. This system is a tool to assist in analysis, not a guarantee of profitable trades.