# Biomedical News Analyzer - Production Version

A production-ready application that analyzes biomedical news to predict stock movements using AI/LLM analysis.

## 🚀 Quick Start

1. **Run Setup**:
   ```bash
   setup.bat
   ```

2. **Add API Keys** to `.env` file (created by setup)

3. **Test Configuration**:
   ```bash
   python diagnose.py
   ```

4. **Run Analysis**:
   ```bash
   python main.py
   ```

## 📋 Requirements

### System Requirements
- **Python 3.8+** (Python 3.10+ recommended)
- **Windows** (setup script is for Windows, but code works on all platforms)
- **Internet connection** for API calls

### API Keys Required

| API | Required | Purpose | Free Tier | Get Key |
|-----|----------|---------|-----------|---------|
| **Gemini AI** | ✅ Yes | News analysis with LLM | Yes | [ai.google.dev](https://ai.google.dev/) |
| **Finnhub** | ⚠️ Optional | News + FDA calendar | 60 calls/min | [finnhub.io](https://finnhub.io/) |
| **FMP** | ⚠️ Optional | Additional news coverage | 250 calls/day | [financialmodelingprep.com](https://financialmodelingprep.com/) |

> **Note**: You need **at least one** of Finnhub or FMP for news data. Both recommended for best coverage.

## 🛠️ Installation

### Automatic Setup (Recommended)
```bash
# Run the setup script
setup.bat

# Follow the prompts to:
# 1. Install Python packages
# 2. Create virtual environment
# 3. Set up configuration files
```

### Manual Setup
```bash
# 1. Create virtual environment
python -m venv bi
call bi\Scripts\activate.bat

# 2. Install packages
pip install -r requirements.txt

# 3. Create .env file from template
copy .env.example .env

# 4. Add your API keys to .env
```

## ⚙️ Configuration

### 1. Get API Keys

**Gemini AI (Required)**:
- Go to [Google AI Studio](https://ai.google.dev/)
- Create account and get API key
- Add to `.env`: `GEMINI_API_KEY=your_key_here`

**Finnhub (Optional)**:
- Sign up at [Finnhub](https://finnhub.io/)
- Get free API key from dashboard
- Add to `.env`: `FINNHUB_API_KEY=your_key_here`

**FMP (Optional)**:
- Sign up at [Financial Modeling Prep](https://financialmodelingprep.com/)
- Get free API key
- Add to `.env`: `FMP_API_KEY=your_key_here`

### 2. Verify Setup
```bash
# Run diagnostics to check everything
python diagnose.py
```

This will test:
- ✅ Package installations
- ✅ API key validity
- ✅ Network connections
- ✅ Application modules

## 🚀 Usage

### Basic Commands
```bash
# Activate environment (if not already active)
call bi\Scripts\activate.bat

# Test configuration
python main.py --dry-run

# Run full analysis
python main.py

# Get help
python main.py --help
```

### Analysis Modes
```bash
# All available news sources (default)
python main.py

# Only general market news
python main.py --mode general

# Only FDA calendar events (requires Finnhub)
python main.py --mode fda

# Specific tickers only
python main.py --mode tickers --tickers "PFE,MRNA,GILD"

# Analyze specific tickers + all other news
python main.py --tickers "PFE,MRNA,GILD"
```

### API Selection
```bash
# Use only Finnhub (disable FMP)
python main.py --disable-fmp

# Use only FMP (disable Finnhub)
python main.py --disable-finnhub

# Use both APIs (default)
python main.py
```

### Output Options
```bash
# Custom output file
python main.py --output my_predictions.json

# Verbose logging
python main.py --log-level DEBUG

# Test mode (no actual analysis)
python main.py --dry-run
```

## 📊 Output

The application generates a JSON file with predictions:

```json
[
  {
    "ticker": "PFE",
    "predicted_movement": "UP",
    "confidence": 8,
    "reasoning": "Positive Phase 3 trial results likely to boost investor confidence",
    "news_headline": "Pfizer Reports Positive Phase 3 Results for Cancer Drug",
    "news_url": "https://...",
    "news_source": "Reuters"
  }
]
```

### Prediction Fields
- **ticker**: Stock symbol
- **predicted_movement**: UP, DOWN, or NEUTRAL
- **confidence**: 1-10 (higher = more confident)
- **reasoning**: AI explanation
- **news_headline**: Original news headline
- **news_url**: Link to full article
- **news_source**: News provider

## 🔧 Troubleshooting

### Common Issues

**"Invalid Finnhub API key"**
- Check your API key at [Finnhub Dashboard](https://finnhub.io/dashboard)
- Ensure key is copied correctly without extra spaces
- Try regenerating the key

**"No predictions found"**
- Check if there's recent biomedical news
- Lower confidence threshold in config
- Try specific tickers: `--tickers "PFE,JNJ,MRNA"`

**"Module not found"**
- Ensure virtual environment is activated: `call bi\Scripts\activate.bat`
- Reinstall packages: `pip install -r requirements.txt`

**Rate limiting errors**
- Free tiers have limits (Finnhub: 60/min, FMP: 250/day)
- Upgrade API plan or wait for reset
- Use only one API if needed

### Diagnostic Tools

```bash
# Full diagnostic check
python diagnose.py

# Test specific components
python main.py --dry-run

# Debug mode for detailed logs
python main.py --log-level DEBUG
```

## 📁 Project Structure

```
biomed_analyzer/
├── analyzer/                   # Main application code
│   ├── config.py              # Configuration management
│   ├── finnhub_client.py      # Finnhub API client
│   ├── fmp_client.py          # FMP API client
│   ├── news_processor.py      # News processing logic
│   ├── models.py              # Data models
│   ├── utils.py               # Utilities
│   └── llm_services/          # LLM integrations
├── main.py                    # Main application
├── diagnose.py                # Diagnostic tool
├── setup.bat                  # Setup script
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── .env                      # Your API keys (created by setup)
└── README.md                 # This file
```

## 🎯 How It Works

### 1. **News Collection**
- Fetches recent news from Finnhub and/or FMP APIs
- Filters for biomedical companies using GICS classifications
- Identifies research-relevant keywords (clinical trials, FDA approvals, etc.)

### 2. **Company Identification**
- Maintains database of ~1000+ biomedical companies
- Uses GICS industry classifications
- Dynamic profile checking for unlisted companies

### 3. **Content Filtering**
- Scans headlines/summaries for research keywords
- Keywords include: "clinical trial", "FDA approval", "Phase I/II/III", etc.
- Only analyzes news relevant to drug development/research

### 4. **AI Analysis**
- Sends relevant news to Gemini AI for sentiment analysis
- AI predicts stock movement (UP/DOWN/NEUTRAL)
- Provides confidence score (1-10) and reasoning

### 5. **Output Generation**
- Deduplicates predictions across news sources
- Sorts by confidence level
- Saves structured JSON results

## 🔒 Security Notes

- **Keep `.env` file private** - contains your API keys
- **Don't commit `.env` to version control**
- **API keys have usage limits** - monitor your usage
- **Rate limiting is built-in** to respect API limits

## 📈 Performance

### Typical Analysis Coverage
- **~50-100 news articles** processed per run
- **~10-20 biomedical companies** analyzed
- **~5-15 predictions** generated (depending on news)
- **2-5 minutes** runtime for full analysis

### API Usage (per run)
- **Finnhub**: ~20-50 calls
- **FMP**: ~20-50 calls  
- **Gemini**: ~10-30 calls

## 🆘 Support

### If You Need Help

1. **Run diagnostics first**: `python diagnose.py`
2. **Check the troubleshooting section** above
3. **Review logs** with `--log-level DEBUG`
4. **Verify API keys** are working at their respective dashboards

### Common Support Questions

**Q: No predictions generated?**
A: This is normal if there's no relevant biomedical news. Try specific tickers or check news keywords.

**Q: API rate limit errors?**
A: Free tiers have limits. Wait for reset or upgrade plan. You can also disable one API.

**Q: Application crashes?**
A: Run `python diagnose.py` to identify the issue. Most crashes are due to missing/invalid API keys.

## 📄 License

This is a production application for biomedical news analysis. Ensure compliance with API terms of service for all integrated services.

---

**🎉 Ready to analyze biomedical news? Run `python diagnose.py` to get started!**