# Financial News Analyzer - Q&A Guide (Enhanced Version)

## Original Questions & Answers

### 1. **What is the application flow once it starts?**

**Flow:** The application runs in continuous 5-minute cycles:
```
Start → Fetch All News → Filter Already Processed → Apply 1000 Limit → 
Group by Ticker → Prioritize Tickers → Analyze (News + Technical) → 
Make Decisions → Log to CSV → Mark as Processed → Wait 5 Minutes → Repeat
```

### 2. **Which sources does it use to pull news?**

**Sources:** Uses **FMP (Financial Modeling Prep) APIs** exclusively:
- **General Stock News** (`stock_news`) - 200 articles per cycle
- **Press Releases** (`press-releases`) - 100 articles per cycle  
- **Earnings Calendar** (`earning_calendar`) - 50 synthetic articles per cycle
- **Market News** (`general-news`) - 50 articles per cycle

**Total potential:** ~400 raw articles per cycle before filtering

### 3. **How are the news APIs configured? Will it pull latest news? Since when?**

**Configuration:**
- **Frequency:** Every 5 minutes automatically
- **Time Range:** Only news from **last 24 hours** (`MAX_NEWS_AGE_HOURS: 24`)
- **Auto-filtering:** Removes articles older than 24 hours
- **Continuous:** Always pulls the "latest" news available from FMP

### 4. **How far back will it get news each time it runs? How does it prevent already processed news?**

**Time Range:** 24 hours maximum lookback

**Reprocessing Prevention:**
- **SQLite Database:** `processed_articles.db` tracks all seen articles
- **Unique Hashing:** Creates SHA256 hash from `ticker + title + url`
- **Filter Step:** `filter_unprocessed_articles()` removes already-seen articles before analysis
- **Efficiency:** Only processes truly new articles each cycle

### 5. **How does it decipher which stock companies the news are about? Is there keyword analysis?**

**Stock Identification:**
- **Primary Method:** FMP API provides pre-tagged `symbol` field in articles
- **No keyword analysis** for ticker extraction from content
- **Intelligent Market News Assignment:** ✅ **Enhanced** - Now uses content analysis:
  - Fed news → TLT (Treasury bonds)
  - Tech news → QQQ (NASDAQ)  
  - Energy news → XLE (Energy ETF)
  - Gold news → GLD (Gold ETF)
  - Unclear content → Skipped (not forced to SPY)

### 6. **Are the news grouped into company buckets?**

**Yes, comprehensive bucketing:**
- **Grouping:** `ticker_aggregator.py` groups all articles by ticker symbol
- **Prioritization:** Ranks tickers by article count + content quality
- **Quality Scoring:** Looks for high-value keywords (earnings, FDA, mergers, etc.)
- **Processing Order:** Analyzes highest-priority tickers first

### 7. **What APIs or methodologies does it use to evaluate news for Long/Short decisions?**

**Multi-layered Analysis Chain:**

**News Analysis (70% weight):**
1. **FinBERT** (Local ML model for financial sentiment)
2. **Gemini API** (Google's LLM) 
3. **OpenAI API** (GPT models)
4. **Claude API** (Anthropic)
5. **Enhanced Keyword Analysis** ✅ **New** - Robust fallback with 80+ keywords

**Technical Analysis (30% weight):**
- RSI, MACD, Bollinger Bands, Moving Averages
- Uses FMP historical price data

### 8. **How is the overall score decided if there's more than one evaluation?**

**Weighted Combination (`decision_engine.py`):**
- **News Analysis:** 70% weight
- **Technical Analysis:** 30% weight  
- **Minimum Thresholds:** News confidence ≥ 0.6, Technical strength ≥ 0.4
- **Decision Logic:**
  - Combined score > 0.5 → LONG
  - Combined score < -0.5 → SHORT  
  - Confidence < 0.7 → NONE (no decision)

### 9. **Is there any mechanism to prevent the same ticker getting alerted as buy/sell signal on the same run?**

**Partial Prevention:**
- **Within Cycle:** Each ticker analyzed only once per 5-minute cycle
- **Across Cycles:** ✅ **SQLite deduplication prevents same news triggering multiple decisions**
- **Limitation:** If different news sources report same event, could theoretically generate multiple signals (rare)

### 10. **Is there anything getting faked out instead of dynamically calculated?**

**✅ Fixed - Removed Fake Elements:**
- **~~Emergency APIs Removed~~:** No longer pretends to use Alpha Vantage/Polygon/Tiingo
- **Enhanced Keywords:** ✅ **Upgraded** from 23 to 80+ keywords with industry-specific terms
- **Honest Fallback:** Now clearly labeled as "enhanced_keyword_analysis"

**Remaining Static Elements:**
- Ticker exclusions: Only `VIX` (volatility index)
- Technical analysis thresholds (industry standard)
- Confidence thresholds (configurable)

---

## Follow-up Questions & Answers

### 11. **When does the 1000 news limit get enforced? Before or after pre-processed news have been discarded?**

**✅ FIXED - Now Applied After Filtering:**

**Previous Flow (Inefficient):**
```
Fetch → Dedupe → Time Filter → 1000 Limit → Filter Processed Articles
```

**New Flow (Efficient):**
```
Fetch → Dedupe → Time Filter → Filter Processed Articles → 1000 Limit
```

**Result:** Now processes up to 1000 **NEW** articles per cycle instead of wasting cycles on already-seen articles.

### 12. **What exactly does "market news gets forced to SPY" mean?**

**✅ FIXED - Intelligent Assignment Now:**

**Previous Behavior:**
```python
item['symbol'] = 'SPY'  # All market news forced to SPY
```

**New Behavior - Intelligent Content-Based Assignment:**
- **Fed/Interest Rate news** → `TLT` (Treasury bonds)
- **S&P 500/Broad market** → `SPY` 
- **Technology/NASDAQ** → `QQQ`
- **Small cap/Russell** → `IWM`
- **Oil/Energy** → `XLE`
- **Gold/Precious metals** → `GLD`
- **Volatility** → `VIX`
- **Unclear content** → Skipped (not forced anywhere)

### 13. **Can we enhance the positive/negative keywords lists for robust fallback analysis?**

**✅ MASSIVELY Enhanced:**

**Previous Keywords:** 11 positive, 12 negative (23 total)

**New Keywords:** 40+ positive, 40+ negative (80+ total) including:

**High-Impact Keywords (weighted 2x):**
- Positive: 'fda approval', 'merger', 'acquisition', 'earnings beat'
- Negative: 'bankruptcy', 'lawsuit', 'investigation', 'earnings miss'

**Industry-Specific Terms:**
- **Biotech:** 'phase 3 success', 'breakthrough therapy', 'trial failed'
- **Tech:** 'ai breakthrough', 'cloud growth', 'security breach'

**Advanced Scoring:**
- Weighted importance levels
- Industry context detection
- Multi-keyword phrase matching

### 14. **In the CSV output, are Alpha Vantage, Polygon, or Tiingo actually getting used?**

**✅ FIXED - Fake Services Removed:**

**Previous Behavior (Misleading):**
```python
# Pretended to use these APIs but actually ran simple keyword analysis
return DirectionalPrediction(source="alpha_vantage")  # FAKE
```

**New Behavior (Honest):**
- **Removed:** Fake Alpha Vantage/Polygon/Tiingo references
- **Replaced:** Single honest "enhanced_keyword_analysis" fallback
- **CSV Output:** Will show actual service used:
  - `finbert`, `gemini`, `openai`, `claude`, or `enhanced_keyword_analysis`

**Result:** CSV `news_source` column now accurately reflects which service provided the analysis.

---

## Summary of Key Improvements

| Issue | Before | After |
|-------|--------|-------|
| **Article Limit** | Applied before filtering processed | Applied after filtering (more efficient) |
| **Market News** | All forced to SPY | Intelligent content-based assignment |
| **Keywords** | 23 basic keywords | 80+ enhanced keywords with weighting |
| **Emergency APIs** | Fake services (misleading) | Honest enhanced keyword analysis |
| **Ticker Exclusions** | SPY, QQQ, IWM, VIX excluded | Only VIX excluded (ETFs are tradeable) |

## Configuration Files to Check

- **Article Limit:** `config.py` → `MAX_NEWS_ARTICLES = 1000`
- **News Age:** `config.py` → `MAX_NEWS_AGE_HOURS = 24` 
- **Confidence:** `config.py` → `MIN_CONFIDENCE_THRESHOLD = 0.7`
- **CSV Output:** `output/trading_decisions.csv`
- **Database:** `data/processed_articles.db`