# Financial News Analysis System with Price Tracking - Architecture Diagrams

## 1. System Overview & Main Flow with Price Tracking

```mermaid
flowchart TD
    Start([Start Application]) --> Init[Initialize Components]
    Init --> InitTracker[🕐 Start Price Tracker<br/>Background Scheduler]
    InitTracker --> Loop{Main Loop<br/>Every 5 Minutes}
    
    Loop --> Fetch[📰 Fetch News<br/>FMP API Sources]
    Fetch --> Filter[🔍 Filter Processed Articles<br/>SQLite Deduplication]
    Filter --> Limit[📊 Apply 1000 Article Limit]
    Limit --> Group[🎯 Group by Ticker<br/>Prioritize by Quality]
    
    Group --> Analyze[🧠 Multi-Source Analysis<br/>News + Technical]
    Analyze --> Decide[⚖️ Decision Engine<br/>Combine Scores]
    Decide --> Log[📝 Log to CSV<br/>High-Confidence Only]
    Log --> Track[📈 Add Price Tracking<br/>LONG/SHORT Only]
    Track --> Mark[✅ Mark as Processed<br/>SQLite Database]
    
    Mark --> Wait[⏱️ Wait 5 Minutes]
    Wait --> Loop
    
    subgraph "🕐 Background Price Tracking"
        PriceLoop[Check Every 5min<br/>For Due Price Reads]
        PriceLoop --> FetchPrice[📈 Fetch Prices<br/>45m, 1hr, 3:50pm]
        FetchPrice --> UpdateCSV[📝 Update CSV<br/>With Price Data]
        UpdateCSV --> PriceLoop
    end
    
    Track -.-> PriceLoop
    
    style Fetch fill:#e1f5fe
    style Analyze fill:#f3e5f5
    style Decide fill:#fff3e0
    style Log fill:#e8f5e8
    style Track fill:#fff8e1
    style PriceLoop fill:#f3e5f5
```

## 2. Enhanced Component Architecture with Price Tracking

```mermaid
graph TB
    subgraph "📊 Data Sources"
        FMP[FMP API<br/>• Stock News<br/>• Press Releases<br/>• Earnings Calendar<br/>• Market News<br/>• Real-time Quotes]
    end
    
    subgraph "🧠 AI Analysis Services"
        FinBERT[FinBERT<br/>Local ML Model<br/>Weight: 41%]
        Gemini[Google Gemini<br/>LLM API<br/>Weight: 33%]
        OpenAI[OpenAI GPT<br/>Optional<br/>Weight: 20%]
        Claude[Anthropic Claude<br/>Optional<br/>Weight: 15%]
        
        subgraph "🚨 Emergency Services"
            AlphaV[Alpha Vantage<br/>News Sentiment<br/>Weight: 13%]
            Polygon[Polygon API<br/>News Analysis<br/>Weight: 10%]
            Tiingo[Tiingo API<br/>News Analysis<br/>Weight: 4%]
        end
        
        Keywords[Enhanced Keywords<br/>80+ Financial Terms<br/>Weight: 3%]
    end
    
    subgraph "📈 Technical Analysis"
        TA[Technical Analyzer<br/>• RSI, MACD<br/>• Bollinger Bands<br/>• Moving Averages<br/>Weight: 30%]
    end
    
    subgraph "💾 Data Management"
        SQLite[(SQLite DB<br/>Article Tracking)]
        CSV[📋 CSV Output<br/>Trading Decisions<br/>+ Price History]
    end
    
    subgraph "🎯 Core Engine"
        Aggregator[Ticker Aggregator<br/>Group & Prioritize]
        MultiLLM[Multi-LLM Analyzer<br/>Combine Predictions]
        DecisionEngine[Decision Engine<br/>News 70% + Tech 30%]
    end
    
    subgraph "📈 NEW: Price Tracking System"
        PriceTracker[Price Tracker<br/>Market Hours Logic]
        Scheduler[Background Scheduler<br/>Monitor Checkpoints]
        PriceAPI[FMP Quote API<br/>Real-time Prices]
    end
    
    FMP --> Aggregator
    FMP --> PriceAPI
    Aggregator --> MultiLLM
    Aggregator --> TA
    
    FinBERT --> MultiLLM
    Gemini --> MultiLLM
    OpenAI --> MultiLLM
    Claude --> MultiLLM
    AlphaV --> MultiLLM
    Polygon --> MultiLLM
    Tiingo --> MultiLLM
    Keywords --> MultiLLM
    
    MultiLLM --> DecisionEngine
    TA --> DecisionEngine
    
    DecisionEngine --> CSV
    DecisionEngine --> SQLite
    DecisionEngine --> PriceTracker
    
    PriceTracker --> Scheduler
    Scheduler --> PriceAPI
    Scheduler --> CSV
    
    style FinBERT fill:#4caf50
    style Gemini fill:#2196f3
    style DecisionEngine fill:#ff9800
    style CSV fill:#9c27b0
    style PriceTracker fill:#ff5722
    style Scheduler fill:#795548
```

## 3. Price Tracking Workflow & Timing Logic

```mermaid
sequenceDiagram
    participant Main as Main Analysis Loop
    participant DE as Decision Engine
    participant PT as Price Tracker
    participant PS as Price Scheduler
    participant FMP as FMP Quote API
    participant CSV as CSV Logger
    
    Note over Main: Every 5 minutes
    Main->>DE: Analyze AAPL articles
    DE->>DE: Decision: LONG (0.85 confidence)
    DE->>PT: Add price tracking for AAPL
    
    PT->>FMP: Get current price: $150.25
    PT->>PT: Calculate schedule:<br/>10:45am (45m), 11:00am (1hr), 3:50pm (close)
    PT->>PS: Store tracking schedule
    PS-->>CSV: Log decision with baseline price
    
    Note over PS: Background monitoring every 5min
    
    loop Every 5 minutes
        PS->>PS: Check pending price reads
        
        alt 45-minute checkpoint due
            PS->>FMP: Fetch AAPL price: $151.30
            PS->>PS: Calculate change: +0.70%
            PS->>CSV: Update with 45m price data
        end
        
        alt 1-hour checkpoint due
            PS->>FMP: Fetch AAPL price: $152.10  
            PS->>PS: Calculate change: +1.23%
            PS->>CSV: Update with 1hr price data
        end
        
        alt 3:50pm close checkpoint due
            PS->>FMP: Fetch AAPL price: $149.80
            PS->>PS: Calculate change: -0.30%
            PS->>CSV: Update with close price data
            PS->>PS: Mark tracking completed
        end
    end
```

## 4. Intelligent Price Timing & Market Hours Logic

```mermaid
flowchart TD
    Recommendation[📊 New LONG/SHORT<br/>Recommendation Made] --> GetTime[Get Current Time<br/>Convert to EST]
    
    GetTime --> CalcTargets[Calculate Target Times<br/>45m, 1hr, 3:50pm]
    
    CalcTargets --> Check45m{45m Target<br/>In Market Hours?}
    Check45m -->|Yes| Add45m[✅ Add 45m Checkpoint]
    Check45m -->|No| Skip45m[❌ Skip 45m]
    
    Add45m --> Check1hr{1hr Target<br/>Before 3:50pm?}
    Skip45m --> Check1hr
    
    Check1hr -->|Yes| Add1hr[✅ Add 1hr Checkpoint]
    Check1hr -->|No| Skip1hr[❌ Skip 1hr]
    
    Add1hr --> CheckClose{Recommendation<br/>Before 3:50pm?}
    Skip1hr --> CheckClose
    
    CheckClose -->|Yes| AddClose[✅ Add 3:50pm Checkpoint]
    CheckClose -->|No| SkipClose[❌ Too Late for Close]
    
    AddClose --> Schedule[📅 Store Schedule<br/>Start Monitoring]
    SkipClose --> Schedule
    
    Schedule --> Background[🕐 Background Scheduler<br/>Monitors All Positions]
    
    subgraph "📅 Example Scenarios"
        Scenario1[10:00am Recommendation<br/>→ 10:45am, 11:00am, 3:50pm]
        Scenario2[3:00pm Recommendation<br/>→ 3:45pm, 3:50pm only]
        Scenario3[3:40pm Recommendation<br/>→ 3:50pm only]
        Scenario4[After 4:00pm<br/>→ Next trading day]
    end
    
    style Add45m fill:#4caf50
    style Add1hr fill:#4caf50  
    style AddClose fill:#4caf50
    style Skip45m fill:#f44336
    style Skip1hr fill:#f44336
    style SkipClose fill:#f44336
```

## 5. Enhanced Decision Engine with Price Tracking Integration

```mermaid
flowchart TD
    Input[📊 Ticker Analysis Input] --> NewsCheck{News Analysis<br/>Available?}
    
    NewsCheck -->|No| NoDecision[❌ NONE Decision<br/>No news analysis]
    NewsCheck -->|Yes| ConfCheck{News Confidence<br/>≥ 0.5?}
    
    ConfCheck -->|No| LowConf[❌ NONE Decision<br/>Low confidence]
    ConfCheck -->|Yes| TechCheck{Technical Analysis<br/>Available?}
    
    TechCheck -->|Yes| Conflict{Signals<br/>Conflict?}
    TechCheck -->|No| NewsOnly[📰 News-Only Analysis<br/>Weight: 100%]
    
    Conflict -->|Yes| ConflictCheck{Conflict Confidence<br/>≥ 0.6?}
    Conflict -->|No| Combine[⚖️ Combine Scores<br/>News: 70% + Tech: 30%]
    
    ConflictCheck -->|No| ConflictDecision[❌ NONE Decision<br/>Conflicting signals]
    ConflictCheck -->|Yes| Combine
    
    NewsOnly --> ScoreCalc[📊 Calculate Final Score]
    Combine --> ScoreCalc
    
    ScoreCalc --> Direction{Combined Score}
    Direction -->|> 0.2| Long[📈 LONG Decision]
    Direction -->|< -0.2| Short[📉 SHORT Decision]
    Direction -->|-0.2 to 0.2| Neutral[➡️ NEUTRAL Decision]
    
    Long --> FinalCheck{Final Confidence<br/>≥ 0.6?}
    Short --> FinalCheck
    Neutral --> FinalCheck
    
    FinalCheck -->|Yes| LogDecision[✅ Log to CSV]
    FinalCheck -->|No| SkipLogging[❌ Skip Logging<br/>Low final confidence]
    
    LogDecision --> TrackingCheck{LONG or SHORT<br/>Decision?}
    TrackingCheck -->|Yes| StartTracking[📈 Start Price Tracking<br/>Get baseline price<br/>Schedule checkpoints]
    TrackingCheck -->|No| Complete[Complete]
    
    StartTracking --> Complete
    SkipLogging --> Complete
    
    style NewsCheck fill:#e3f2fd
    style Combine fill:#fff3e0
    style LogDecision fill:#e8f5e8
    style StartTracking fill:#fff8e1
    style NoDecision fill:#ffebee
    style LowConf fill:#ffebee
    style ConflictDecision fill:#ffebee
```

## 6. Price Tracking Data Pipeline & CSV Schema

```mermaid
graph LR
    subgraph "📥 Input Stage"
        Decision[Trading Decision<br/>LONG/SHORT]
        CurrentPrice[Get Current Price<br/>via FMP API]
    end
    
    subgraph "🕐 Scheduling Stage"
        TimeCalc[Calculate Target Times<br/>45m, 1hr, 3:50pm]
        MarketCheck[Check Market Hours<br/>Apply Intelligent Logic]
        Schedule[Store Checkpoint<br/>Schedule]
    end
    
    subgraph "📈 Monitoring Stage"
        Background[Background Scheduler<br/>Every 5 minutes]
        CheckDue[Check Due<br/>Checkpoints]
        FetchPrice[Fetch Real-time<br/>Price]
        CalcChange[Calculate %<br/>Change]
    end
    
    subgraph "💾 Output Stage"
        UpdateCSV[Update CSV<br/>With Price Data]
        AuditTrail[Complete Audit<br/>Trail]
    end
    
    Decision --> CurrentPrice
    CurrentPrice --> TimeCalc
    TimeCalc --> MarketCheck
    MarketCheck --> Schedule
    
    Schedule --> Background
    Background --> CheckDue
    CheckDue --> FetchPrice
    FetchPrice --> CalcChange
    CalcChange --> UpdateCSV
    UpdateCSV --> AuditTrail
    
    style Decision fill:#e1f5fe
    style Schedule fill:#f3e5f5
    style Background fill:#e8f5e8
    style UpdateCSV fill:#fff3e0
```

## 7. Enhanced CSV Output Schema with Price Tracking

```mermaid
erDiagram
    TRADING_DECISIONS {
        string timestamp
        string ticker
        string decision "LONG|SHORT|NONE"
        float confidence "0.0-1.0"
        string reasoning "Combined analysis"
        float news_score "-1.0 to 1.0"
        float technical_score "-1.0 to 1.0"
        float combined_score "-1.0 to 1.0"
        int article_count "Articles analyzed"
        string news_direction "BUY|SELL|NEUTRAL"
        float news_confidence "0.0-1.0"
        string news_reasoning "Service details"
        string news_source "Service name or multi_source"
        string technical_direction "BUY|SELL|NEUTRAL"
        float technical_strength "0.0-1.0"
        string technical_reasoning "Indicator details"
        string analysis_method "standard_analysis"
        string sources_used "finbert,gemini,alpha_vantage..."
        string analysis_timestamp "ISO timestamp"
        
        float recommendation_price "Baseline price when decision made"
        string recommendation_timestamp "When decision was made"
        float price_45m "Price after 45 minutes"
        string price_45m_timestamp "When 45m price was fetched"
        float price_45m_change_pct "Percentage change at 45m"
        float price_1hr "Price after 1 hour"
        string price_1hr_timestamp "When 1hr price was fetched"
        float price_1hr_change_pct "Percentage change at 1hr"
        float price_close "Price at 3:50pm EST"
        string price_close_timestamp "When close price was fetched"
        float price_close_change_pct "Percentage change at close"
        string tracking_status "pending|completed|partial"
    }
```

## 8. Service Integration & Fallback Chain with Price Data

```mermaid
graph TD
    subgraph "🎯 Primary Analysis Services"
        FB[FinBERT<br/>✅ Always Available<br/>Local Model]
        GM[Gemini<br/>✅ 1000 req/day<br/>Google API]
    end
    
    subgraph "🔄 Optional Analysis Services"
        OAI[OpenAI<br/>❓ 500 req/day<br/>Quota dependent]
        CL[Claude<br/>❓ 300 req/day<br/>Key dependent]
    end
    
    subgraph "🚨 Emergency Analysis Fallbacks"
        AV[Alpha Vantage<br/>✅ 500 req/day<br/>Sentiment API]
        PG[Polygon<br/>✅ 500 req/day<br/>News Analysis]
        TG[Tiingo<br/>❓ 1000 req/day<br/>403 error prone]
    end
    
    subgraph "🔤 Always Available"
        KW[Enhanced Keywords<br/>✅ 80+ Terms<br/>No API limits]
    end
    
    subgraph "📈 NEW: Price Data Services"
        FMP_Quote[FMP Quote API<br/>✅ Real-time Prices<br/>300 req/min]
        PriceBackup[Price Fallbacks<br/>❓ Alpha Vantage<br/>❓ Yahoo Finance]
    end
    
    Start[🎯 Ticker Analysis] --> FB
    Start --> GM
    Start --> OAI
    Start --> CL
    Start --> AV
    Start --> PG
    Start --> TG
    Start --> KW
    
    FB --> Combine[⚖️ Weighted Combination]
    GM --> Combine
    OAI --> Combine
    CL --> Combine
    AV --> Combine
    PG --> Combine
    TG --> Combine
    KW --> Combine
    
    Combine --> Decision[📊 Trading Decision<br/>Direction + Confidence]
    
    Decision -->|LONG/SHORT| PriceTrack[📈 Start Price Tracking]
    PriceTrack --> FMP_Quote
    FMP_Quote -->|Backup| PriceBackup
    
    style FB fill:#4caf50
    style GM fill:#2196f3
    style AV fill:#ff9800
    style PG fill:#ff9800
    style KW fill:#9c27b0
    style FMP_Quote fill:#ff5722
    style Decision fill:#e8f5e8
```

## 9. Configuration & Service Controls with Price Tracking

```mermaid
flowchart LR
    subgraph "⚙️ Environment Configuration"
        ENV[.env File<br/>API Keys & Toggles]
    end
    
    subgraph "🔧 Analysis Service Controls"
        FINBERT[ENABLE_FINBERT=true]
        GEMINI[ENABLE_GEMINI=true]
        OPENAI[ENABLE_OPENAI=false]
        CLAUDE[ENABLE_CLAUDE=false]
        ALPHAV[ENABLE_ALPHA_VANTAGE=true]
        POLYGON[ENABLE_POLYGON=true]
        TIINGO[ENABLE_TIINGO=false]
        KEYWORDS[ENABLE_KEYWORD_ANALYSIS=true]
    end
    
    subgraph "📈 NEW: Price Tracking Controls"
        PRICE_CHECK_1[PRICE_CHECK_1_MINUTES=45]
        PRICE_CHECK_2[PRICE_CHECK_2_MINUTES=60]
        CLOSE_HOUR[CLOSE_PRICE_HOUR=15]
        CLOSE_MIN[CLOSE_PRICE_MINUTE=50]
        CHECK_FREQ[PRICE_TRACKER_CHECK_INTERVAL=5]
    end
    
    subgraph "📊 Current Active Services"
        Active1[✅ FinBERT - 41% weight]
        Active2[✅ Gemini - 33% weight]
        Active3[✅ Alpha Vantage - 13% weight]
        Active4[✅ Polygon - 10% weight]
        Active5[✅ Keywords - 3% weight]
        Active6[✅ Price Tracking - ENABLED]
    end
    
    subgraph "❌ Disabled Services"
        Disabled1[❌ OpenAI - Quota issues]
        Disabled2[❌ Claude - Invalid key]
        Disabled3[❌ Tiingo - 403 errors]
    end
    
    ENV --> FINBERT
    ENV --> GEMINI
    ENV --> OPENAI
    ENV --> CLAUDE
    ENV --> ALPHAV
    ENV --> POLYGON
    ENV --> TIINGO
    ENV --> KEYWORDS
    ENV --> PRICE_CHECK_1
    ENV --> PRICE_CHECK_2
    ENV --> CLOSE_HOUR
    ENV --> CLOSE_MIN
    ENV --> CHECK_FREQ
    
    FINBERT --> Active1
    GEMINI --> Active2
    ALPHAV --> Active3
    POLYGON --> Active4
    KEYWORDS --> Active5
    PRICE_CHECK_1 --> Active6
    
    OPENAI --> Disabled1
    CLAUDE --> Disabled2
    TIINGO --> Disabled3
    
    style Active1 fill:#4caf50
    style Active2 fill:#4caf50
    style Active3 fill:#4caf50
    style Active4 fill:#4caf50
    style Active5 fill:#4caf50
    style Active6 fill:#ff5722
    style Disabled1 fill:#f44336
    style Disabled2 fill:#f44336
    style Disabled3 fill:#f44336
```

## 10. Real-time Price Tracking Timeline Example

```mermaid
gantt
    title Price Tracking Timeline for Multiple Positions
    dateFormat  HH:mm
    axisFormat %H:%M
    
    section AAPL (LONG at 10:00)
    Baseline Price $150.25    :milestone, m1, 10:00, 0m
    45m Check $151.30 (+0.70%) :milestone, m2, 10:45, 0m
    1hr Check $152.10 (+1.23%) :milestone, m3, 11:00, 0m
    Close $149.80 (-0.30%)     :milestone, m4, 15:50, 0m
    
    section TSLA (SHORT at 14:00)
    Baseline Price $200.50     :milestone, t1, 14:00, 0m
    45m Check $198.20 (-1.15%) :milestone, t2, 14:45, 0m
    1hr Check $195.80 (-2.34%) :milestone, t3, 15:00, 0m
    Close $196.50 (-2.00%)     :milestone, t4, 15:50, 0m
    
    section MSFT (LONG at 15:30)
    Baseline Price $300.00     :milestone, ms1, 15:30, 0m
    Close Only $301.50 (+0.50%) :milestone, ms2, 15:50, 0m
    
    section Background Scheduler
    Monitoring All Positions   :active, sched, 10:00, 06:00
```

## 11. Performance Metrics & Scaling

```mermaid
graph TB
    subgraph "📊 Input Metrics"
        Articles[600 Articles/Cycle<br/>~100 Tickers/Cycle]
        Decisions[8-25 Decisions/Cycle<br/>~6-20 LONG/SHORT]
    end
    
    subgraph "⚖️ Processing Performance"
        Analysis[~12 sec/ticker<br/>69 tickers in 15min]
        Confidence[36% Success Rate<br/>0.74 Avg Confidence]
    end
    
    subgraph "📈 Price Tracking Load"
        Positions[10-50 Active Positions<br/>Throughout Trading Day]
        PriceReads[~150 Price Fetches/Day<br/>3 reads × 50 positions]
        APIUsage[Well Within Limits<br/>300/min FMP capacity]
    end
    
    subgraph "💾 Data Volume"
        CSVGrowth[~25 rows/cycle<br/>~200 rows/day]
        Storage[Audit Trail<br/>Complete Price History]
    end
    
    Articles --> Analysis
    Analysis --> Decisions
    Decisions --> Positions
    Positions --> PriceReads
    PriceReads --> APIUsage
    Decisions --> CSVGrowth
    PriceReads --> Storage
    
    style Analysis fill:#e3f2fd
    style Confidence fill:#e8f5e8
    style Positions fill:#fff8e1
    style APIUsage fill:#e8f5e8
```

---

## 🚀 Quick Start Guide

### Prerequisites
```bash
pip install pandas numpy requests torch transformers google-generativeai pytz
```

### Configuration
Create `.env` file with your API keys:
```bash
FMP_API_KEY=your_fmp_key
GEMINI_API_KEY=your_gemini_key

# Price Tracking Configuration (Optional)
PRICE_CHECK_1_MINUTES=45
PRICE_CHECK_2_MINUTES=60
CLOSE_PRICE_HOUR=15
CLOSE_PRICE_MINUTE=50
MAX_TICKERS_TO_ANALYZE=100
```

### Run the System
```bash
python main.py
```

## 📈 Price Tracking Features

### ✅ **Automatic Price Monitoring**
- **Baseline price** captured when LONG/SHORT decision is made
- **45-minute checkpoint** - tracks short-term price movement
- **1-hour checkpoint** - confirms trend direction
- **Market close price** - final daily performance

### 🕐 **Intelligent Timing**
- **Market hours aware** - only fetches during trading hours (9:30am-4:00pm EST)
- **Adaptive scheduling** - adjusts checkpoints based on recommendation time
- **Weekend handling** - schedules for next trading day
- **Configurable intervals** - customize timing via environment variables

### 📊 **Performance Analytics**
- **Percentage changes** calculated automatically
- **Complete audit trail** in CSV format
- **Background monitoring** doesn't impact main analysis
- **Multi-position tracking** - monitors dozens of positions simultaneously

## 🎯 Output Examples

### Trading Decision with Price Tracking
```csv
ticker,decision,confidence,recommendation_price,price_45m,price_45m_change_pct,price_1hr,price_1hr_change_pct,price_close,price_close_change_pct
AAPL,LONG,0.85,150.25,151.30,+0.70,152.10,+1.23,149.80,-0.30
TSLA,SHORT,0.72,200.50,198.20,-1.15,195.80,-2.34,196.50,-2.00
```

### Log Output
```
📊 Added price tracking for AAPL: $150.25 baseline, 3 checkpoints
📈 45m: AAPL $151.30 (+0.70%)
📈 1hr: AAPL $152.10 (+1.23%)
📈 Close: AAPL $149.80 (-0.30%)
📊 Active price tracking: 25 positions being monitored
```

## 🔧 Configuration Options

### Analysis Thresholds
- `MIN_CONFIDENCE_THRESHOLD=0.6` - Minimum confidence for CSV logging
- `MAX_TICKERS_TO_ANALYZE=100` - Maximum tickers per cycle
- `MAX_NEWS_ARTICLES=1000` - Article limit per cycle

### Price Tracking Settings
- `PRICE_CHECK_1_MINUTES=45` - First price check interval
- `PRICE_CHECK_2_MINUTES=60` - Second price check interval
- `CLOSE_PRICE_HOUR=15` - Hour for close price (EST)
- `CLOSE_PRICE_MINUTE=50` - Minute for close price (EST)
- `PRICE_TRACKER_CHECK_INTERVAL=5` - Background check frequency (minutes)

### Service Toggles
- `ENABLE_FINBERT=true` - Local FinBERT model (recommended)
- `ENABLE_GEMINI=true` - Google Gemini API
- `ENABLE_ALPHA_VANTAGE=true` - Alpha Vantage sentiment
- `ENABLE_POLYGON=true` - Polygon news analysis

## 📊 System Statistics

- **Decision Volume**: 25-50 high-confidence decisions per day
- **Processing Speed**: ~12 seconds per ticker analysis
- **API Efficiency**: Well within all service limits
- **Price Tracking**: Real-time monitoring of 20-100 positions
- **Data Retention**: Complete audit trail in CSV format
- **Uptime**: Continuous operation with graceful error handling

---

## Architecture Highlights

- **📊 Multi-source Analysis**: Combines 5+ AI services for robust predictions
- **🎯 Intelligent Filtering**: Processes only new articles, avoids reprocessing
- **⚖️ Sophisticated Scoring**: 70% news + 30% technical analysis weighting
- **📈 Real-time Price Tracking**: Automatic performance monitoring
- **🕐 Background Processing**: Price tracking runs independently
- **💾 Complete Audit Trail**: Every decision and price change logged
- **🔧 Highly Configurable**: Customize thresholds, intervals, and services
- **🛡️ Robust Error Handling**: Graceful degradation and service fallbacks

This system provides institutional-grade financial news analysis with comprehensive price tracking capabilities, perfect for algorithmic trading, research, and performance analytics.