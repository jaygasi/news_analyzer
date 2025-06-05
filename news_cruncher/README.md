# Financial News Analysis System - Architecture Diagrams

## 1. System Overview & Main Flow

```mermaid
flowchart TD
    Start([Start Application]) --> Init[Initialize Components]
    Init --> Loop{Main Loop<br/>Every 5 Minutes}
    
    Loop --> Fetch[📰 Fetch News<br/>FMP API Sources]
    Fetch --> Filter[🔍 Filter Processed Articles<br/>SQLite Deduplication]
    Filter --> Limit[📊 Apply 1000 Article Limit]
    Limit --> Group[🎯 Group by Ticker<br/>Prioritize by Quality]
    
    Group --> Analyze[🧠 Multi-Source Analysis<br/>News + Technical]
    Analyze --> Decide[⚖️ Decision Engine<br/>Combine Scores]
    Decide --> Log[📝 Log to CSV<br/>High-Confidence Only]
    Log --> Mark[✅ Mark as Processed<br/>SQLite Database]
    
    Mark --> Wait[⏱️ Wait 5 Minutes]
    Wait --> Loop
    
    style Fetch fill:#e1f5fe
    style Analyze fill:#f3e5f5
    style Decide fill:#fff3e0
    style Log fill:#e8f5e8
```

## 2. Component Architecture

```mermaid
graph TB
    subgraph "📊 Data Sources"
        FMP[FMP API<br/>• Stock News<br/>• Press Releases<br/>• Earnings Calendar<br/>• Market News]
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
        CSV[📋 CSV Output<br/>Trading Decisions]
    end
    
    subgraph "🎯 Core Engine"
        Aggregator[Ticker Aggregator<br/>Group & Prioritize]
        MultiLLM[Multi-LLM Analyzer<br/>Combine Predictions]
        DecisionEngine[Decision Engine<br/>News 70% + Tech 30%]
    end
    
    FMP --> Aggregator
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
    
    style FinBERT fill:#4caf50
    style Gemini fill:#2196f3
    style DecisionEngine fill:#ff9800
    style CSV fill:#9c27b0
```

## 3. Multi-Source Analysis Flow

```mermaid
sequenceDiagram
    participant TC as Ticker Aggregator
    participant MLA as Multi-LLM Analyzer
    participant FB as FinBERT
    participant GM as Gemini
    participant AV as Alpha Vantage
    participant PG as Polygon
    participant KW as Keywords
    participant DE as Decision Engine
    
    TC->>MLA: Articles for AAPL
    
    par Parallel Analysis
        MLA->>FB: Analyze sentiment
        FB-->>MLA: BUY (0.85)
        
        MLA->>GM: Analyze with LLM
        GM-->>MLA: BUY (0.78)
        
        MLA->>AV: News sentiment API
        AV-->>MLA: BUY (0.72)
        
        MLA->>PG: News analysis
        PG-->>MLA: BUY (0.68)
        
        MLA->>KW: Keyword analysis
        KW-->>MLA: BUY (0.60)
    end
    
    MLA->>MLA: Weight & Combine<br/>FB(41%) + GM(33%) + AV(13%)<br/>+ PG(10%) + KW(3%)
    MLA->>DE: Combined: BUY (0.79)
    DE->>DE: Add Technical Analysis<br/>News(70%) + Tech(30%)
    DE-->>TC: Final Decision: BUY (0.82)
```

## 4. Decision Engine Workflow

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
    
    style NewsCheck fill:#e3f2fd
    style Combine fill:#fff3e0
    style LogDecision fill:#e8f5e8
    style NoDecision fill:#ffebee
    style LowConf fill:#ffebee
    style ConflictDecision fill:#ffebee
```

## 5. Data Processing Pipeline

```mermaid
graph LR
    subgraph "📥 Input Stage"
        API1[Stock News<br/>~20 articles]
        API2[Press Releases<br/>~100 articles]
        API3[Earnings Calendar<br/>~450 articles]
        API4[Market News<br/>~50 articles]
    end
    
    subgraph "🔍 Filtering Stage"
        Raw[📊 Raw Articles<br/>~620 total]
        Dedup[🔄 Deduplicate<br/>Remove duplicates]
        TimeFilter[⏰ Time Filter<br/>Last 24-72 hours]
        ProcessedFilter[✅ Processed Filter<br/>SQLite lookup]
        Limited[📋 Limited Set<br/>≤1000 articles]
    end
    
    subgraph "🎯 Analysis Stage"
        Group[📊 Group by Ticker<br/>~100 buckets]
        Priority[🏆 Prioritize<br/>Quality scoring]
        Analyze[🧠 Multi-Source Analysis<br/>Top 50 tickers]
    end
    
    subgraph "📈 Output Stage"
        Decisions[⚖️ Trading Decisions<br/>~5-15 high-confidence]
        CSV[📋 CSV Log<br/>Permanent record]
        DB[💾 SQLite Update<br/>Mark processed]
    end
    
    API1 --> Raw
    API2 --> Raw
    API3 --> Raw
    API4 --> Raw
    
    Raw --> Dedup
    Dedup --> TimeFilter
    TimeFilter --> ProcessedFilter
    ProcessedFilter --> Limited
    
    Limited --> Group
    Group --> Priority
    Priority --> Analyze
    
    Analyze --> Decisions
    Decisions --> CSV
    Decisions --> DB
    
    style Raw fill:#e1f5fe
    style Limited fill:#f3e5f5
    style Decisions fill:#e8f5e8
    style CSV fill:#fff3e0
```

## 6. Service Integration & Fallback Chain

```mermaid
graph TD
    subgraph "🎯 Primary Services"
        FB[FinBERT<br/>✅ Always Available<br/>Local Model]
        GM[Gemini<br/>✅ 1000 req/day<br/>Google API]
    end
    
    subgraph "🔄 Optional Services"
        OAI[OpenAI<br/>❓ 500 req/day<br/>Quota dependent]
        CL[Claude<br/>❓ 300 req/day<br/>Key dependent]
    end
    
    subgraph "🚨 Emergency Fallbacks"
        AV[Alpha Vantage<br/>✅ 500 req/day<br/>Sentiment API]
        PG[Polygon<br/>✅ 500 req/day<br/>News Analysis]
        TG[Tiingo<br/>❓ 1000 req/day<br/>403 error prone]
    end
    
    subgraph "🔤 Always Available"
        KW[Enhanced Keywords<br/>✅ 80+ Terms<br/>No API limits]
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
    
    Combine --> Result[📊 Final Prediction<br/>Direction + Confidence]
    
    style FB fill:#4caf50
    style GM fill:#2196f3
    style AV fill:#ff9800
    style PG fill:#ff9800
    style KW fill:#9c27b0
    style Result fill:#e8f5e8
```

## 7. Configuration & Service Toggles

```mermaid
flowchart LR
    subgraph "⚙️ Environment Configuration"
        ENV[.env File<br/>API Keys & Toggles]
    end
    
    subgraph "🔧 Service Controls"
        FINBERT[ENABLE_FINBERT=true]
        GEMINI[ENABLE_GEMINI=true]
        OPENAI[ENABLE_OPENAI=false]
        CLAUDE[ENABLE_CLAUDE=false]
        ALPHAV[ENABLE_ALPHA_VANTAGE=true]
        POLYGON[ENABLE_POLYGON=true]
        TIINGO[ENABLE_TIINGO=false]
        KEYWORDS[ENABLE_KEYWORD_ANALYSIS=true]
    end
    
    subgraph "📊 Current Active Services"
        Active1[✅ FinBERT - 41% weight]
        Active2[✅ Gemini - 33% weight]
        Active3[✅ Alpha Vantage - 13% weight]
        Active4[✅ Polygon - 10% weight]
        Active5[✅ Keywords - 3% weight]
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
    
    FINBERT --> Active1
    GEMINI --> Active2
    ALPHAV --> Active3
    POLYGON --> Active4
    KEYWORDS --> Active5
    
    OPENAI --> Disabled1
    CLAUDE --> Disabled2
    TIINGO --> Disabled3
    
    style Active1 fill:#4caf50
    style Active2 fill:#4caf50
    style Active3 fill:#4caf50
    style Active4 fill:#4caf50
    style Active5 fill:#4caf50
    style Disabled1 fill:#f44336
    style Disabled2 fill:#f44336
    style Disabled3 fill:#f44336
```

## 8. CSV Output Schema

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
    }
```

---

## Usage Instructions

1. **Copy any diagram** you want to your README.md
2. **Mermaid renders automatically** on GitHub, GitLab, and most modern markdown viewers
3. **Customize as needed** - modify colors, add/remove components
4. **Live preview** available at [Mermaid Live Editor](https://mermaid.live/)

## Diagram Highlights

- **📊 System Overview**: Perfect for explaining the main flow to users
- **🧠 Component Architecture**: Shows how all pieces fit together  
- **⚖️ Decision Engine**: Explains the scoring and threshold logic
- **🔄 Multi-Source Analysis**: Demonstrates the AI service integration
- **📈 Data Pipeline**: Shows data transformation stages
- **⚙️ Configuration**: Explains service toggles and weights

These diagrams will make your README much more professional and help users understand the sophisticated multi-layered analysis system you've built!