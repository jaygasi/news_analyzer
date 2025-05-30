from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Company(Base):
    """Company information"""
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), unique=True, index=True, nullable=False)
    company_name = Column(String(255), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    news_articles = relationship("NewsArticle", back_populates="company", cascade="all, delete-orphan")
    earnings_calendar = relationship("EarningsCalendar", back_populates="company", cascade="all, delete-orphan")
    trading_signals = relationship("TradingSignal", back_populates="company", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Company(ticker='{self.ticker}', name='{self.company_name}')>"

class NewsArticle(Base):
    """News articles with sentiment analysis"""
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text)
    url = Column(String(1000), unique=True, index=True)
    published_date = Column(DateTime, nullable=False, index=True)
    source = Column(String(100), nullable=False)

    # Company relationship
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    company = relationship("Company", back_populates="news_articles")

    # Analysis results
    sentiment_score = Column(Float)      # Combined sentiment (-1 to 1)
    textblob_score = Column(Float)       # TextBlob sentiment
    vader_score = Column(Float)          # VADER sentiment
    llm_score = Column(Float)            # LLM sentiment (Claude/GPT/etc)
    confidence_score = Column(Float)     # Analysis confidence (0 to 1)
    llm_provider_used = Column(String(50))   # Which LLM was used

    # Enhanced fields from EnhancedNewsCollector integration
    initial_sentiment_score = Column(Float) # Quick sentiment score from collector
    earnings_confidence_score = Column(Float) # How confident it's earnings related
    is_earnings_related = Column(Boolean, default=False) # Quick check if article seems earnings related

    # Processing status
    processed = Column(Boolean, default=False, index=True)
    processing_error = Column(Text)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trading_signals = relationship("TradingSignal", back_populates="news_article", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<NewsArticle(id={self.id}, title='{self.title[:50]}...')>"

class EarningsCalendar(Base):
    """Upcoming earnings dates"""
    __tablename__ = "earnings_calendar"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    company = relationship("Company", back_populates="earnings_calendar")

    earnings_date = Column(DateTime, nullable=False, index=True)
    earnings_time = Column(String(10))  # 'BMO', 'AMC', 'TAS' (Time After/Not Specified), 'DMH'
    fiscal_quarter = Column(String(10))
    fiscal_year = Column(Integer)

    # Estimates
    eps_estimate = Column(Float)
    revenue_estimate = Column(BigInteger)

    # Actual results (filled after earnings)
    actual_eps = Column(Float)
    actual_revenue = Column(BigInteger)
    eps_surprise = Column(Float)
    revenue_surprise = Column(Float)

    # Source of the earnings event data
    source_api = Column(String(50), nullable=True) # e.g., "Finnhub", "FMP", "Yahoo Finance"

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<EarningsCalendar(ticker='{self.company.ticker if self.company else 'N/A'}', date='{self.earnings_date}')>"

class TradingSignal(Base):
    """Generated trading signals"""
    __tablename__ = "trading_signals"

    id = Column(Integer, primary_key=True, index=True)

    # Company relationship
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    company = relationship("Company", back_populates="trading_signals")

    # News article that triggered this signal
    news_article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=True)
    news_article = relationship("NewsArticle", back_populates="trading_signals")

    # Signal details
    signal_type = Column(String(10), nullable=False, index=True)  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence_score = Column(Float, nullable=False, index=True)
    reasoning = Column(Text, nullable=False)
    llm_provider_used = Column(String(50))

    # Market data at signal generation
    stock_price = Column(Float)
    volume = Column(BigInteger)
    market_cap = Column(BigInteger)
    pe_ratio = Column(Float)
    price_change_30d = Column(Float)
    sector = Column(String(100))
    industry = Column(String(100))

    # Risk management
    suggested_position_size = Column(Float)
    stop_loss_price = Column(Float)
    take_profit_price = Column(Float)

    # Signal quality metrics
    textblob_score = Column(Float)
    vader_score = Column(Float)
    llm_score = Column(Float)
    combined_sentiment = Column(Float)

    # Status tracking
    executed = Column(Boolean, default=False, index=True)
    execution_price = Column(Float)
    execution_date = Column(DateTime)

    # Performance tracking
    current_pnl = Column(Float)
    realized_pnl = Column(Float)
    max_favorable_excursion = Column(Float)
    max_adverse_excursion = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<TradingSignal(id={self.id}, ticker='{self.company.ticker if self.company else 'N/A'}', signal='{self.signal_type}', confidence={self.confidence_score})>"

class SystemLog(Base):
    """System activity log"""
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(10), nullable=False, index=True)
    component = Column(String(50), nullable=False, index=True)
    message = Column(Text, nullable=False)
    details = Column(Text)

    def __repr__(self):
        return f"<SystemLog(id={self.id}, level='{self.level}', component='{self.component}')>"

class ProviderUsage(Base):
    """Track LLM provider usage and costs"""
    __tablename__ = "provider_usage"

    id = Column(Integer, primary_key=True, index=True)
    provider_name = Column(String(50), nullable=False, index=True)
    model_name = Column(String(50), nullable=False)

    tokens_used = Column(Integer, default=0)
    requests_made = Column(Integer, default=0)
    estimated_cost = Column(Float, default=0.0)

    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)

    usage_date = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ProviderUsage(provider='{self.provider_name}', model='{self.model_name}', date='{self.usage_date}')>"
