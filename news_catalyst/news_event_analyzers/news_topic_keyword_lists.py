from enum import Enum


class NewsTopicSentiment(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNKNOWN = "UNKNOWN"


class NewsTopic(Enum):
    POSITIVE_FINANCIAL_PERFORMANCE = "POSITIVE_FINANCIAL_PERFORMANCE"
    NEGATIVE_FINANCIAL_PERFORMANCE = "NEGATIVE_FINANCIAL_PERFORMANCE"
    MARKET_GROWTH = "MARKET_GROWTH"
    INNOVATION = "INNOVATION"
    STOCK_BUYBACK = "STOCK_BUYBACK"
    STOCK_SPLIT = "STOCK_SPLIT"
    LAWSUIT_WIN = "LAWSUIT_WIN"
    POSITIVE_PHASE2_RESULT = "POSITIVE_PHASE2_RESULT"
    NEGATIVE_PHASE2_RESULT = "NEGATIVE_PHASE2_RESULT"
    POSITIVE_PHASE3_RESULT = "POSITIVE_PHASE3_RESULT"
    NEGATIVE_PHASE3_RESULT = "NEGATIVE_PHASE3_RESULT"
    FDA_APPROVAL = "FDA_APPROVAL"
    FDA_REJECTION = "FDA_REJECTION"
    UNKNOWN = "UNKNOWN"


news_topic_sentiment_map = {
    NewsTopic.POSITIVE_FINANCIAL_PERFORMANCE: NewsTopicSentiment.POSITIVE,
    NewsTopic.MARKET_GROWTH: NewsTopicSentiment.POSITIVE,
    NewsTopic.INNOVATION: NewsTopicSentiment.POSITIVE,
    NewsTopic.STOCK_BUYBACK: NewsTopicSentiment.POSITIVE,
    NewsTopic.STOCK_SPLIT: NewsTopicSentiment.POSITIVE,
    NewsTopic.LAWSUIT_WIN: NewsTopicSentiment.POSITIVE,
    NewsTopic.POSITIVE_PHASE2_RESULT: NewsTopicSentiment.POSITIVE,
    NewsTopic.NEGATIVE_PHASE2_RESULT: NewsTopicSentiment.NEGATIVE,
    NewsTopic.POSITIVE_PHASE3_RESULT: NewsTopicSentiment.POSITIVE,
    NewsTopic.NEGATIVE_PHASE3_RESULT: NewsTopicSentiment.NEGATIVE,
    NewsTopic.FDA_APPROVAL: NewsTopicSentiment.POSITIVE,
    NewsTopic.FDA_REJECTION: NewsTopicSentiment.NEGATIVE
}

biotech_topic_list = [NewsTopic.POSITIVE_PHASE2_RESULT, NewsTopic.NEGATIVE_PHASE2_RESULT,
                      NewsTopic.POSITIVE_PHASE3_RESULT, NewsTopic.NEGATIVE_PHASE3_RESULT,
                      NewsTopic.FDA_APPROVAL, NewsTopic.FDA_REJECTION]

# Content keywords for filtering titles
fda_approval_title_list = [
    "fda approval",
    "fda approves",
    "fda approved",
    "federal drug administration approved",
    "federal drug administration approves",
    "federal drug administration approval",
    "federal drug administration (fda) approved",
    "federal drug administration (fda) approves",
    "federal drug administration (fda) approval",
    "approved by the fda",
    "fda grants approval",
    "fda clearance",
    "fda clears",
    "cleared by fda",
    "fda authorizes",
    "fda authorized",
    "fda marketing approval",
    "new drug approved by fda",
    "fda okays",
    "fda greenlights",
    "approved by federal drug administration",
    "fda endorsement",
    "fda approval granted",
    "new treatment approved by fda",
    "fda nod",
    "fda grants nod",
    "fda regulatory approval",
    "approved under fda",
    "breakthrough therapy approval by fda",
    "fast track approval by fda",
    "priority review approval by fda",
    "accelerated approval by fda",
    "drug approved by fda",
    "treatment approved by fda",
    "fda new drug approval",
    "fda indication approval",
    "fda new indication approval",
    "approval for commercialization by fda",
    "new therapy gets fda approval",
]

fda_rejection_title_list = [
    "fda rejection",
    "fda rejected",
    "fda denies",
    "fda declined",
    "fda does not approve",
    "fda challenges",
    "fda fails to approve",
    "fda refuses approval",
    "fda no approval",
    "fda declines approval",
    "fda refuses to approve",
    "fda rejects application",
    "fda issues rejection",
    "fda disapproves",
    "fda withholds approval",
    "fda turns down",
    "fda setback",
    "federal drug administration rejected",
    "federal drug administration rejection",
    "federal drug administration denies",
    "federal drug administration declined",
    "federal drug administration does not approve",
    "federal drug administration challenges",
    "federal drug administration fails to approve",
    "federal drug administration refuses approval",
    "federal drug administration (fda) rejection",
    "federal drug administration (fda) rejected",
    "federal drug administration (fda) denies",
    "federal drug administration (fda) declined",
    "federal drug administration (fda) does not approve",
    "federal drug administration (fda) challenges",
    "federal drug administration (fda) fails to approve",
    "federal drug administration (fda) refuses approval",
    "drug rejected by fda",
    "drug refusal by fda",
    "fda decision against approval",
    "fda regulatory denial",
    "fda rules against approval",
    "fda review ends in rejection",
    "fda marketing application rejected",
    "fda declines marketing approval",
    "fda new drug rejection",
    "fda denies new drug approval",
    "fda denial for new drug",
    "new therapy rejected by fda",
    "treatment rejected by fda",
    "fda application denied",
    "fda application rejected",
]

phase2_title_list = [
    # General Phase 2 Terms
    "phase 2",
    "phase 2a",
    "phase 2b",
    "phase two",
    "phase II",
    "phase IIa",
    "phase IIb",

    # Announcements and Milestones
    "phase 2 results",
    "phase II results",
    "phase 2a results",
    "phase 2b results",
    "phase IIa results",
    "phase IIb results",
    "phase 2 data",
    "phase II data",
    "phase 2a data",
    "phase 2b data",
    "phase IIa data",
    "phase IIb data",
    "phase 2 clinical trial",
    "phase II clinical trial",
    "phase 2a clinical trial",
    "phase 2b clinical trial",
    "phase IIa clinical trial",
    "phase IIb clinical trial",
    "phase 2 study",
    "phase II study",
    "phase 2a study",
    "phase 2b study",
    "phase IIa study",
    "phase IIb study",
    "phase 2 trial",
    "phase II trial",
    "phase 2 announcement",
    "phase II announcement",
    "phase 2 readout",
    "phase II readout",
    "phase 2 topline results",
    "phase II topline results",
    "phase 2 efficacy data",
    "phase II efficacy data",
    "phase 2 safety data",
    "phase II safety data",
    "phase 2 progress",
    "phase II progress",
    "phase 2 trial completed",
    "phase II trial completed",
    "phase 2 interim results",
    "phase II interim results",
    "phase 2 exploratory results",
    "phase II exploratory results",
]

phase3_title_list = [
    # General Phase 3 Terms
    "phase 3",
    "phase 3a",
    "phase 3b",
    "phase three",
    "phase III",
    "phase IIIa",
    "phase IIIb",

    # Announcements and Milestones
    "phase 3 results",
    "phase III results",
    "phase 3a results",
    "phase 3b results",
    "phase IIIa results",
    "phase IIIb results",
    "phase 3 data",
    "phase III data",
    "phase 3a data",
    "phase 3b data",
    "phase IIIa data",
    "phase IIIb data",
    "phase 3 clinical trial",
    "phase III clinical trial",
    "phase 3a clinical trial",
    "phase 3b clinical trial",
    "phase IIIa clinical trial",
    "phase IIIb clinical trial",
    "phase 3 study",
    "phase III study",
    "phase 3a study",
    "phase 3b study",
    "phase IIIa study",
    "phase IIIb study",
    "phase 3 trial",
    "phase III trial",

    # Announcements and Progress Updates
    "phase 3 announcement",
    "phase III announcement",
    "phase 3 readout",
    "phase III readout",
    "phase 3 topline results",
    "phase III topline results",
    "phase 3 efficacy data",
    "phase III efficacy data",
    "phase 3 safety data",
    "phase III safety data",
    "phase 3 progress",
    "phase III progress",
    "phase 3 trial completed",
    "phase III trial completed",
    "phase 3 interim results",
    "phase III interim results",
    "phase 3 exploratory results",
    "phase III exploratory results",
]

lawsuit_win_title_list = [
    "won lawsuit",
    "won shareholder lawsuit",
    "wins lawsuit",
    "wins shareholder lawsuit",
    "wins dismissal of lawsuit",
    "wins dismissal of shareholder lawsuit",
    "lawsuit victory",
    "lawsuit dismissal",
    "wins legal battle",
    "wins court case",
    "won court case",
    "legal victory",
    "court victory",
    "success in lawsuit",
    "lawsuit resolved in favor",
    "prevails in lawsuit",
    "prevails in court case",
    "dismissal of lawsuit",
    "dismissal of shareholder lawsuit",
    "victorious in lawsuit",
    "victorious in court case",
    "favorable lawsuit outcome",
    "court rules in favor",
    "jury rules in favor",
    "judge rules in favor",
    "court dismisses lawsuit",
    "lawsuit settled favorably",
    "lawsuit outcome favorable",
]


positive_financial_performance_title_list = [
    # Revenue and Profit
    "record revenue",
    "profit surge",
    "record profits",
    "revenue jump",
    "profit growth",
    "surging profits",
    "soaring revenue",
    "revenue milestone",
    "exceeds revenue expectations",
    "revenue beats estimates",
    "profit beats estimates",
    "exceeds profit projections",
    "record-breaking revenue",
    "record-breaking profits",
    "revenue expansion",
    "strong revenue growth",
    "unprecedented revenue growth",

    # Earnings
    "beats earnings",
    "beat earnings",
    "beat earnings estimates",
    "impressive earnings surprise",
    "positive earnings",
    "strong earnings report",
    "earnings exceed expectations",
    "better-than-expected earnings",
    "outperforms earnings forecast",
    "earnings beat consensus",
    "solid earnings report",
    "stellar earnings results",
    "record-breaking earnings",
    "surprise profit report",

    # Financial Performance
    "strong financial performance",
    "strong financial",
    "impressive financial results",
    "positive financial results",
    "outstanding financial performance",
    "financial results beat expectations",
    "upward financial trend",
    "financial outperformance",
    "record financial results",
    "exceptional financial results",

    # Growth
    "revenue growth",
    "profit growth",
    "double-digit growth",
    "sustained growth",
    "strong quarterly growth",
    "accelerated growth",
    "strong sales growth",
    "improved financial outlook",
    "expanding margins",

    # Market and Forecasts
    "earnings surprise",
    "beat upcoming",
    "beat quarterly",
    "quarterly revenue beat",
    "quarterly profit beat",
    "market outperformance",
    "guidance raised",
    "raises financial outlook",
    "upgraded revenue forecast",
    "upgraded profit forecast",
    "upgraded earnings outlook",
    "positive quarterly report",
    "exceeds market expectations",
    "outperforms market estimates",
]

negative_financial_performance_title_list = [
    # Revenue Decline
    "revenue decline",
    "revenue drop",
    "revenue falls",
    "revenue plunges",
    "revenue slowdown",
    "revenue shortfall",
    "revenue contraction",
    "revenue misses estimates",
    "declining revenue",
    "falling revenue",
    "weak revenue performance",
    "underwhelming revenue",
    "disappointing revenue",
    "lower-than-expected revenue",
    "revenue slump",
    "revenue underperformance",

    # Profit and Earnings Miss
    "profit declines",
    "earnings miss",
    "profit drop",
    "profit falls",
    "profit plunges",
    "weak profit report",
    "earnings disappointment",
    "earnings miss expectations",
    "earnings fall short",
    "earnings decline",
    "earnings slump",
    "profit misses estimates",
    "profit shortfall",
    "misses earnings estimates",
    "missed earnings estimates",
    "missed earnings",
    "earnings underperformance",
    "earnings underwhelm",
    "profit underpressure",
    "profit contraction",
    "earnings erosion",
    "earnings weaken",
    "unexpected earnings miss",
    "earnings miss consensus",

    # Weak Financial Performance
    "weak financial performance",
    "weak financial",
    "disappointing financial results",
    "poor financial performance",
    "underwhelming financial results",
    "financial results miss expectations",
    "declining financial results",
    "deteriorating financial health",
    "financial shortfall",
    "negative financial results",
    "negative financial report",
    "underwhelming financial report",
    "subpar financial results",
    "loss-making quarter",
    "financial performance falls short",

    # Market and Forecasts
    "misses revenue estimates",
    "missed revenue estimates",
    "missed revenue",
    "cuts financial guidance",
    "lowers revenue forecast",
    "downgrades profit outlook",
    "profit warning",
    "issues profit warning",
    "issues earnings warning",
    "cuts earnings outlook",
    "guidance downgrade",
    "forecast revision downward",
    "weaker-than-expected earnings",
    "weaker-than-expected revenue",
]


market_growth_title_list = [
    # General Market Growth
    "market growth",
    "market expansion",
    "industry expansion",
    "global expansion",
    "international growth",
    "increasing market size",
    "market demand growth",
    "growing market share",
    "expanding market presence",
    "expanding market share",
    "rapid market growth",
    "accelerating market growth",
    "strong market growth",
    "double-digit market growth",
    "surging market growth",
    "steady market growth",
    "new market opportunities",

    # Demand Growth
    "increasing demand",
    "increased demand",
    "fueled demand",
    "rising demand",
    "demand surge",
    "growing demand",
    "surging demand",
    "strong demand",
    "demand boost",
    "demand momentum",
    "accelerated demand growth",
    "sustained demand growth",
    "increased adoption",
    "boosted demand",
    "high market demand",
    "soaring demand",
    "demand-driven growth",

    # Market Expansion
    "market boost",
    "global market expansion",
    "international expansion",
    "market penetration",
    "increased market penetration",
    "new market entry",
    "expanding global footprint",
    "regional expansion",
    "geographical expansion",
    "emerging market growth",
    "targeting new markets",
    "new markets unlocked",
    "growth in emerging markets",
    "new customer base",
    "broader market reach",

    # Adoption and Market Share
    "rising market share",
    "increasing market share",
    "growing adoption",
    "strong adoption rates",
    "adoption gains",
    "expanding customer base",
    "expanding user base",
    "increased product adoption",
    "adoption surge",
    "rising consumer interest",
    "widening market appeal",
    "market uptake growth",
    "adoption growth momentum",
]


innovation_title_list = [
    "new product launch",
    "introduces new technology",
    "breakthrough innovation",
    "cutting-edge development",
    "revolutionary product",
    "revolutionary new product",
    "revolutionary new service",
    "revolutionary service",
    "cutting-edge service",
    "game-changing technology",
    "next-gen technology",
    "pioneering innovation",
    "state-of-the-art product",
    "state-of-the-art service",
    "technological milestone",
    "industry-first innovation",
    "disruptive technology",
    "unveils new product",
    "unveils breakthrough technology",
    "first-of-its-kind product",
    "first-of-its-kind service",
    "breakthrough in technology",
    "next-generation service",
    "introduces game-changing product",
    "announces major innovation",
    "launches innovative product",
    "launches groundbreaking service",
    "trailblazing technology",
    "launches cutting-edge solution",
    "disruptive innovation",
    "introduces cutting-edge service",
    "announces innovative breakthrough",
    "launches new generation product",
    "transformational technology",
    "redefines industry standards",
    "introduces advanced solution",
    "technological breakthrough",
    "technology-first innovation",
    "new era in technology",
    "next-level innovation",
]

stock_buyback_title_list = [
    # General Terms
    "stock buyback",
    "share buyback",
    "share repurchase",
    "stock repurchase",
    "repurchase program",
    "buyback program",
    "repurchase plan",
    "buyback plan",
    "authorized buyback",
    "authorizes buyback",
    "authorizes repurchase",
    "repurchase authorization",
    "share repurchase program",
    "share repurchase plan",
    "stock buyback authorization",
    "buyback authorization",
    "stock buyback program",
    "share buyback authorization",

    # Announcements
    "announces buyback",
    "announces share repurchase",
    "announces stock buyback",
    "announces repurchase program",
    "initiates buyback",
    "company initiates buyback",
    "launches buyback program",
    "launches stock repurchase",
    "launches repurchase program",
    "announces share buyback",
    "company announces repurchase",
    "board approves buyback",
    "board authorizes buyback",
    "board announces repurchase",

    # Execution and Progress
    "buying back shares",
    "repurchasing stock",
    "repurchasing shares",
    "commences buyback",
    "begins stock repurchase",
    "commences stock buyback",
    "continues buyback program",
    "expands buyback program",
    "extends repurchase plan",
    "increases buyback authorization",
    "updates stock repurchase program",
    "updates buyback plan",
    "ongoing buyback",

    # Related Financial Terms
    "returns capital to shareholders",
    "boosting shareholder value",
    "capital return program",
    "returns to shareholders",
    "enhancing shareholder returns",
    "enhancing shareholder value",
    "shareholder-focused buyback",
]

stock_split_title_list = [
    # General Terms
    "stock split",
    "share split",
    "splitting shares",
    "forward split",
    "reverse split",
    "split effective",
    "stock split announcement",
    "announced split",
    "announces split",
    "stock split plan",
    "split plan",
    "split decision",
    "stock split program",
    "share split program",
    "stock split ratio",

    # Announcements
    "company announces stock split",
    "announces share split",
    "company announces split",
    "board approves stock split",
    "board announces split",
    "approves stock split",
    "approves share split",
    "company splits shares",
    "shareholders approve stock split",
    "declares stock split",
    "declares share split",
    "board declares split",

    # Split Execution
    "split becomes effective",
    "split date announced",
    "split date set",
    "effective split date",
    "effective date for stock split",
    "stock split takes effect",
    "share split takes effect",
    "shares split on record date",
    "stock split record date",
    "share split record date",
    "effective date announced",
    "split implementation",
    "split ratio confirmed",

    # Forward and Reverse Splits
    "forward stock split",
    "forward share split",
    "reverse stock split",
    "reverse share split",
    "stock split forward",
    "stock split reverse",
    "forward split plan",
    "reverse split plan",
    "reverse split effective",
]

# Content keyword lists
positive_phase_result_content_keywords = [
    "successful Phase", "positive Phase", "positive trial results", "positive topline",
    "announces positive", "announce positive","announces positive data","announce positive data",
    "positive topline results","positive top-line results","presents new pivotal data",
    "successful topline results", "successful top-line results","top-line results", "topline results",
    "pivotal trial data", "improved", "final scheduled review" "positive trial data",
    "positive data", "announces achievement", "plans to submit",
    "meets primary endpoint", "meets secondary endpoint", "achieves primary endpoint",
    "achieves secondary endpoint", "statistically significant", "p-value < 0.05", "clinically meaningful",
    "superior to placebo", "superior efficacy", "treatment effect", "progression-free survival",
    "overall survival", "reduces disease progression", "improved quality of life", "well-tolerated",
    "favorable safety profile", "no serious adverse events", "superior to standard of care",
    "novel therapy", "breakthrough treatment", "regulatory submission planned", "supports BLA submission",
    "NDA submission", "fast track designation", "priority review expected", "orphan drug designation",
    "breakthrough therapy designation", "first-in-class treatment", "potential new treatment option",
    "significant milestone", "compelling data", "strong efficacy results", "key clinical milestone",
    "new drug application"]

negative_phase_result_content_keywords = [
    "failed Phase","negative Phase","negative trial results","does not meet primary endpoint",
    "does not meet secondary endpoint","fails primary endpoint","fails secondary endpoint",
    "not statistically significant","p-value > 0.05","not clinically meaningful","inferior to placebo",
    "inferior efficacy","no significant treatment effect","disease progression continues",
    "no improvement in quality of life","poor safety profile","serious adverse events",
    "high discontinuation rate","inferior to standard of care","lacks efficacy","treatment failure",
    "unexpected safety issues","regulatory submission unlikely","BLA submission delayed",
    "NDA submission delayed","loss of fast track designation","priority review not granted",
    "orphan drug designation revoked","breakthrough therapy designation revoked",
    "not a first-in-class treatment","no significant milestone achieved","disappointing data",
    "weak efficacy results", "failed clinical milestone", "fails in phase", "discontinue phase",
    "discontinues phase 3"
]

fda_approval_content_keywords = [
    "federal drug administration approved",
    "federal drug administration approves",
    "federal drug administration approval",
    "federal drug administration (fda) approved",
    "federal drug administration (fda) approves",
    "federal drug administration (fda) approval",
    "FDA approval","get fda approval", "gets fda approval", "announce fda approval", "approved treatment"
    "announces fda approval", "to receive fda approval","wins fda approval","fda approval call",
    "receives FDA orphan drug designation", "received FDA orphan drug designation",
    "receives FDA approval","approved by the FDA","granted FDA approval","favorable regulatory outcome",
    "receives regulatory approval","BLA approval","NDA approval","approved for marketing",
    "announces clearance", "announce clearance", "fda clearance", "announces approval",
    "regulatory milestone achieved", "first-in-class approval","breakthrough therapy approval",
    "orphan drug approval","priority review approval","accelerated approval","fast track approval",
    "positive FDA decision","green light from FDA","marketing authorization granted",
    "new treatment option approved","FDA clears drug","approved for commercialization",
    "new drug approval","favorable regulatory outcome","EMA and FDA approval",
    "meets all regulatory requirements","to receive FDA approval","officially approved by the FDA",
    "U.S. launch following FDA approval","US launch following FDA approval","fda approved"
]

fda_rejection_content_keywords = [
    "fda rejection",
    "fda rejected",
    "fda denies",
    "fda declined",
    "fda does not approve",
    "fda challenges",
    "federal drug administration rejected",
    "federal drug administration rejection",
    "federal drug administration denies",
    "federal drug administration declined",
    "federal drug administration does not approve",
    "federal drug administration challenges",
    "federal drug administration (fda) rejection",
    "federal drug administration (fda) rejected",
    "federal drug administration (fda) denies",
    "federal drug administration (fda) declined",
    "federal drug administration (fda) does not approve",
    "federal drug administration (fda) challenges",
    "fails to secure FDA approval",
    "fails FDA review","does not meet FDA requirements","FDA raises concerns","CRL issued by FDA",
    "regulatory setback","BLA submission rejected","NDA submission rejected","submission withdrawn",
    "regulatory challenges","approval not granted","fails regulatory standards", "regulatory delay",
    "unfavorable FDA decision","FDA identifies deficiencies","FDA outlines safety concerns",
    "clinical hold issued","drug not approved","application not accepted", "sinks despite fda approval"
]

lawsuit_win_content_keywords = [
    "won lawsuit",
    "won shareholder lawsuit",
    "wins lawsuit",
    "wins shareholder lawsuit",
    "wins dismissal of lawsuit",
    "wins dismissal of shareholder lawsuit",
    "lawsuit victory",
    "lawsuit dismissal",
    "wins legal battle",
    "wins court case",
    "won court case",
    "legal victory",
    "court victory",
    "success in lawsuit",
    "lawsuit resolved in favor",
    "prevails in lawsuit",
    "prevails in court case",
    "dismissal of lawsuit",
    "dismissal of shareholder lawsuit",
    "victorious in lawsuit",
    "victorious in court case",
    "favorable lawsuit outcome",
    "court rules in favor",
    "jury rules in favor",
    "judge rules in favor",
    "court dismisses lawsuit",
    "lawsuit settled favorably",
    "lawsuit outcome favorable",
]


positive_financial_performance_content_keywords = [
    # General Positive Performance
    "financial performance", "financial growth", "quarterly earnings", "earnings beat",
    "earnings surprise", "EPS", "earnings per share", "adjusted EPS", "GAAP EPS",
    "revenue beat", "revenue growth", "year-over-year revenue", "top-line growth",
    "bottom-line growth",

    # Positive Guidance and Projections
    "raises guidance", "updated guidance", "full-year guidance", "fiscal guidance",
    "profit forecast", "profit projection", "revenue guidance", "earnings outlook",
    "guidance upgrade", "exceeds expectations", "surpasses estimates", "forecast adjustment",

    # Positive Financial Metrics
    "operating income", "net income", "gross profit", "gross margin", "net profit",
    "net profit margin", "operating margin", "profit margin", "EBITDA", "adjusted EBITDA",
    "cash flow", "free cash flow", "return on investment", "ROI", "shares surge",
    "stock soars", "stock jumps", "shares increase", "stock rises",

    # Comparative and Market Terms
    "beats estimates", "meets expectations", "positive market reaction", "sector performance",
    "market share", "market growth", "strong demand", "productivity gains",
    "business performance", "operational efficiency"
]

negative_financial_performance_content_keywords = [
    # General Negative Performance
    "earnings miss", "revenue miss", "revenue decline", "below expectations",
    "falls short", "misses estimates", "stock falls", "shares drop", "stock declines",
    "stock plummets", "weak demand",

    # Negative Guidance and Projections
    "lowers guidance", "guidance downgrade", "below expectations", "negative market reaction",

    # Negative Financial Metrics
    "cost-cutting measures", "expense reduction", "cost management", "cost of goods sold",
    "COGS", "sales decline", "financial struggles"
]

market_growth_content_keywords = [
    # General Market Growth and Expansion
    "market growth", "market size", "market expansion", "market forecast", "industry growth",
    "growth projection", "market demand", "surging demand", "increased demand", "expanding market",
    "emerging markets", "global expansion", "industry outlook", "growth trend", "market outlook",
    "market potential", "future growth", "growth in demand", "market value", "rising demand",

    # Specific Markets and Sectors
    "cloud market", "cloud content delivery network", "energy storage", "renewable energy",
    "solar power", "semiconductor industry", "automation market", "laboratory automation",
    "neurostimulation market", "biotechnology market", "pharmaceuticals market", "retail expansion",
    "electric sector growth", "food and beverage expansion", "entertainment industry growth",
    "data center market", "clean energy", "AI data centers", "green energy",

    # Keywords for Emerging Markets and Geographies
    "emerging economies", "emerging markets", "new markets", "global market", "international market",
    "foreign investment", "global reach", "cross-border expansion", "new regions", "regional hub",
    "Asia-Pacific", "APAC", "Europe market", "Latin America", "Africa market", "Gulf region",
    "Middle East", "regional growth", "North America expansion",

    # Terms Related to Business Expansion and Investment
    "new plant", "new facility", "new hub", "expanding operations", "expanding presence",
    "business growth", "facility expansion", "capacity expansion", "new investments",
    "investment plans", "capital investment", "foreign direct investment", "FDI", "new unit",
    "production increase", "new projects", "projected growth", "long-term growth", "scaling up",
    "scale expansion", "scaling operations",

    # Growth Drivers
    "driven by demand", "boosting industry", "growth drivers", "expanding market size",
    "projected to grow", "projected to reach", "market boost", "surging adoption", "industry demand",
    "growth catalysts", "increased adoption", "industry expansion", "driven by adoption",
    "large-scale adoption", "rising prevalence", "boost market", "market opportunity",
    "growth opportunity", "future demand",

    # Specific Financial Targets and Goals
    "projected revenue", "market worth", "USD growth", "revenue target", "billion dollar market",
    "multi-billion dollar market", "investment target", "revenue growth", "financial forecast",
    "market estimate", "growth target", "target market size", "demand estimate",

    # Expansion Indicators and News Terms
    "breaks ground", "new market entry", "new venture", "launching operations", "infrastructure growth",
    "strategic expansion", "breaking ground", "launching projects", "large-scale projects",
    "growth in infrastructure", "groundbreaking", "pipeline projects", "new project start",
    "construction underway", "development phase", "project expansion", "on track for growth"
]

innovation_content_keywords = [
    # General Product Launch and Innovation
    "product launch", "new product", "latest product", "product introduction", "innovative product",
    "new release", "new version", "next generation", "cutting-edge technology", "latest model",
    "technology upgrade", "breakthrough product", "advanced features", "new solution",
    "product debut", "innovation", "technology innovation", "launch announcement", "new lineup",
    "product advancement", "technology evolution", "product upgrade", "enhanced version",
    "next-gen technology", "revamped model", "product offering", "revolutionary product",

    # AI and Generative AI Technologies
    "AI-powered", "artificial intelligence", "generative AI", "AI-driven", "machine learning",
    "deep learning", "natural language processing", "NLP", "computer vision", "predictive analytics",
    "data-driven insights", "AI-enhanced", "AI-based", "AI technology", "smart technology",
    "intelligent solutions", "AI tools", "AI algorithms", "automated intelligence",
    "AI-powered platform", "AI-powered tools", "intelligent automation", "self-learning technology",
    "AI innovation", "AI functionality", "AI application", "AI model",

    # Industry-Specific AI Applications and Trends
    "industry 4.0", "smart manufacturing", "intelligent manufacturing", "automation technology",
    "digital transformation", "AI in manufacturing", "AI in healthcare", "AI in finance",
    "retail AI solutions", "automated advertising", "AI-powered marketing", "AI in marketing",
    "predictive modeling", "AI-powered recommendations", "smart technology", "smart factory",
    "AI in customer experience", "AI for personalization", "AI in advertising",

    # Key Phrases for Tech Adoption and Upgrades
    "technology adoption", "upgraded technology", "enhanced capabilities", "digital innovation",
    "technology adoption trend", "smart technology trends", "digital upgrade", "technology-driven",
    "next-gen technology", "technology adoption growth", "digital journey", "adopting AI",
    "expanding AI capabilities", "smart systems", "technology boost", "intelligent systems",
    "technology-driven solutions",

    # Market Expansion and Reach with New Technology
    "expanding market reach", "new functionality", "global expansion with AI", "entering new markets",
    "regional expansion", "expanding AI", "technology expansion", "increased functionality",
    "global AI rollout", "market with AI capabilities", "AI in new market", "brand reach",
    "technology reach", "technology accessibility", "AI in APAC market", "expanding technology footprint",

    # Specific Financial Goals Related to Innovation
    "market value", "revenue growth through AI", "projected revenue from AI", "multi-billion dollar market",
    "market size projection", "AI market growth", "technology market projection", "target market size",
    "revenue from AI", "market growth in AI technology", "financial impact of AI", "projected revenue boost",

    # Product Launch and AI News Indicators
    "launches AI product", "introduces AI", "new AI capabilities", "announces new AI", "AI integration",
    "AI launch", "AI-powered product launch", "technology launch", "AI debut", "announces AI functionality",
    "AI-powered features", "introduces new AI", "AI-enabled", "technology release", "next-gen AI",
    "AI announcement", "smart technology release", "introduces advanced AI", "AI solution",
    "launches advanced model", "AI-powered release", "AI for enterprise", "AI for consumers",
    "technology debut", "unveils new product"
]

regulatory_content_challenges = [
    # Regulatory and Government Investigations
    "investigation", "regulatory challenge", "regulatory probe", "regulatory scrutiny",
    "under investigation", "ongoing investigation", "government probe", "DOJ investigation",
    "Department of Justice probe", "federal investigation", "state investigation",
    "SEC investigation", "FTC probe", "antitrust investigation", "price-fixing probe",
    "bribery scandal", "fraud investigation", "compliance investigation", "FBI inquiry",
    "regulatory action", "government oversight", "financial misconduct", "compliance issues",
    "fraud allegations", "SEC filing", "trade commission", "regulatory review",
    "watchdog review", "investigative review", "regulatory hurdles", "scrutiny by regulators",
    "under regulatory scrutiny", "privacy watchdog", "compliance breach", "policy violation",

    # Penalties and Fines
    "penalties", "fines", "financial penalty", "monetary fines", "settlement costs",
    "settlement payout", "financial settlement", "compensation payout", "restitution",
    "regulatory fines", "enforcement actions", "penalty fees", "hefty fine", "sanctions",
    "compliance fine", "restitution payment", "penalized", "financial penalties", "punitive damages",
    "liabilities", "settlement charges", "charges filed", "penalty settlement", "fined for",
    "compensation payment", "reimbursement",

    # Specific Types of Legal and Regulatory Cases
    "price-fixing", "bribery scandal", "fraud charges", "antitrust lawsuit", "intellectual property lawsuit",
    "patent infringement", "privacy breach", "cybersecurity breach", "data privacy violation",
    "discrimination lawsuit", "labor dispute", "wrongful termination case", "insider trading",
    "false advertising", "false claims", "environmental violation", "financial misconduct",
    "whistleblower lawsuit", "fraud investigation", "unfair competition", "criminal charges",
    "corruption charges", "data breach lawsuit", "privacy lawsuit", "intellectual property dispute",

    # International Regulatory and Legal Terms
    "European Union scrutiny", "EU investigation", "foreign regulatory probe", "cross-border case",
    "international lawsuit", "global regulatory compliance", "international regulations",
    "cross-border regulatory compliance", "international legal challenges", "foreign sanctions",
    "foreign investigation", "international oversight", "international penalties",
    "global regulatory scrutiny", "foreign trade commission", "international legal review",

    # Government and Agency Names Related to Investigations
    "Department of Justice", "DOJ", "Federal Trade Commission", "FTC", "Securities and Exchange Commission",
    "SEC", "Environmental Protection Agency", "EPA", "Federal Bureau of Investigation", "FBI",
    "Office of Foreign Assets Control", "OFAC", "Financial Conduct Authority", "FCA",
    "European Commission", "European Union regulators", "privacy watchdog", "trade commission",
    "consumer protection agency", "competition authority", "Justice Department", "regulatory body",
    "government agency"
]

analyst_upgrade_content_keywords = [
    # General Upgrade Terms
    "analyst upgrade", "upgraded to", "raises rating", "revised rating", "upgraded by",
    "boosts rating", "increases rating", "raises recommendation", "positive rating",
    "bullish outlook", "raised outlook", "market outperform", "outperform rating",
    "buy rating", "strong buy", "positive outlook", "target increase", "raises price target",
    "price target change", "target boost", "upward revision", "adjusted price target",
    "new price target", "boosts price target", "target increase", "price forecast",
]

analyst_downgrade_keywords = [
    # General Downgrade Terms
    "analyst downgrade", "downgraded to", "lowers rating", "revised rating", "downgraded by",
    "cuts rating", "reduces rating", "lowers recommendation", "negative rating",
    "bearish outlook", "lowered outlook", "market underperform", "underperform rating",
    "sell rating", "target cut", "lowers price target", "target reduction", "downward revision",
    "price target change", "cuts price target", "target reduction", "negative outlook",
]

mergers_acquisitions_keyword_content_list = [
    # General Deal and Acquisition Terms
    "acquisition", "merger", "merger and acquisition", "M&A", "acquires", "acquired by",
    "to acquire", "stake purchase", "takeover", "buyout", "purchase deal", "deal finalized",
    "deal agreement", "deal announcement", "deal closing", "deal worth", "asset purchase",
    "strategic acquisition", "acquisition agreement", "business acquisition", "asset acquisition",

    # Partnership and Collaboration Terms
    "partnership", "collaboration", "strategic partnership", "strategic alliance", "joint venture",
    "alliance with", "partners with", "team up with", "strategic relationship", "collaborative deal",
    "partnership deal", "global partnership", "technology partnership", "business alliance",
    "partnership agreement", "partnering with", "expands partnership", "enhances partnership",

    # Stake and Equity Terms
    "stake in", "stake acquisition", "equity purchase", "equity stake", "share purchase",
    "ownership stake", "majority stake", "minority stake", "equity investment", "takes stake in",
    "increased stake", "buying stake", "sells stake", "stake sale", "acquires stake", "acquiring stake",

    # Market Expansion and Reach Terms
    "market expansion", "market entry", "expanding reach", "global expansion", "regional expansion",
    "market presence", "expanding footprint", "geographic expansion", "international expansion",
    "new market entry", "expanding operations", "strengthening market position", "expanding portfolio",
    "increased market presence", "enhancing reach", "business expansion", "global footprint",

    # Financial and Deal Structure Terms
    "deal valued at", "deal worth", "deal amount", "worth $", "deal value", "multi-million dollar deal",
    "billion dollar acquisition", "valuation", "financial terms", "purchase amount", "buyout cost",
    "transaction value", "transaction amount", "deal size", "purchase price", "total value",
    "purchase consideration", "acquisition price",

    # Industry-Specific M&A Terms
    "technology acquisition", "energy sector deal", "real estate acquisition", "healthcare acquisition",
    "pharma merger", "insurance deal", "financial services acquisition", "retail acquisition",
    "telecom merger", "automotive merger", "energy acquisition", "oil and gas acquisition",
    "renewable energy acquisition", "banking merger", "technology merger", "cloud technology acquisition",

    # Regulatory and Legal Terms Related to M&A
    "merger approval", "regulatory approval", "antitrust review", "antitrust approval",
    "government approval", "regulatory clearance", "subject to regulatory approval", "FTC review",
    "DOJ review", "competition review", "merger conditions", "merger regulation", "legal approval",
    "subject to approval", "regulatory requirements", "clearance required",

    # Specific Action Words Indicating Deals
    "agrees to buy", "agreed to acquire", "signs deal", "enters into agreement", "announces acquisition",
    "to be acquired by", "closes acquisition", "completes acquisition", "completes merger",
    "finalizes deal", "finalizes acquisition", "consummates merger", "announces merger",
    "to merge with", "agrees to purchase", "acquisition bid", "offer to acquire", "signs partnership",
    "enters partnership", "forges alliance", "builds partnership", "forms partnership"
]


layoffs_keyword_content_list = [
    # General Layoff and Job Cut Terms
    "layoffs", "job cuts", "redundancies", "workforce reduction", "staff reduction",
    "downsizing", "job losses", "cutting jobs", "job reductions", "employee reduction",
    "employee layoffs", "staff layoffs", "reducing workforce", "reducing staff",
    "reducing headcount", "laying off workers", "laying off employees", "job terminations",
    "reducing workforce size", "downsizing operations", "headcount reduction",

    # Terms Related to Restructuring and Reorganization
    "restructuring", "organizational restructuring", "corporate restructuring",
    "business restructuring", "restructuring plan", "strategic restructuring",
    "organizational changes", "reorganization", "restructuring effort",
    "business reorganization", "restructuring initiative", "internal reorganization",
    "reorganizing workforce", "streamlining operations", "cost-cutting measures",
    "operational changes",

    # Specific Words for Layoffs and Firing
    "furloughs", "dismissals", "terminations", "job furlough", "letting go of staff",
    "firing workers", "firing employees", "letting go of employees", "letting go of workers",
    "staff dismissal", "job separation", "staff termination", "employee termination",
    "workforce dismissal", "job cessation", "ceasing employment", "employee separation",

    # Economic and Financial Drivers of Layoffs
    "economic downturn", "financial struggles", "cost reduction", "cutting expenses",
    "budget cuts", "revenue decline", "revenue drop", "decline in sales", "reduced demand",
    "profit decline", "financial losses", "fall in demand", "economic challenges",
    "macroeconomic impact", "slowdown in demand", "economic impact", "revenue shortfall",

    # Company-Specific Layoff Terms
    "third round of layoffs", "second round of layoffs", "additional layoffs", "more layoffs",
    "further layoffs", "recent round of layoffs", "new layoffs", "announces layoffs",
    "mass layoffs", "mass job cuts", "companywide layoffs", "significant layoffs",
    "large-scale layoffs", "major layoffs", "major job cuts", "layoff announcement",
    "mass workforce reduction", "large workforce cut", "wave of layoffs",

    # Specific Phrasing for Layoff Announcements
    "affected employees", "reduction in force", "reducing headcount", "cutting workforce",
    "eliminating positions", "elimination of roles", "eliminating jobs", "reducing positions",
    "trimming workforce", "downsizing team", "workforce adjustments", "headcount adjustments",
    "adjusting workforce", "reduced staffing", "scaled-back operations", "staffing changes",
    "adjustments to workforce",

    # Contextual Terms Indicating Economic Challenges
    "agriculture downturn", "technology sector slowdown", "market slowdown",
    "decline in sector", "slowing market demand", "economic pressures",
    "challenging market conditions", "market pressures", "economic headwinds",
    "sector downturn", "industry downturn", "shift in market demand", "changes in demand"
]

stock_buyback_content_keywords = [
    "stock buyback",
    "share repurchase",
    "repurchase program"
    "buying back shares",
    "repurchasing stock",
    "authorized buyback",
    "authorizes repurchase",
    "repurchase plan",
    "repurchase authorization"
    # General Stock Buyback Terms
    "buyback", "stock buyback", "share buyback", "buyback plan",
    "share repurchase", "stock repurchase", "buyback authorization", "repurchase authorization",
    "buyback announcement", "buyback initiative", "buyback strategy", "buyback program",
    "authorized buyback", "authorized repurchase", "approved buyback", "approved repurchase",

    # Terms Related to Buyback Value and Volume
    "announces buyback", "announces repurchase", "launches buyback", "initiates buyback",
    "initiates repurchase", "declares buyback", "approves buyback", "approves repurchase",
    "buyback worth", "repurchase worth", "buyback valued at", "repurchase valued at",
    "$ buyback", "$ repurchase", "million buyback", "billion buyback", "million repurchase",
    "billion repurchase", "share reduction", "reducing share count", "reducing shares outstanding",

    # Specific Buyback Action Phrases
    "to buy back shares", "to repurchase shares", "plans to buy back", "plans to repurchase",
    "intends to buy back", "intends to repurchase", "boosts buyback", "expands buyback",
    "increases buyback", "extends buyback program", "enhances repurchase program",
    "increased buyback", "accelerated buyback", "accelerated repurchase", "shareholder return",
    "returning value to shareholders", "capital return", "shareholder distribution",

    # Contextual Terms Related to Buybacks
    "authorized stock buyback", "approved stock repurchase", "launches stock buyback",
    "board approved buyback", "board authorized buyback", "under buyback program",
    "in buyback program", "within repurchase program", "through buyback plan",
    "buying back shares", "repurchasing shares", "additional buyback", "expanded buyback program"
]


stock_split_content_keywords = [
    "stock split", "share split", "splitting shares", "forward stock split",
    "reverse stock split", "split adjustment", "shares outstanding increase",
    "shares outstanding decrease", "post-split price", "pre-split price",
    "effective date of split", "adjusted share price", "dividing shares",
    "multiplying shares", "split announcement", "proposed stock split",
    "approved stock split", "split plan", "shareholder approval for split"
]


fundraising_content_keywords = [
    # General Fundraising Terms
    "fundraising", "capital raising", "capital raise", "fundraising round", "funding round",
    "raised funds", "fundraising success", "successful funding", "capital injection",
    "funding success", "capital secured", "capital obtained", "secured funding", "funding closed",
    "capital acquisition", "financing success", "investment round", "successful fundraising",

    # Terms Related to Amounts Raised and Valuations
    "raises $", "raised $", "seeks $", "targets $", "million raised", "billion raised",
    "million valuation", "billion valuation", "valuation of $", "up to $", "targets valuation of",
    "capital valued at", "project financing", "equity financing", "debt financing", "mezzanine financing",
    "capital amount", "financing amount", "EUR raised", "USD raised", "million-dollar financing",
    "funding amount", "funds secured", "valuation increase", "achieves valuation", "high valuation",

    # IPO and Public Offering Terms
    "IPO", "initial public offering", "public offering", "seeks IPO", "files for IPO",
    "IPO valuation", "IPO plans", "IPO success", "successful IPO", "seeks public listing",
    "public market entry", "IPO targets", "IPO filing", "IPO valued at", "NYSE IPO", "Nasdaq IPO",
    "public listing", "market debut", "trading debut", "successful public offering", "IPO launch",
    "raised in IPO", "IPO capital",

    # Venture Capital and Series Funding Terms
    "seed funding", "seed round", "Series A", "Series B", "Series C", "Series D",
    "venture funding", "venture capital", "VC funding", "series funding", "secured funding round",
    "investment round", "round of funding", "funding series", "growth funding", "growth equity",
    "early-stage funding", "late-stage funding", "mezzanine round", "angel funding", "private equity",

    # Project Financing and Investment
    "project financing", "project funding", "infrastructure financing", "financing success",
    "project capital", "development financing", "investment secured", "investment success",
    "capital for expansion", "secured project financing", "expansion financing",
    "project funding round", "strategic funding", "investment closed", "financing for project",

    # Investment and Investor-Related Terms
    "investment in", "secured investment", "raised investment", "investment success",
    "backed by investors", "financing from investors", "funding led by", "round led by",
    "supported by investors", "funds from investors", "investor capital", "funding participation",
    "investment amount", "investor support", "investment for growth", "backed by venture capital",
    "secured capital from investors", "financing from institutional investors", "equity investment",

    # Terms Indicating Success and Growth Potential
    "successful capital raise", "growth financing", "expansion capital", "capital to grow",
    "capital for expansion", "growth investment", "strategic investment", "financing for growth",
    "funds to expand", "accelerates growth", "enables expansion", "capital injection for growth",
    "fueling expansion", "growth capital", "expansion financing", "boosts valuation",
    "achieves high valuation", "capital for new projects", "capitalizing on growth"
]


combined_topic_content_keywords = {
    NewsTopic.POSITIVE_FINANCIAL_PERFORMANCE: positive_financial_performance_content_keywords,
    NewsTopic.MARKET_GROWTH: market_growth_content_keywords,
    NewsTopic.INNOVATION: innovation_content_keywords,
    NewsTopic.STOCK_BUYBACK: stock_buyback_content_keywords,
    NewsTopic.STOCK_SPLIT: stock_split_content_keywords,
    NewsTopic.LAWSUIT_WIN: lawsuit_win_content_keywords,
    NewsTopic.POSITIVE_PHASE2_RESULT: positive_phase_result_content_keywords,
    NewsTopic.NEGATIVE_PHASE2_RESULT: negative_phase_result_content_keywords,
    NewsTopic.POSITIVE_PHASE3_RESULT: positive_phase_result_content_keywords,
    NewsTopic.NEGATIVE_PHASE3_RESULT: negative_phase_result_content_keywords,
    NewsTopic.FDA_APPROVAL: fda_approval_content_keywords,
    NewsTopic.FDA_REJECTION: fda_rejection_content_keywords,
}

combined_topic_title_keywords = {
    NewsTopic.POSITIVE_FINANCIAL_PERFORMANCE: positive_financial_performance_title_list,
    NewsTopic.NEGATIVE_FINANCIAL_PERFORMANCE: negative_financial_performance_title_list,
    NewsTopic.MARKET_GROWTH: market_growth_title_list,
    NewsTopic.INNOVATION: innovation_title_list,
    NewsTopic.STOCK_BUYBACK: stock_buyback_title_list,
    NewsTopic.STOCK_SPLIT: stock_split_title_list,
    NewsTopic.LAWSUIT_WIN: lawsuit_win_title_list,
    NewsTopic.POSITIVE_PHASE2_RESULT: phase2_title_list,
    NewsTopic.NEGATIVE_PHASE2_RESULT: phase2_title_list,
    NewsTopic.POSITIVE_PHASE3_RESULT: phase3_title_list,
    NewsTopic.NEGATIVE_PHASE3_RESULT: phase3_title_list,
    NewsTopic.FDA_APPROVAL: fda_approval_title_list,
    NewsTopic.FDA_REJECTION: fda_rejection_title_list,
}


