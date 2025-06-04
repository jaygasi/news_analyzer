from utils.utils import get_logger, clean_text_for_analysis
from config import (
    GEMINI_API_KEY, OPENAI_API_KEY, CLAUDE_API_KEY, GROK_API_KEY, LLAMA_API_KEY,
    ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY, TIINGO_API_KEY,
    LLM_PROMPT_TEMPLATE
)
from .llm_clients.base_llm_client import BaseLLMClient
from .llm_clients.finbert_client import FinBERTClient
from .llm_clients.gemini_client import GeminiClient
from .llm_clients.openai_client import OpenAIClient
from .llm_clients.claude_client import ClaudeClient
from utils.exceptions import LLMAPIError

logger = get_logger(__name__)

class MultiLLMAnalyzer:
    """
    Orchestrates multiple AI services (LLMs and traditional APIs) for news analysis.
    Prioritizes local models/cheaper APIs, with fallbacks to more expensive ones.
    Focuses on directional impact rather than sentiment.
    """
    def __init__(self):
        # Initialize actual LLM clients
        self.llm_clients: dict[str, BaseLLMClient] = {
            "finbert": FinBERTClient(),
            "gemini": GeminiClient(api_key=GEMINI_API_KEY, prompt_template=LLM_PROMPT_TEMPLATE),
            "openai": OpenAIClient(api_key=OPENAI_API_KEY, prompt_template=LLM_PROMPT_TEMPLATE),
            "claude": ClaudeClient(api_key=CLAUDE_API_KEY, prompt_template=LLM_PROMPT_TEMPLATE),
            # Add GrokClient, LLaMAClient here if implemented
        }
        # Filter out unconfigured clients immediately
        self.llm_clients = {name: client for name, client in self.llm_clients.items() if client.is_configured}

        self.llm_priority = ["finbert", "gemini", "openai", "claude"] # Order of preference
        # Ensure only configured clients are in priority list
        self.llm_priority = [client_name for client_name in self.llm_priority if client_name in self.llm_clients]

        # Placeholder for non-LLM API clients (e.g., Alpha Vantage, Polygon, Tiingo)
        # These would ideally have their own client classes similar to LLM clients.
        self.non_llm_fallback_clients = {
            # "alphavantage": self._init_alphavantage(), # Still simulated for brevity
            # "polygon": self._init_polygon(),
            # "tiingo": self._init_tiingo(),
        }
        logger.info(f"Initialized MultiLLMAnalyzer with configured LLMs: {list(self.llm_clients.keys())}")


    def _analyze_with_non_llm_api(self, client_name: str, text: str, ticker: str) -> tuple[float, str]:
        """
        Generic function to call a non-LLM API (e.g., Alpha Vantage, Polygon)
        for sentiment proxies or simplified directional analysis.
        These are typically not as robust as LLMs for deep semantic understanding.
        """
        # if not self.non_llm_fallback_clients.get(client_name): # Check if configured
        #     return 0.0, f"API client '{client_name}' not available."

        logger.debug(f"Attempting analysis with non-LLM API {client_name} for {ticker} (SIMULATED)...")
        
        # Simulate non-LLM API response (very basic)
        simulated_scores = {
            "alphavantage": 0.1, # Could provide basic sentiment or news impact
            "polygon": -0.2,
            "tiingo": 0.05,
        }
        
        score = simulated_scores.get(client_name, 0.0)
        reason = f"Simulated {client_name} news impact analysis."
        
        return score, reason


    def analyze_news_with_llms(self, article: dict, ticker: str) -> tuple[float, str, str]:
        """
        Analyzes a single news article using a prioritized chain of LLMs.
        Returns the best score, reason, and the LLM used.
        """
        content = article.get('content') or article.get('text') or article.get('title', '')
        title = article.get('title', 'No Title')
        
        if not content:
            return 0.0, "No content for AI analysis.", "N/A"

        # Try LLMs in priority order
        for llm_name in self.llm_priority:
            client = self.llm_clients.get(llm_name)
            if client and client.is_configured:
                try:
                    score, reason = client.analyze_text(text=content, ticker=ticker, title=title)
                    if score is not None and reason:
                        logger.debug(f"Analysis successful with {llm_name} for {ticker}: Score={score:.2f}, Reason='{reason}'")
                        return score, reason, llm_name
                except LLMAPIError as e:
                    logger.warning(f"LLM API error for {llm_name} on {ticker}: {e}. Trying next LLM.")
                except Exception as e:
                    logger.error(f"Unexpected error calling {llm_name} for {ticker}: {e}. Trying next LLM.")
        
        # Fallback to non-LLM APIs if all LLMs fail or are not configured
        # This section is still largely simulated for brevity in this interaction.
        # It would need dedicated client implementations for each API.
        for api_name in ["alphavantage", "polygon", "tiingo"]: # Example fallback order
            # if self.non_llm_fallback_clients.get(api_name):
            try:
                score, reason = self._analyze_with_non_llm_api(api_name, content, ticker)
                if score is not None and reason:
                    logger.debug(f"Analysis successful with {api_name} (fallback) for {ticker}: Score={score:.2f}, Reason='{reason}'")
                    return score, reason, api_name
            except Exception as e:
                logger.error(f"Error calling {api_name} (fallback) for {ticker}: {e}")

        logger.warning(f"No AI/API analysis performed for {ticker} due to missing configurations or errors.")
        return 0.0, "No AI/API analysis available.", "N/A"