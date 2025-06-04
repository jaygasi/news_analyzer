from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from utils.utils import get_logger
from utils.exceptions import LLMAPIError

logger = get_logger(__name__)

class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients."""

    def __init__(self, api_key: str, prompt_template: str):
        if not api_key or api_key.startswith("YOUR_"):
            logger.warning(f"API key for {self.__class__.__name__} is not set. This client will be skipped.")
            self.is_configured = False
        else:
            self.api_key = api_key
            self.prompt_template = prompt_template
            self.is_configured = True

    @abstractmethod
    def analyze_text(self, text: str, ticker: str, title: str) -> Tuple[float, str]:
        """
        Analyzes the given text for directional impact.
        Returns a directional score (-1.0 to 1.0) and a reason.
        Raises LLMAPIError on API failures.
        """
        pass

    def _prepare_prompt(self, text: str, ticker: str, title: str) -> str:
        """Fills the prompt template with relevant data."""
        # Truncate content if too long for typical LLM context windows
        max_content_length = 3000 # Adjust based on LLM's context window
        truncated_content = text[:max_content_length] + ("..." if len(text) > max_content_length else "")
        
        return self.prompt_template.format(
            ticker=ticker,
            title=title,
            content=truncated_content
        )

    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        """
        Parses the LLM's raw text response into a directional score and reason.
        Expects a JSON object with 'direction', 'confidence', 'reason'.
        """
        try:
            import json
            response_json = json.loads(response_text)
            
            direction_map = {"LONG": 1.0, "SHORT": -1.0, "NEUTRAL": 0.0}
            
            direction = response_json.get("direction", "NEUTRAL").upper()
            confidence = float(response_json.get("confidence", 0.0))
            reason = response_json.get("reason", "No specific reason provided.")

            base_score = direction_map.get(direction, 0.0)
            directional_score = base_score * confidence # Weight score by confidence

            return directional_score, reason
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}. Response: {response_text[:200]}")
            raise LLMAPIError("Failed to parse JSON response from LLM.") from e
        except ValueError as e:
            logger.error(f"Invalid data in LLM JSON response: {e}. Response: {response_text[:200]}")
            raise LLMAPIError("Invalid data in LLM JSON response.") from e
        except KeyError as e:
            logger.error(f"Missing expected key in LLM JSON response: {e}. Response: {response_text[:200]}")
            raise LLMAPIError(f"Missing expected key '{e}' in LLM JSON response.") from e