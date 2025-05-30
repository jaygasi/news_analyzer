# biomed_analyzer/analyzer/models.py
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
import re

class StockPrediction(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    confidence: int = Field(..., ge=1, le=10, description="Confidence score from 1 to 10")
    predicted_movement: Literal["UP", "DOWN", "NEUTRAL"] = Field(..., description="Predicted stock movement")
    reasoning: str = Field(..., description="Main reasoning points for the predicted stock movement")
    news_headline: str = Field(..., description="Headline of the news article analyzed")
    news_url: Optional[str] = Field(None, description="URL of the news article")
    news_source: Optional[str] = Field(None, description="Source of the news article")

class LLMAnalysisResult(BaseModel):
    predicted_movement: Literal["UP", "DOWN", "NEUTRAL"]
    confidence: int
    reasoning: str

    @validator('confidence', pre=True, always=True)
    def clean_confidence_from_str(cls, v):
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                match = re.search(r'\b(\d+)\b', v)
                if match:
                    return int(match.group(1))
                raise ValueError(f"Confidence string '{v}' could not be converted to int")
        if not isinstance(v, int):
             raise ValueError(f"Confidence value '{v}' is not an integer")
        return v

    @validator('confidence')
    def confidence_must_be_in_range(cls, v):
        if not (1 <= v <= 10):
            # Instead of raising error, clamp it, as LLM might sometimes go out of spec.
            # logger.warning(f"Confidence {v} out of range, clamping to 1-10.") # Can't use logger here easily
            return max(1, min(10, v))
        return v