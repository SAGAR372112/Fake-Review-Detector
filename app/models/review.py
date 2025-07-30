from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ReviewInput(BaseModel):
    """Input model for review analysis"""
    text: str = Field(..., min_length=1, max_length=5000, description="Review text content")
    rating: int = Field(..., ge=1, le=5, description="Rating given (1-5 stars)")
    reviewer_id: Optional[str] = Field(None, description="Unique reviewer identifier")
    product_id: Optional[str] = Field(None, description="Product/business identifier")
    review_date: Optional[datetime] = Field(None, description="When review was posted")
    reviewer_total_reviews: Optional[int] = Field(0, description="Total reviews by this reviewer")
    reviewer_account_age_days: Optional[int] = Field(0, description="Reviewer account age in days")

class DetectionResult(BaseModel):
    """Result of fake review detection"""
    is_fake: bool = Field(description="Whether review is likely fake")
    confidence_score: float = Field(ge=0, le=100, description="Confidence score (0-100)")
    fake_probability: float = Field(ge=0, le=1, description="Probability of being fake (0-1)")
    flags: List[str] = Field(default=[], description="Specific red flags detected")
    explanation: str = Field(description="Human-readable explanation")

class BatchReviewInput(BaseModel):
    """Input for batch review analysis"""
    reviews: List[ReviewInput] = Field(..., max_items=100, description="List of reviews to analyze")

class BatchDetectionResult(BaseModel):
    """Result of batch review detection"""
    results: List[DetectionResult] = Field(description="Detection results for each review")
    summary: dict = Field(description="Summary statistics")

class ModelInfo(BaseModel):
    """Information about the detection model"""
    model_version: str = Field(description="Current model version")
    features_used: List[str] = Field(description="Features considered in detection")
    accuracy_metrics: dict = Field(description="Model performance metrics")
    last_updated: datetime = Field(description="When model was last updated")