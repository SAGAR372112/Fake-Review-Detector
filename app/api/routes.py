from fastapi import APIRouter, HTTPException
import time
from datetime import datetime

from ..models.review import (
    ReviewInput, DetectionResult, BatchReviewInput, 
    BatchDetectionResult, ModelInfo
)
from ..services.detector import FakeReviewDetector

router = APIRouter()
detector = FakeReviewDetector()

@router.post("/analyze/single", response_model=DetectionResult)
async def analyze_single_review(review: ReviewInput):
    """
    Analyze a single review for fake indicators
    
    Returns detection result with confidence score and explanation
    """
    try:
        start_time = time.time()
        result = detector.analyze_review(review)
        processing_time = time.time() - start_time
        
        # Add processing time to response (useful for monitoring)
        result.explanation += f" (Analysis took {processing_time:.3f}s)"
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/batch", response_model=BatchDetectionResult)
async def analyze_batch_reviews(batch_input: BatchReviewInput):
    """
    Analyze multiple reviews in batch
    
    Maximum 100 reviews per request for performance
    """
    try:
        start_time = time.time()
        
        if len(batch_input.reviews) > 100:
            raise HTTPException(
                status_code=400, 
                detail="Too many reviews. Maximum 100 reviews per batch."
            )
        
        results = detector.analyze_batch(batch_input.reviews)
        processing_time = time.time() - start_time
        
        # Calculate summary statistics
        total_reviews = len(results)
        fake_count = sum(1 for r in results if r.is_fake)
        avg_confidence = sum(r.confidence_score for r in results) / total_reviews
        
        summary = {
            "total_reviews": total_reviews,
            "fake_reviews_detected": fake_count,
            "fake_percentage": (fake_count / total_reviews) * 100,
            "average_confidence": round(avg_confidence, 2),
            "processing_time_seconds": round(processing_time, 3),
            "reviews_per_second": round(total_reviews / processing_time, 2)
        }
        
        return BatchDetectionResult(results=results, summary=summary)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the current detection model
    """
    try:
        model_data = detector.get_model_info()
        return ModelInfo(**model_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": detector.model_version,
        "service": "fake-review-detector"
    }

@router.post("/analyze/quick")
async def quick_analyze(text: str, rating: int):
    """
    Quick analysis endpoint for simple text + rating input
    
    Useful for testing and simple integrations
    """
    try:
        if not (1 <= rating <= 5):
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
            
        review = ReviewInput(text=text, rating=rating)
        result = detector.analyze_review(review)
        
        return {
            "is_fake": result.is_fake,
            "confidence": result.confidence_score,
            "summary": result.explanation.split('.')[0],  # First sentence only
            "top_flags": result.flags[:3]  # Top 3 flags
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")

@router.get("/stats/demo")
async def get_demo_stats():
    """
    Demo endpoint showing some fake review statistics
    
    In a real application, this would query actual database
    """
    return {
        "platform_stats": {
            "total_reviews_analyzed": 50000,
            "fake_reviews_detected": 8500,
            "fake_percentage": 17.0,
            "accuracy_rate": 82.5
        },
        "common_fake_indicators": [
            "Excessive positive language",
            "New reviewer accounts", 
            "Sentiment-rating mismatch",
            "Generic/template-like text",
            "Suspicious timing patterns"
        ],
        "detection_performance": {
            "avg_processing_time_ms": 45,
            "throughput_per_minute": 1300,
            "false_positive_rate": 12.0,
            "false_negative_rate": 8.0
        }
    }