from typing import List, Dict
from datetime import datetime

from ..models.review import ReviewInput, DetectionResult
from .features import FeatureExtractor

class FakeReviewDetector:
    """Core fake review detection logic"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model_version = "1.0.0"
        
        # Rule-based scoring weights (can be tuned)
        self.weights = {
            'text_quality': 0.25,
            'sentiment_analysis': 0.20,
            'reviewer_behavior': 0.25,
            'pattern_matching': 0.30
        }
        
    def analyze_review(self, review: ReviewInput) -> DetectionResult:
        """Analyze a single review for fake indicators"""
        
        # Extract features
        features = self.feature_extractor.extract_all_features(
            review.text,
            review.rating,
            review.reviewer_total_reviews or 0,
            review.reviewer_account_age_days or 0,
            review.review_date
        )
        
        # Check for fake patterns
        pattern_flags = self.feature_extractor.check_fake_patterns(review.text)
        
        # Calculate component scores
        text_score = self._calculate_text_quality_score(features)
        sentiment_score = self._calculate_sentiment_score(features)
        reviewer_score = self._calculate_reviewer_behavior_score(features)
        pattern_score = self._calculate_pattern_score(pattern_flags, features)
        
        # Weighted final score
        fake_probability = (
            self.weights['text_quality'] * text_score +
            self.weights['sentiment_analysis'] * sentiment_score +
            self.weights['reviewer_behavior'] * reviewer_score +
            self.weights['pattern_matching'] * pattern_score
        )
        
        # Collect all flags
        all_flags = pattern_flags + self._get_behavior_flags(features)
        
        # Generate explanation
        explanation = self._generate_explanation(
            fake_probability, text_score, sentiment_score, 
            reviewer_score, pattern_score, all_flags
        )
        
        return DetectionResult(
            is_fake=fake_probability > 0.6,  # Threshold for classification
            confidence_score=fake_probability * 100,
            fake_probability=fake_probability,
            flags=all_flags,
            explanation=explanation
        )
    
    def analyze_batch(self, reviews: List[ReviewInput]) -> List[DetectionResult]:
        """Analyze multiple reviews"""
        return [self.analyze_review(review) for review in reviews]
    
    def _calculate_text_quality_score(self, features: Dict) -> float:
        """Calculate text quality suspicion score (0-1, higher = more suspicious)"""
        score = 0.0
        
        # Length-based indicators
        if features['text_length'] < 20:
            score += 0.3  # Too short
        elif features['text_length'] > 2000:
            score += 0.2  # Unusually long
            
        # Repetition indicators
        if features['repeated_words'] > 3:
            score += 0.2
            
        if features['repeated_chars'] > 0:
            score += 0.1
            
        # Grammar/style indicators
        if features['capitalization_ratio'] > 0.15:  # Too much CAPS
            score += 0.2
            
        if features['exclamation_count'] > 3:
            score += 0.15
            
        # Sentence structure
        if features['avg_sentence_length'] < 3:  # Too simple
            score += 0.1
        elif features['avg_sentence_length'] > 30:  # Too complex
            score += 0.1
            
        return min(score, 1.0)
    
    def _calculate_sentiment_score(self, features: Dict) -> float:
        """Calculate sentiment-based suspicion score"""
        score = 0.0
        
        # Extreme sentiment often indicates fake reviews
        if features['is_extreme_sentiment']:
            score += 0.4
            
        # Sentiment-rating mismatch
        if features['high_mismatch']:
            score += 0.3
            
        # Very extreme ratings
        if features['is_extreme_rating']:
            score += 0.2
            
        return min(score, 1.0)
    
    def _calculate_reviewer_behavior_score(self, features: Dict) -> float:
        """Calculate reviewer behavior suspicion score"""
        score = 0.0
        
        # New accounts are more suspicious
        if features['is_new_reviewer']:
            score += 0.3
            
        # Very active reviewers might be bots
        if features['reviews_per_day'] > 2:
            score += 0.4
        elif features['reviews_per_day'] > 1:
            score += 0.2
            
        # Very few reviews also suspicious
        if features['reviewer_total_reviews'] == 1:
            score += 0.2
        elif features['reviewer_total_reviews'] < 5:
            score += 0.1
            
        return min(score, 1.0)
    
    def _calculate_pattern_score(self, pattern_flags: List[str], features: Dict) -> float:
        """Calculate pattern-based suspicion score"""
        base_score = len(pattern_flags) * 0.2
        
        # Additional pattern-based scoring
        if 'excessive_positive_words' in pattern_flags:
            base_score += 0.1
            
        if 'excessive_exclamation' in pattern_flags:
            base_score += 0.1
            
        return min(base_score, 1.0)
    
    def _get_behavior_flags(self, features: Dict) -> List[str]:
        """Get behavioral red flags"""
        flags = []
        
        if features['is_new_reviewer']:
            flags.append("new_account")
            
        if features['reviews_per_day'] > 2:
            flags.append("high_review_frequency")
            
        if features['high_mismatch']:
            flags.append("sentiment_rating_mismatch")
            
        if features['is_extreme_sentiment']:
            flags.append("extreme_sentiment")
            
        if features['capitalization_ratio'] > 0.15:
            flags.append("excessive_caps")
            
        if features['text_length'] < 20:
            flags.append("very_short_text")
            
        return flags
    
    def _generate_explanation(self, fake_prob: float, text_score: float, 
                            sentiment_score: float, reviewer_score: float, 
                            pattern_score: float, flags: List[str]) -> str:
        """Generate human-readable explanation"""
        
        if fake_prob < 0.3:
            base = "This review appears authentic"
        elif fake_prob < 0.6:
            base = "This review shows some suspicious indicators"
        else:
            base = "This review appears likely to be fake"
            
        reasons = []
        
        if text_score > 0.4:
            reasons.append("poor text quality")
        if sentiment_score > 0.4:
            reasons.append("suspicious sentiment patterns")
        if reviewer_score > 0.4:
            reasons.append("questionable reviewer behavior")
        if pattern_score > 0.4:
            reasons.append("matches known fake review patterns")
            
        if reasons:
            explanation = f"{base} due to: {', '.join(reasons)}."
        else:
            explanation = f"{base}."
            
        if flags:
            top_flags = flags[:3]  # Show top 3 flags
            explanation += f" Key indicators: {', '.join(top_flags)}."
            
        return explanation
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            "model_version": self.model_version,
            "features_used": [
                "text_quality", "sentiment_analysis", 
                "reviewer_behavior", "pattern_matching"
            ],
            "accuracy_metrics": {
                "precision": 0.82,  # Placeholder - would be from actual testing
                "recall": 0.78,
                "f1_score": 0.80
            },
            "last_updated": datetime.now()
        }