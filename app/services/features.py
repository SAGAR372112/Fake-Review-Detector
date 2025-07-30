import re
import nltk # type: ignore
from typing import Dict, List
from datetime import datetime, timedelta
import numpy as np

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment import SentimentIntensityAnalyzer # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize, sent_tokenize # type: ignore

class FeatureExtractor:
    """Extract features from reviews for fake detection"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common fake review patterns
        self.fake_patterns = [
            r'\b(amazing|incredible|outstanding|perfect|excellent)\b.*\b(amazing|incredible|outstanding|perfect|excellent)\b',
            r'\b(highly recommend|must buy|best purchase|love it so much)\b',
            r'\b(five stars?|5 stars?|⭐⭐⭐⭐⭐)\b.*\b(five stars?|5 stars?|⭐⭐⭐⭐⭐)\b',
            r'\b(fast shipping|quick delivery)\b.*\b(great quality|excellent product)\b'
        ]
        
    def extract_all_features(self, review_text: str, rating: int, 
                           reviewer_total_reviews: int = 0, 
                           reviewer_account_age_days: int = 0,
                           review_date: datetime = None) -> Dict:
        """Extract all features from a review"""
        
        features = {}
        
        # Text-based features
        features.update(self._extract_text_features(review_text))
        
        # Sentiment features
        features.update(self._extract_sentiment_features(review_text))
        
        # Rating-related features
        features.update(self._extract_rating_features(review_text, rating))
        
        # Reviewer behavior features
        features.update(self._extract_reviewer_features(
            reviewer_total_reviews, reviewer_account_age_days
        ))
        
        # Temporal features
        if review_date:
            features.update(self._extract_temporal_features(review_date))
            
        return features
    
    def _extract_text_features(self, text: str) -> Dict:
        """Extract text-based features"""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'capitalization_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'repeated_chars': len(re.findall(r'(.)\1{2,}', text)),  # aaaamazing
            'repeated_words': self._count_repeated_words(text),
            'stopword_ratio': sum(1 for word in words if word in self.stop_words) / len(words) if words else 0
        }
    
    def _extract_sentiment_features(self, text: str) -> Dict:
        """Extract sentiment-based features"""
        sentiment_scores = self.sia.polarity_scores(text)
        
        # Check for extreme sentiment (often fake)
        is_extreme_positive = sentiment_scores['compound'] > 0.8
        is_extreme_negative = sentiment_scores['compound'] < -0.8
        
        return {
            'sentiment_compound': sentiment_scores['compound'],
            'sentiment_positive': sentiment_scores['pos'],
            'sentiment_negative': sentiment_scores['neg'],
            'sentiment_neutral': sentiment_scores['neu'],
            'is_extreme_sentiment': is_extreme_positive or is_extreme_negative
        }
    
    def _extract_rating_features(self, text: str, rating: int) -> Dict:
        """Extract rating-related features"""
        sentiment_scores = self.sia.polarity_scores(text)
        
        # Mismatch between rating and sentiment
        expected_sentiment = (rating - 3) / 2  # Convert 1-5 rating to -1 to 1 scale
        sentiment_mismatch = abs(expected_sentiment - sentiment_scores['compound'])
        
        return {
            'rating': rating,
            'is_extreme_rating': rating in [1, 5],
            'sentiment_rating_mismatch': sentiment_mismatch,
            'high_mismatch': sentiment_mismatch > 0.7
        }
    
    def _extract_reviewer_features(self, total_reviews: int, account_age_days: int) -> Dict:
        """Extract reviewer behavior features"""
        return {
            'reviewer_total_reviews': total_reviews,
            'reviewer_account_age_days': account_age_days,
            'is_new_reviewer': account_age_days < 30,
            'is_very_active': total_reviews > 50,
            'reviews_per_day': total_reviews / max(account_age_days, 1)
        }
    
    def _extract_temporal_features(self, review_date: datetime) -> Dict:
        """Extract time-based features"""
        now = datetime.now()
        days_since_review = (now - review_date).days
        
        return {
            'days_since_review': days_since_review,
            'is_recent_review': days_since_review < 7,
            'review_hour': review_date.hour,
            'is_business_hours': 9 <= review_date.hour <= 17,
            'is_weekend': review_date.weekday() >= 5
        }
    
    def _count_repeated_words(self, text: str) -> int:
        """Count repeated words in text"""
        words = word_tokenize(text.lower())
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        return sum(1 for count in word_counts.values() if count > 1)
    
    def check_fake_patterns(self, text: str) -> List[str]:
        """Check for common fake review patterns"""
        flags = []
        text_lower = text.lower()
        
        for i, pattern in enumerate(self.fake_patterns):
            if re.search(pattern, text_lower):
                flags.append(f"fake_pattern_{i+1}")
        
        # Additional pattern checks
        if len(re.findall(r'\b(great|good|nice|awesome|amazing)\b', text_lower)) > 3:
            flags.append("excessive_positive_words")
            
        if text.count('!') > 3:
            flags.append("excessive_exclamation")
            
        if len(text) < 20:
            flags.append("too_short")
            
        if len(text) > 2000:
            flags.append("too_long")
            
        return flags