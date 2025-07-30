import re
import nltk
from typing import Dict, List
from datetime import datetime, timedelta
import numpy as np

# Download required NLTK data with better error handling
def ensure_nltk_data():
    """Ensure NLTK data is available"""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('vader_lexicon', 'vader_lexicon')
    ]
    
    for data_path, download_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            try:
                print(f"Downloading NLTK data: {download_name}")
                nltk.download(download_name, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {download_name}: {e}")

# Import with fallback handling
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: NLTK features limited due to import error: {e}")
    NLTK_AVAILABLE = False

class FeatureExtractor:
    """Extract features from reviews for fake detection"""
    
    def __init__(self):
        if NLTK_AVAILABLE:
            try:
                self.sia = SentimentIntensityAnalyzer()
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                print(f"Warning: NLTK initialization failed: {e}")
                self.sia = None
                self.stop_words = set()
        else:
            self.sia = None
            self.stop_words = set()
        
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
        # Basic tokenization fallback if NLTK not available
        if NLTK_AVAILABLE:
            try:
                words = word_tokenize(text.lower())
                sentences = sent_tokenize(text)
            except Exception:
                # Fallback to simple tokenization
                words = text.lower().split()
                sentences = text.split('.')
        else:
            words = text.lower().split()
            sentences = text.split('.')
        
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
            'stopword_ratio': sum(1 for word in words if word in self.stop_words) / len(words) if words and self.stop_words else 0
        }
    
    def _extract_sentiment_features(self, text: str) -> Dict:
        """Extract sentiment-based features"""
        if self.sia:
            try:
                sentiment_scores = self.sia.polarity_scores(text)
            except Exception:
                # Fallback sentiment analysis
                sentiment_scores = self._simple_sentiment(text)
        else:
            sentiment_scores = self._simple_sentiment(text)
        
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
        if NLTK_AVAILABLE:
            try:
                words = word_tokenize(text.lower())
            except Exception:
                words = text.lower().split()
        else:
            words = text.lower().split()
            
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        return sum(1 for count in word_counts.values() if count > 1)
    
    def _simple_sentiment(self, text: str) -> Dict:
        """Simple sentiment analysis fallback"""
        text_lower = text.lower()
        
        # Simple positive/negative word counting
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect', 'love', 'best', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointing', 'poor', 'useless']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        
        pos_ratio = pos_count / len(text_lower.split())
        neg_ratio = neg_count / len(text_lower.split())
        compound = (pos_count - neg_count) / max(total, 1)
        
        return {
            'compound': max(-1.0, min(1.0, compound)),
            'pos': min(1.0, pos_ratio * 3),  # Scale up for visibility
            'neg': min(1.0, neg_ratio * 3),
            'neu': max(0.0, 1.0 - pos_ratio * 3 - neg_ratio * 3)
        }
    
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