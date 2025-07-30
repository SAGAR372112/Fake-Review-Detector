import requests
import json

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

def test_single_review():
    """Test single review analysis"""
    print("=== Testing Single Review Analysis ===")
    
    # Test authentic review
    authentic_review = {
        "text": "I've been using this laptop for 3 months now. The performance is good for programming and light gaming. Battery life is around 6-7 hours which is acceptable. The keyboard is comfortable but the trackpad could be more responsive. Overall a decent machine for the price.",
        "rating": 4,
        "reviewer_total_reviews": 15,
        "reviewer_account_age_days": 400
    }
    
    response = requests.post(f"{BASE_URL}/analyze/single", json=authentic_review)
    print(f"Authentic Review Result: {response.json()}")
    print()
    
    # Test fake review
    fake_review = {
        "text": "AMAZING AMAZING AMAZING!!! This is the BEST laptop EVER!!! Five stars! Highly recommend! Must buy! Perfect quality! Fast shipping! Love it so much!!! BEST PURCHASE EVER!!!",
        "rating": 5,
        "reviewer_total_reviews": 1,
        "reviewer_account_age_days": 5
    }
    
    response = requests.post(f"{BASE_URL}/analyze/single", json=fake_review)
    print(f"Fake Review Result: {response.json()}")
    print()

def test_quick_analyze():
    """Test quick analysis endpoint"""
    print("=== Testing Quick Analysis ===")
    
    test_cases = [
        {"text": "Great product! Works perfectly!", "rating": 5},
        {"text": "Terrible quality. Broke after one day. Complete waste of money!", "rating": 1},
        {"text": "Average product. Does what it's supposed to do. Nothing special but works fine.", "rating": 3}
    ]
    
    for i, case in enumerate(test_cases):
        response = requests.post(f"{BASE_URL}/analyze/quick", params=case)
        print(f"Test Case {i+1}: {response.json()}")
    print()

def test_batch_analysis():
    """Test batch review analysis"""
    print("=== Testing Batch Analysis ===")
    
    batch_reviews = {
        "reviews": [
            {
                "text": "Excellent product! Highly recommend!",
                "rating": 5,
                "reviewer_total_reviews": 1,
                "reviewer_account_age_days": 2
            },
            {
                "text": "The product works as expected. Good build quality and reasonable price. Delivery was on time.",
                "rating": 4,
                "reviewer_total_reviews": 12,
                "reviewer_account_age_days": 300
            },
            {
                "text": "Perfect perfect perfect! Amazing! Best buy ever! Five stars!",
                "rating": 5,
                "reviewer_total_reviews": 2,
                "reviewer_account_age_days": 7
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/analyze/batch", json=batch_reviews)
    result = response.json()
    
    print("Batch Analysis Summary:")
    print(json.dumps(result["summary"], indent=2))
    print()
    
    print("Individual Results:")
    for i, review_result in enumerate(result["results"]):
        print(f"Review {i+1}: Fake={review_result['is_fake']}, Confidence={review_result['confidence_score']:.1f}%")
    print()

def test_model_info():
    """Test model info endpoint"""
    print("=== Testing Model Info ===")
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(json.dumps(response.json(), indent=2, default=str))
    print()

def test_health_check():
    """Test health check"""
    print("=== Testing Health Check ===")
    
    response = requests.get(f"{BASE_URL}/health")
    print(json.dumps(response.json(), indent=2))
    print()

def test_demo_stats():
    """Test demo statistics"""
    print("=== Testing Demo Stats ===")
    
    response = requests.get(f"{BASE_URL}/stats/demo")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("🔍 Fake Review Detector API Test Suite")
    print("="*50)
    
    try:
        # Test basic endpoints
        test_health_check()
        test_model_info()
        test_demo_stats()
        
        # Test analysis endpoints
        test_single_review()
        test_quick_analyze()
        test_batch_analysis()
        
        print("✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("Make sure the server is running: python -m uvicorn app.main:app --reload")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")