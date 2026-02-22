"""
Quick test script for the Free Deepfake Detection API
"""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        print(f"   Make sure the backend is running on {API_URL}")
        return False

def test_analyze(image_path):
    """Test analyze endpoint with an image"""
    print(f"\nTesting analyze endpoint with: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/analyze", files=files)
        
        if response.status_code == 200:
            print("✅ Analysis successful!")
            result = response.json()
            print(f"\n   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence'] * 100:.1f}%")
            print(f"   Probability: {result['probability'] * 100:.1f}%")
            print(f"   Explanation: {result['explanation']}")
            print(f"\n   Technical Details:")
            for key, value in result['details'].items():
                print(f"      - {key}: {value}")
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Free Deepfake Detection API - Test Script")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("\n⚠️  Backend is not running!")
        print("   Start it with: python deepfake_free.py")
        sys.exit(1)
    
    # Test analyze
    print("\n" + "=" * 60)
    if len(sys.argv) > 1:
        # Use provided image path
        image_path = sys.argv[1]
        test_analyze(image_path)
    else:
        print("\nℹ️  To test image analysis, run:")
        print(f"   python test_free_api.py path/to/your/image.jpg")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

