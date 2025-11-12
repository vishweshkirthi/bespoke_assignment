#!/usr/bin/env python3
"""
Test script for the FastAPI document quality service
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the root endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_model_status():
    """Test model status endpoint"""
    print("Testing model status...")
    response = requests.get(f"{BASE_URL}/model/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_score_text(text):
    """Test scoring a text"""
    print(f"Testing score for text: '{text[:50]}...'")
    
    payload = {"text": text}
    response = requests.post(f"{BASE_URL}/score", json=payload)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"High Quality: {result['is_high_quality']}")
    else:
        print(f"Error: {response.text}")
    print("-" * 50)

def test_train_model(file_path):
    """Test training with a file"""
    print(f"Testing training with file: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': ('positive_examples.txt', f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/train", files=files)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Message: {result['message']}")
            print(f"Training examples: {result['training_examples']}")
            print(f"Model saved to: {result['model_saved']}")
        else:
            print(f"Error: {response.text}")
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")
        
    print("-" * 50)

def main():
    print("FastAPI Document Quality Service Test")
    print("=" * 50)
    
    # Test health check
    test_health_check()
    
    # Test model status
    test_model_status()
    
    # Test scoring (this will fail if no model is loaded)
    print("Testing scoring (may fail if no model loaded)...")
    test_score_text("This is a well-written, comprehensive article with detailed analysis and proper structure.")
    test_score_text("bad doc very short no info")
    
    # Test training (uncomment if you have a positive examples file)
    # test_train_model("data/train_positive.txt")
    
    print("Test completed!")
    print("\nTo run the API server:")
    print("python app.py")
    print("\nOr with uvicorn:")
    print("pip install fastapi uvicorn")
    print("uvicorn app:app --reload")

if __name__ == "__main__":
    main()