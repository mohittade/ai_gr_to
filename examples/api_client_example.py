"""
Example API client for interacting with the translation service
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """
    Test the health check endpoint
    """
    print("=" * 80)
    print("Testing Health Check Endpoint")
    print("=" * 80)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ API is healthy!")
            print(f"  Status: {data['status']}")
            print(f"  Timestamp: {data['timestamp']}")
        else:
            print(f"✗ Health check failed with status: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server!")
        print("  Please start the server first: python api/api_server.py")
        return False
    
    return True

def test_supported_languages():
    """
    Test the supported languages endpoint
    """
    print("\n" + "=" * 80)
    print("Testing Supported Languages Endpoint")
    print("=" * 80)
    
    try:
        response = requests.get(f"{BASE_URL}/supported-languages")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Supported languages:")
            for lang, info in data['languages'].items():
                print(f"  {lang}: {info['name']} ({info['native']})")
            print(f"\nPipeline: {' → '.join(data['pipeline'])}")
        else:
            print(f"✗ Request failed with status: {response.status_code}")
    
    except Exception as e:
        print(f"✗ Error: {e}")

def test_single_translation():
    """
    Test single translation endpoint
    """
    print("\n" + "=" * 80)
    print("Testing Single Translation Endpoint")
    print("=" * 80)
    
    test_sentences = [
        "Guten Morgen, wie geht es Ihnen?",
        "Ich liebe die deutsche Sprache.",
        "Die Wissenschaft ist sehr wichtig für unsere Zukunft."
    ]
    
    for i, german_text in enumerate(test_sentences, 1):
        print(f"\nExample {i}:")
        print(f"  🇩🇪 German: {german_text}")
        
        # Prepare request
        payload = {
            "text": german_text,
            "source_lang": "de",
            "target_lang": "mr"
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/translate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"  🇬🇧 English: {data['intermediate_translation']}")
                print(f"  🇮🇳 Marathi: {data['translation']}")
                print(f"  ⏱️  Time: {elapsed_time:.3f}s (API reports: {data['translation_time']:.3f}s)")
            else:
                print(f"  ✗ Translation failed: {response.status_code}")
                print(f"    {response.text}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_batch_translation():
    """
    Test batch translation endpoint
    """
    print("\n" + "=" * 80)
    print("Testing Batch Translation Endpoint")
    print("=" * 80)
    
    german_texts = [
        "Hallo",
        "Danke",
        "Auf Wiedersehen",
        "Guten Tag",
        "Wie geht es dir?"
    ]
    
    print(f"\nTranslating {len(german_texts)} sentences in batch...")
    
    # Prepare request
    payload = {
        "texts": german_texts,
        "source_lang": "de",
        "target_lang": "mr"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/batch-translate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Batch translation successful!")
            print(f"  Total time: {elapsed_time:.3f}s")
            print(f"  Average per sentence: {elapsed_time/len(german_texts):.3f}s")
            
            print("\n  Results:")
            for result in data['translations']:
                print(f"    🇩🇪 {result['text']}")
                print(f"    🇮🇳 {result['translation']}\n")
        else:
            print(f"✗ Batch translation failed: {response.status_code}")
            print(f"  {response.text}")
    
    except Exception as e:
        print(f"✗ Error: {e}")

def test_stats_endpoint():
    """
    Test statistics endpoint
    """
    print("\n" + "=" * 80)
    print("Testing Statistics Endpoint")
    print("=" * 80)
    
    try:
        response = requests.get(f"{BASE_URL}/stats")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ API Statistics:")
            print(f"  Total translations: {data['total_translations']}")
            print(f"  Average translation time: {data['average_translation_time']:.3f}s")
            
            if data['language_pairs']:
                print("\n  Translation by language pair:")
                for pair, count in data['language_pairs'].items():
                    print(f"    {pair}: {count}")
        else:
            print(f"✗ Request failed with status: {response.status_code}")
    
    except Exception as e:
        print(f"✗ Error: {e}")

def test_error_handling():
    """
    Test API error handling
    """
    print("\n" + "=" * 80)
    print("Testing Error Handling")
    print("=" * 80)
    
    # Test 1: Invalid language pair
    print("\n[Test 1] Invalid language pair:")
    payload = {
        "text": "Hello world",
        "source_lang": "xx",
        "target_lang": "yy"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/translate", json=payload)
        print(f"  Status: {response.status_code}")
        if response.status_code != 200:
            print(f"  ✓ Error properly handled: {response.json()['detail']}")
    except Exception as e:
        print(f"  Response: {e}")
    
    # Test 2: Empty text
    print("\n[Test 2] Empty text:")
    payload = {
        "text": "",
        "source_lang": "de",
        "target_lang": "mr"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/translate", json=payload)
        print(f"  Status: {response.status_code}")
        if response.status_code != 200:
            print(f"  ✓ Error properly handled: {response.json()['detail']}")
    except Exception as e:
        print(f"  Response: {e}")
    
    # Test 3: Missing fields
    print("\n[Test 3] Missing required fields:")
    payload = {
        "text": "Hello world"
        # Missing source_lang and target_lang
    }
    
    try:
        response = requests.post(f"{BASE_URL}/translate", json=payload)
        print(f"  Status: {response.status_code}")
        if response.status_code != 200:
            print(f"  ✓ Error properly handled")
    except Exception as e:
        print(f"  Response: {e}")

def main():
    """
    Run all API tests
    """
    print("\n🚀 Starting API Client Tests\n")
    
    # Check if API is running
    if not test_health_check():
        print("\n⚠️  Please start the API server first:")
        print("   python api/api_server.py")
        return
    
    # Run tests
    test_supported_languages()
    test_single_translation()
    test_batch_translation()
    test_stats_endpoint()
    test_error_handling()
    
    print("\n" + "=" * 80)
    print("✓ All API tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
