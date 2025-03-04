import requests
import json

def test_routes():
    base_url = 'http://localhost:5000'
    
    # Test 1: Main page load
    print("\n1. Testing main page load...")
    response = requests.get(base_url)
    print(f"Status: {response.status_code}")
    
    # Test 2: Story generation
    print("\n2. Testing story generation...")
    data = {
        'prompt': 'A man wakes up to find himself transformed',
        'genre': 'surreal',
        'length': 'short',
        'darkness': '70'
    }
    response = requests.post(base_url, data=data)
    print(f"Status: {response.status_code}")
    
    # Test 3: Audio generation
    print("\n3. Testing audio generation...")
    audio_data = {
        'text': 'Test story for audio generation',
        'lang': 'en',
        'slow': False
    }
    response = requests.post(f'{base_url}/generate-audio', json=audio_data)
    print(f"Status: {response.status_code}")
    
    # Test 4: Analytics page
    print("\n4. Testing analytics page...")
    response = requests.get(f'{base_url}/analytics')
    print(f"Status: {response.status_code}")

if __name__ == '__main__':
    test_routes() 