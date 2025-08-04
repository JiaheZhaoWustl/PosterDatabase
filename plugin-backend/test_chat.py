#!/usr/bin/env python3
"""
Test script for the chat bot functionality.
Run this to test the chat endpoint before integrating with the frontend.
"""

import requests
import json

# Test the chat endpoint
def test_chat():
    url = "http://localhost:5000/chat"
    
    # Test data
    test_data = {
        "message": "Hello! I'm working on a poster design. Can you help me with some layout suggestions?",
        "conversation_history": [],
        "temperature": 0.7
    }
    
    print("🧪 Testing chat endpoint...")
    print(f"📤 Sending message: {test_data['message']}")
    
    try:
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"🤖 Response: {result['response']}")
            print(f"📝 Model used: {result.get('model_used', 'Unknown')}")
            print(f"💬 History length: {len(result.get('conversation_history', []))}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the Flask server is running on port 5000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_chat_with_history():
    url = "http://localhost:5000/chat"
    
    # Test with conversation history
    test_data = {
        "message": "What about the color scheme?",
        "conversation_history": [
            {"role": "user", "content": "Hello! I'm working on a poster design. Can you help me with some layout suggestions?"},
            {"role": "assistant", "content": "Of course! I'd be happy to help you with your poster design. What kind of poster are you creating? Is it for an event, product, or something else? This will help me give you more specific layout suggestions."}
        ],
        "temperature": 0.7
    }
    
    print("\n🧪 Testing chat with history...")
    print(f"📤 Sending follow-up message: {test_data['message']}")
    
    try:
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"🤖 Response: {result['response']}")
            print(f"💬 Updated history length: {len(result.get('conversation_history', []))}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the Flask server is running on port 5000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_reset_chat():
    url = "http://localhost:5000/chat/reset"
    
    print("\n🧪 Testing chat reset...")
    
    try:
        response = requests.post(url)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"📝 Message: {result['message']}")
            print(f"💬 History length: {len(result.get('conversation_history', []))}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the Flask server is running on port 5000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Starting chat bot functionality tests...")
    print("=" * 50)
    
    test_chat()
    test_chat_with_history()
    test_reset_chat()
    
    print("\n" + "=" * 50)
    print("✅ Chat bot functionality tests completed!") 