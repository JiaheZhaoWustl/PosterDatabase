#!/usr/bin/env python3
"""
Test script for the prompt refinement functionality.
This demonstrates the multi-turn conversation flow.
"""

import requests
import json

def test_prompt_refinement():
    url = "http://localhost:5000/chat/refine"
    
    # Test with an initial prompt
    test_data = {
        "prompt": "I want to create a poster for a tech conference",
        "conversation_history": [],
        "temperature": 0.7
    }
    
    print("🧪 Testing prompt refinement...")
    print(f"📤 Initial prompt: {test_data['prompt']}")
    
    try:
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"📝 Status: {result.get('status', 'Unknown')}")
            print(f"🤖 AI Response: {result.get('conversation_flow', [{}])[0].get('ai_response', 'No response')}")
            print(f"💬 Is Question: {result.get('conversation_flow', [{}])[0].get('is_question', False)}")
            
            if result.get('status') == 'waiting_for_user_response':
                print("⏳ Waiting for user response to continue...")
            elif result.get('status') == 'completed':
                print(f"✅ Final prompt: {result.get('final_prompt', 'No final prompt')}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the Flask server is running on port 5000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_regular_chat():
    url = "http://localhost:5000/chat"
    
    # Test regular chat functionality
    test_data = {
        "message": "I want to create a poster for a tech conference",
        "conversation_history": [],
        "temperature": 0.7
    }
    
    print("\n🧪 Testing regular chat...")
    print(f"📤 Message: {test_data['message']}")
    
    try:
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"🤖 Response: {result.get('response', 'No response')}")
            print(f"📝 Is Question: {result.get('is_question', False)}")
            
            if 'updated_prompt' in result:
                print(f"🔄 Updated Prompt: {result['updated_prompt']}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the Flask server is running on port 5000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Starting prompt refinement tests...")
    print("=" * 60)
    
    test_prompt_refinement()
    test_regular_chat()
    
    print("\n" + "=" * 60)
    print("✅ Prompt refinement tests completed!")
    print("\n📝 Usage:")
    print("1. Use /chat/refine for multi-turn prompt refinement")
    print("2. Use /chat for regular conversation")
    print("3. The AI will ask questions and generate updated prompts") 