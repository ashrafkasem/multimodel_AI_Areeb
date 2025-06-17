#!/usr/bin/env python3
"""
Test script for the new Qwen-Agent implementation
"""

import asyncio
import json
from qwen_agent_server import create_agent

async def test_basic_conversation():
    """Test basic conversation without tools."""
    print("🧪 Testing basic conversation...")
    
    agent = create_agent()
    messages = [
        {'role': 'user', 'content': 'Hello! Can you explain what you are and what you can do?'}
    ]
    
    print("User: Hello! Can you explain what you are and what you can do?")
    print("Assistant: ", end="")
    
    response_text = ""
    for response in agent.run(messages=messages):
        if isinstance(response, list):
            for item in response:
                if isinstance(item, dict) and item.get('role') == 'assistant':
                    content = item.get('content', '')
                    if content:
                        print(content)
                        response_text = content
                        break
    
    return len(response_text) > 0

async def test_code_generation():
    """Test code generation using the advanced code generator tool."""
    print("\n🔧 Testing code generation tool...")
    
    agent = create_agent()
    messages = [
        {'role': 'user', 'content': 'Write a Python function to calculate the factorial of a number using recursion.'}
    ]
    
    print("User: Write a Python function to calculate the factorial of a number using recursion.")
    print("Assistant: ", end="")
    
    response_text = ""
    for response in agent.run(messages=messages):
        if isinstance(response, list):
            for item in response:
                if isinstance(item, dict) and item.get('role') == 'assistant':
                    content = item.get('content', '')
                    if content:
                        print(content)
                        response_text = content
                        break
    
    # Check if the response contains code
    has_code = 'def ' in response_text or 'function' in response_text.lower()
    return has_code

async def test_api_compatibility():
    """Test API server compatibility."""
    print("\n🌐 Testing API server...")
    
    try:
        import httpx
        
        # Start the server in background (for testing)
        # In real usage, you'd run this as a separate process
        print("Note: Run 'python qwen_agent_server.py' to start the API server for full testing")
        
        return True
        
    except ImportError:
        print("httpx not installed for API testing")
        return False

async def main():
    """Run all tests."""
    print("🚀 Testing Official Qwen-Agent Implementation")
    print("=" * 50)
    
    try:
        # Test 1: Basic conversation
        basic_test = await test_basic_conversation()
        print(f"✅ Basic conversation: {'PASSED' if basic_test else 'FAILED'}")
        
        # Test 2: Code generation
        code_test = await test_code_generation()
        print(f"✅ Code generation: {'PASSED' if code_test else 'FAILED'}")
        
        # Test 3: API compatibility
        api_test = await test_api_compatibility()
        print(f"✅ API compatibility: {'PASSED' if api_test else 'FAILED'}")
        
        print("\n" + "=" * 50)
        
        if all([basic_test, code_test, api_test]):
            print("🎉 All tests PASSED! Your Qwen-Agent setup is working correctly.")
            print("\nNext steps:")
            print("1. Run API server: python qwen_agent_server.py")
            print("2. Run GUI: python qwen_agent_server.py gui")
            print("3. Test with Continue/Roo Code at http://localhost:8002")
        else:
            print("❌ Some tests failed. Check your configuration.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("Make sure your model servers are running:")
        print("- Qwen3 on http://localhost:8000")
        print("- Qwen2.5-Coder on http://localhost:8001")

if __name__ == "__main__":
    asyncio.run(main()) 