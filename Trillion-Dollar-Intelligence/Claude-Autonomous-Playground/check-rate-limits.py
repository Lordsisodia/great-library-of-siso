#!/usr/bin/env python3
"""
Check actual rate limits from API response headers
"""

import asyncio
import aiohttp
import json

async def check_cerebras_limits():
    """Check Cerebras rate limits from headers"""
    headers = {
        'Authorization': 'Bearer csk-3k6trr428thrppejwexnep65kh8m3nccmx5p92np3x8rr2wr',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'llama3.1-8b',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'max_tokens': 10
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.cerebras.ai/v1/chat/completions',
                headers=headers,
                json=payload
            ) as response:
                print("🧠 CEREBRAS API RATE LIMITS:")
                print(f"Status: {response.status}")
                
                # Check for rate limit headers
                rate_headers = {}
                for header, value in response.headers.items():
                    if 'rate' in header.lower() or 'limit' in header.lower():
                        rate_headers[header] = value
                        print(f"  {header}: {value}")
                
                if not rate_headers:
                    print("  No rate limit headers found in response")
                
                return rate_headers
    except Exception as e:
        print(f"Cerebras error: {e}")
        return {}

async def check_gemini_limits():
    """Check Gemini rate limits from headers"""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyDnuBN9ZzW3HnH_-3RAlOZu3GUs9zTz6HM"
    
    payload = {
        'contents': [{'parts': [{'text': 'Hello'}]}],
        'generationConfig': {'maxOutputTokens': 10}
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                print("\n🤖 GEMINI API RATE LIMITS:")
                print(f"Status: {response.status}")
                
                # Check for rate limit headers  
                rate_headers = {}
                for header, value in response.headers.items():
                    if 'rate' in header.lower() or 'limit' in header.lower() or 'quota' in header.lower():
                        rate_headers[header] = value
                        print(f"  {header}: {value}")
                
                if not rate_headers:
                    print("  No rate limit headers found in response")
                
                return rate_headers
    except Exception as e:
        print(f"Gemini error: {e}")
        return {}

async def main():
    print("🔍 CHECKING API RATE LIMITS")
    print("=" * 50)
    
    cerebras_limits = await check_cerebras_limits()
    gemini_limits = await check_gemini_limits()
    
    print("\n📊 RATE LIMIT RECOMMENDATIONS:")
    print("Based on documentation:")
    print("🧠 Cerebras: Conservative approach - 30 requests/minute")
    print("🤖 Gemini: Free tier - 15 requests/minute")
    print("\n⚠️  SAFE AUTONOMOUS SETTINGS:")
    print("- Cerebras: 1 request every 2 minutes (30/hour)")
    print("- Gemini: 1 request every 4 minutes (15/hour)")
    print("- Total API calls: ~45/hour for both models")
    print("- 12-hour session: ~540 API calls total")

if __name__ == "__main__":
    asyncio.run(main())