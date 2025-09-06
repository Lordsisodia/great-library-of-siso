#!/usr/bin/env python3
"""
Test the generation rate to estimate overnight output
"""

import asyncio
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import aiohttp
import random

async def test_generation_rate():
    """Test generation rate for 5 minutes to estimate overnight output"""
    print("🧪 TESTING GENERATION RATE")
    print("=" * 50)
    
    start_time = datetime.now()
    session_dir = f"test_sessions/{start_time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(session_dir, exist_ok=True)
    
    # Working APIs
    cerebras_key = 'csk-3k6trr428thrppejwexnep65kh8m3nccmx5p92np3x8rr2wr'
    
    total_chars = 0
    files_created = 0
    
    # Test topics
    test_topics = [
        "Machine learning optimization strategies",
        "Distributed system architecture patterns", 
        "Advanced neural network implementations"
    ]
    
    print(f"⏰ Testing for 5 minutes starting {start_time.strftime('%H:%M:%S')}")
    print(f"📁 Session: {session_dir}")
    
    for i, topic in enumerate(test_topics):
        print(f"\n🔍 Generating: {topic}")
        
        # Generate content using Cerebras
        response = await query_cerebras(cerebras_key, 
            f"Provide comprehensive technical analysis of: {topic}. "
            f"Include detailed explanations, code examples, algorithms, "
            f"implementation strategies, and best practices. "
            f"Make this thorough and educational - at least 2000 words."
        )
        
        # Create file
        safe_topic = topic[:30].replace(' ', '_').replace('/', '_')
        filename = f"{session_dir}/test_{i+1}_{safe_topic}.md"
        
        content = f"""# {topic}
*Test Generation - Sample {i+1}*
*Generated: {datetime.now().isoformat()}*

## Comprehensive Analysis
{response}

## Summary
This represents the quality and depth of content that will be generated
continuously throughout the 12-hour autonomous session.

*Content Length: {len(response)} characters*
"""
        
        with open(filename, 'w') as f:
            f.write(content)
        
        chars_this_file = len(content)
        total_chars += chars_this_file
        files_created += 1
        
        print(f"✅ Generated {chars_this_file:,} characters")
        print(f"📊 Running total: {total_chars:,} characters")
        
        # Small delay
        await asyncio.sleep(10)
    
    end_time = datetime.now()
    test_duration = end_time - start_time
    
    # Calculate rates
    chars_per_second = total_chars / test_duration.total_seconds()
    chars_per_hour = chars_per_second * 3600
    chars_per_12_hours = chars_per_hour * 12
    
    print(f"\n📊 GENERATION RATE ANALYSIS:")
    print(f"⏱️  Test Duration: {test_duration}")
    print(f"📝 Total Characters: {total_chars:,}")
    print(f"📁 Files Created: {files_created}")
    print(f"⚡ Rate: {chars_per_second:.1f} chars/sec")
    print(f"📈 Hourly Rate: {chars_per_hour:,.0f} chars/hour")
    print(f"🎯 12-Hour Projection: {chars_per_12_hours:,.0f} characters")
    
    if chars_per_12_hours >= 5_000_000:
        print(f"✅ EXCEEDS 5M character target!")
    elif chars_per_12_hours >= 3_000_000:
        print(f"✅ Will generate 3M+ characters")
    else:
        print(f"⚠️  May need optimization")
    
    # Calculate files projection
    files_per_hour = files_created / (test_duration.total_seconds() / 3600)
    files_per_12_hours = files_per_hour * 12
    print(f"📁 12-Hour File Projection: {files_per_12_hours:.0f} files")
    
    return {
        'chars_per_hour': chars_per_hour,
        'projected_12_hour_chars': chars_per_12_hours,
        'projected_12_hour_files': files_per_12_hours
    }

async def query_cerebras(api_key: str, prompt: str) -> str:
    """Query Cerebras API"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'llama3.1-8b',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 2000,
        'temperature': 0.7
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.cerebras.ai/v1/chat/completions',
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                if 'choices' in data and data['choices']:
                    return data['choices'][0]['message']['content']
                return "Response unavailable"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    asyncio.run(test_generation_rate())