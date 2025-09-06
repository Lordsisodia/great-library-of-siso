#!/usr/bin/env python3
"""
FREE API HARVESTER
Discover and test ALL available free compute resources
"""

import requests
import asyncio
import aiohttp
import json
from typing import Dict, List
from datetime import datetime

class FreeAPIHarvester:
    """Discover and harvest ALL free AI APIs and compute resources"""
    
    def __init__(self):
        self.discovered_apis = {}
        self.test_results = {}
        
        # Known free APIs to test
        self.free_apis_database = {
            'groq': {
                'name': 'Groq (Ultra Fast)',
                'endpoint': 'https://api.groq.com/openai/v1/chat/completions',
                'key': 'YOUR_GROQ_API_KEY_HERE',
                'models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
                'rate_limit': '30 RPM',
                'tokens_per_sec': 500,
                'specialization': 'speed'
            },
            'cerebras': {
                'name': 'Cerebras (Efficient)',
                'endpoint': 'https://api.cerebras.ai/v1/chat/completions',
                'key': 'csk-3k6trr428thrppejwexnep65kh8m3nccmx5p92np3x8rr2wr',
                'models': ['llama3.1-70b', 'llama3.1-8b'],
                'rate_limit': '30 RPM',
                'tokens_per_sec': 400,
                'specialization': 'efficiency'
            },
            'gemini': {
                'name': 'Google Gemini Pro',
                'endpoint': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
                'key': 'AIzaSyDnuBN9ZzW3HnH_-3RAlOZu3GUs9zTz6HM',
                'models': ['gemini-pro', 'gemini-pro-vision'],
                'rate_limit': '60 RPM',
                'tokens_per_sec': 150,
                'specialization': 'reasoning'
            },
            'together': {
                'name': 'Together AI',
                'endpoint': 'https://api.together.xyz/v1/chat/completions',
                'key': 'NEED_TO_OBTAIN',
                'models': ['meta-llama/Llama-3-70b-chat-hf', 'mistralai/Mixtral-8x7B-Instruct-v0.1'],
                'rate_limit': '600 RPM free',
                'tokens_per_sec': 200,
                'specialization': 'variety'
            },
            'replicate': {
                'name': 'Replicate',
                'endpoint': 'https://api.replicate.com/v1/predictions',
                'key': 'NEED_TO_OBTAIN',
                'models': ['llama-2-70b-chat', 'mixtral-8x7b-instruct-v0.1'],
                'rate_limit': '$10 free credits',
                'tokens_per_sec': 100,
                'specialization': 'variety'
            },
            'huggingface': {
                'name': 'HuggingFace Inference API',
                'endpoint': 'https://api-inference.huggingface.co/models/',
                'key': 'NEED_TO_OBTAIN',
                'models': ['microsoft/DialoGPT-large', 'facebook/blenderbot-1B-distill'],
                'rate_limit': '30k tokens/month free',
                'tokens_per_sec': 50,
                'specialization': 'experimentation'
            }
        }
        
        # APIs to discover and test
        self.discovery_targets = {
            'cohere': 'https://api.cohere.ai/v1/generate',
            'ai21': 'https://api.ai21.com/studio/v1/j2-ultra/complete',
            'anthropic_claude': 'https://api.anthropic.com/v1/messages',
            'perplexity': 'https://api.perplexity.ai/chat/completions',
            'mistral': 'https://api.mistral.ai/v1/chat/completions',
            'fireworks': 'https://api.fireworks.ai/inference/v1/chat/completions'
        }
    
    async def harvest_all_free_apis(self):
        """Discover and test ALL available free compute"""
        print("🔍 FREE API HARVESTER STARTING")
        print("="*50)
        
        # Test known APIs
        print("🧪 Testing known free APIs...")
        for api_name, config in self.free_apis_database.items():
            result = await self._test_api(api_name, config)
            self.test_results[api_name] = result
            
        # Discover new APIs
        print("\n🕵️  Discovering additional free APIs...")
        for api_name, endpoint in self.discovery_targets.items():
            result = await self._discover_api(api_name, endpoint)
            if result['accessible']:
                self.discovered_apis[api_name] = result
                
        # Generate comprehensive report
        return self._generate_harvest_report()
    
    async def _test_api(self, api_name: str, config: Dict) -> Dict:
        """Test specific API for accessibility and performance"""
        print(f"  🧪 Testing {config['name']}...")
        
        if config['key'] == 'NEED_TO_OBTAIN':
            return {
                'status': 'requires_key',
                'name': config['name'],
                'message': 'API key needed - potential free tier available'
            }
            
        try:
            if api_name == 'gemini':
                return await self._test_gemini(config)
            else:
                return await self._test_openai_compatible(config)
        except Exception as e:
            return {
                'status': 'error',
                'name': config['name'],
                'error': str(e)
            }
    
    async def _test_gemini(self, config: Dict) -> Dict:
        """Test Gemini API specifically"""
        url = f"{config['endpoint']}?key={config['key']}"
        test_prompt = "Say 'Hello from Gemini!' if you're working."
        
        payload = {
            'contents': [{
                'parts': [{
                    'text': test_prompt
                }]
            }]
        }
        
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                end_time = asyncio.get_event_loop().time()
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': 'working',
                        'name': config['name'],
                        'response_time': end_time - start_time,
                        'specialization': config['specialization'],
                        'tokens_per_sec': config['tokens_per_sec'],
                        'test_response': data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response')[:100]
                    }
                else:
                    return {
                        'status': 'failed',
                        'name': config['name'],
                        'error': f"HTTP {response.status}"
                    }
    
    async def _test_openai_compatible(self, config: Dict) -> Dict:
        """Test OpenAI-compatible APIs"""
        headers = {
            'Authorization': f'Bearer {config["key"]}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': config['models'][0],
            'messages': [{'role': 'user', 'content': f'Say "Hello from {config["name"]}!" if you\'re working.'}],
            'max_tokens': 50,
            'temperature': 0.3
        }
        
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config['endpoint'], headers=headers, json=payload) as response:
                end_time = asyncio.get_event_loop().time()
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': 'working',
                        'name': config['name'],
                        'response_time': end_time - start_time,
                        'specialization': config['specialization'],
                        'tokens_per_sec': config['tokens_per_sec'],
                        'test_response': data.get('choices', [{}])[0].get('message', {}).get('content', 'No response')[:100]
                    }
                else:
                    return {
                        'status': 'failed',
                        'name': config['name'],
                        'error': f"HTTP {response.status}: {await response.text()}"
                    }
    
    async def _discover_api(self, api_name: str, endpoint: str) -> Dict:
        """Attempt to discover new free API access"""
        print(f"  🕵️  Discovering {api_name}...")
        
        # Basic endpoint check
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, timeout=5) as response:
                    return {
                        'accessible': response.status != 404,
                        'status_code': response.status,
                        'endpoint': endpoint,
                        'discovery_note': f'Endpoint accessible, may offer free tier'
                    }
        except:
            return {
                'accessible': False,
                'endpoint': endpoint,
                'discovery_note': 'Endpoint not accessible or requires authentication'
            }
    
    def _generate_harvest_report(self) -> Dict:
        """Generate comprehensive free API harvest report"""
        working_apis = {k: v for k, v in self.test_results.items() if v.get('status') == 'working'}
        potential_apis = {k: v for k, v in self.test_results.items() if v.get('status') == 'requires_key'}
        discovered_apis = {k: v for k, v in self.discovered_apis.items() if v.get('accessible')}
        
        total_tokens_per_sec = sum(api.get('tokens_per_sec', 0) for api in working_apis.values())
        
        report = {
            'harvest_timestamp': datetime.now().isoformat(),
            'summary': {
                'working_apis': len(working_apis),
                'potential_apis': len(potential_apis),
                'discovered_apis': len(discovered_apis),
                'total_tokens_per_sec': total_tokens_per_sec,
                'compound_intelligence_ready': len(working_apis) >= 2
            },
            'working_apis': working_apis,
            'potential_apis': potential_apis,
            'discovered_apis': discovered_apis,
            'optimization_recommendations': self._generate_optimization_recommendations(working_apis)
        }
        
        return report
    
    def _generate_optimization_recommendations(self, working_apis: Dict) -> List[str]:
        """Generate optimization recommendations based on available APIs"""
        recommendations = []
        
        if len(working_apis) >= 3:
            recommendations.append("🚀 COMPOUND INTELLIGENCE: Use parallel model coordination for 10x effects")
            
        if any(api.get('tokens_per_sec', 0) > 400 for api in working_apis.values()):
            recommendations.append("⚡ SPEED OPTIMIZATION: Use ultra-fast APIs for real-time applications")
            
        if any(api.get('specialization') == 'reasoning' for api in working_apis.values()):
            recommendations.append("🧠 REASONING BOOST: Use reasoning specialist for complex problems")
            
        if len(working_apis) >= 2:
            recommendations.append("🔄 LOAD BALANCING: Distribute queries across APIs for maximum throughput")
            
        recommendations.append("💡 COST OPTIMIZATION: 100% free compute - unlimited experimentation possible")
        
        return recommendations

async def main():
    """Run comprehensive free API harvest"""
    harvester = FreeAPIHarvester()
    
    print("🔍 STARTING COMPREHENSIVE FREE API HARVEST")
    print("="*60)
    
    report = await harvester.harvest_all_free_apis()
    
    print("\n" + "="*60)
    print("🎯 FREE API HARVEST COMPLETE")
    print("="*60)
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"free_api_harvest_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"💾 Detailed report saved: {filename}")
    
    # Print summary
    print("\n🎯 HARVEST SUMMARY:")
    print(f"✅ Working APIs: {report['summary']['working_apis']}")
    print(f"⏳ Potential APIs: {report['summary']['potential_apis']}")  
    print(f"🔍 Discovered APIs: {report['summary']['discovered_apis']}")
    print(f"⚡ Total Speed: {report['summary']['total_tokens_per_sec']} tokens/sec")
    print(f"🧠 Compound Ready: {report['summary']['compound_intelligence_ready']}")
    
    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
    for rec in report['optimization_recommendations']:
        print(f"  {rec}")
    
    print("\n🚀 FREE COMPUTE HARVESTED - READY FOR AUTONOMOUS INNOVATION!")

if __name__ == "__main__":
    asyncio.run(main())