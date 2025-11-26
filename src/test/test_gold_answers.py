#!/usr/bin/env python3
"""
Test script that runs questions from test.csv through the Klara API
and uses an LLM judge to compare responses with gold answers.

The judge only flags answers that are completely irrelevant (blocked, refused, no data).
It does NOT check for exact matches or length - only direction/semantic alignment.
"""

import os
import sys
import csv
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

def load_test_questions(csv_path: str) -> List[Dict[str, Any]]:
    """Load test questions from CSV file"""
    questions = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'id': row.get('ID', ''),
                'question': row.get('Question', ''),
                'route_expected': row.get('route_expected', ''),
                'tag': row.get('tag', ''),
                'gold_answer': row.get('gold_answer', '')
            })
    
    return questions

def query_klara_api(question: str, user_id: str = "test_user") -> Dict[str, Any]:
    """Query the Klara API with a question"""
    try:
        payload = {
            "question": question,
            "user_id": user_id,
            "max_chunks": 5,
            "include_sources": True
        }
        
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            'error': str(e),
            'answer': '',
            'route': 'error',
            'success': False
        }

def judge_answer_similarity(
    question: str,
    gold_answer: str,
    actual_answer: str,
    route_expected: str,
    actual_route: str
) -> Dict[str, Any]:
    """
    Use LLM to judge if actual answer is similar to gold answer.
    Only flags completely irrelevant answers (blocked, refused, no data).
    Returns: {'score': 'extremely_similar'|'somewhat_similar'|'not_similar', 'reason': str}
    """
    
    # Check for obvious failures
    if not actual_answer or len(actual_answer.strip()) < 10:
        return {
            'score': 'not_similar',
            'reason': 'Answer is empty or too short - likely blocked or no data returned'
        }
    
    # Check for common failure patterns
    failure_indicators = [
        'i found no related data',
        'i could not find',
        'i am unable to',
        'i cannot answer',
        'blocked by',
        'guardrails',
        'content policy',
        'i apologize, but',
        'i\'m sorry, but i cannot',
        'no information available',
        'no data found'
    ]
    
    actual_lower = actual_answer.lower()
    if any(indicator in actual_lower for indicator in failure_indicators):
        return {
            'score': 'not_similar',
            'reason': 'Answer indicates failure (no data, blocked, or refused)'
        }
    
    # Use LLM to judge semantic similarity (lenient)
    judge_prompt = f"""You are a lenient judge comparing two answers to the same question.

Question: "{question}"

Gold Answer (reference):
{gold_answer}

Actual Answer (to evaluate):
{actual_answer}

Your task:
- Determine if the actual answer addresses the question in a similar direction as the gold answer
- Be LENIENT - only flag as "not_similar" if the answer is completely irrelevant, off-topic, or indicates failure
- Do NOT penalize for:
  * Different wording or phrasing
  * Different length
  * Missing minor details
  * Different formatting
  * Slightly different focus areas

Scoring levels:
1. "extremely_similar" - Answer addresses the same topic and provides similar information, even if worded differently
2. "somewhat_similar" - Answer is in the right direction but may be missing key points or has different emphasis
3. "not_similar" - Answer is completely irrelevant, off-topic, or indicates the system failed to answer (blocked, no data, refused)

Respond with ONLY valid JSON:
{{"score": "extremely_similar"|"somewhat_similar"|"not_similar", "reason": "brief explanation"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a lenient judge that only flags completely irrelevant answers. Respond with valid JSON only."
                },
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            # Remove markdown code blocks if present
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            result = json.loads(result_text)
            return {
                'score': result.get('score', 'somewhat_similar'),
                'reason': result.get('reason', 'LLM evaluation completed')
            }
        except json.JSONDecodeError:
            # Fallback: try to extract score from text
            if 'extremely_similar' in result_text.lower():
                return {'score': 'extremely_similar', 'reason': 'Extracted from LLM response'}
            elif 'not_similar' in result_text.lower():
                return {'score': 'not_similar', 'reason': 'Extracted from LLM response'}
            else:
                return {'score': 'somewhat_similar', 'reason': 'Default fallback'}
                
    except Exception as e:
        return {
            'score': 'somewhat_similar',
            'reason': f'Judge error: {str(e)}'
        }

def run_tests(csv_path: str, output_path: str = None):
    """Run all test questions and save results"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.dirname(csv_path), f"test_results_{timestamp}.json")
    
    questions = load_test_questions(csv_path)
    results = []
    
    print(f"🧪 Running {len(questions)} test questions...")
    print("=" * 60)
    
    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] Testing: {q['question'][:60]}...")
        print(f"   Expected route: {q['route_expected']}, Tag: {q['tag']}")
        
        # Query API
        start_time = time.time()
        api_response = query_klara_api(q['question'])
        elapsed_time = time.time() - start_time
        
        actual_answer = api_response.get('answer', '')
        actual_route = api_response.get('route', 'unknown')
        
        print(f"   Route: {actual_route}, Time: {elapsed_time:.2f}s")
        print(f"   Answer preview: {actual_answer[:100]}...")
        
        # Judge similarity
        print("   Judging similarity...")
        judgment = judge_answer_similarity(
            question=q['question'],
            gold_answer=q['gold_answer'],
            actual_answer=actual_answer,
            route_expected=q['route_expected'],
            actual_route=actual_route
        )
        
        print(f"   Score: {judgment['score']} - {judgment['reason']}")
        
        # Store result
        result = {
            'id': q['id'],
            'question': q['question'],
            'route_expected': q['route_expected'],
            'actual_route': actual_route,
            'tag': q['tag'],
            'gold_answer': q['gold_answer'],
            'actual_answer': actual_answer,
            'judgment': judgment,
            'api_response_time': round(elapsed_time, 2),
            'api_success': api_response.get('success', True),
            'api_error': api_response.get('error'),
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
        
        # Be nice to the API
        time.sleep(1)
    
    # Calculate summary statistics
    scores = [r['judgment']['score'] for r in results]
    extremely_similar = scores.count('extremely_similar')
    somewhat_similar = scores.count('somewhat_similar')
    not_similar = scores.count('not_similar')
    
    route_matches = sum(1 for r in results if r['route_expected'] == r['actual_route'])
    
    summary = {
        'total_questions': len(questions),
        'extremely_similar': extremely_similar,
        'somewhat_similar': somewhat_similar,
        'not_similar': not_similar,
        'route_match_count': route_matches,
        'route_match_percentage': round((route_matches / len(questions)) * 100, 2),
        'test_timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"Total questions: {len(questions)}")
    print(f"✅ Extremely similar: {extremely_similar} ({extremely_similar/len(questions)*100:.1f}%)")
    print(f"⚠️  Somewhat similar: {somewhat_similar} ({somewhat_similar/len(questions)*100:.1f}%)")
    print(f"❌ Not similar: {not_similar} ({not_similar/len(questions)*100:.1f}%)")
    print(f"🎯 Route matches: {route_matches}/{len(questions)} ({route_matches/len(questions)*100:.1f}%)")
    print(f"\n📁 Results saved to: {output_path}")
    
    # List not_similar cases
    if not_similar > 0:
        print("\n⚠️  Questions flagged as 'not_similar':")
        for r in results:
            if r['judgment']['score'] == 'not_similar':
                print(f"   [{r['id']}] {r['question'][:60]}...")
                print(f"      Reason: {r['judgment']['reason']}")
    
    return summary

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Klara API against gold answers')
    parser.add_argument(
        '--csv',
        default=os.path.join(os.path.dirname(__file__), 'test.csv'),
        help='Path to test.csv file'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSON file path (default: test_results_TIMESTAMP.json)'
    )
    
    args = parser.parse_args()
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API health check failed. Is the server running at {API_BASE_URL}?")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API at {API_BASE_URL}")
        print(f"   Error: {e}")
        print("\nPlease start the server with: python run_server.py")
        return
    
    print("✅ API is healthy and ready")
    
    # Run tests
    run_tests(args.csv, args.output)

if __name__ == "__main__":
    main()

