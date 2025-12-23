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
from io import StringIO
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

def load_test_questions(csv_path_or_url: str) -> List[Dict[str, Any]]:
    """Load test questions from CSV file or Google Sheets URL"""
    questions = []
    
    if csv_path_or_url.startswith(('http://', 'https://')):
        url = csv_path_or_url
        if '/edit' in url or '/spreadsheets/d/' in url:
            import re
            sheet_id_match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
            if sheet_id_match:
                sheet_id = sheet_id_match.group(1)
                gid_match = re.search(r'[#&]gid=(\d+)', url)
                gid = gid_match.group(1) if gid_match else '0'
                url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        print(f"📥 Fetching CSV from Google Sheets: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            csv_content = response.text
            f = StringIO(csv_content)
        except Exception as e:
            print(f"❌ Failed to fetch Google Sheets: {e}")
            raise
    else:
        print(f"📂 Loading CSV from local file: {csv_path_or_url}")
        f = open(csv_path_or_url, 'r', encoding='utf-8')
    
    try:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            question_text = row.get('Question', '').strip()
            question_id = row.get('ID', '').strip()
            
            if not question_text:
                print(f"⚠️  Skipping row {row_num}: Empty question (ID: {question_id})")
                continue
            
            if not question_id:
                question_id = f"row_{row_num}"
            
            questions.append({
                'id': question_id,
                'question': question_text,
                'route_expected': row.get('route_expected', '').strip(),
                'tag': row.get('tag', '').strip(),
                'gold_answer': row.get('gold_answer', '').strip()
            })
    finally:
        if not csv_path_or_url.startswith(('http://', 'https://')):
            f.close()
    
    return questions

def query_klara_api(question: str, user_id: str = "test_user", max_retries: int = 2) -> Dict[str, Any]:
    """Query the Klara API with a question"""
    if not question or not question.strip():
        return {
            'error': 'Empty question provided',
            'answer': '',
            'route': 'error',
            'success': False
        }
    
    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "question": question,
                "user_id": user_id,
                "client_ip": "127.0.0.1",
                "max_chunks": 5,
                "include_sources": True
            }
            
            timeout = 120 if attempt == 1 else 180
            response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result
        except requests.exceptions.Timeout as e:
            partial_response = None
            if hasattr(e, 'response') and e.response is not None:
                try:
                    partial_response = e.response.json()
                except:
                    pass
            
            if attempt < max_retries:
                print(f"   ⚠️  Request timeout (attempt {attempt}/{max_retries}), retrying with longer timeout...")
                time.sleep(5)
                continue
            else:
                answer = ''
                route = 'error'
                if partial_response:
                    answer = partial_response.get('answer', '')
                    route = partial_response.get('route', 'error')
                return {
                    'error': f'Request timeout after {max_retries} attempts',
                    'answer': answer,
                    'route': route,
                    'success': False
                }
        except requests.exceptions.HTTPError as e:
            answer = ''
            route = 'error'
            error_detail = "Unknown error"
            
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    answer = error_data.get('answer', '')
                    route = error_data.get('route', 'error')
                    error_detail = error_data.get('detail', error_data.get('error', str(e)))
                except:
                    try:
                        error_detail = e.response.text[:200]
                    except:
                        error_detail = str(e)
            
            if e.response and e.response.status_code == 422:
                return {
                    'error': f'422 Validation Error: {error_detail}',
                    'answer': answer,
                    'route': route,
                    'success': False
                }
            elif attempt < max_retries and e.response and e.response.status_code >= 500:
                print(f"   ⚠️  Server error {e.response.status_code} (attempt {attempt}/{max_retries}), retrying...")
                time.sleep(5)
                continue
            else:
                return {
                    'error': f'HTTP {e.response.status_code if e.response else "unknown"}: {error_detail}',
                    'answer': answer,
                    'route': route,
                    'success': False
                }
        except Exception as e:
            if attempt < max_retries:
                print(f"   ⚠️  Error (attempt {attempt}/{max_retries}): {str(e)[:100]}, retrying...")
                time.sleep(5)
                continue
            else:
                return {
                    'error': str(e),
                    'answer': '',
                    'route': 'error',
                    'success': False
                }
    
    return {
        'error': 'Failed after all retries',
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
    
    if not actual_answer or len(actual_answer.strip()) < 10:
        return {
            'score': 'not_similar',
            'reason': 'Answer is empty or too short - likely blocked or no data returned'
        }
    
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
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.core.config import Config
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
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
        
        try:
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
        if csv_path.startswith(('http://', 'https://')):
            output_dir = os.path.join(os.path.dirname(__file__), 'data')
        else:
            output_dir = os.path.dirname(csv_path)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"test_results_{timestamp}.json")
    
    questions = load_test_questions(csv_path)
    results = []
    
    print(f"🧪 Running {len(questions)} test questions...")
    print("=" * 60)
    
    for idx, q in enumerate(questions, 1):
        if not q.get('question') or not q['question'].strip():
            print(f"\n[{idx}/{len(questions)}] ⚠️  Skipping: Empty question (ID: {q.get('id', 'unknown')})")
            continue
        
        print(f"\n[{idx}/{len(questions)}] Testing: {q['question'][:60]}...")
        print(f"   ID: {q.get('id', 'unknown')}, Expected route: {q['route_expected']}, Tag: {q['tag']}")
        
        start_time = time.time()
        api_response = query_klara_api(q['question'])
        elapsed_time = time.time() - start_time
        
        actual_answer = api_response.get('answer', '') or ''
        actual_route = api_response.get('route', 'unknown') or 'error'
        
        if not actual_answer:
            error_msg = api_response.get('error', '')
            if error_msg:
                actual_answer = f"[Error: {error_msg}]"
            else:
                actual_answer = "[No answer returned]"
        
        print(f"   Route: {actual_route}, Time: {elapsed_time:.2f}s")
        print(f"   Answer preview: {actual_answer[:100]}...")
        
        print("   Judging similarity...")
        judgment = judge_answer_similarity(
            question=q['question'],
            gold_answer=q['gold_answer'],
            actual_answer=actual_answer,
            route_expected=q['route_expected'],
            actual_route=actual_route
        )
        
        print(f"   Score: {judgment['score']} - {judgment['reason']}")
        
        result = {
            'id': q.get('id', f'row_{idx}'),
            'question': q.get('question', ''),
            'route_expected': q.get('route_expected', ''),
            'actual_route': actual_route,
            'tag': q.get('tag', ''),
            'gold_answer': q.get('gold_answer', ''),
            'actual_answer': actual_answer,
            'judgment': judgment,
            'api_response_time': round(elapsed_time, 2),
            'api_success': api_response.get('success', True),
            'api_error': api_response.get('error'),
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
        
        scores = [r['judgment']['score'] for r in results]
        extremely_similar = scores.count('extremely_similar')
        somewhat_similar = scores.count('somewhat_similar')
        not_similar = scores.count('not_similar')
        route_matches = sum(1 for r in results if r['route_expected'] == r['actual_route'])
        
        summary = {
            'total_questions': len(questions),
            'completed_questions': len(results),
            'extremely_similar': extremely_similar,
            'somewhat_similar': somewhat_similar,
            'not_similar': not_similar,
            'route_match_count': route_matches,
            'route_match_percentage': round((route_matches / len(results)) * 100, 2) if results else 0,
            'test_timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Progress saved ({len(results)}/{len(questions)})")
        
        time.sleep(1)
    
    scores = [r['judgment']['score'] for r in results]
    extremely_similar = scores.count('extremely_similar')
    somewhat_similar = scores.count('somewhat_similar')
    not_similar = scores.count('not_similar')
    
    route_matches = sum(1 for r in results if r['route_expected'] == r['actual_route'])
    
    summary = {
        'total_questions': len(questions),
        'completed_questions': len(results),
        'extremely_similar': extremely_similar,
        'somewhat_similar': somewhat_similar,
        'not_similar': not_similar,
        'route_match_count': route_matches,
        'route_match_percentage': round((route_matches / len(questions)) * 100, 2) if len(questions) > 0 else 0,
        'test_timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"Total questions: {len(questions)}")
    if len(questions) > 0:
        print(f"✅ Extremely similar: {extremely_similar} ({extremely_similar/len(questions)*100:.1f}%)")
        print(f"⚠️  Somewhat similar: {somewhat_similar} ({somewhat_similar/len(questions)*100:.1f}%)")
        print(f"❌ Not similar: {not_similar} ({not_similar/len(questions)*100:.1f}%)")
        print(f"🎯 Route matches: {route_matches}/{len(questions)} ({route_matches/len(questions)*100:.1f}%)")
    else:
        print("⚠️  No questions loaded - check CSV format and column names")
    print(f"\n📁 Final results saved to: {output_path}")
    print(f"   Total questions: {len(questions)}, Completed: {len(results)}")
    
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
    default_csv = os.getenv('TEST_CSV_URL') or os.path.join(os.path.dirname(__file__), 'data', 'test.csv')
    parser.add_argument(
        '--csv',
        default=default_csv,
        help='Path to test.csv file or Google Sheets CSV export URL (can also set TEST_CSV_URL env var)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSON file path (default: test_results_TIMESTAMP.json)'
    )
    
    args = parser.parse_args()
    
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
    
    run_tests(args.csv, args.output)

if __name__ == "__main__":
    main()

