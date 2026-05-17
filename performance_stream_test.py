#!/usr/bin/env python3
"""
Measure time-to-first-token and tokens/sec for a streaming /v1/messages endpoint.

This script sends a POST with `stream: true` and parses SSE `data:` events
to capture content deltas. Token counting is approximated by splitting on
whitespace (words). Usage:

  python performance_stream_test.py --url http://192.168.68.112:8080/v1/messages --model mlx-community/gemma-4-e4b-it-4bit

Options allow repeating the test and printing JSON output.
"""
import argparse
import time
import requests
import json
import sys


def parse_sse_lines(response):
    """Yield each server-sent data payload as parsed JSON."""
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if line.startswith('data:'):
            payload = line[len('data:'):].strip()
            # Some servers send '[DONE]' or similar; ignore non-json
            try:
                obj = json.loads(payload)
                yield obj
            except Exception:
                continue


def measure_stream(url, model, message, timeout, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": True,
        "max_tokens": max_tokens,
    }
    start = time.perf_counter()
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            first_token_time = None
            first_token_ts = None
            last_token_ts = None
            full_text = ""
            for obj in parse_sse_lines(resp):
                now = time.perf_counter()
                # look for content_block_delta events
                if obj.get('type') == 'content_block_delta':
                    delta = obj.get('delta') or obj.get('content_block') or {}
                    # delta may be nested under different keys depending on server
                    text = None
                    if 'delta' in obj and isinstance(obj['delta'], dict):
                        text = obj['delta'].get('text')
                    if text is None:
                        # try content_block_delta structure
                        text = obj.get('content_block', {}).get('text') or obj.get('delta', {}).get('text')
                    if text:
                        if first_token_time is None:
                            first_token_time = now - start
                            first_token_ts = now
                        full_text += text
                        last_token_ts = now
                # also handle message_stop to finish early
                if obj.get('type') == 'message_stop' or obj.get('type') == 'message_delta':
                    # continue until stream end; some servers send message_stop then stop
                    pass
            end = time.perf_counter()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

    total_time = end - start
    if first_token_time is None:
        return {"error": "no tokens received", "total_time": total_time}

    char_count = len(full_text)
    # word count (useful for English; misleading for Chinese)
    word_count = len(full_text.split())
    tokens_duration = (last_token_ts - first_token_ts) if (first_token_ts and last_token_ts and last_token_ts > first_token_ts) else (end - (start + first_token_time))
    chars_per_sec = char_count / tokens_duration if tokens_duration and tokens_duration > 0 else None

    return {
        "time_to_first_token_s": first_token_time,
        "total_time_s": total_time,
        "char_count": char_count,
        "word_count": word_count,
        "chars_per_sec": chars_per_sec,
        "text_snippet": (full_text[:400] + '...') if len(full_text) > 400 else full_text,
    }


def parse_args():
    p = argparse.ArgumentParser(description='Streaming perf tester for /v1/messages')
    p.add_argument('--url', required=True, help='Full endpoint URL (e.g. http://host:port/v1/messages)')
    p.add_argument('--model', required=True, help='Model id to request')
    p.add_argument('--message', default='Please write a short paragraph describing the benefits of AI in software development.', help='Message content (prompt) to send to the model')
    p.add_argument('--max-tokens', type=int, default=128, help='Maximum number of tokens to generate')
    p.add_argument('--timeout', type=float, default=120.0, help='Request timeout seconds')
    p.add_argument('--repeat', type=int, default=1, help='Number of repeated runs')
    p.add_argument('--out', default=None, help='Optional output JSON file')
    return p.parse_args()


def main():
    args = parse_args()
    results = []
    for i in range(args.repeat):
        print(f'Run {i+1}/{args.repeat} ...', file=sys.stderr)
        r = measure_stream(args.url, args.model, args.message, args.timeout, args.max_tokens)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        results.append(r)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
