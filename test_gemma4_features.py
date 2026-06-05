#!/usr/bin/env python3
"""
Gemma 4 最佳實踐功能驗證測試腳本。
"""

import json
import httpx
import time

BASE_URL = "http://127.0.0.1:8080"

def test_health():
    print("📋 檢查健康狀況...")
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=10)
        print(f"  ✅ /health 回傳: {resp.status_code}, 內容: {resp.json()}")
        return resp.json()
    except Exception as e:
        print(f"  ❌ 失敗: {e}")
        return None

def test_history_cleanup_openai():
    print("\n💬 測試 OpenAI 格式歷史思維鏈清洗 (/v1/chat/completions)...")
    # 模擬多輪對話歷史，其中 assistant 包含思維鏈
    payload = {
        "model": "mlx-community/gemma-4-12B-it-4bit",
        "messages": [
            {"role": "user", "content": "1 + 1 等於多少？"},
            {"role": "assistant", "content": "<|channel>thought\n這是先前多餘的推理過程，不應該被傳給模型。<channel|>1 + 1 等於 2。"},
            {"role": "user", "content": "那再加上 2 呢？"}
        ],
        "max_tokens": 100,
        "temperature": 0.0
    }
    
    try:
        # 發送請求，並在此時觀察 server.log 以確認中間件是否觸發
        resp = httpx.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30)
        print(f"  ✅ 回應狀態碼: {resp.status_code}")
        res_data = resp.json()
        print(f"  ✅ 模型回應: {res_data['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"  ❌ 失敗: {e}")

def test_history_cleanup_anthropic():
    print("\n💬 測試 Anthropic 格式歷史思維鏈清洗 (/v1/messages)...")
    payload = {
        "model": "mlx-community/gemma-4-12B-it-4bit",
        "messages": [
            {"role": "user", "content": "1 + 1 等於多少？"},
            {"role": "assistant", "content": "<|channel>thought\n這是先前多餘的推理過程，不應該被傳給模型。<channel|>1 + 1 等於 2。"},
            {"role": "user", "content": "那再加上 2 呢？"}
        ],
        "max_tokens": 100,
        "temperature": 0.0
    }
    
    try:
        resp = httpx.post(f"{BASE_URL}/v1/messages", json=payload, timeout=30)
        print(f"  ✅ 回應狀態碼: {resp.status_code}")
        res_data = resp.json()
        print(f"  ✅ 模型回應: {res_data['content'][0]['text']}")
    except Exception as e:
        print(f"  ❌ 失敗: {e}")

def test_stream_filter_anthropic():
    print("\n🌊 測試 Anthropic 格式串流與空標籤過濾 (/v1/messages)...")
    payload = {
        "model": "mlx-community/gemma-4-12B-it-4bit",
        "messages": [
            {"role": "user", "content": "請用繁體中文回答，你好！"}
        ],
        "max_tokens": 100,
        "stream": True,
        "temperature": 0.0
    }
    
    try:
        print("  [串流內容開始]")
        with httpx.stream("POST", f"{BASE_URL}/v1/messages", json=payload, timeout=30) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta":
                            text = data["delta"].get("text", "")
                            # 印出收到的片段，看看有沒有含有特殊的標籤
                            print(text, end="", flush=True)
                    except Exception:
                        pass
        print("\n  [串流內容結束]")
    except Exception as e:
        print(f"  ❌ 失敗: {e}")

def test_stream_filter_openai():
    print("\n🌊 測試 OpenAI 格式串流與空標籤過濾 (/v1/chat/completions)...")
    payload = {
        "model": "mlx-community/gemma-4-12B-it-4bit",
        "messages": [
            {"role": "user", "content": "請用繁體中文回答，你好！"}
        ],
        "max_tokens": 100,
        "stream": True,
        "temperature": 0.0
    }
    
    try:
        print("  [串流內容開始]")
        with httpx.stream("POST", f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            text = choices[0].get("delta", {}).get("content", "")
                            print(text, end="", flush=True)
                    except Exception:
                        pass
        print("\n  [串流內容結束]")
    except Exception as e:
        print(f"  ❌ 失敗: {e}")

if __name__ == "__main__":
    test_health()
    test_history_cleanup_openai()
    test_history_cleanup_anthropic()
    test_stream_filter_anthropic()
    test_stream_filter_openai()
