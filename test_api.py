#!/usr/bin/env python3
"""
MLX-VLM OpenAI API 測試腳本。

在伺服器啟動後執行，驗證各端點是否正常運作。

使用方式:
    uv run python test_api.py
    uv run python test_api.py --base-url http://localhost:9090/v1
"""

import argparse
import json
import sys

try:
    from openai import OpenAI
except ImportError:
    print("❌ 請先安裝 openai 套件: uv add openai")
    sys.exit(1)


def test_models(client: OpenAI):
    """測試 /v1/models 端點。"""
    print("\n📋 測試 /v1/models ...")
    try:
        models = client.models.list()
        print(f"  ✅ 可用模型數量: {len(models.data)}")
        for m in models.data:
            print(f"     - {m.id}")
        return True
    except Exception as e:
        print(f"  ❌ 失敗: {e}")
        return False


def test_chat_text(client: OpenAI, model: str):
    """測試純文字 Chat Completions。"""
    print(f"\n💬 測試 /v1/chat/completions (文字) ...")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一個有幫助的助手。請用繁體中文回答。"},
                {"role": "user", "content": "你好！請用一句話介紹自己。"},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        content = resp.choices[0].message.content
        print(f"  ✅ 回應: {content[:200]}")
        print(f"  📊 Token 使用: prompt={resp.usage.prompt_tokens}, "
              f"completion={resp.usage.completion_tokens}")
        return True
    except Exception as e:
        print(f"  ❌ 失敗: {e}")
        return False


def test_chat_stream(client: OpenAI, model: str):
    """測試串流 Chat Completions。"""
    print(f"\n🌊 測試 /v1/chat/completions (串流) ...")
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "數 1 到 5。"},
            ],
            max_tokens=50,
            stream=True,
        )
        print("  ✅ 串流回應: ", end="")
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"  ❌ 失敗: {e}")
        return False


def test_chat_image(client: OpenAI, model: str):
    """測試圖像輸入 Chat Completions。"""
    print(f"\n🖼️  測試 /v1/chat/completions (圖像) ...")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "請描述這張圖片。"},
                        {
                            "type": "input_image",
                            "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
                        },
                    ],
                },
            ],
            max_tokens=200,
        )
        content = resp.choices[0].message.content
        print(f"  ✅ 回應: {content[:200]}")
        return True
    except Exception as e:
        print(f"  ❌ 失敗: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="測試 MLX-VLM API")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="API base URL",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="指定測試用模型 ID（預設使用伺服器上的模型）",
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="跳過圖像測試",
    )
    args = parser.parse_args()

    # 初始化 OpenAI 客戶端
    # 針對本地自簽署憑證，我們建立一個不驗證 SSL 的 httpx 客戶端
    import httpx
    http_client = httpx.Client(verify=False) if args.base_url.startswith("https") else None
    
    client = OpenAI(
        base_url=args.base_url,
        api_key="mlx-vlm",
        http_client=http_client
    )
    print(f"🔗 連線至: {args.base_url}")

    # 取得模型列表
    results = []
    results.append(("列出模型", test_models(client)))

    # 取得要測試的模型名稱（優先從 /health 端點取得當前載入的模型）
    model = args.model
    if not model:
        try:
            import httpx
            base = args.base_url.rstrip("/v1").rstrip("/")
            # 針對本地測試忽略 SSL 驗證
            h_resp = httpx.get(f"{base}/health", timeout=5, verify=False)
            health = h_resp.json()
            model = health.get("loaded_model")
        except Exception:
            pass
    if not model:
        try:
            models = client.models.list()
            if models.data:
                model = models.data[0].id
        except Exception:
            pass

    if not model:
        print("\n⚠️  無法取得模型名稱，請使用 --model 參數指定")
        sys.exit(1)

    print(f"\n🤖 使用模型: {model}")

    results.append(("文字對話", test_chat_text(client, model)))
    results.append(("串流對話", test_chat_stream(client, model)))
    if not args.skip_image:
        results.append(("圖像對話", test_chat_image(client, model)))

    # 摘要
    print("\n" + "=" * 40)
    print("  測試結果摘要")
    print("=" * 40)
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 所有測試通過！")
    else:
        print("⚠️  部分測試未通過，請檢查伺服器日誌。")
        sys.exit(1)


if __name__ == "__main__":
    main()
