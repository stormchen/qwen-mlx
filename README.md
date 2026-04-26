# MLX-VLM OpenAI API 伺服器 🚀

在 Mac（Apple Silicon）上透過 **OpenAI 相容 API** 使用視覺語言模型（VLM），
基於 [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 封裝。

支援任何 OpenAI SDK 或相容客戶端直接呼叫本地模型，包括文字、圖像、音訊等多模態輸入。

## ✨ 功能特色

- 🔌 **完整 OpenAI API 相容** — 使用 OpenAI SDK 直接連線
- 🖼️ **多模態支援** — 文字、圖像、音訊輸入
- ⚡ **Continuous Batching** — 自動批次處理多個並行請求
- 🧠 **推測解碼（DFlash）** — 2-3x 加速生成
- 💾 **KV Cache 量化** — TurboQuant / Uniform 減少記憶體使用
- 📸 **Vision Feature Cache** — 多輪對話中自動快取圖像特徵
- 🔧 **工具呼叫（Function Calling）** — 支援 OpenAI 格式的工具呼叫
- 🌊 **串流回應** — SSE 串流輸出

## 📦 安裝

```bash
# 建立虛擬環境
uv venv
source .venv/bin/activate

# 安裝依賴
uv sync
```

## 🚀 快速啟動

```bash
# 使用預設模型（gemma-4-e4b-it-4bit）啟動
uv run python server.py

# 指定模型
uv run python server.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit

# 指定連接埠
uv run python server.py --port 9090

# 啟用 KV Cache 量化（節省記憶體）
uv run python server.py --kv-bits 8

# 啟用推測解碼（加速生成）
uv run python server.py --model Qwen/Qwen3.5-4B --draft-model z-lab/Qwen3.5-4B-DFlash
```

## 🔗 API 端點

啟動後可用以下端點：

| 端點 | 方法 | 說明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI Chat Completions API |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/models` | GET | 列出可用模型 |
| `/health` | GET | 健康檢查 |
| `/unload` | POST | 卸載當前模型 |
| `/docs` | GET | 互動式 API 文件（Swagger UI）|

## 💻 使用範例

### Python（OpenAI SDK）

```python
from openai import OpenAI

# 連線至本地伺服器
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="mlx-vlm"  # 任意值即可
)

# 文字對話
response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    messages=[
        {"role": "system", "content": "你是一個有幫助的助手。"},
        {"role": "user", "content": "你好！"},
    ],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

### 圖像輸入

```python
response = client.chat.completions.create(
    model="mlx-community/gemma-4-e4b-it-4bit",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "請描述這張圖片"},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/image.jpg",
                },
            ],
        }
    ],
    max_tokens=500,
)
print(response.choices[0].message.content)
```

### 串流回應

```python
stream = client.chat.completions.create(
    model="mlx-community/gemma-4-e4b-it-4bit",
    messages=[{"role": "user", "content": "寫一首短詩"}],
    stream=True,
    max_tokens=200,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### cURL

```bash
# 文字對話
curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-e4b-it-4bit",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'

# 健康檢查
curl http://localhost:8080/health
```

## 🧪 測試 API

伺服器啟動後，在另一個終端機執行：

```bash
uv run python test_api.py

# 指定 base URL
uv run python test_api.py --base-url http://localhost:9090/v1

# 跳過圖像測試
uv run python test_api.py --skip-image
```

## ⚙️ 設定

### 命令列參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--host` | 監聽位址 | `0.0.0.0` |
| `--port` | 監聽埠號 | `8080` |
| `--model` | 模型路徑/HF ID | `mlx-community/gemma-4-e4b-it-4bit` |
| `--adapter-path` | Adapter 權重路徑 | - |
| `--trust-remote-code` | 信任遠端程式碼 | `false` |
| `--kv-bits` | KV Cache 量化位元 | `4.0` |
| `--kv-quant-scheme` | 量化方案 | `turboquant` |
| `--draft-model` | DFlash drafter 路徑 | - |
| `--vision-cache-size` | Vision Cache 大小 | `20` |
| `--log-level` | 日誌等級 | `INFO` |

### 環境變數

也可用環境變數設定（加上 `MLX_VLM_` 前綴），參考 `.env.example`。

設定優先順序：**命令列參數 > 環境變數 > 預設值**

## 📂 專案結構

```
qwen-mlx/
├── server.py        # 主要服務啟動器
├── config.py        # 設定管理模組
├── test_api.py      # API 測試腳本
├── run_model.py     # 離線推論腳本（單次呼叫）
├── .env.example     # 環境變數範本
├── pyproject.toml   # 專案設定與依賴
└── README.md        # 本文件
```

## 🤝 相關專案

- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — 核心推論引擎
- [MLX](https://github.com/ml-explore/mlx) — Apple 機器學習框架

## 📄 授權

MIT License
