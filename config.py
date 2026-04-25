"""
MLX-VLM OpenAI 相容 API 服務設定模組。

提供預設設定值與環境變數覆蓋機制，讓使用者可以
透過 .env 檔案或環境變數來自訂伺服器行為。
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    """伺服器設定，支援環境變數覆蓋。"""

    # === 伺服器基本設定 ===
    host: str = "0.0.0.0"
    port: int = 8080

    # === 模型設定 ===
    # 預設載入的模型（HuggingFace repo 或本機路徑）
    model: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    # Adapter 權重路徑（可選）
    adapter_path: Optional[str] = None
    # 是否信任遠端程式碼
    trust_remote_code: bool = False

    # === 推論效能設定 ===
    # KV Cache 量化位元數（例如 8, 4, 3.5）
    kv_bits: Optional[float] = None
    # KV Cache 量化方案（uniform 或 turboquant）
    kv_quant_scheme: str = "uniform"
    # KV Cache 量化 group size
    kv_group_size: int = 64
    # 最大 KV Cache 大小（token 數）
    max_kv_size: Optional[int] = None

    # === 推測解碼（Speculative Decoding）===
    draft_model: Optional[str] = None
    draft_kind: str = "dflash"
    draft_block_size: Optional[int] = None

    # === Vision Feature Cache ===
    vision_cache_size: int = 20

    # === 日誌等級 ===
    log_level: str = "INFO"

    # === Top Logprobs ===
    top_logprobs_k: int = 0

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """從環境變數建立設定實例。

        環境變數名稱統一使用 MLX_VLM_ 前綴，例如：
          - MLX_VLM_HOST=127.0.0.1
          - MLX_VLM_PORT=9090
          - MLX_VLM_MODEL=mlx-community/Qwen2-VL-2B-Instruct-4bit
        """
        def _env(key: str, default=None):
            return os.environ.get(f"MLX_VLM_{key}", default)

        def _env_int(key: str, default: Optional[int] = None) -> Optional[int]:
            val = _env(key)
            return int(val) if val is not None else default

        def _env_float(key: str, default: Optional[float] = None) -> Optional[float]:
            val = _env(key)
            return float(val) if val is not None else default

        def _env_bool(key: str, default: bool = False) -> bool:
            val = _env(key)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        return cls(
            host=_env("HOST", cls.host),
            port=_env_int("PORT", cls.port),
            model=_env("MODEL", cls.model),
            adapter_path=_env("ADAPTER_PATH"),
            trust_remote_code=_env_bool("TRUST_REMOTE_CODE", cls.trust_remote_code),
            kv_bits=_env_float("KV_BITS"),
            kv_quant_scheme=_env("KV_QUANT_SCHEME", cls.kv_quant_scheme),
            kv_group_size=_env_int("KV_GROUP_SIZE", cls.kv_group_size),
            max_kv_size=_env_int("MAX_KV_SIZE"),
            draft_model=_env("DRAFT_MODEL"),
            draft_kind=_env("DRAFT_KIND", cls.draft_kind),
            draft_block_size=_env_int("DRAFT_BLOCK_SIZE"),
            vision_cache_size=_env_int("VISION_CACHE_SIZE", cls.vision_cache_size),
            log_level=_env("LOG_LEVEL", cls.log_level),
            top_logprobs_k=_env_int("TOP_LOGPROBS_K", cls.top_logprobs_k),
        )

    def to_env_vars(self) -> dict[str, str]:
        """將設定轉換為 mlx-vlm server 所需的環境變數。"""
        env = {}

        if self.trust_remote_code:
            env["MLX_TRUST_REMOTE_CODE"] = "true"

        if self.kv_bits is not None:
            env["KV_BITS"] = str(self.kv_bits)

        env["KV_QUANT_SCHEME"] = self.kv_quant_scheme
        env["KV_GROUP_SIZE"] = str(self.kv_group_size)

        if self.max_kv_size is not None:
            env["MAX_KV_SIZE"] = str(self.max_kv_size)

        if self.draft_model:
            env["MLX_VLM_DRAFT_MODEL"] = self.draft_model
            env["MLX_VLM_DRAFT_KIND"] = self.draft_kind
            if self.draft_block_size is not None:
                env["MLX_VLM_DRAFT_BLOCK_SIZE"] = str(self.draft_block_size)

        env["MLX_VLM_VISION_CACHE_SIZE"] = str(self.vision_cache_size)
        env["TOP_LOGPROBS_K"] = str(self.top_logprobs_k)

        return env

    def print_summary(self):
        """印出設定摘要。"""
        print("=" * 60)
        print("  MLX-VLM OpenAI 相容 API 服務設定")
        print("=" * 60)
        print(f"  🌐 主機位址:       {self.host}:{self.port}")
        print(f"  🤖 模型:           {self.model}")
        if self.adapter_path:
            print(f"  🔧 Adapter:        {self.adapter_path}")
        if self.draft_model:
            print(f"  ⚡ 推測解碼:       {self.draft_model} ({self.draft_kind})")
        if self.kv_bits:
            print(f"  💾 KV Cache 量化:  {self.kv_bits} bits ({self.kv_quant_scheme})")
        print(f"  📸 Vision Cache:   {self.vision_cache_size} 個")
        print(f"  📝 日誌等級:       {self.log_level}")
        if self.trust_remote_code:
            print(f"  ⚠️  信任遠端程式碼: 是")
        print("=" * 60)
        print()
        print("  API 端點:")
        print(f"    POST  http://{self.host}:{self.port}/v1/chat/completions")
        print(f"    POST  http://{self.host}:{self.port}/v1/responses")
        print(f"    GET   http://{self.host}:{self.port}/v1/models")
        print(f"    GET   http://{self.host}:{self.port}/health")
        print()
        print("  OpenAI SDK 連線方式:")
        print(f'    client = OpenAI(base_url="http://{self.host}:{self.port}/v1", api_key="mlx-vlm")')
        print("=" * 60)
