#!/usr/bin/env python3
"""
MLX-VLM OpenAI 相容 API 服務啟動器。

使用方式:
    uv run python server.py
    uv run python server.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    uv run python server.py --port 9090 --kv-bits 8
"""

import argparse
import logging
import os
import sys

from config import ServerConfig


def parse_args():
    """解析命令列參數。"""
    p = argparse.ArgumentParser(description="MLX-VLM OpenAI 相容 API 伺服器")
    p.add_argument("--host", type=str, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--adapter-path", type=str, default=None)
    p.add_argument("--trust-remote-code", action="store_true", default=None)
    p.add_argument("--kv-bits", type=float, default=None)
    p.add_argument("--kv-quant-scheme", type=str, default=None)
    p.add_argument("--kv-group-size", type=int, default=None)
    p.add_argument("--max-kv-size", type=int, default=None)
    p.add_argument("--draft-model", type=str, default=None)
    p.add_argument("--draft-kind", type=str, default=None)
    p.add_argument("--draft-block-size", type=int, default=None)
    p.add_argument("--vision-cache-size", type=int, default=None)
    p.add_argument("--log-level", type=str, default=None)
    p.add_argument("--top-logprobs-k", type=int, default=None)
    return p.parse_args()


def build_config(args):
    """合併環境變數與命令列參數。優先順序：CLI > 環境變數 > 預設值"""
    cfg = ServerConfig.from_env()
    for attr in [
        "host", "port", "model", "adapter_path", "trust_remote_code",
        "kv_bits", "kv_quant_scheme", "kv_group_size", "max_kv_size",
        "draft_model", "draft_kind", "draft_block_size",
        "vision_cache_size", "log_level", "top_logprobs_k",
    ]:
        val = getattr(args, attr.replace("-", "_"), None)
        if val is not None:
            setattr(cfg, attr, val)
    return cfg


def main():
    """啟動伺服器。"""
    args = parse_args()
    cfg = build_config(args)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    cfg.print_summary()

    # 注入環境變數供 mlx-vlm server 使用
    for k, v in cfg.to_env_vars().items():
        os.environ[k] = v
    if cfg.model:
        os.environ["PRELOAD_MODEL"] = cfg.model
    if cfg.adapter_path:
        os.environ["PRELOAD_ADAPTER"] = cfg.adapter_path

    try:
        from mlx_vlm.server import app
        import uvicorn

        print(f"\n🚀 啟動伺服器 http://{cfg.host}:{cfg.port}")
        print(f"📖 API 文件: http://{cfg.host}:{cfg.port}/docs\n")
        uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level.lower())
    except ImportError as e:
        print(f"\n❌ 匯入失敗: {e}\n請執行: uv add mlx-vlm")
        sys.exit(1)


if __name__ == "__main__":
    main()
