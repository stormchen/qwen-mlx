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
    p.add_argument("--ssl-certfile", type=str, default=None)
    p.add_argument("--ssl-keyfile", type=str, default=None)
    return p.parse_args()


def build_config(args):
    """合併環境變數與命令列參數。優先順序：CLI > 環境變數 > 預設值"""
    cfg = ServerConfig.from_env()
    for attr in [
        "host", "port", "model", "adapter_path", "trust_remote_code",
        "kv_bits", "kv_quant_scheme", "kv_group_size", "max_kv_size",
        "draft_model", "draft_kind", "draft_block_size",
        "vision_cache_size", "log_level", "top_logprobs_k",
        "ssl_certfile", "ssl_keyfile",
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
        from mlx_vlm.version import __version__
        from fastapi.responses import JSONResponse
        import uvicorn

        # 注入根路由，讓 Claude Desktop 等工具的健康檢查能正常通過
        @app.get("/", include_in_schema=False)
        async def root():
            return JSONResponse({
                "name": "MLX-VLM API",
                "version": __version__,
                "status": "ok",
                "docs": f"http://{cfg.host}:{cfg.port}/docs",
            })

        # ── Anthropic Messages API 相容層 ──────────────────────────────
        # Claude Desktop 使用 Anthropic 格式 (POST /v1/messages)
        # 這裡將其轉換成 OpenAI 格式，再交給 mlx-vlm 處理
        from fastapi import Request
        from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
        from mlx_vlm.server import get_cached_model, stream_generate, generate
        import mlx_vlm.server as _srv
        import asyncio, time, uuid, json as _json

        @app.post("/v1/messages", include_in_schema=False)
        async def anthropic_messages(request: Request):
            """將 Anthropic Messages API 格式轉換為 OpenAI 格式並回應。"""
            body = await request.json()
            model_id  = body.get("model", cfg.model)
            messages  = body.get("messages", [])
            max_tokens = body.get("max_tokens", 1024)
            stream    = body.get("stream", False)
            system    = body.get("system", None)

            # 建構 OpenAI 格式的 messages
            oai_messages = []
            if system:
                oai_messages.append({"role": "system", "content": system})
            for msg in messages:
                role    = msg.get("role", "user")
                content = msg.get("content", "")
                # content 可能是 string 或 list
                if isinstance(content, list):
                    text_parts = [
                        c.get("text", "") for c in content
                        if c.get("type") == "text"
                    ]
                    content = " ".join(text_parts)
                oai_messages.append({"role": role, "content": content})

            # 取得或載入模型
            try:
                model, processor, model_config = get_cached_model(model_id)
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"type": "error", "error": {"type": "api_error", "message": str(e)}}
                )

            from mlx_vlm.prompt_utils import apply_chat_template
            prompt = apply_chat_template(processor, model_config, oai_messages)

            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            gen_args = _srv.GenerationArguments(
                max_tokens=max_tokens,
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 1.0),
            )

            if stream:
                # 串流回應（SSE 格式）
                async def event_stream():
                    yield f"data: {_json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':model_id,'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
                    yield f"data: {_json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"

                    full_text = ""
                    output_tokens = 0

                    if _srv.response_generator is not None:
                        ctx, token_iter = await asyncio.to_thread(
                            _srv.response_generator.generate,
                            prompt,
                            None,  # images
                            None,  # audio
                            gen_args,
                        )

                        def _next_token():
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        while True:
                            token = await asyncio.to_thread(_next_token)
                            if token is None:
                                break
                            output_tokens += 1
                            t = token.text if hasattr(token, "text") else str(token)
                            full_text += t
                            yield f"data: {_json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':t}})}\n\n"
                            await asyncio.sleep(0.001)
                            if hasattr(token, "finish_reason") and token.finish_reason:
                                break
                    else:
                        # Fallback to stream_generate
                        for token in stream_generate(model, processor, prompt, image=None, max_tokens=max_tokens):
                            output_tokens += 1
                            t = token.text if hasattr(token, "text") else str(token)
                            full_text += t
                            yield f"data: {_json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':t}})}\n\n"
                            await asyncio.sleep(0.001)

                    yield f"data: {_json.dumps({'type':'content_block_stop','index':0})}\n\n"
                    yield f"data: {_json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':output_tokens}})}\n\n"
                    yield f"data: {_json.dumps({'type':'message_stop'})}\n\n"

                return FastAPIStreamingResponse(event_stream(), media_type="text/event-stream")
            else:
                # 非串流回應
                if _srv.response_generator is not None:
                    ctx, token_iter = await asyncio.to_thread(
                        _srv.response_generator.generate,
                        prompt,
                        None,  # images
                        None,  # audio
                        gen_args,
                    )
                    def _consume():
                        res = []
                        for tk in token_iter:
                            res.append(tk.text if hasattr(tk, "text") else str(tk))
                            if hasattr(tk, "finish_reason") and tk.finish_reason:
                                break
                        return "".join(res)
                    text = await asyncio.to_thread(_consume)
                else:
                    response = generate(model, processor, prompt, image=None, max_tokens=max_tokens)
                    text = response.text if hasattr(response, "text") else str(response)

                return JSONResponse({
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "model": model_id,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": len(text.split()),
                    }
                })
        # ── Anthropic 相容層結束 ────────────────────────────────────────

        print(f"\n🚀 啟動伺服器 http://{cfg.host}:{cfg.port}")
        print(f"📖 API 文件: http://{cfg.host}:{cfg.port}/docs\n")
        uvicorn.run(
            app,
            host=cfg.host,
            port=cfg.port,
            log_level=cfg.log_level.lower(),
            ssl_certfile=cfg.ssl_certfile,
            ssl_keyfile=cfg.ssl_keyfile
        )
    except ImportError as e:
        print(f"\n❌ 匯入失敗: {e}\n請執行: uv add mlx-vlm")
        sys.exit(1)


if __name__ == "__main__":
    main()
