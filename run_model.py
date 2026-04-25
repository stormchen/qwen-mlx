from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import sys

# 模型名稱
#model_path = "Intel/Qwen3.6-27B-4.5b-mlx-AutoRound"
model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"

def run_inference(image_url, prompt_text):
    print(f"正在載入模型 {model_path}... (這可能需要一點時間，且需下載模型檔案)")
    try:
        # 載入模型與處理器
        model, processor = load(model_path)
        mlx_cfg = load_config(model_path)

        # 格式化提示詞
        formatted = apply_chat_template(processor, mlx_cfg, prompt_text, num_images=1)

        print("正在生成回應...\n")
        # 生成結果
        response = generate(model, processor, formatted, image=[image_url], max_tokens=1024)
        
        print("-" * 30)
        print("模型回應：")
        print(response.text)
        print("-" * 30)
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")

if __name__ == "__main__":
    # 預設範例
    default_image = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
    default_prompt = "請用繁體中文詳細描述這張圖片的內容。"
    
    run_inference(default_image, default_prompt)
