#!/usr/bin/env python3
import os
import getpass
from pathlib import Path

# 設定參數
USER = getpass.getuser()
PROJECT_DIR = "/Users/storm/Projects/qwen-mlx"
PYTHON_PATH = f"{PROJECT_DIR}/.venv/bin/python"
SERVER_SCRIPT = f"{PROJECT_DIR}/server.py"
MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit" # 建議用 Qwen，視覺能力更強
PLIST_NAME = "com.storm.qwen_mlx.plist"
PLIST_PATH = f"/Users/{USER}/Library/LaunchAgents/{PLIST_NAME}"

# 檢查路徑
if not os.path.exists(PYTHON_PATH):
    print(f"❌ 找不到虛擬環境: {PYTHON_PATH}")
    exit(1)

# 產生 Plist 內容
plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.storm.qwen_mlx</string>
    <key>ProgramArguments</key>
    <array>
        <string>{PYTHON_PATH}</string>
        <string>{SERVER_SCRIPT}</string>
        <string>--model</string>
        <string>{MODEL}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{PROJECT_DIR}/server.log</string>
    <key>StandardErrorPath</key>
    <string>{PROJECT_DIR}/server.log</string>
    <key>WorkingDirectory</key>
    <string>{PROJECT_DIR}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HF_HUB_OFFLINE</key>
        <string>0</string>
    </dict>
</dict>
</plist>
"""

# 寫入檔案
with open(PLIST_PATH, "w") as f:
    f.write(plist_content)

print(f"✅ 已建立服務設定檔: {PLIST_PATH}")
print("\n請執行以下指令來啟用開機啟動：")
print(f"launchctl load {PLIST_PATH}")
print("\n如果要停止並停用：")
print(f"launchctl unload {PLIST_PATH}")
print(f"\n你可以透過查看此檔案來確認執行日誌: {PROJECT_DIR}/server.log")
