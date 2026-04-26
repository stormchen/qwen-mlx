#!/usr/bin/env python3
import os
import getpass

# 設定參數
USER = getpass.getuser()
PROJECT_DIR = "/Users/storm/Projects/qwen-mlx"
PYTHON_PATH = f"{PROJECT_DIR}/.venv/bin/python"
SERVER_SCRIPT = f"{PROJECT_DIR}/server.py"
MODEL = "mlx-community/gemma-4-e4b-it-4bit"
PLIST_NAME = "com.storm.qwen_mlx.plist"
PLIST_PATH = f"/Users/{USER}/Library/LaunchAgents/{PLIST_NAME}"

# 檢查路徑
if not os.path.exists(PYTHON_PATH):
    print(f"❌ 找不到虛擬環境: {PYTHON_PATH}")
    exit(1)

# SSL 自動偵測
CERT_FILE = f"{PROJECT_DIR}/cert.pem"
KEY_FILE = f"{PROJECT_DIR}/key.pem"
ssl_lines = ""
if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
    ssl_lines = (
        f"        <string>--ssl-certfile</string>\n"
        f"        <string>{CERT_FILE}</string>\n"
        f"        <string>--ssl-keyfile</string>\n"
        f"        <string>{KEY_FILE}</string>\n"
    )
    print("🔒 偵測到 SSL 憑證，將自動啟用 HTTPS 支援。")

# 產生 Plist 內容（逐行組合，避免多餘的換行符問題）
lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
    '<plist version="1.0">',
    '<dict>',
    '    <key>Label</key>',
    '    <string>com.storm.qwen_mlx</string>',
    '    <key>ProgramArguments</key>',
    '    <array>',
    f'        <string>{PYTHON_PATH}</string>',
    f'        <string>{SERVER_SCRIPT}</string>',
    '        <string>--model</string>',
    f'        <string>{MODEL}</string>',
]

# 加入 SSL 參數（若有）
if ssl_lines:
    for line in ssl_lines.strip().split("\n"):
        lines.append(line)

lines += [
    '    </array>',
    '    <key>RunAtLoad</key>',
    '    <true/>',
    '    <key>KeepAlive</key>',
    '    <true/>',
    '    <key>StandardOutPath</key>',
    f'    <string>{PROJECT_DIR}/server.log</string>',
    '    <key>StandardErrorPath</key>',
    f'    <string>{PROJECT_DIR}/server.log</string>',
    '    <key>WorkingDirectory</key>',
    f'    <string>{PROJECT_DIR}</string>',
    '    <key>EnvironmentVariables</key>',
    '    <dict>',
    '        <key>PATH</key>',
    '        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>',
    '        <key>HF_HUB_OFFLINE</key>',
    '        <string>0</string>',
    '    </dict>',
    '</dict>',
    '</plist>',
    '',  # 結尾換行
]

plist_content = "\n".join(lines)

# 寫入並驗證格式
with open(PLIST_PATH, "w", newline="\n") as f:
    f.write(plist_content)

# 用 plutil 驗證 plist 格式
import subprocess
result = subprocess.run(["plutil", "-lint", PLIST_PATH], capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ 已建立服務設定檔（格式驗證通過）: {PLIST_PATH}")
else:
    print(f"❌ Plist 格式有誤: {result.stderr}")
    exit(1)

print("\n請執行以下指令來啟用開機啟動：")
print(f"launchctl load {PLIST_PATH}")
print("\n如果要停止並停用：")
print(f"launchctl unload {PLIST_PATH}")
print(f"\n你可以透過查看此檔案來確認執行日誌: {PROJECT_DIR}/server.log")
