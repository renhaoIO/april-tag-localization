#!/usr/bin/env python3
"""
半自动 Git 提交脚本 — 手动确认版本号后自动 commit + push

用法:
    python commit.py              # 交互式输入版本号
    python commit.py v1.2.3       # 直接指定版本号
"""

import subprocess
import sys
from datetime import datetime


def run(cmd: list[str]) -> str:
    """执行命令并返回输出"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ 失败: {' '.join(cmd)}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def main():
    # 1. 确认版本号
    if len(sys.argv) > 1:
        version = sys.argv[1]
    else:
        version = input("输入版本号 (如 v1.2.3): ").strip()
        if not version:
            print("❌ 版本号不能为空")
            sys.exit(1)

    # 2. 检查是否有改动
    status = run(["git", "status", "--porcelain"])
    if not status:
        print("ℹ️ 无改动，无需提交")
        return

    print(f"\n📋 待提交文件:")
    for line in status.splitlines():
        print(f"   {line}")

    # 3. 确认
    confirm = input(f"\n⚠️ 确认以版本 [{version}] 提交并推送? [y/N]: ").strip().lower()
    if confirm != "y":
        print("已取消")
        return

    # 4. 提交 + 推送
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"{version} — {timestamp}"
    run(["git", "add", "-A"])
    run(["git", "commit", "-m", msg])
    run(["git", "push"])

    print(f"\n✅ 已提交并推送: {msg}")


if __name__ == "__main__":
    main()
