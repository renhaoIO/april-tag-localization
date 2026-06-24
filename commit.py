#!/usr/bin/env python3
"""
半自动 Git 提交脚本 — 手动确认版本号和修改内容后自动 commit + push

用法:
    python commit.py                           # 交互式
    python commit.py v1.2.3                    # 指定版本，交互式输入描述
    python commit.py v1.2.3 "修复坐标映射bug"   # 指定版本和描述
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_git() -> str:
    """按优先级查找 git: PATH → 标准安装 → PortableGit"""
    candidates = [
        shutil.which("git"),
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\bin\git.exe",
        os.path.expandvars(r"%USERPROFILE%\.workbuddy\vendor\PortableGit\mingw64\bin\git.exe"),
        os.path.expandvars(r"%USERPROFILE%\.workbuddy\vendor\PortableGit\bin\git.exe"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("找不到 git.exe，请安装 Git for Windows")


GIT = _find_git()


def run(cmd: list[str]) -> str:
    """执行 git 命令并返回输出"""
    result = subprocess.run(
        [GIT] + cmd,
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        cwd=REPO_DIR,
    )
    if result.returncode != 0:
        print(f"❌ 失败: git {' '.join(cmd)}\n{result.stderr}")
        sys.exit(1)
    return (result.stdout or "").strip()


def main():
    # 1. 版本号
    if len(sys.argv) > 1:
        version = sys.argv[1]
    else:
        version = input("版本号 (如 v1.2.3): ").strip()
        if not version:
            print("❌ 版本号不能为空")
            sys.exit(1)

    # 2. 修改内容描述
    if len(sys.argv) > 2:
        desc = sys.argv[2]
    else:
        desc = input("修改内容 (简要描述本次改动): ").strip()

    # 3. 检查改动
    status = run(["status", "--porcelain"])
    if not status:
        print("ℹ️ 无改动，无需提交")
        return

    print(f"\n📋 待提交文件:")
    for line in status.splitlines():
        print(f"   {line}")

    # 4. 确认
    msg = f"{version} — {desc}"
    confirm = input(f"\n⚠️ 确认提交并推送?\n   {msg}\n   [y/N]: ").strip().lower()
    if confirm != "y":
        print("已取消")
        return

    # 5. 提交 + 推送
    run(["add", "-A"])
    run(["commit", "-m", msg])
    run(["push"])

    print(f"\n✅ 已推送: {msg}")


if __name__ == "__main__":
    main()
