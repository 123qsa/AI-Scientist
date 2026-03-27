#!/usr/bin/env python3
"""
验证 Kimi OAuth 配置的测试脚本。
运行此脚本检查 OAuth 登录是否正常工作。
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_scientist.llm import (
    validate_kimi_oauth,
    get_kimi_oauth_token,
    KIMI_OAUTH_CREDENTIALS_PATH,
    create_client
)


def main():
    print("=" * 60)
    print("Kimi OAuth 验证工具")
    print("=" * 60)
    print()

    # 检查凭证文件是否存在
    print(f"1. 检查凭证文件...")
    print(f"   路径: {KIMI_OAUTH_CREDENTIALS_PATH}")
    if os.path.exists(KIMI_OAUTH_CREDENTIALS_PATH):
        print(f"   状态: ✓ 文件存在")
    else:
        print(f"   状态: ✗ 文件不存在")
        print()
        print("解决方案:")
        print("  1. 安装 Kimi CLI: pip install kimi-cli")
        print("  2. 登录: kimi login")
        return 1

    # 验证 OAuth 状态
    print()
    print("2. 验证 OAuth 凭证...")
    is_valid, message = validate_kimi_oauth()
    if is_valid:
        print(f"   状态: ✓ {message}")
    else:
        print(f"   状态: ✗ {message}")
        return 1

    # 尝试读取 token
    print()
    print("3. 读取 OAuth Token...")
    token = get_kimi_oauth_token()
    if token:
        # 只显示 token 的前 10 个字符和后 5 个字符
        masked = token[:10] + "..." + token[-5:] if len(token) > 15 else "***"
        print(f"   Token: {masked}")
        print(f"   长度: {len(token)} 字符")
    else:
        print("   状态: ✗ 无法读取 token")
        return 1

    # 尝试创建客户端并测试连接
    print()
    print("4. 测试 API 连接 (使用 Kimi CLI)...")
    try:
        client, model = create_client("kimi-k2.5")
        # 发送一个简单的测试请求
        response = client.chat.completions.create(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10
        )
        print(f"   状态: ✓ 连接成功")
        print(f"   模型响应: {response.choices[0].message.content[:50]}...")
    except Exception as e:
        print(f"   状态: ✗ 连接失败")
        print(f"   错误: {e}")
        return 1

    print()
    print("=" * 60)
    print("✓ 所有检查通过！OAuth 配置正确。")
    print("=" * 60)
    print()
    print("现在可以运行 AI-Scientist:")
    print('  python launch_scientist.py --model "kimi-k2.5" --experiment nanoGPT_lite --num-ideas 2')

    return 0


if __name__ == "__main__":
    sys.exit(main())
