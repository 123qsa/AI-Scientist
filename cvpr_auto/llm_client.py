"""
LLM 客户端模块 - 支持多种大语言模型
支持：Kimi (kimi-cli), Claude (Anthropic API), OpenAI, DeepSeek
"""

import os
import subprocess
import json
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass
import time


@dataclass
class LLMResponse:
    """LLM 响应封装"""
    content: str
    model: str
    usage: Dict = None
    error: str = None


class BaseLLMClient:
    """LLM 客户端基类"""

    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        raise NotImplementedError

    def generate_stream(self, prompt: str, system_prompt: str = None) -> Generator[str, None, None]:
        """流式生成"""
        raise NotImplementedError


class KimiClient(BaseLLMClient):
    """Kimi 客户端 - 通过 kimi-cli 调用"""

    def __init__(self, model: str = "kimi-k2.5", temperature: float = 0.7):
        super().__init__(model, temperature)
        self._check_kimi_cli()

    def _check_kimi_cli(self):
        """检查 kimi-cli 是否安装"""
        try:
            result = subprocess.run(
                ["kimi", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("kimi-cli not properly configured")
        except FileNotFoundError:
            raise RuntimeError(
                "kimi-cli not found. Please install: pip install kimi-cli && kimi login"
            )

    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        """生成文本"""
        try:
            # 构建 kimi 命令
            cmd = ["kimi", "chat", "--model", self.model, "-t", str(self.temperature)]

            if system_prompt:
                cmd.extend(["--system", system_prompt])

            # 添加 prompt
            cmd.append(prompt)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode == 0:
                return LLMResponse(
                    content=result.stdout.strip(),
                    model=self.model
                )
            else:
                return LLMResponse(
                    content="",
                    model=self.model,
                    error=result.stderr
                )

        except subprocess.TimeoutExpired:
            return LLMResponse(
                content="",
                model=self.model,
                error="Timeout after 300 seconds"
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                error=str(e)
            )

    def generate_stream(self, prompt: str, system_prompt: str = None) -> Generator[str, None, None]:
        """流式生成（kimi-cli 不支持真正的流式，这里模拟）"""
        response = self.generate(prompt, system_prompt)
        if response.error:
            yield f"Error: {response.error}"
        else:
            # 逐字输出模拟流式
            for char in response.content:
                yield char
                time.sleep(0.001)


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude 客户端"""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022",
                 api_key: str = None, temperature: float = 0.7):
        super().__init__(model, temperature)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var.")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install: pip install anthropic")

    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        """生成文本"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                system=system_prompt or "",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return LLMResponse(
                content=message.content[0].text,
                model=self.model,
                usage={
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens
                }
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                error=str(e)
            )

    def generate_stream(self, prompt: str, system_prompt: str = None) -> Generator[str, None, None]:
        """流式生成"""
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            yield f"Error: {str(e)}"


class OpenAIClient(BaseLLMClient):
    """OpenAI 客户端"""

    def __init__(self, model: str = "gpt-4o",
                 api_key: str = None, temperature: float = 0.7):
        super().__init__(model, temperature)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")

    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        """生成文本"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=4096
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                error=str(e)
            )

    def generate_stream(self, prompt: str, system_prompt: str = None) -> Generator[str, None, None]:
        """流式生成"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=4096,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error: {str(e)}"


class LLMClientFactory:
    """LLM 客户端工厂"""

    @staticmethod
    def create_client(provider: str = None, model: str = None, **kwargs) -> BaseLLMClient:
        """
        创建 LLM 客户端

        Args:
            provider: 'kimi', 'anthropic', 'openai'
            model: 模型名称
            **kwargs: 额外参数

        Returns:
            LLM 客户端实例
        """
        # 从环境变量获取默认 provider
        if provider is None:
            provider = os.environ.get("LLM_PROVIDER", "kimi")

        if provider.lower() == "kimi":
            model = model or "kimi-k2.5"
            return KimiClient(model=model, **kwargs)

        elif provider.lower() in ["anthropic", "claude"]:
            model = model or "claude-3-5-sonnet-20241022"
            return AnthropicClient(model=model, **kwargs)

        elif provider.lower() in ["openai", "gpt"]:
            model = model or "gpt-4o"
            return OpenAIClient(model=model, **kwargs)

        else:
            raise ValueError(f"Unknown provider: {provider}")


# 便捷函数
def get_llm_client(provider: str = None, model: str = None) -> BaseLLMClient:
    """获取 LLM 客户端"""
    return LLMClientFactory.create_client(provider, model)


def generate_text(prompt: str, system_prompt: str = None,
                  provider: str = None, model: str = None) -> str:
    """
    简单文本生成函数

    Example:
        >>> response = generate_text("Explain deep learning", provider="kimi")
        >>> print(response)
    """
    client = get_llm_client(provider, model)
    response = client.generate(prompt, system_prompt)

    if response.error:
        raise RuntimeError(f"LLM generation failed: {response.error}")

    return response.content


if __name__ == "__main__":
    # 测试
    print("Testing LLM clients...")

    # 测试 Kimi
    try:
        client = KimiClient()
        response = client.generate("Hello, what is 2+2?")
        print(f"Kimi response: {response.content}")
    except Exception as e:
        print(f"Kimi test failed: {e}")

    # 测试 Claude (需要 API key)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            client = AnthropicClient()
            response = client.generate("Hello, what is 2+2?")
            print(f"Claude response: {response.content}")
        except Exception as e:
            print(f"Claude test failed: {e}")
