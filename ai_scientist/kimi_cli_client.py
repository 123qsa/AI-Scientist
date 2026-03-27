"""
Subprocess-based client for Kimi CLI.
This wraps the kimi CLI command to make chat completion requests.
"""
import json
import subprocess
from typing import List, Dict, Any, Optional


class KimiCLIClient:
    """A client that uses the kimi CLI subprocess for chat completions."""

    def __init__(self, model: str = "kimi-k2.5"):
        self.model = model
        self.chat = KimiChat()


class KimiChat:
    """Chat interface using kimi CLI."""

    def __init__(self):
        self.completions = KimiChatCompletions()


class KimiChatCompletions:
    """Chat completions interface using kimi CLI."""

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: int = 1,
        stop: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """Create a chat completion using kimi CLI."""
        # For n > 1, we need to make multiple calls
        if n > 1:
            results = []
            for _ in range(n):
                result = self._create_single(model, messages, temperature, max_tokens, max_completion_tokens, stop, seed)
                results.append(result)
            # Combine results - return first one but log warning
            return results[0]

        return self._create_single(model, messages, temperature, max_tokens, max_completion_tokens, stop, seed)

    def _create_single(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        stop: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """Create a single chat completion using kimi CLI."""
        # Convert messages to a prompt
        prompt = self._messages_to_prompt(messages)

        # Build kimi CLI command - use --quiet for non-interactive mode
        cmd = ["kimi", "--quiet", "--model", "kimi-code/kimi-for-coding"]

        # Add prompt
        cmd.extend(["--prompt", prompt])

        # Execute kimi CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode != 0:
            raise RuntimeError(f"kimi CLI error: {result.stderr}")

        content = result.stdout.strip()

        # Return in OpenAI-compatible format
        return KimiResponse(content, model)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a single prompt string."""
        parts = []
        system_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        # Prepend system messages at the beginning
        if system_parts:
            system_prompt = "\n".join(system_parts)
            parts.insert(0, f"System: {system_prompt}")

        return "\n\n".join(parts)


class KimiResponse:
    """OpenAI-compatible response object."""

    def __init__(self, content: str, model: str):
        self.choices = [KimiChoice(content)]
        self.model = model


class KimiChoice:
    """OpenAI-compatible choice object."""

    def __init__(self, content: str):
        self.message = KimiMessage(content)


class KimiMessage:
    """OpenAI-compatible message object."""

    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"
