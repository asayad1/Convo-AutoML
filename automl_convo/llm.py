"""
This module provides a minimal wrapper around Portkey's chat completion API,
including logging of reasoning content in a readable format.
"""

from typing import Optional
from portkey_ai import Portkey
from utils.logger import Logger
from config import Config, PortkeyConfig, OllamaConfig
import ollama


class PortkeyLLM:
    logger = Logger()

    def __init__(
        self,
        model: str = PortkeyConfig.PORTKEY_MODEL,
        base_url: str = PortkeyConfig.PORTKEY_BASE_URL,
        api_key: str = PortkeyConfig.PORTKEY_API_KEY,
        max_tokens: int = PortkeyConfig.PORTKEY_MAX_TOKENS,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = Portkey(base_url=base_url, api_key=api_key)

    def invoke(self, system_prompt: str, human_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        choice = response.choices[0]
        message = choice.message

        # Show reasoning in a Rich panel if available
        reasoning: Optional[str] = message.reasoning_content
        if reasoning:
            self.logger.reasoning(reasoning)

        #! getting reasoning traces is so incredibly annoying from portkey this doesnt work like 40% of the time
        
        # Try normal Chat Completions content
        content: Optional[str] = message.content

        # If that's empty, try the text field on the choice
        if not content:
            text_field: Optional[str] = choice.text
            
            if text_field:
                content = text_field

        if content:
            return content

        # If still nothing, return nothing and log a warning
        self.logger.info(
            "[PortkeyLLM] WARNING: no 'content' or 'text' returned by model; "
            "reasoning was logged but not used as answer.",
            style="yellow",
        )
        return ""


class OllamaLLM:
    logger = Logger()

    def __init__(
        self,
        model: str = OllamaConfig.OLLAMA_MODEL,
        max_tokens: int = OllamaConfig.OLLAMA_MAX_TOKENS,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.think = True

    def invoke(self, system_prompt: str, human_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]

        chat_kwargs = {
            "model": self.model,
            "messages": messages,
            "think": True,
            "options": {"num_predict": self.max_tokens}
        }
        
        response = ollama.chat(**chat_kwargs)
        message = response.message

        reasoning: Optional[str] = getattr(message, "thinking", None)
        if reasoning:
            self.logger.reasoning(reasoning)

        content: Optional[str] = getattr(message, "content", None)
        if content:
            return content

        self.logger.info(
            "[OllamaLLM] WARNING: no 'content' returned by model; "
            "reasoning was logged but not used as answer.",
            style="yellow",
        )
        return ""
    
class LLM:
    """
    Generic LLM wrapper that chooses the correct backend based on Config.SERVING_METHOD.
    """

    def __init__(self, serving_method: str = "ollama"):
        if serving_method is None:
            serving_method = Config.SERVING_METHOD

        if serving_method == "portkey":
            self._llm = PortkeyLLM()
        elif serving_method == "ollama":
            self._llm = OllamaLLM()
        else:
            raise ValueError(
                f"Unsupported SERVING_METHOD '{serving_method}'. "
                "Expected 'portkey' or 'ollama'."
            )

    @property
    def logger(self) -> Logger:
        # Expose the underlying logger
        return self._llm.logger

    def invoke(self, system_prompt: str, human_prompt: str) -> str:
        return self._llm.invoke(system_prompt, human_prompt)