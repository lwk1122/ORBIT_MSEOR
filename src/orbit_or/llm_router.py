import logging
from typing import Optional

from .broker import PROFILE_MINIMAX, llm_call, llm_call_with_web

logger = logging.getLogger(__name__)


async def query_with_fallback(
    prompt: str,
    *,
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 65535,
    system_instruction: Optional[str] = None,
    thinking_level: str = "NONE",
    use_search: bool = False,
    enable_fallback: bool = True,
    fallback_role: str = "skynet",
) -> str:
    """Compatibility wrapper that routes orchestration calls through MiniMax."""
    if thinking_level and thinking_level.upper() != "NONE":
        logger.debug(
            "[LLMRouter] MiniMax path ignores explicit thinking_level=%s.",
            thinking_level,
        )
    if not enable_fallback:
        logger.debug(
            "[LLMRouter] MiniMax path ignores enable_fallback=False and uses broker behavior."
        )
    if use_search:
        result = await llm_call_with_web(
            prompt,
            system_prompt=system_instruction or "",
            provider_profile=PROFILE_MINIMAX,
            role=fallback_role,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result.text
    result = await llm_call(
        prompt,
        system_prompt=system_instruction or "",
        provider_profile=PROFILE_MINIMAX,
        role=fallback_role,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return result.text
