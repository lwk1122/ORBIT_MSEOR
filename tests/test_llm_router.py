from unittest.mock import AsyncMock, patch

import pytest

from orbit_or.broker import BrokerResponse
from orbit_or.llm_router import query_with_fallback


@pytest.mark.asyncio
async def test_query_with_fallback_returns_minimax_result_when_available():
    with patch(
        "orbit_or.llm_router.llm_call",
        new=AsyncMock(
            return_value=BrokerResponse(
                text="minimax answer", provider_used="minimax"
            )
        ),
    ) as broker_call:
        result = await query_with_fallback(
            "prompt", model="minimax-3.0-flash", fallback_role="skynet"
        )

    assert result == "minimax answer"
    broker_call.assert_awaited_once_with(
        "prompt",
        system_prompt="",
        provider_profile="minimax",
        role="skynet",
        model="minimax-3.0-flash",
        temperature=0.7,
        max_tokens=65535,
    )


@pytest.mark.asyncio
async def test_query_with_fallback_routes_non_grounded_calls_through_broker():
    with patch(
        "orbit_or.llm_router.llm_call",
        new=AsyncMock(
            return_value=BrokerResponse(
                text="minimax answer", provider_used="minimax"
            )
        ),
    ) as broker_call:
        result = await query_with_fallback(
            "prompt",
            model="minimax-3.0-flash",
            use_search=False,
            fallback_role="skynet",
        )

    assert result == "minimax answer"
    broker_call.assert_awaited_once_with(
        "prompt",
        system_prompt="",
        provider_profile="minimax",
        role="skynet",
        model="minimax-3.0-flash",
        temperature=0.7,
        max_tokens=65535,
    )


@pytest.mark.asyncio
async def test_query_with_fallback_routes_grounded_calls_through_broker():
    with patch(
        "orbit_or.llm_router.llm_call_with_web",
        new=AsyncMock(
            return_value=BrokerResponse(
                text="grounded minimax answer",
                provider_used="minimax",
                search_used=True,
            )
        ),
    ) as broker_call:
        result = await query_with_fallback(
            "prompt",
            model="minimax-3.1-pro-preview",
            use_search=True,
            fallback_role="writer",
        )

    assert result == "grounded minimax answer"
    broker_call.assert_awaited_once_with(
        "prompt",
        system_prompt="",
        provider_profile="minimax",
        role="writer",
        model="minimax-3.1-pro-preview",
        temperature=0.7,
        max_tokens=65535,
    )


@pytest.mark.asyncio
async def test_query_with_fallback_propagates_broker_errors():
    with patch(
        "orbit_or.llm_router.llm_call",
        new=AsyncMock(side_effect=RuntimeError("Error: No API key.")),
    ):
        with pytest.raises(RuntimeError, match="Error: No API key."):
            await query_with_fallback(
                "prompt",
                model="minimax-3.0-flash",
                use_search=False,
                fallback_role="skynet",
            )


@pytest.mark.asyncio
async def test_query_with_fallback_propagates_grounded_broker_errors():
    with patch(
        "orbit_or.llm_router.llm_call_with_web",
        new=AsyncMock(side_effect=RuntimeError("Error: MiniMax search timed out")),
    ):
        with pytest.raises(RuntimeError, match="Error: MiniMax search timed out"):
            await query_with_fallback(
                "prompt",
                model="minimax-3.1-pro-preview",
                use_search=True,
                fallback_role="writer",
            )
