import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from orbit_or.broker import (
    BrokerRequest,
    BrokerResponse,
    PROFILE_MINIMAX,
    call_text,
    call_via_broker,
    get_or_collect_search_evidence_item,
    llm_call,
    llm_call_with_web,
    reset_broker_state,
    shutdown_broker,
)


@pytest.mark.asyncio
async def test_broker_routes_minimax_web_requests_through_react_loop():
    with patch(
        "orbit_or.broker.react_search_loop_with_evidence",
        new=AsyncMock(
            return_value=BrokerResponse(text="web-result", search_failed=False)
        ),
    ) as react_search_loop:
        result = await llm_call_with_web(
            "Prompt",
            provider_profile=PROFILE_MINIMAX,
            system_prompt="sys",
            role="dreamer",
        )

    assert result.text == "web-result"
    assert result.provider_used == PROFILE_MINIMAX
    react_search_loop.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_routes_minimax_direct_requests_through_query_minimax():
    with patch(
        "orbit_or.broker.query_minimax",
        new=AsyncMock(return_value=("plain-result", False)),
    ) as query_minimax:
        result = await llm_call(
            "Prompt",
            provider_profile=PROFILE_MINIMAX,
            system_prompt="sys",
            role="dreamer",
        )

    assert result.text == "plain-result"
    assert result.provider_used == PROFILE_MINIMAX
    query_minimax.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_text_normalizes_legacy_provider_to_minimax():
    with patch(
        "orbit_or.broker.query_minimax",
        new=AsyncMock(return_value=("minimax-result", False)),
    ) as query_minimax:
        text = await call_text("Prompt", provider="legacy-provider")

    assert text == "minimax-result"
    query_minimax.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_reuses_cached_web_evidence_before_running_fresh_search():
    cached_rows = [
        {
            "id": 17,
            "query_text": "query",
            "title": "Cached title",
            "snippet": "Cached snippet",
            "url": "https://example.com/a",
            "source_domain": "example.com",
            "content": "Cached title Cached snippet query example.com",
        }
    ]

    with patch(
        "orbit_or.broker.api.search_web_evidence_same_topic", return_value=cached_rows
    ):
        with patch(
            "orbit_or.broker.api.search_web_evidence_cross_topic", return_value=[]
        ):
            with patch(
                "orbit_or.broker.arerank", new=AsyncMock(return_value=[(0, 0.9)])
            ):
                with patch(
                    "orbit_or.broker._validate_cache_with_agent",
                    new=AsyncMock(return_value=True),
                ):
                    with patch(
                        "orbit_or.broker.minimax_search", new=AsyncMock()
                    ) as minimax_search:
                        item = await get_or_collect_search_evidence_item(
                            "query",
                            topic_id=1,
                            subtopic_id=2,
                            role="dreamer",
                        )

    assert item.web_ids == (17,)
    assert "[W17]" in item.rendered_results
    minimax_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_broker_ignores_low_scoring_cached_web_evidence_and_runs_fresh_search():
    cached_rows = [
        {
            "id": 17,
            "query_text": "query",
            "title": "Cached title",
            "snippet": "Cached snippet",
            "url": "https://example.com/a",
            "source_domain": "example.com",
            "content": "Cached title Cached snippet query example.com",
        }
    ]

    with patch(
        "orbit_or.broker.api.search_web_evidence_same_topic", return_value=cached_rows
    ):
        with patch(
            "orbit_or.broker.api.search_web_evidence_cross_topic", return_value=[]
        ):
            with patch(
                "orbit_or.broker.arerank", new=AsyncMock(return_value=[(0, 0.35)])
            ):
                with patch(
                    "orbit_or.broker.minimax_search",
                    new=AsyncMock(
                        return_value={
                            "organic": [
                                {
                                    "title": "Fresh",
                                    "snippet": "Fresh snippet",
                                    "link": "https://fresh.example.com",
                                }
                            ]
                        }
                    ),
                ) as minimax_search:
                    with patch(
                        "orbit_or.broker.api.insert_web_evidence", return_value=21
                    ):
                        item = await get_or_collect_search_evidence_item(
                            "query",
                            topic_id=1,
                            subtopic_id=2,
                            role="dreamer",
                        )

    minimax_search.assert_awaited_once()
    assert item.web_ids == (21,)
    assert "[W21]" in item.rendered_results


@pytest.mark.asyncio
async def test_broker_reuses_cross_topic_cache_above_threshold():
    cached_rows = [
        {
            "id": 30,
            "query_text": "GPU market share",
            "title": "GPU market share 2025",
            "snippet": "NVIDIA dominates GPU market",
            "url": "https://example.com/gpu",
            "source_domain": "example.com",
            "content": "GPU market share 2025 NVIDIA dominates example.com",
        }
    ]

    with patch("orbit_or.broker.api.search_web_evidence_same_topic", return_value=[]):
        with patch(
            "orbit_or.broker.api.search_web_evidence_cross_topic",
            return_value=cached_rows,
        ):
            with patch(
                "orbit_or.broker.arerank", new=AsyncMock(return_value=[(0, 0.70)])
            ):
                with patch(
                    "orbit_or.broker._validate_cache_with_agent",
                    new=AsyncMock(return_value=True),
                ):
                    with patch(
                        "orbit_or.broker.minimax_search", new=AsyncMock()
                    ) as minimax_search:
                        with patch(
                            "orbit_or.broker.api.clone_web_evidence_to_topic",
                            return_value={30: 99},
                        ) as mock_clone:
                            item = await get_or_collect_search_evidence_item(
                                "GPU market share analysis",
                                topic_id=1,
                                subtopic_id=2,
                                role="dreamer",
                            )

    minimax_search.assert_not_awaited()
    mock_clone.assert_called_once()
    assert item.web_ids == (99,)
    assert "[W99]" in item.rendered_results


@pytest.mark.asyncio
async def test_broker_passes_pseudo_tool_recovery_flag_to_direct_minimax_calls():
    with patch(
        "orbit_or.broker.query_minimax",
        new=AsyncMock(return_value=("recovered-query", False)),
    ) as query_minimax:
        result = await llm_call(
            "Prompt",
            provider_profile=PROFILE_MINIMAX,
            recover_pseudo_tool_query=True,
        )

    assert result.text == "recovered-query"
    assert query_minimax.await_args.kwargs["recover_pseudo_tool_query"] is True


@pytest.mark.asyncio
async def test_broker_coalesces_identical_inflight_requests():
    await reset_broker_state()
    gate = AsyncMock(return_value=("shared-result", False))
    request = BrokerRequest(
        prompt="Prompt",
        system_instruction="sys",
        provider="minimax",
        allow_web=False,
        fallback_role="skynet",
    )

    with patch("orbit_or.broker.query_minimax", new=gate):
        first, second = await asyncio.gather(
            call_via_broker(request),
            call_via_broker(request),
        )

    assert first.text == "shared-result"
    assert second.text == "shared-result"
    gate.assert_awaited_once()


@pytest.mark.asyncio
async def test_broker_shields_shared_request_from_single_waiter_cancellation():
    await reset_broker_state()

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_query(*args, **kwargs):
        started.set()
        await release.wait()
        return ("shared-result", False)

    request = BrokerRequest(prompt="Prompt", provider="minimax", fallback_role="skynet")

    with patch("orbit_or.broker.query_minimax", new=slow_query):
        task1 = asyncio.create_task(call_via_broker(request))
        task2 = asyncio.create_task(call_via_broker(request))
        await started.wait()
        task1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task1
        release.set()
        assert (await task2).text == "shared-result"


@pytest.mark.asyncio
async def test_broker_repairs_invalid_json_from_direct_minimax_plain_path():
    with patch(
        "orbit_or.broker.query_minimax",
        new=AsyncMock(side_effect=[("not valid json", []), ('{"ok": true}', [])]),
    ) as query_minimax:
        result = await llm_call(
            "Prompt",
            provider_profile=PROFILE_MINIMAX,
            system_prompt="sys",
            role="dreamer",
            require_json=True,
        )

    assert result.text == '{"ok": true}'
    assert result.provider_used == PROFILE_MINIMAX
    assert result.fallback_used is True
    assert query_minimax.await_count == 2


@pytest.mark.asyncio
async def test_broker_raises_on_minimax_error_sentinel():
    with patch(
        "orbit_or.broker.query_minimax",
        new=AsyncMock(return_value=("Error: nope", False)),
    ):
        with pytest.raises(RuntimeError, match="Error: nope"):
            await call_text("Prompt", provider="minimax")


@pytest.mark.asyncio
async def test_shutdown_broker_closes_minimax_client():
    with patch(
        "orbit_or.broker.close_minimax_client", new=AsyncMock()
    ) as close_minimax:
        await shutdown_broker()

    close_minimax.assert_awaited_once()
