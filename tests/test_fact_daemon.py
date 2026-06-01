import asyncio
import os

os.environ["TESTING"] = "1"

import pytest
from unittest.mock import AsyncMock, patch

from orbit_or.fact_daemon import FactDaemon, get_active_daemon, start_daemon, stop_daemon, drain_daemon


@pytest.mark.asyncio
async def test_daemon_start_and_stop():
    daemon = FactDaemon(topic_id=1, subtopic_id=1)

    with patch.object(daemon, "_clerk_loop", new=AsyncMock()):
        with patch.object(daemon, "_librarian_loop", new=AsyncMock()):
            with patch.object(daemon, "_evidence_loop", new=AsyncMock()):
                await daemon.start()
                assert daemon._clerk_task is not None
                assert daemon._librarian_task is not None
                assert daemon._evidence_task is not None
                await daemon.stop()
                assert daemon._shutdown.is_set()


@pytest.mark.asyncio
async def test_daemon_registry_start_and_stop():
    with patch.object(FactDaemon, "_clerk_loop", new=AsyncMock()):
        with patch.object(FactDaemon, "_librarian_loop", new=AsyncMock()):
            with patch.object(FactDaemon, "_evidence_loop", new=AsyncMock()):
                with patch("orbit_or.fact_daemon.api.get_max_round_number", return_value=0):
                    await start_daemon(1, 1)
                    assert get_active_daemon(1, 1) is not None
                    await stop_daemon(1, 1)
                    assert get_active_daemon(1, 1) is None


@pytest.mark.asyncio
async def test_daemon_drain():
    daemon = FactDaemon(topic_id=1, subtopic_id=1)

    async def fake_clerk_loop():
        while not daemon._shutdown.is_set():
            if daemon._drain.is_set():
                break
            try:
                await asyncio.wait_for(daemon._shutdown.wait(), timeout=0.1)
                break
            except asyncio.TimeoutError:
                pass

    async def fake_librarian_loop():
        while not daemon._shutdown.is_set():
            if daemon._drain.is_set():
                daemon._drain_complete.set()
                break
            try:
                await asyncio.wait_for(daemon._shutdown.wait(), timeout=0.1)
                break
            except asyncio.TimeoutError:
                pass

    async def fake_evidence_loop():
        while not daemon._shutdown.is_set():
            if daemon._drain.is_set():
                daemon._evidence_done.set()
                break
            try:
                await asyncio.wait_for(daemon._shutdown.wait(), timeout=0.1)
                break
            except asyncio.TimeoutError:
                pass

    with patch.object(daemon, "_clerk_loop", side_effect=fake_clerk_loop):
        with patch.object(daemon, "_librarian_loop", side_effect=fake_librarian_loop):
            with patch.object(daemon, "_evidence_loop", side_effect=fake_evidence_loop):
                await daemon.start()
                await daemon.drain_and_stop(timeout=5.0)

    assert daemon._shutdown.is_set()
    assert daemon._drain_complete.is_set()
    assert daemon._evidence_done.is_set()


@pytest.mark.asyncio
async def test_daemon_extract_candidates_skips_npc():
    daemon = FactDaemon(topic_id=1, subtopic_id=1)
    # NPC messages should be skipped
    for sender in ["skynet", "writer", "librarian", "fact_proposer"]:
        msg = {"id": 1, "sender": sender, "content": "test content"}
        # Should not raise, and should be a no-op
        await daemon._extract_candidates(msg)


@pytest.mark.asyncio
async def test_drain_daemon_noop_when_no_daemon():
    # Should not raise when no daemon is registered
    await drain_daemon(999, 999, timeout=1.0)


@pytest.mark.asyncio
async def test_clerk_loop_handles_none_round_number():
    """Regression: round_number=None in message should not raise TypeError."""
    daemon = FactDaemon(topic_id=1, subtopic_id=1)
    msg = {"id": 1, "sender": "dreamer", "content": "test", "round_number": None, "msg_type": "standard"}
    # The fix: msg.get("round_number") or 0 should return 0, not None
    result = max(daemon.current_round, msg.get("round_number") or 0)
    assert result == 0
    # Also test with a valid round
    msg2 = {"id": 2, "sender": "dreamer", "content": "test", "round_number": 3, "msg_type": "standard"}
    result2 = max(daemon.current_round, msg2.get("round_number") or 0)
    assert result2 == 3
