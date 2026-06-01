from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .broker import DEFAULT_MAX_TOKENS, PROFILE_MINIMAX, llm_call
from .json_utils import extract_json_object as _extract_json_vote
from .prompts import PROMPTS
from .structured_retry import retry_structured_output, usable_text_output

logger = logging.getLogger(__name__)

_VOTE_FORMAT_STRIP_RE = re.compile(
    r"(Format: \{.*?\}|【MANDATORY REASONING DRAFTING】.*)",
    flags=re.DOTALL,
)

ORCHESTRATOR = "orchestrator"
DELIBERATOR = "deliberator"
SPECIAL = "special"
NPC = "npc"

SKYNET = "skynet"
SPECTATOR = "spectator"

DELIBERATORS = [
    "dreamer",
    "scientist",
    "engineer",
    "analyst",
    "critic",
    "contrarian",
]

SPECIALS = ["cat", "dog", "tron", SPECTATOR]
NPCS = ["writer", "fact_proposer", "librarian"]
VOTING_PARTICIPANTS = [SKYNET] + DELIBERATORS + SPECIALS
TARGETABLE_DELIBERATORS = set(DELIBERATORS)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    agent_class: str
    role_prompt: str
    default_provider: str = PROFILE_MINIMAX
    default_strategy: str = "direct"
    default_tools: tuple[str, ...] = ()
    can_vote: bool = False
    can_target: bool = False
    can_be_targeted: bool = False


def _build_registry() -> dict[str, AgentSpec]:
    specs = {
        SKYNET: AgentSpec(
            name=SKYNET,
            agent_class=ORCHESTRATOR,
            role_prompt=PROMPTS["skynet"],
            default_provider=PROFILE_MINIMAX,
            default_strategy="direct",
            can_vote=True,
        ),
        "dreamer": AgentSpec(
            name="dreamer",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["dreamer"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "scientist": AgentSpec(
            name="scientist",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["scientist"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "engineer": AgentSpec(
            name="engineer",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["engineer"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "analyst": AgentSpec(
            name="analyst",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["analyst"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "critic": AgentSpec(
            name="critic",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["critic"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "contrarian": AgentSpec(
            name="contrarian",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["contrarian"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "cat": AgentSpec(
            name="cat",
            agent_class=SPECIAL,
            role_prompt=PROMPTS["cat"],
            can_vote=True,
            can_target=True,
        ),
        "dog": AgentSpec(
            name="dog",
            agent_class=SPECIAL,
            role_prompt=PROMPTS["dog"],
            can_vote=True,
            can_target=True,
        ),
        "tron": AgentSpec(
            name="tron",
            agent_class=SPECIAL,
            role_prompt=PROMPTS["tron"],
            can_vote=True,
            can_target=True,
        ),
        SPECTATOR: AgentSpec(
            name=SPECTATOR,
            agent_class=SPECIAL,
            role_prompt=PROMPTS["spectator"],
            can_vote=True,
            can_target=True,
        ),
        "writer": AgentSpec(
            name="writer",
            agent_class=NPC,
            role_prompt=PROMPTS["writer"],
            default_provider=PROFILE_MINIMAX,
        ),
        "fact_proposer": AgentSpec(
            name="fact_proposer",
            agent_class=NPC,
            role_prompt=PROMPTS["fact_proposer"],
        ),
        "librarian": AgentSpec(
            name="librarian",
            agent_class=NPC,
            role_prompt=PROMPTS["librarian"],
        ),
    }
    return specs


AGENT_REGISTRY = _build_registry()

# AR-4: Sparse communication — each deliberator sees only a subset of roles
ROLE_VISIBILITY: dict[str, set[str] | None] = {
    "dreamer": {"dreamer", "scientist", "engineer", "skynet", "writer", "tron"},
    "scientist": {
        "scientist",
        "analyst",
        "critic",
        "contrarian",
        "skynet",
        "writer",
        "tron",
    },
    "engineer": {"engineer", "analyst", "dreamer", "skynet", "writer", "tron"},
    "analyst": {
        "analyst",
        "scientist",
        "engineer",
        "critic",
        "skynet",
        "writer",
        "tron",
    },
    "critic": {
        "critic",
        "scientist",
        "analyst",
        "contrarian",
        "skynet",
        "writer",
        "tron",
    },
    "contrarian": {"contrarian", "critic", "scientist", "skynet", "writer", "tron"},
    "dog": None,  # sees everything
    "cat": None,
    "tron": None,
    "spectator": None,
    "skynet": None,
    "writer": None,
    "librarian": None,
    "fact_proposer": None,
}


def get_agent_spec(name: str) -> AgentSpec:
    return AGENT_REGISTRY[name]


def can_special_target(target: str) -> bool:
    return target in TARGETABLE_DELIBERATORS


def voting_agents() -> list[str]:
    return [name for name in VOTING_PARTICIPANTS if AGENT_REGISTRY[name].can_vote]


def ordinary_deliberators() -> list[str]:
    return list(DELIBERATORS)


def special_agents() -> list[str]:
    return list(SPECIALS)


def is_deliberator(name: str) -> bool:
    return AGENT_REGISTRY.get(name, AgentSpec("", "", "")).agent_class == DELIBERATOR


def is_special(name: str) -> bool:
    return AGENT_REGISTRY.get(name, AgentSpec("", "", "")).agent_class == SPECIAL


def is_npc(name: str) -> bool:
    return AGENT_REGISTRY.get(name, AgentSpec("", "", "")).agent_class == NPC


def parse_vote_response(text: str) -> Optional[bool]:
    payload = parse_vote_payload(text)
    return None if payload is None else payload["decision"]


def parse_vote_payload(text: str) -> Optional[dict]:
    parsed = _extract_json_vote(text)
    if isinstance(parsed, dict):
        raw_vote = parsed.get("vote")
        if isinstance(raw_vote, bool):
            reason = parsed.get("reason")
            return {
                "decision": raw_vote,
                "decision_label": "yes" if raw_vote else "no",
                "reason": reason.strip() if isinstance(reason, str) else "",
            }
        if isinstance(raw_vote, str):
            normalized = raw_vote.strip().lower()
            if normalized in {
                "yes",
                "true",
                "continue",
                "approve",
                "select",
                "replan",
                "close",
            }:
                reason = parsed.get("reason")
                return {
                    "decision": True,
                    "decision_label": "yes",
                    "reason": reason.strip() if isinstance(reason, str) else "",
                }
            if normalized in {"no", "false", "reject", "skip", "deny"}:
                reason = parsed.get("reason")
                return {
                    "decision": False,
                    "decision_label": "no",
                    "reason": reason.strip() if isinstance(reason, str) else "",
                }
            return None
    normalized_text = (text or "").strip().lower()
    if (
        normalized_text in {"yes", "true"}
        or normalized_text.startswith("yes ")
        or normalized_text.startswith("true ")
    ):
        return {"decision": True, "decision_label": "yes", "reason": ""}
    if (
        normalized_text in {"no", "false"}
        or normalized_text.startswith("no ")
        or normalized_text.startswith("false ")
    ):
        return {"decision": False, "decision_label": "no", "reason": ""}
    return None


class Agent:
    def __init__(self, spec: AgentSpec):
        self.spec = spec

    async def call(
        self,
        prompt: str,
        *,
        allow_web: bool = False,
        require_json: bool = False,
        model: str = "",
        provider_profile: str = "",
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        boost: Optional[str] = None,
    ) -> str:
        del require_json
        selected_provider = provider_profile or self.spec.default_provider
        if allow_web:
            from .broker import llm_call_with_web

            result = await llm_call_with_web(
                prompt,
                system_prompt=self.spec.role_prompt,
                provider_profile=selected_provider,
                require_json=False,
                role=self.spec.name,
                temperature=temperature,
                max_tokens=max_tokens,
                boost=boost or "",
            )
            return result.text
        result = await llm_call(
            prompt,
            system_prompt=self.spec.role_prompt,
            provider_profile=selected_provider,
            require_json=False,
            role=self.spec.name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            boost=boost or "",
        )
        return result.text

    async def vote(
        self, prompt: str, *, allow_web: bool = False, provider_profile: str = ""
    ) -> Optional[bool]:
        payload = await self.vote_detail(
            prompt, allow_web=allow_web, provider_profile=provider_profile
        )
        return None if payload is None else payload["decision"]

    async def vote_detail(
        self, prompt: str, *, allow_web: bool = False, provider_profile: str = ""
    ) -> Optional[dict]:
        del allow_web
        selected_provider = provider_profile or self.spec.default_provider
        base_prompt = _VOTE_FORMAT_STRIP_RE.sub("", self.spec.role_prompt)
        vote_instruction = (
            f"{base_prompt}\n\n"
            "VOTING MODE:\n"
            "You are not posting a normal workspace message. "
            'You must reply with strict JSON only using this schema: {"vote":"yes|no","reason":"short sentence"}. '
            "The `reason` must be brief and concrete."
        )
        result = await retry_structured_output(
            stage_name=f"{self.spec.name} vote",
            logger=logger,
            is_usable=lambda item: parse_vote_payload(item.text) is not None,
            invoke=lambda: llm_call(
                prompt,
                system_prompt=vote_instruction,
                provider_profile=selected_provider,
                require_json=True,
                role=self.spec.name,
                temperature=0.7,
                max_tokens=DEFAULT_MAX_TOKENS,
            ),
        )
        if result is None:
            logger.info(
                "[Vote] agent=%s parsed=%s decision=%s reason=%s raw_response=%s",
                self.spec.name,
                None,
                None,
                "",
                "",
            )
            return None
        parsed_vote = parse_vote_payload(result.text)
        if parsed_vote is None:
            logger.warning(
                "[Vote] agent=%s retry passed but second parse returned None; raw=%s",
                self.spec.name,
                result.text,
            )
            return None
        parsed_vote = {**parsed_vote, "raw_response": result.text}
        logger.info(
            "[Vote] agent=%s parsed=%s decision=%s reason=%s raw_response=%s",
            self.spec.name,
            True,
            parsed_vote.get("decision_label"),
            parsed_vote.get("reason", ""),
            result.text,
        )
        return parsed_vote

    async def governance_vote(self, prompt: str, *, provider_profile: str = "") -> str:
        selected_provider = provider_profile or self.spec.default_provider
        base_prompt = _VOTE_FORMAT_STRIP_RE.sub("", self.spec.role_prompt)

        vote_instruction = (
            f"{base_prompt}\n\n"
            "GOVERNANCE VOTING MODE:\n"
            "You are not posting a normal workspace message. "
            "You must analyze governance state and reply with strict JSON only using the exact schema requested in the user prompt. "
            "Do not output markdown, prose outside JSON, or extra keys. IGNORE any previous JSON format instructions."
        )
        result = await retry_structured_output(
            stage_name=f"{self.spec.name} governance_vote",
            logger=logger,
            is_usable=lambda item: usable_text_output(item.text)
            and _extract_json_vote(item.text) is not None,
            invoke=lambda: llm_call(
                prompt,
                system_prompt=vote_instruction,
                provider_profile=selected_provider,
                require_json=True,
                role=self.spec.name,
                temperature=0.3,
                max_tokens=DEFAULT_MAX_TOKENS,
            ),
        )
        return result.text if result is not None else ""


def get_agent(name: str) -> Agent:
    return Agent(get_agent_spec(name))
