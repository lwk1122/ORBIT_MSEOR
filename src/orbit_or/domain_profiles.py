"""Domain profile prompt additions."""

from __future__ import annotations

from .domain_ontology import build_domain_ontology_prompt

MSE_PROFILE_VALUES = {"mse", "management_science_engineering"}

MSE_COMMON_PROMPT = """\
DOMAIN PROFILE: Management Science and Engineering.
- Work as a modeling workflow, not an open discussion loop. Produce one useful artifact-oriented answer for this round.
- Frame problems as decisions under constraints, uncertainty, and stakeholder trade-offs.
- Separate evidence from assumptions. Mark missing data explicitly.
- When relevant, identify alternatives, objectives/KPIs, constraints, datasets, methods, boundary conditions, and managerial implications.
- For optimization problems, preserve sets, parameters, decision variables, objective direction, constraints, units, and data requirements.
- Prefer falsifiable statements and executable checks over broad strategic prose.
"""

MSE_ROLE_PROMPTS = {
    "analyst": "MSE ROLE: OR Modeler. MSE TASK ROLE: OR Modeler. Own the decision frame, sets, parameters, decision variables, objective direction, constraints, units, and model class.",
    "scientist": "MSE TASK ROLE: Parameter and Data Auditor. Check data validity, units, uncertainty, missing parameters, and whether components are source-backed.",
    "engineer": "MSE TASK ROLE: Solver Engineer. Produce solver-safe LP/MPS artifacts, diagnose syntax/runtime failures, and keep solver output inspectable.",
    "critic": "MSE TASK ROLE: Validity Reviewer. Stress-test assumptions, feasibility, objective misspecification, external validity, and unsupported generalization.",
    "contrarian": "MSE TASK ROLE: Managerial Decision Analyst. Translate model results into trade-offs, adoption constraints, stakeholder implications, and decision boundaries.",
    "dreamer": "MSE TASK ROLE: Decision Framer. Use only when the decision problem itself is under-specified; convert ambiguity into explicit alternatives and evaluation criteria.",
}


def get_domain_prompt_additions(profile: str, actor: str) -> list[str]:
    normalized = (profile or "base").strip().lower()
    if normalized not in MSE_PROFILE_VALUES:
        return []
    additions = [MSE_COMMON_PROMPT, build_domain_ontology_prompt("mse")]
    role_prompt = MSE_ROLE_PROMPTS.get(actor)
    if role_prompt:
        additions.append(role_prompt)
    return additions
