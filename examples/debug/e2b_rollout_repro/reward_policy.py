"""Classify trainable agent terminations separately from infrastructure failures."""
import json
from pathlib import Path
from typing import Any


def apply_reward_policy(result: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    trial: dict[str, Any] = {}
    trial_dir = result.get("trial_dir")
    if trial_dir:
        path = Path(trial_dir) / "result.json"
        if path.exists():
            trial = json.loads(path.read_text())
    exception = trial.get("exception_info") or {}
    exception_type = exception.get("exception_type")
    traceback = exception.get("exception_traceback") or ""
    context_limit = exception_type == "ContextLengthExceededError"
    parser_failure = (
        exception_type == "IndexError"
        and "terminus_xml_plain_parser.py" in traceback
        and "_find_top_level_tags" in traceback
        and "tag_content.split()[0]" in traceback
    )
    agent_timeout = exception_type == "AgentTimeoutError"
    truncated = result.get("exit_status") == "SequenceLengthLimitExceeded"
    if context_limit or parser_failure:
        result["reward"] = 0.0
        result["reward_source"] = "context_budget_exhausted" if context_limit else "agent_response_parse_failure"
        result["pilot_trainable_failure"] = exception_type
    elif truncated:
        result["reward"] = 0.0
        result["reward_source"] = "token_budget_exhausted"
    elif agent_timeout:
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        result["reward"] = float(rewards.get("reward", next(iter(rewards.values()), 0.0)))
        result["reward_source"] = "verifier" if rewards else "agent_time_budget_exhausted"
        result["pilot_agent_timeout"] = True
    result["pilot_valid_for_training"] = context_limit or parser_failure or truncated or agent_timeout or (
        result.get("exit_status") == "Submitted" and bool(result.get("eval_report"))
    )
    return result
