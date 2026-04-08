from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeProgress:
    actions_taken: list[str]
    discovered_sources: set[str]
    marked_root_cause: str | None = None
    applied_fix: str | None = None
    resolved: bool = False
    failed: bool = False


def get_relevant_sources(task_id: str) -> set[str]:
    mapping = {
        "easy_batch_01": {"config"},
        "medium_tokenizer_01": {"logs", "recent_deploy"},
        "hard_timeout_01": {"config"},
    }
    return mapping.get(task_id, set())


def score_step(
    task: Any,
    progress: EpisodeProgress,
    action_str: str,
    action_type: str,
    target: str | None,
    is_repeat: bool,
    is_invalid: bool,
) -> float:
    if is_invalid:
        return -0.05

    if action_type.startswith("inspect_"):
        if is_repeat:
            return -0.05

        source = action_type.replace("inspect_", "", 1)
        score_gain = 0.10

        if source in get_relevant_sources(task.task_id):
            score_gain += 0.10

        return score_gain

    if action_type == "mark_root_cause":
        if target == task.true_root_cause:
            return 0.30
        return -0.10

    if action_type == "apply_fix":
        if target == task.true_fix:
            return 0.30
        return -0.15

    if action_type == "resolve_incident":
        return 0.0

    return 0.0


def clamp_score(score: float) -> float:
    return max(0.01, min(0.99, score))


def finalize_score(
    task: Any,
    progress: EpisodeProgress,
    accumulated_score: float,
) -> tuple[float, bool, str]:
    if progress.resolved:
        if (
            progress.marked_root_cause == task.true_root_cause
            and progress.applied_fix == task.true_fix
        ):
            return clamp_score(accumulated_score), True, "incident resolved correctly"

        return (
            clamp_score(accumulated_score - 0.20),
            False,
            "incident resolved incorrectly",
        )

    if progress.failed:
        return clamp_score(accumulated_score), False, "episode failed"

    return clamp_score(accumulated_score), False, "episode ended without resolution"
