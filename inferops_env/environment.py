import json
from pathlib import Path
from typing import Any

from .models import Action, Observation, StepResult, TaskDefinition
from .grader import EpisodeProgress, score_step, finalize_score


def strict_score(score: float) -> float:
    return max(0.01, min(0.99, score))


class InferOpsEnv:
    def __init__(self, task_id: str | None = None) -> None:
        self.tasks: dict[str, TaskDefinition] = self._load_tasks()

        if task_id is not None and task_id in self.tasks:
            self.task_id = task_id
        else:
            self.task_id = sorted(self.tasks.keys())[0]

        self.current_task = self.tasks[self.task_id]

        self.step_count = 0
        self.accumulated_score = 0.01
        self.status = "investigating"
        self.last_action_error: str | None = None
        self.progress = EpisodeProgress(
            actions_taken=[],
            discovered_sources=set(),
        )

    def _load_tasks(self) -> dict[str, TaskDefinition]:
        tasks_dir = Path(__file__).parent / "tasks"
        loaded: dict[str, TaskDefinition] = {}

        for json_path in tasks_dir.glob("*.json"):
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                task_def = TaskDefinition(**data)
                loaded[task_def.task_id] = task_def

        return loaded

    def _build_observation(self) -> Observation:
        discovered_metrics = None
        if "metrics" in self.progress.discovered_sources:
            discovered_metrics = self.current_task.metrics

        discovered_logs = None
        if "logs" in self.progress.discovered_sources:
            discovered_logs = self.current_task.logs

        discovered_config = None
        if "config" in self.progress.discovered_sources:
            discovered_config = self.current_task.config

        discovered_recent_deploy = None
        if "recent_deploy" in self.progress.discovered_sources:
            discovered_recent_deploy = self.current_task.recent_deploy

        return Observation(
            task_id=self.current_task.task_id,
            difficulty=self.current_task.difficulty,
            incident_summary=self.current_task.incident_summary,
            discovered_metrics=discovered_metrics,
            discovered_logs=discovered_logs,
            discovered_config=discovered_config,
            discovered_recent_deploy=discovered_recent_deploy,
            actions_taken=list(self.progress.actions_taken),
            step_count=self.step_count,
            status=self.status,
            last_action_error=self.last_action_error,
        )

    def reset(self) -> Observation:
        self.accumulated_score = 0.01
        self.step_count = 0
        self.last_action_error = None
        self.status = "investigating"

        self.progress = EpisodeProgress(
            actions_taken=[],
            discovered_sources=set(),
        )

        return self._build_observation()

    def state(self) -> dict[str, Any]:
        return {
            "task_id": self.current_task.task_id,
            "difficulty": self.current_task.difficulty,
            "actions_taken": list(self.progress.actions_taken),
            "step_count": self.step_count,
            "score": strict_score(self.accumulated_score),
            "status": self.status,
            "discovered_sources": list(self.progress.discovered_sources),
            "marked_root_cause": self.progress.marked_root_cause,
            "applied_fix": self.progress.applied_fix,
            "last_action_error": self.last_action_error,
        }

    def step(self, action: Action) -> StepResult:
        if self.status != "investigating":
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"message": "episode already finished"},
            )

        self.step_count += 1
        self.last_action_error = None

        if action.target:
            action_str = f"{action.action_type.value}:{action.target}"
        else:
            action_str = action.action_type.value

        action_type_val = action.action_type.value

        is_inspect = action_type_val.startswith("inspect_")
        is_repeat = False

        if is_inspect:
            source = action_type_val.replace("inspect_", "", 1)
            if source in self.progress.discovered_sources:
                is_repeat = True
            else:
                self.progress.discovered_sources.add(source)

        elif action_type_val == "mark_root_cause":
            self.progress.marked_root_cause = action.target

        elif action_type_val == "apply_fix":
            self.progress.applied_fix = action.target

        elif action_type_val == "resolve_incident":
            self.progress.resolved = True
            self.status = "resolved"

        self.progress.actions_taken.append(action_str)

        reward = score_step(
            task=self.current_task,
            progress=self.progress,
            action_str=action_str,
            action_type=action_type_val,
            target=action.target,
            is_repeat=is_repeat,
            is_invalid=False,
        )

        self.accumulated_score += reward
        self.accumulated_score = strict_score(self.accumulated_score)

        if self.step_count >= self.current_task.max_steps and not self.progress.resolved:
            self.progress.failed = True
            self.status = "failed"

        done = self.progress.resolved or self.progress.failed

        info: dict[str, Any] = {}
        if done:
            final_score, success, reason = finalize_score(
                task=self.current_task,
                progress=self.progress,
                accumulated_score=self.accumulated_score,
            )
            self.accumulated_score = strict_score(final_score)
            info["success"] = success
            info["reason"] = reason
        else:
            info["success"] = False
            info["reason"] = "in_progress"

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=done,
            info=info,
        )

    def close(self) -> None:
        pass
