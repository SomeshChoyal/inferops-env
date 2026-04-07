from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    inspect_metrics = "inspect_metrics"
    inspect_logs = "inspect_logs"
    inspect_config = "inspect_config"
    inspect_recent_deploy = "inspect_recent_deploy"
    mark_root_cause = "mark_root_cause"
    apply_fix = "apply_fix"
    resolve_incident = "resolve_incident"


class Action(BaseModel):
    action_type: ActionType
    target: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "Action":
        requires_none = {
            ActionType.inspect_metrics,
            ActionType.inspect_logs,
            ActionType.inspect_config,
            ActionType.inspect_recent_deploy,
            ActionType.resolve_incident,
        }

        if self.action_type in requires_none:
            if self.target is not None:
                raise ValueError(f"target must be None for {self.action_type}")

        elif self.action_type == ActionType.mark_root_cause:
            valid_targets = {
                "batch_size_too_high",
                "tokenizer_version_mismatch",
                "request_timeout_too_low",
            }
            if self.target not in valid_targets:
                raise ValueError(
                    f"target must be one of {valid_targets} for {self.action_type}"
                )

        elif self.action_type == ActionType.apply_fix:
            valid_targets = {
                "reduce_batch_size",
                "rollback_tokenizer",
                "restore_timeout_config",
                "restart_service",
            }
            if self.target not in valid_targets:
                raise ValueError(
                    f"target must be one of {valid_targets} for {self.action_type}"
                )

        return self


class Observation(BaseModel):
    task_id: str
    difficulty: str
    incident_summary: str
    discovered_metrics: dict[str, Any] | None = None
    discovered_logs: str | None = None
    discovered_config: dict[str, Any] | None = None
    discovered_recent_deploy: str | None = None
    actions_taken: list[str] = Field(default_factory=list)
    step_count: int = 0
    status: str = "investigating"
    last_action_error: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "Observation":
        valid_difficulties = {"easy", "medium", "hard"}
        valid_statuses = {"investigating", "resolved", "failed"}

        if self.difficulty not in valid_difficulties:
            raise ValueError(f"difficulty must be one of {valid_difficulties}")

        if self.status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")

        return self


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class TaskDefinition(BaseModel):
    task_id: str
    difficulty: str
    objective: str
    incident_summary: str
    true_root_cause: str
    true_fix: str
    metrics: dict[str, Any]
    logs: str
    config: dict[str, Any]
    recent_deploy: str
    max_steps: int

    @model_validator(mode="after")
    def validate_fields(self) -> "TaskDefinition":
        valid_difficulties = {"easy", "medium", "hard"}
        valid_root_causes = {
            "batch_size_too_high",
            "tokenizer_version_mismatch",
            "request_timeout_too_low",
        }
        valid_fixes = {
            "reduce_batch_size",
            "rollback_tokenizer",
            "restore_timeout_config",
            "restart_service",
        }

        if self.difficulty not in valid_difficulties:
            raise ValueError(f"difficulty must be one of {valid_difficulties}")

        if self.true_root_cause not in valid_root_causes:
            raise ValueError(f"true_root_cause must be one of {valid_root_causes}")

        if self.true_fix not in valid_fixes:
            raise ValueError(f"true_fix must be one of {valid_fixes}")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")

        return self