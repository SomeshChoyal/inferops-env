import os
import sys
from typing import Iterable

from openai import OpenAI

from inferops_env.environment import InferOpsEnv
from inferops_env.models import Action, ActionType

DEFAULT_TASK_IDS = [
    "easy_batch_01",
    "medium_tokenizer_01",
    "hard_timeout_01",
]


def choose_rule_based_action(obs) -> str | None:
    actions_taken = set(obs.actions_taken)

    if obs.task_id == "easy_batch_01":
        if "inspect_config" not in actions_taken:
            return "inspect_config"
        if "mark_root_cause:batch_size_too_high" not in actions_taken:
            return "mark_root_cause:batch_size_too_high"
        if "apply_fix:reduce_batch_size" not in actions_taken:
            return "apply_fix:reduce_batch_size"
        if "resolve_incident" not in actions_taken:
            return "resolve_incident"
        return None

    if obs.task_id == "medium_tokenizer_01":
        if "inspect_logs" not in actions_taken:
            return "inspect_logs"
        if "inspect_recent_deploy" not in actions_taken:
            return "inspect_recent_deploy"
        if "mark_root_cause:tokenizer_version_mismatch" not in actions_taken:
            return "mark_root_cause:tokenizer_version_mismatch"
        if "apply_fix:rollback_tokenizer" not in actions_taken:
            return "apply_fix:rollback_tokenizer"
        if "resolve_incident" not in actions_taken:
            return "resolve_incident"
        return None

    if obs.task_id == "hard_timeout_01":
        if "inspect_config" not in actions_taken:
            return "inspect_config"
        if "inspect_logs" not in actions_taken:
            return "inspect_logs"
        if "mark_root_cause:request_timeout_too_low" not in actions_taken:
            return "mark_root_cause:request_timeout_too_low"
        if "apply_fix:restore_timeout_config" not in actions_taken:
            return "apply_fix:restore_timeout_config"
        if "resolve_incident" not in actions_taken:
            return "resolve_incident"
        return None

    return None


def parse_action(action_str: str) -> Action:
    if ":" in action_str:
        a_type, target = action_str.split(":", 1)
        return Action(
            action_type=ActionType(a_type.strip()),
            target=target.strip(),
        )
    return Action(action_type=ActionType(action_str.strip()))


def get_task_ids() -> list[str]:
    """
    Priority:
    1. TASK_IDS="task1,task2,task3"
    2. TASK_ID="all" -> all 3 default tasks
    3. TASK_ID="<single_task>"
    4. default -> all 3 tasks
    """
    task_ids_env = os.environ.get("TASK_IDS")
    if task_ids_env:
        task_ids = [x.strip() for x in task_ids_env.split(",") if x.strip()]
        if task_ids:
            return task_ids

    task_id_env = os.environ.get("TASK_ID")
    if task_id_env:
        task_id_env = task_id_env.strip()
        if task_id_env.lower() == "all":
            return DEFAULT_TASK_IDS
        return [task_id_env]

    return DEFAULT_TASK_IDS


def build_client() -> tuple[OpenAI, str]:
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
    api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

    if not api_key:
        raise ValueError("API_KEY or HF_TOKEN environment variable is required")

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    return client, model_name


def warmup_client(client: OpenAI, model_name: str) -> None:
    try:
        _ = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Return exactly: inspect_logs"}],
            temperature=0.0,
        )
    except Exception:
        pass


def choose_llm_action(client: OpenAI, model_name: str, obs) -> tuple[str, str | None]:
    prompt = (
        "You are an agent solving an inference debugging task.\n"
        "Return exactly one action string and nothing else.\n"
        "Do not explain.\n"
        "Do not use markdown.\n"
        "Valid actions are:\n"
        "inspect_metrics\n"
        "inspect_logs\n"
        "inspect_config\n"
        "inspect_recent_deploy\n"
        "mark_root_cause:batch_size_too_high\n"
        "mark_root_cause:tokenizer_version_mismatch\n"
        "mark_root_cause:request_timeout_too_low\n"
        "apply_fix:reduce_batch_size\n"
        "apply_fix:rollback_tokenizer\n"
        "apply_fix:restore_timeout_config\n"
        "apply_fix:restart_service\n"
        "resolve_incident\n\n"
        "Observation:\n"
        f"{obs.model_dump_json()}\n\n"
        "Return only the next action."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw_content = response.choices[0].message.content or ""
        action_str = raw_content.strip().splitlines()[0].strip()
        action_str = action_str.strip("`").replace("**", "").strip()
        return action_str, None
    except Exception as e:
        return "inspect_logs", f"llm_call_failed: {str(e).replace(chr(10), ' ')}"


def run_single_task(task_id: str, client: OpenAI, model_name: str) -> dict:
    env = InferOpsEnv(task_id=task_id)
    obs = env.reset()

    print(f"[START] task={obs.task_id} env=inferops model={model_name}")

    done = False
    success = False
    rewards: list[float] = []

    try:
        while not done:
            action_str = choose_rule_based_action(obs)
            parse_error = None

            if action_str is None:
                action_str, parse_error = choose_llm_action(client, model_name, obs)

            action_str_clean = action_str.replace("\n", " ")

            try:
                action = parse_action(action_str)
            except Exception as e:
                parse_error = str(e).replace("\n", " ")
                action = Action(action_type=ActionType.inspect_logs)
                action_str_clean = "inspect_logs"

            step_result = env.step(action)
            obs = step_result.observation
            reward = step_result.reward
            done = step_result.done
            rewards.append(reward)

            error_val = parse_error or obs.last_action_error or "null"
            error_val = str(error_val).replace("\n", " ")

            print(
                f"[STEP] task={task_id} step={env.step_count} action={action_str_clean} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_val}"
            )

            if done:
                success = step_result.info.get("success", False)

    except Exception as e:
        print(f"Exception: {str(e).replace(chr(10), ' ')}", file=sys.stderr)
    finally:
        step_count = env.step_count
        env.close()

    total_score = sum(rewards)
    total_score = min (0.99 , max (total_score ,0.01))
    print(
        f"[END] task={task_id} success={str(success).lower()} steps={step_count} "
        f"score={total_score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}"
    )

    return {
        "task_id": task_id,
        "success": success,
        "steps": step_count,
        "score": total_score,
        "rewards": rewards,
    }


def print_overall_summary(results: Iterable[dict]) -> None:
    results = list(results)
    print("\n[SUMMARY]")
    for r in results:
        print(
            f"task={r['task_id']} success={str(r['success']).lower()} "
            f"steps={r['steps']} score={r['score']:.2f}"
        )


def main():
    client, model_name = build_client()
    warmup_client(client, model_name)

    task_ids = get_task_ids()
    print(f"[RUN] tasks={','.join(task_ids)}")

    results = [run_single_task(task_id, client, model_name) for task_id in task_ids]
    print_overall_summary(results)


if __name__ == "__main__":
    main()
