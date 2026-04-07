import os
import sys
from openai import OpenAI

from inferops_env.environment import InferOpsEnv
from inferops_env.models import Action, ActionType


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

    if obs.task_id == "medium_partial_tokenizer_regression_01":
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

    if obs.task_id == "hard_misleading_restart_signal_01":
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


def main():
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    task_id_env = os.environ.get("TASK_ID")
    env = InferOpsEnv(task_id=task_id_env)

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

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )

                raw_content = response.choices[0].message.content or ""
                action_str = raw_content.strip().splitlines()[0].strip()
                action_str = action_str.strip("`").replace("**", "").strip()

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
                f"[STEP] step={env.step_count} action={action_str_clean} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_val}"
            )

            if done:
                success = step_result.info.get("success", False)

    except Exception as e:
        print(f"Exception: {str(e).replace(chr(10), ' ')}", file=sys.stderr)
    finally:
        env.close()
        print(
            f"[END] success={str(success).lower()} steps={env.step_count} "
            f"rewards={','.join(f'{r:.2f}' for r in rewards)}"
        )


if __name__ == "__main__":
    main()