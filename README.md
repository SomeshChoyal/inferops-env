---
title: InferOps Env
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# InferOps Env

InferOps Env is a real-world OpenEnv-style environment designed for debugging machine learning inference incidents. It evaluates an AI agent's ability to logically investigate, diagnose, and resolve production failures in an inference service.

## Real-World Task Simulation

Unlike generic coding benchmarks, InferOps Env operates exactly how a Site Reliability Engineer (SRE) or ML Engineer approaches a live incident. The agent must traverse logs, read configuration states, examine metrics, and cross-reference recent deployments to identify the root cause before applying the correct fix. All tasks follow strict deterministic rules with hidden state until inspected.

## Observation Space

At each step, the agent receives an observation structured as follows:
- `task_id`: Unique identifier for the current incident.
- `difficulty`: Expected complexity (easy, medium, hard).
- `incident_summary`: High-level summary of the observed degradation.
- `discovered_metrics`: Revealed only after inspecting metrics.
- `discovered_logs`: Revealed only after inspecting logs.
- `discovered_config`: Revealed only after inspecting configs.
- `discovered_recent_deploy`: Revealed only after inspecting recent deployments.
- `actions_taken`: Ordered history of actions the agent has performed.
- `step_count`: Current step number.
- `status`: One of `investigating`, `resolved`, or `failed`.
- `last_action_error`: Error message from the previous action, if invalid.

## Action Space

The model must output one of the following string payloads:
- `inspect_metrics`: Reveals current APM metrics.
- `inspect_logs`: Reveals relevant service logs.
- `inspect_config`: Reveals the current service configuration.
- `inspect_recent_deploy`: Reveals global deployment history.
- `mark_root_cause:<value>`: Declares the underlying issue.
- `apply_fix:<value>`: Deploys a mitigation or fix.
- `resolve_incident`: Finalizes the episode.

## Tasks

1. **easy_batch_01 (Easy)**
   - *Incident:* Inference latency spiked after a configuration update.
   - *Challenge:* Read the config and metrics to spot an aggressively high batch size.
2. **medium_tokenizer_01 (Medium)**
   - *Incident:* Error rate increased after a deployment.
   - *Challenge:* Correlate recent deployment data with logs to find a tokenizer version mismatch.
3. **medium_partial_tokenizer_regression_01 (Medium)**
   - *Incident:* Some requests started failing after a tokenizer-related deployment.
   - *Challenge:* Partial failure with misleading success rates requires cross-referencing logs, config, and deploy data.
4. **hard_timeout_01 (Hard)**
   - *Incident:* Requests are backing up after a recent config deploy.
   - *Challenge:* Identify a hidden timeout regression causing queue pileup while GPU utilization remains low.
5. **hard_misleading_restart_signal_01 (Hard)**
   - *Incident:* Requests are timing out and operators suspect a worker restart issue.
   - *Challenge:* Misleading worker heartbeat signals distract from the real root cause: an aggressively low timeout.

## Reward and Grading

The environment is deterministic, scoring episodes strictly between `0.0` and `1.0`:
- **Positive Actions:** Correct sequence of root cause isolation (+0.30) and applying the right fix (+0.30). Identifying relevant telemetry for the first time adds incremental rewards (+0.10 to +0.20).
- **Negative Penalties:** Invalid actions, repeated inspects, or wrong root causes/fixes apply small deductions (-0.05 to -0.15).
- **Final Outcome:** Reaching `resolve_incident` with the correct root cause and fix finalizes the score. If resolved incorrectly, a -0.20 penalty is taken. The total is clamped between `0.0` and `1.0`.

## Setup Instructions

1. Clone or download the repository.
2. Ensure you have Python 3.11+ installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Alternatively, build the provided Docker container:
```bash
docker build -t inferops-env .
```

## Running the Agent

The entrypoint is `inference.py`, which initializes the environment and loops the OpenAI client against the configured model.

### Required Environment Variables

- `HF_TOKEN`: (Required) API Key for the OpenAI-compatible endpoint.
- `API_BASE_URL`: The endpoint URL (defaults to `https://api.openai.com/v1`).
- `MODEL_NAME`: The model version string to request (defaults to `gpt-4.1-mini`).

### Execution

```bash
export HF_TOKEN="your-api-token"
export API_BASE_URL="https://your.custom.endpoint/v1"
export MODEL_NAME="your-custom-model"

python inference.py
```

If using Docker:
```bash
docker run --rm -it \
  -e HF_TOKEN="your-api-token" \
  -e API_BASE_URL="https://your.custom.endpoint/v1" \
  -e MODEL_NAME="your-custom-model" \
  inferops-env
```

## API Endpoints (FastAPI)

When running the container, the environment is available as a REST API over port `7860`:
- `GET /` — Returns basic info, current status, and available tasks.
- `GET /tasks` — Returns a JSON array of all valid task IDs.
- `POST /reset` — Resets the environment. Use `?task_id=<id>` to switch to a specific scenario (e.g. `POST /reset?task_id=hard_timeout_01`).
- `GET /state` — Returns the current episode state.
- `POST /step` — Submit an action (`{"action": {"action_type": "...", "target": "..."}}`). Returns the result observation.

## Baseline Scores

| Task | Difficulty | Score | Steps | Success |
|------|-----------|-------|-------|---------|
| easy_batch_01 | Easy | 0.80 | 4 | true |
| medium_tokenizer_01 | Medium | 1.00 | 5 | true |
| medium_partial_tokenizer_regression_01 | Medium | 1.00 | 5 | true |
| hard_timeout_01 | Hard | 0.90 | 5 | true |
| hard_misleading_restart_signal_01 | Hard | 1.00 | 5 | true |
