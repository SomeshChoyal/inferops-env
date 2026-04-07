from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ValidationError

from inferops_env.environment import InferOpsEnv
from inferops_env.models import Action

app = FastAPI(title="InferOps Env API")
env = InferOpsEnv()
env.reset()


class StepRequest(BaseModel):
    action: dict


@app.get("/")
def read_root():
    return {
        "name": "InferOps Env",
        "status": env.status,
        "current_task": env.task_id,
        "available_tasks": sorted(env.tasks.keys()),
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {"tasks": sorted(env.tasks.keys())}


@app.post("/reset")
def reset_env(task_id: str | None = Query(default=None)):
    global env
    if task_id is not None:
        if task_id not in env.tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id '{task_id}'. Available: {sorted(env.tasks.keys())}",
            )
        env = InferOpsEnv(task_id=task_id)
    obs = env.reset()
    return obs.model_dump()


@app.get("/state")
def get_state():
    return env.state()


@app.post("/step")
def step_env(req: StepRequest):
    try:
        action_obj = Action(**req.action)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parsing error: {str(e)}")

    try:
        result = env.step(action_obj)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Runtime error: {str(e)}")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
