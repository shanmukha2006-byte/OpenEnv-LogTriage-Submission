# server/app.py
# FastAPI server exposing the RL inference engine via HTTP.

from fastapi import FastAPI, HTTPException
import uvicorn

from core.inference import run_inference   # ← correct import path

app = FastAPI(
    title       = "OpenEnv Log Triage",
    description = "RL-based DevOps log triage inference server.",
    version     = "0.1.0",
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "openenv-logtriage"}


@app.get("/health")
def health():
    """Detailed health check."""
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/reset")
def reset():
    """Run one RL inference episode and return results."""
    try:
        result = run_inference()
        return {
            "status" : "reset successful",
            "result" : result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer")
def infer(steps: int = 10):
    """
    Run inference for a configurable number of steps.
    Query param: ?steps=N  (default 10, max 50)
    """
    steps = min(max(steps, 1), 50)
    try:
        result = run_inference(max_steps=steps)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Entrypoint called by the 'server' console script."""
    uvicorn.run(
        "server.app:app",
        host    = "0.0.0.0",
        port    = 7860,
        reload  = False,
    )


if __name__ == "__main__":
    main()
