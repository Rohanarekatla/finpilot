"""Run the FinPilot server: `python main.py` (or `uv run main.py`).

Equivalent to `uvicorn app.main:app`; HOST, PORT and RELOAD can be set in the environment.
"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",
    )
