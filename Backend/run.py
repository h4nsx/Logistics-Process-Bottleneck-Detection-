"""
Windows-compatible uvicorn launcher.

On Windows, psycopg (async) requires SelectorEventLoop.
This must be set BEFORE uvicorn creates its event loop.

Usage:
    python run.py
    python run.py --port 8001
"""
import asyncio
import sys

# Must be set before uvicorn starts — uvicorn otherwise creates ProactorEventLoop on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop="asyncio",
    )
