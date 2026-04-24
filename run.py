"""
Main entry point for the Win Go prediction system.
"""
import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HOST, PORT


def main():
    print("""
    ==================================================
    |        WIN GO PREDICTION SYSTEM v1.0            |
    |        Small (0-4) vs Big (5-9)                 |
    |=================================================|
    |  Dashboard:  http://127.0.0.1:8000              |
    |  API Docs:   http://127.0.0.1:8000/docs         |
    ==================================================
    """)

    uvicorn.run(
        "server.app:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
