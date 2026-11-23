"""Launch the local app and optionally open its UI."""
import argparse
import os
from pathlib import Path
import json
import threading
import time
import urllib.request
import webbrowser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    os.environ["MOTIONBRIDGE_DATA"] = str(project_root / "data/physics-jobs")
    url = f"http://127.0.0.1:{args.port}"
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
            existing = json.load(response)
        if existing.get("instance_id") != "motionbridge-physics-dev" or Path(existing.get("project_root", ".")).resolve() != project_root:
            raise SystemExit("端口已被其他工程占用，未复用或停止其他服务。")
        if existing.get("version") and "model_ready" in existing:
            if not args.no_browser:
                webbrowser.open(url)
            print("MotionBridge is already running:", url)
            return
    except Exception:
        pass
    if not args.no_browser:
        def open_when_ready():
            for _ in range(50):
                try:
                    urllib.request.urlopen(url + "/api/health", timeout=.3).close()
                    webbrowser.open(url)
                    return
                except Exception:
                    time.sleep(.2)
        threading.Thread(target=open_when_ready, daemon=True).start()
    import uvicorn
    uvicorn.run("motionbridge.app:app", host="127.0.0.1", port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
