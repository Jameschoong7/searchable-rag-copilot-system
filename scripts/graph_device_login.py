import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.connectors.graph_auth import run_device_login


if __name__ == "__main__":
    result = run_device_login()
    print(result)