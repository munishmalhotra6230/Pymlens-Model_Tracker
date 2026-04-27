import subprocess
import os
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        dashboard_path = os.path.join(
            os.path.dirname(__file__),
            "dashboard.py"
        )
        subprocess.run(["streamlit", "run", dashboard_path])
    else:
        print("Usage: pymlens dashboard")