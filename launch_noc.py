import subprocess
import webbrowser
import time
import sys
import os

print("Starting Network Cell Health Monitor...")

# Start FastAPI in background
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn",
     "network_health_api:app",
     "--host", "0.0.0.0",
     "--port", "8000"],
    cwd=os.path.dirname(os.path.abspath(__file__))
)

# Wait for server to start
time.sleep(2)

# Open dashboard in browser
dashboard = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "kpi_upload_dashboard.html"
)
webbrowser.open(f"file:///{dashboard}")

print("Dashboard opened. Press Ctrl+C to stop.")

try:
    server.wait()
except KeyboardInterrupt:
    server.terminate()
    print("Stopped.")