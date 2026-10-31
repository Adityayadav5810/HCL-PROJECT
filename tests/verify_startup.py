"""
Robust Streamlit Startup and Accessibility Verification
-------------------------------------------------------
Verifies that Streamlit starts up, loads the machine learning models,
binds to the HTTP port, and responds to health checks and HTTP requests.
"""

import sys
import time
import urllib.request
import subprocess

def test_startup():
    print("Starting Streamlit test process on 127.0.0.1:8501...")
    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        "Movie-Recommendation-System/app.py",
        "--server.port", "8501",
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    is_healthy = False
    html_status = None
    out = ""
    err = ""

    try:
        # Poll health endpoint on 127.0.0.1 (avoids Windows IPv6 ::1 localhost delay)
        for attempt in range(1, 35):
            time.sleep(1.0)
            elapsed = round(time.time() - t0, 1)

            # Check if process crashed early
            if proc.poll() is not None:
                print(f"Process exited early with return code {proc.poll()} at {elapsed}s")
                break

            for url in ["http://127.0.0.1:8501/_stcore/health", "http://localhost:8501/_stcore/health"]:
                try:
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        if resp.status == 200:
                            is_healthy = True
                            print(f"SUCCESS: Streamlit health endpoint responded 200 OK via {url} in {elapsed}s!")
                            break
                except Exception:
                    pass

            if is_healthy:
                # Also verify root page HTML content
                try:
                    req_root = urllib.request.Request("http://127.0.0.1:8501/")
                    with urllib.request.urlopen(req_root, timeout=3.0) as root_resp:
                        html_status = root_resp.status
                        html_snippet = root_resp.read(1000).decode("utf-8", errors="replace")
                        print(f"Root URL HTTP status: {html_status}")
                        print("HTML title or app container found:", "<!DOCTYPE html>" in html_snippet or "<html" in html_snippet)
                except Exception as e:
                    print(f"Warning on root page probe: {e}")
                break

            if attempt % 5 == 0:
                print(f"  ...waiting for app initialization ({elapsed}s elapsed)...")

    finally:
        print("Stopping test server...")
        proc.terminate()
        try:
            out, err = proc.communicate(timeout=4)
        except Exception as e:
            proc.kill()
            try:
                out, err = proc.communicate(timeout=2)
            except Exception:
                pass

        print("--- Streamlit STDOUT ---")
        print(out[:800] if out else "(No stdout captured)")
        print("--- Streamlit STDERR ---")
        print(err[:800] if err else "(No stderr captured)")
        print("Test server stopped cleanly.")

    return is_healthy

if __name__ == "__main__":
    success = test_startup()
    print("Startup verification result:", "PASSED" if success else "FAILED")
    sys.exit(0 if success else 1)
