import subprocess
import atexit
import time
import signal
import os

procs = []

def cleanup():
    for proc in procs:
        if proc and proc.poll() is None:
            print("Terminating process:", proc.pid)
            os.kill(proc.pid, signal.SIGTERM)  # Windows

if __name__ == "__main__":
    atexit.register(cleanup)

    strengths = [25, 50, 100, 200, 400, 600]

    i = True
    for strength in strengths:
        # Run the watermarking script with the specified parameters.
        pipe = subprocess.PIPE if i else None
        procs.append(subprocess.Popen(["uv", "run", "krawtchouk_watermark_all.py", "0.5", "0.5", str(strength), "./datasets/ctscan/raw"], shell=True, stdout=pipe))
        i = False
    
    for proc in procs:
        if proc and proc.poll() is None:
            proc.wait()
        print("Process finished:", proc.pid)
    
    cleanup()
    print("All processes finished.")
