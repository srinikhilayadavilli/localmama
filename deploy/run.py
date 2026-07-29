"""Container entrypoint: runs the API server and the LiveKit worker together.

**Why one container and not two services.** Leads and transcripts are JSON
files on disk (`persistence.py`), and a Railway volume attaches to exactly one
service. Split the two processes and the worker writes leads to a disk the
`/admin` page cannot read — the admin view would sit empty while real calls
were being captured. Keeping them together keeps one `DATA_DIR` and one truth.

The honest cost: they restart together, and they scale together. The fix, when
leads outgrow JSON files, is to move `persistence.py` onto Postgres and split
this into two Railway services — at which point this file goes away.

Either child dying takes the whole container down, so the platform restarts it
rather than leaving a half-dead deploy answering health checks with no agent
behind it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

#: (label, argv). The worker's `start` subcommand is the production mode of
#: LiveKit's CLI — `dev` enables file-watching and reload, which is wrong here.
PROCESSES: list[tuple[str, list[str]]] = [
    ("api", [sys.executable, "-m", "backend.app.main"]),
    ("agent", [sys.executable, "-m", "backend.app.agent", "start"]),
]

children: list[tuple[str, subprocess.Popen]] = []


def _shutdown(signum: int, _frame) -> None:
    """Forward the platform's stop signal so calls end cleanly."""
    print(f"[supervisor] signal {signum}, stopping children", flush=True)
    for label, proc in children:
        if proc.poll() is None:
            print(f"[supervisor] terminating {label}", flush=True)
            proc.terminate()
    deadline = time.monotonic() + 10
    for label, proc in children:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[supervisor] {label} ignored SIGTERM, killing", flush=True)
            proc.kill()
    sys.exit(0)


def main() -> int:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    for label, argv in PROCESSES:
        print(f"[supervisor] starting {label}: {' '.join(argv)}", flush=True)
        children.append((label, subprocess.Popen(argv, env=os.environ.copy())))

    # Poll rather than os.wait(), so the label of whichever child died is known
    # and the log says which one — the first thing you want on a failed deploy.
    while True:
        for label, proc in children:
            code = proc.poll()
            if code is None:
                continue
            print(f"[supervisor] {label} exited with {code}; "
                  "stopping the container", flush=True)
            for other_label, other in children:
                if other is not proc and other.poll() is None:
                    other.terminate()
            for _, other in children:
                try:
                    other.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    other.kill()
            return code or 1
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
