import subprocess
import pytest


def test_trainrtdetr():
    process = subprocess.run(
        ["bash", "distributed_main_trainyolox.sh"],
        capture_output=True,
        text=True
    )

    assert process.returncode == 0, f"trainyolox .\nStdout:\n{process.stdout}\nStderr:\n{process.stderr}"