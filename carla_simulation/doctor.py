#!/usr/bin/env python3
"""Read-only preflight checks for the CARLA starter environment."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def command_output(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=8, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode == 0, output[0] if output else f"exit {result.returncode}"


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("CARLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CARLA_PORT", "2000")))
    args = parser.parse_args()

    system = platform.system()
    machine = platform.machine()
    py = sys.version_info
    checks = [
        Check("OS", system == "Linux", f"{system} {machine}; server requires Linux x86_64 or Windows x86_64"),
        Check("CPU", machine in {"x86_64", "AMD64"}, machine),
        Check("Python", (py.major, py.minor) in {(3, 10), (3, 11), (3, 12)}, platform.python_version()),
        Check("CARLA Python API", importlib.util.find_spec("carla") is not None, "installed" if importlib.util.find_spec("carla") else "not installed"),
        Check("Docker", shutil.which("docker") is not None, shutil.which("docker") or "not installed"),
        Check("CARLA RPC", port_open(args.host, args.port), f"{args.host}:{args.port}"),
    ]

    if shutil.which("nvidia-smi"):
        ok, detail = command_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        checks.append(Check("NVIDIA GPU", ok, detail))
    else:
        checks.append(Check("NVIDIA GPU", False, "nvidia-smi not found"))

    width = max(len(check.name) for check in checks)
    for check in checks:
        marker = "OK" if check.ok else "--"
        print(f"[{marker}] {check.name:<{width}}  {check.detail}")

    server_ready = checks[0].ok and checks[1].ok and checks[4].ok and checks[-1].ok
    client_ready = checks[2].ok and checks[3].ok and checks[5].ok
    print()
    if system == "Darwin":
        print("This Mac can open the browser UI, but cannot run the official CARLA server or Python wheel.")
        print("Run ./launch.sh on an Ubuntu x86_64/NVIDIA machine, then open http://SERVER_IP:8080 here.")
    elif server_ready and client_ready:
        print("Environment is ready.")
    else:
        print("Environment is not ready yet. Follow README.md.")
    return 0 if (server_ready and client_ready) else 1


if __name__ == "__main__":
    raise SystemExit(main())
