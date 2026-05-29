from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen



def wait_for_bee_api(
    base_url: str,
    timeout: int = 300,
    check_interval: int = 5,
    logger=print,
) -> bool:
    """Poll Bee API model endpoint until it responds successfully or timeout elapses."""
    endpoint = f"{base_url.rstrip('/')}/v1/models"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            req = Request(endpoint, method="GET")
            with urlopen(req, timeout=10) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if 200 <= int(status) < 300:
                    logger(f"Bee API is reachable at {endpoint} (status={status}).")
                    return True
                logger(f"Bee API not ready yet at {endpoint} (status={status}).")
        except HTTPError as exc:
            logger(f"Bee API returned HTTP {exc.code} at {endpoint}; retrying...")
        except URLError as exc:
            logger(f"Bee API not reachable at {endpoint}: {exc}; retrying...")
        except Exception as exc:  # pragma: no cover - defensive
            logger(f"Unexpected error while checking Bee API: {exc}; retrying...")

        time.sleep(check_interval)

    logger(f"Timed out after {timeout}s waiting for Bee API at {endpoint}.")
    return False



def build_bee_failure_guidance(base_url: str, start_command: str | None = None) -> str:
    """Build actionable guidance when Bee API is unavailable."""
    endpoint = f"{base_url.rstrip('/')}/v1/models"
    lines = [
        f"Bee API is not reachable at {endpoint}.",
        "Verify the server is running and accessible from this environment.",
    ]
    if start_command:
        lines.append(f"Try starting Bee with: {start_command}")
    else:
        lines.append("Provide a Bee start command and retry with automatic startup enabled.")
    return "\n".join(lines)



def start_bee_server(start_command: str, logger=print):
    """Start Bee server process using shell command and return Popen handle."""
    logger(f"Starting Bee server with command: {start_command}")
    process = subprocess.Popen(start_command, shell=True)
    return process


@dataclass
class BeeStartupResult:
    ready: bool
    started: bool
    message: str



def ensure_bee_api_ready(
    base_url: str,
    start_bee: bool = False,
    start_command: str | None = None,
    timeout: int = 300,
    check_interval: int = 5,
    logger=print,
) -> BeeStartupResult:
    """Ensure Bee API is reachable, optionally starting Bee when unavailable."""
    if wait_for_bee_api(
        base_url=base_url,
        timeout=timeout,
        check_interval=check_interval,
        logger=logger,
    ):
        return BeeStartupResult(
            ready=True,
            started=False,
            message=f"Bee API is ready at {base_url.rstrip('/')}/v1/models.",
        )

    if not start_bee:
        return BeeStartupResult(
            ready=False,
            started=False,
            message=build_bee_failure_guidance(base_url, start_command=start_command),
        )

    if not start_command:
        return BeeStartupResult(
            ready=False,
            started=False,
            message=build_bee_failure_guidance(base_url, start_command=None),
        )

    start_bee_server(start_command=start_command, logger=logger)

    if wait_for_bee_api(
        base_url=base_url,
        timeout=timeout,
        check_interval=check_interval,
        logger=logger,
    ):
        return BeeStartupResult(
            ready=True,
            started=True,
            message=f"Bee API started and is ready at {base_url.rstrip('/')}/v1/models.",
        )

    return BeeStartupResult(
        ready=False,
        started=True,
        message=build_bee_failure_guidance(base_url, start_command=start_command),
    )
