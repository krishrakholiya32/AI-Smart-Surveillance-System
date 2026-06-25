"""
In-memory login lockout. Safe because the backend runs as a single
uvicorn worker process (see entrypoint.sh) — if that ever changes to
multiple workers, this state needs to move to Redis/the DB instead.
"""
import time
from typing import Dict, Tuple

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60

# username -> (failed_count, locked_until_epoch)
_attempts: Dict[str, Tuple[int, float]] = {}


def seconds_until_unlocked(username: str) -> float:
    _, locked_until = _attempts.get(username, (0, 0.0))
    remaining = locked_until - time.time()
    return remaining if remaining > 0 else 0.0


def record_failure(username: str) -> None:
    count, _ = _attempts.get(username, (0, 0.0))
    count += 1
    locked_until = time.time() + LOCKOUT_SECONDS if count >= MAX_ATTEMPTS else 0.0
    _attempts[username] = (count, locked_until)


def record_success(username: str) -> None:
    _attempts.pop(username, None)
