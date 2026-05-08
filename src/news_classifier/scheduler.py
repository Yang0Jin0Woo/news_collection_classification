from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class SimpleIntervalScheduler:
    interval_seconds: int
    job: Callable[[], None]
    max_runs: int | None = None

    def start(self) -> None:
        runs = 0
        while True:
            self.job()
            runs += 1
            if self.max_runs is not None and runs >= self.max_runs:
                break
            time.sleep(self.interval_seconds)
