from __future__ import annotations

import random
import time


def iteration_id_random(prefix: str = "iteration") -> str:
    """
    Similar ao seu R:
      sprintf("iteration_%s", runif(1, 10000, 99999))
    """
    r = random.randint(10000, 99999)
    return f"{prefix}_{r}_{int(time.time() * 1000)}"
