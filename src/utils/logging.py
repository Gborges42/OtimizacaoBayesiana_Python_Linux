from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def append_log(logfile: Union[str, os.PathLike], msg: str) -> None:
    path = Path(logfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg)
