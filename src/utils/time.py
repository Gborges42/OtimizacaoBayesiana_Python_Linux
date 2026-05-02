from __future__ import annotations

from datetime import datetime
from typing import Union


def calcular_tempo_dec(start_time: Union[datetime, float]) -> str:
    """
    Porta do R calcular_tempo_dec() com saída tipo:
      "D: X H: Y M: Z S: W"
    """
    if isinstance(start_time, (int, float)):
        start_dt = datetime.fromtimestamp(float(start_time))
    else:
        start_dt = start_time

    end_dt = datetime.now()
    diff_sec = (end_dt - start_dt).total_seconds()

    days = int(diff_sec // (24 * 3600))
    remainder = diff_sec % (24 * 3600)
    hours = int(remainder // 3600)
    remainder = remainder % 3600
    minutes = int(remainder // 60)
    seconds = remainder % 60

    parts = []
    if days != 0:
        parts.append(f"D: {days}")
    if hours != 0:
        parts.append(f"H: {hours}")
    if minutes != 0:
        parts.append(f"M: {minutes}")
    if seconds != 0:
        parts.append(f"S: {int(round(seconds))}")

    return " ".join(parts) if parts else "S: 0"
