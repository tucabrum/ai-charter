"""notes.confidence.json: os compassos em que o modelo de bateria ficou mais em dúvida.

O editor marca esses compassos e pula de um para o outro, para a revisão começar por onde há mais
chance de erro. Medido em 61 músicas do Rock Band: os 20% de compassos com mais dúvida (soma da
dúvida dos golpes) concentram ~38% dos erros da bateria.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NAME = "notes.confidence.json"


def doubtful_bars(doubt: dict[int, float], bar_ticks: np.ndarray, share: float = 0.2) -> list[dict]:
    """Os `share` compassos com mais dúvida: [{"inicio", "fim", "duvida" (0-1, relativa à música)}]."""
    if not doubt or len(bar_ticks) < 2:
        return []
    ticks = np.array(sorted(doubt))
    bars = np.searchsorted(bar_ticks, ticks, side="right") - 1
    total = np.zeros(len(bar_ticks))
    np.add.at(total, np.clip(bars, 0, len(bar_ticks) - 1), [doubt[t] for t in ticks])
    used = np.flatnonzero(total > 0)
    if not len(used):
        return []
    top = used[np.argsort(-total[used])][: max(1, int(round(share * len(used))))]
    peak = float(total[top].max())
    ends = np.append(bar_ticks[1:], bar_ticks[-1] + (bar_ticks[-1] - bar_ticks[-2]))
    return [{"inicio": int(bar_ticks[b]), "fim": int(ends[b]), "duvida": round(float(total[b]) / peak, 3)}
            for b in sorted(top)]


def write_confidence(path: Path, drum_doubt: dict[int, float], bar_ticks: np.ndarray, shift: int = 0) -> None:
    """Grava o arquivo (ou apaga um antigo, se não há dúvida para mostrar). `shift`: ticks somados a
    tudo (a entrada de silêncio posta antes da música)."""
    bars = doubtful_bars(drum_doubt, bar_ticks)
    if not bars:
        path.unlink(missing_ok=True)
        return
    for b in bars:
        b["inicio"] += shift
        b["fim"] += shift
    path.write_text(json.dumps({"drums": bars}, ensure_ascii=False, indent=1), encoding="utf-8")
