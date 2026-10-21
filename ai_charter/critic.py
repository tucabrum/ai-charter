"""Avaliador de jogabilidade: aponta defeitos de uma chart SEM comparar com outra.

Uso:  ai-charter-critic pasta_da_musica/notes.mid

Em vez de medir se a chart "bate" com uma chart oficial (o que depende muito do
gosto de quem fez), mede coisas que deixam qualquer chart ruim de tocar:

- acordes de 3+ notas demais: a chart fica "inflada" e pesada;
- trocas relâmpago: um acorde de UMA palhetada só no meio de dois grupos
  estáveis (o som de uma troca, não uma parte musical);
- picos: a nota pula longe (3+ lanes) e volta logo em seguida;
- densidade: o trecho mais rápido (notas por segundo, em janelas de 2 s).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ai_charter.compare import NAMES, Event, read_chart

STABLE_GROUP = 3     # palhetadas seguidas do mesmo formato que contam como "grupo estável"
SPIKE_LANES = 3      # pulo que conta como "pico" (ida e volta)
DENSITY_WINDOW = 2.0


@dataclass
class Critique:
    notes: int
    triples: float          # fração das notas que são acordes de 3+ notas
    quick_changes: int      # acordes de uma palhetada só entre dois grupos estáveis
    spikes: int             # pulos de 3+ lanes que voltam na nota seguinte
    max_nps: float          # notas por segundo no trecho mais denso


def _groups(events: list[Event]) -> list[tuple[tuple[int, ...], int]]:
    """Formatos seguidos iguais: [(formato, quantas vezes), ...]."""
    groups: list[list] = []
    for e in events:
        if groups and groups[-1][0] == e.lanes:
            groups[-1][1] += 1
        else:
            groups.append([e.lanes, 1])
    return [(lanes, n) for lanes, n in groups]


def critique(events: list[Event]) -> Critique:
    if not events:
        return Critique(0, 0.0, 0, 0, 0.0)
    triples = sum(len(e.lanes) >= 3 for e in events) / len(events)
    groups = _groups(events)
    quick = sum(
        1 for (a, na), (b, nb), (c, nc) in zip(groups, groups[1:], groups[2:])
        if nb == 1 and len(b) >= 2 and na >= STABLE_GROUP and nc >= STABLE_GROUP
    )
    low = [min(e.lanes) for e in events]
    spikes = sum(
        1 for a, b, c in zip(low, low[1:], low[2:])
        if abs(b - a) >= SPIKE_LANES and abs(b - c) >= SPIKE_LANES and (b - a) * (b - c) > 0
    )
    times = [e.time for e in events]
    max_nps, start = 0.0, 0
    for end, t in enumerate(times):
        while t - times[start] > DENSITY_WINDOW:
            start += 1
        max_nps = max(max_nps, (end - start + 1) / DENSITY_WINDOW)
    return Critique(len(events), triples, quick, spikes, max_nps)


def report(path: Path) -> str:
    chart = read_chart(path)
    lines = ["| Faixa | Dificuldade | Notas | Acordes de 3+ | Trocas relâmpago | Picos | Notas/s (máx.) |",
             "|---|---|---|---|---|---|---|"]
    for part, name in NAMES.items():
        if part == "PART DRUMS" or part not in chart.tracks:
            continue
        for difficulty, events in chart.tracks[part].items():
            c = critique(events)
            lines.append(f"| {name} | {difficulty} | {c.notes} | {c.triples:.0%} | {c.quick_changes} "
                         f"| {c.spikes} | {c.max_nps:.1f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-charter-critic",
                                     description="Aponta defeitos de jogabilidade de uma chart (notes.mid).")
    parser.add_argument("chart", type=Path, help="notes.mid a avaliar")
    args = parser.parse_args(argv)
    print(report(args.chart))


if __name__ == "__main__":
    main()
