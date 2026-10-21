"""Leitor simples de arquivos .chart, usado só nos testes para conferir o resultado."""

from __future__ import annotations

import re
from pathlib import Path

LINE = re.compile(r"^\s*(\S+)\s*=\s*(.*?)\s*$")


def parse_chart(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Retorna {nome_da_seção: [(chave, valor), ...]}."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current = None
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif line in ("{", "}") or not line:
            continue
        else:
            match = LINE.match(line)
            assert match and current is not None, f"linha inválida: {raw!r}"
            sections[current].append((match.group(1), match.group(2)))
    return sections


def notes_by_tick(entries: list[tuple[str, str]]) -> dict[int, list[int]]:
    """{tick: [números de nota]} a partir das linhas "tick = N nota duração"."""
    result: dict[int, list[int]] = {}
    for key, value in entries:
        parts = value.split()
        if parts[0] == "N":
            result.setdefault(int(key), []).append(int(parts[1]))
    return result
