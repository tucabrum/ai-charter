"""Qual stem vira cada instrumento da chart (uma ocasião incomum).

Numa música sem guitarra, o sintetizador (que o Demucs põe no stem "other") pode virar a pista
de guitarra:  --stem-source guitar=other

O áudio acompanha: o stem escolhido vai para o guitar.ogg e o da guitarra vai para o lugar
dele (song.ogg). Os dois trocam de posição, então a mixagem total não muda, e o som que
silencia quando o jogador erra é o do instrumento que ele está tocando.

A bateria não troca (nem como destino nem como fonte): o jogo separa as peças dela em
arquivos próprios e a transcrição depende do stem da bateria.
"""

from __future__ import annotations

from pathlib import Path

TARGETS = ("guitar", "bass", "keys")                      # pistas que podem usar outro stem
SOURCES = ("guitar", "bass", "keys", "vocals", "other")   # stems que podem virar uma dessas pistas
INI_KEY = "stem_sources"                                  # no song.ini: a troca feita nos arquivos

_NAMES = {
    "guitar": ("guitar", "guitarra", "violao"),
    "bass": ("bass", "baixo"),
    "keys": ("keys", "piano", "teclado", "teclas"),
    "vocals": ("vocals", "voz", "vocais", "vocal"),
    "other": ("other", "resto", "synth", "sintetizador", "sintetizadores", "song"),
}


def _name(text: str) -> str | None:
    text = text.strip().lower()
    return next((stem for stem, names in _NAMES.items() if text in names), None)


def validate(mapping: dict[str, str]) -> dict[str, str]:
    """Confere a troca e tira as entradas sem efeito (guitar=guitar)."""
    out = {}
    for target, source in mapping.items():
        if target == "drums" or source == "drums":
            raise ValueError("a bateria não pode trocar de stem")
        if target not in TARGETS:
            raise ValueError(f"só guitarra, baixo e teclado podem usar outro stem (não {target})")
        if source not in SOURCES:
            raise ValueError(f"stem desconhecido para {target}: {source}")
        if source != target:
            out[target] = source
    used = list(out.values())
    for source in set(used):
        if used.count(source) > 1:
            raise ValueError(f"o stem {source} não pode virar dois instrumentos ao mesmo tempo")
    return out


def parse(text: str) -> dict[str, str]:
    """"guitar=other, keys=vocals" (ou em português: "guitarra=resto") -> {"guitar": "other", ...}."""
    mapping = {}
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        target, sep, source = part.partition("=")
        t, s = _name(target), _name(source)
        if not sep or t is None or s is None:
            raise ValueError(f"troca de stem inválida: {part} (use, por exemplo, guitar=other)")
        mapping[t] = s
    return validate(mapping)


def to_text(mapping: dict[str, str]) -> str:
    return ",".join(f"{t}={s}" for t, s in mapping.items())


def layout(mapping: dict[str, str]) -> dict[str, str]:
    """Para cada lugar de áudio que muda, de qual stem original vem o som.

    O destino recebe o stem escolhido; o som original do destino vai para o lugar que ficou vago.
    {"guitar": "other"} -> {"guitar": "other", "other": "guitar"}."""
    mapping = validate(mapping)
    moved = dict(mapping)
    vacated = [s for s in mapping.values() if s not in mapping]   # fontes que não recebem nada
    displaced = [t for t in mapping if t not in mapping.values()]  # destinos cujo som sobrou
    moved.update(zip(vacated, displaced))
    return moved


def arrange(stems: dict[str, list[Path]], mapping: dict[str, str]) -> dict[str, list[Path]]:
    """Os arquivos de cada lugar de áudio depois da troca (os lugares sem som ficam de fora)."""
    moved = layout(mapping)
    out = {slot: stems.get(moved.get(slot, slot), []) for slot in set(stems) | set(moved)}
    return {slot: files for slot, files in out.items() if files}


def original(current: dict[str, list[Path]], mapping: dict[str, str]) -> dict[str, list[Path]]:
    """Desfaz a troca de uma pasta já trocada: o stem original de cada lugar."""
    moved = layout(mapping)
    out = {moved.get(slot, slot): files for slot, files in current.items()}
    return {stem: files for stem, files in out.items() if files}


def recorded(folder: Path) -> dict[str, str]:
    """A troca feita nos arquivos da pasta (anotada no song.ini)."""
    ini = Path(folder) / "song.ini"
    if not ini.is_file():
        return {}
    for line in ini.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip().lower() == INI_KEY:
            try:
                return parse(value)
            except ValueError:
                return {}
    return {}


def record(folder: Path, mapping: dict[str, str]) -> None:
    from ai_charter.augment import update_song_ini

    update_song_ini(Path(folder) / "song.ini", {INI_KEY: to_text(mapping)})


def rearrange_folder(folder: Path, current: dict[str, list[Path]], old: dict[str, str],
                     new: dict[str, str]) -> list[str]:
    """Troca os áudios da pasta de uma música da troca `old` para a `new`. Devolve os arquivos gravados.

    Um lugar de um arquivo .ogg só é copiado; vários arquivos (ex.: duas guitarras) viram um só."""
    import shutil

    from ai_charter.export import audio
    from ai_charter.pipeline import STEM_FILES

    folder = Path(folder)
    stems = original(current, old)
    after = arrange(stems, new)
    changed = [slot for slot in set(current) | set(after)
               if sorted(current.get(slot, [])) != sorted(after.get(slot, []))]
    temps = {}
    for slot in changed:   # primeiro, cópias ao lado; os originais ainda são lidos
        if slot not in after:
            continue
        tmp = folder / f".troca-{slot}.ogg"
        files = after[slot]
        if len(files) == 1 and files[0].suffix.lower() == ".ogg":
            shutil.copy2(files[0], tmp)
        else:
            audio.mix_to_ogg(files, tmp)
        temps[slot] = tmp
    for slot in changed:
        for file in current.get(slot, []):
            file.unlink(missing_ok=True)
    written = []
    for slot, tmp in temps.items():
        name = STEM_FILES[slot][0]
        tmp.replace(folder / name)
        written.append(name)
    record(folder, new)
    return sorted(written)
