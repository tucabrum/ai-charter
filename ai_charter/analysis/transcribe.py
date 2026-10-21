"""Transcrição automática da letra com o Whisper (opcional: --auto-lyrics).

Usa o faster-whisper (uma versão mais leve do Whisper, que roda bem na CPU).
Instale com:  pip install "ai-charter[lyrics]"
Na primeira vez, o modelo é baixado da internet; depois funciona offline.

O Whisper foi treinado principalmente com fala, então erra mais em canto:
o resultado é um RASCUNHO da letra, que deve ser revisado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class TimedWord:
    text: str
    start: float
    end: float


def words_to_lines(words: Iterable[TimedWord], gap_seconds: float, max_words: int = 12) -> list[str]:
    """Agrupa as palavras em linhas (frases): uma nova linha começa depois de uma pausa
    de pelo menos `gap_seconds`, ou quando a linha já está longa demais."""
    lines: list[list[str]] = []
    previous_end = None
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        new_line = (previous_end is None or word.start - previous_end >= gap_seconds
                    or len(lines[-1]) >= max_words)
        if new_line:
            lines.append([])
        lines[-1].append(text)
        previous_end = word.end
    return [" ".join(line) for line in lines]


def transcribe_lyrics(vocals: Path, language: str, model_size: str, gap_seconds: float) -> list[str]:
    """Ouve o stem de voz e devolve a letra, uma linha por frase cantada."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            'O pacote "faster-whisper" não está instalado. Rode:  pip install "ai-charter[lyrics]"\n'
            "ou forneça a letra com --lyrics letra.txt."
        ) from exc

    try:
        model = WhisperModel(model_size, device="auto")
    except Exception as exc:  # o erro varia (sem internet, proxy, nome de modelo errado...)
        raise RuntimeError(
            f'Não foi possível carregar o modelo "{model_size}" do Whisper ({exc}).\n'
            "Na primeira vez ele é baixado de huggingface.co, então é preciso internet. "
            "Confira também o nome do modelo (whisper_model nas regras)."
        ) from exc
    segments, _ = model.transcribe(
        str(vocals), language=language, word_timestamps=True,
        vad_filter=True,                  # ignora trechos sem voz
        condition_on_previous_text=False,  # evita que um erro "contamine" o resto da letra
    )
    words = [TimedWord(w.word, w.start, w.end) for segment in segments for w in (segment.words or [])]
    return words_to_lines(words, gap_seconds)
