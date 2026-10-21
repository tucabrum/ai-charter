"""Interface de linha de comando:  ai-charter musica.flac [opções]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_charter.pipeline import ALL_INSTRUMENTS, Options, run


def parse_args(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(
        prog="ai-charter",
        description="Gera charts de Clone Hero / YARG (Easy a Expert) a partir de um MP3 ou FLAC.",
    )
    parser.add_argument("input", type=Path, help="arquivo de música (.mp3, .flac, .wav, .ogg)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"),
                        help="pasta onde a música será criada (padrão: ./output)")
    parser.add_argument("--separator", choices=("demucs", "folder", "none"), default="demucs",
                        help="como separar os stems (padrão: demucs)")
    parser.add_argument("--stems", type=Path, help="pasta com stems prontos (use com --separator folder)")
    parser.add_argument("--rules", type=Path, help="arquivo .toml com regras de charting personalizadas")
    parser.add_argument("--instruments", default=",".join(ALL_INSTRUMENTS),
                        help="instrumentos a gerar, separados por vírgula (padrão: guitar,bass,drums,keys)")
    parser.add_argument("--tempo", choices=("auto", "constant", "variable"), default="auto",
                        help="mapa de tempo constante ou variável (padrão: auto)")
    parser.add_argument("--title", help="nome da música (padrão: lido das tags do arquivo)")
    parser.add_argument("--artist", help="artista (padrão: lido das tags do arquivo)")
    parser.add_argument("--zip", action="store_true", help="também cria um .zip da pasta da música")
    parser.add_argument("--keep-work", action="store_true", help="não apaga os arquivos temporários")
    args = parser.parse_args(argv)

    instruments = tuple(i.strip() for i in args.instruments.split(",") if i.strip())
    invalid = set(instruments) - set(ALL_INSTRUMENTS)
    if invalid:
        parser.error(f"instrumento(s) inválido(s): {', '.join(sorted(invalid))}")
    if not args.input.exists():
        parser.error(f"arquivo não encontrado: {args.input}")

    return Options(
        input=args.input, output_dir=args.output, separator=args.separator, stems_dir=args.stems,
        rules_path=args.rules, instruments=instruments, tempo_mode=args.tempo,
        title=args.title, artist=args.artist, make_zip=args.zip, keep_work=args.keep_work,
    )


def main(argv: list[str] | None = None) -> None:
    opts = parse_args(argv)
    try:
        result = run(opts)
    except (RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"\nPronto! Música criada em: {result.song_dir}")
    if result.zip_path:
        print(f"Arquivo .zip: {result.zip_path}")
    print("Copie essa pasta para a pasta de músicas do Clone Hero ou do YARG e faça um rescan.")


if __name__ == "__main__":
    main()
