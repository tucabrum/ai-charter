"""Interface de linha de comando:  ai-charter musica.flac [opções]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_charter.config import list_genres
from ai_charter.pipeline import ALL_INSTRUMENTS, Options, run


def parse_args(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(
        prog="ai-charter",
        description="Gera charts de Clone Hero / YARG (Easy a Expert) a partir de um MP3 ou FLAC.",
    )
    parser.add_argument("input", help="arquivo de música (.mp3, .flac, .wav, .ogg, .opus, .m4a) ou link de um "
                                      "vídeo do YouTube (o áudio é baixado e guardado em --downloads)")
    parser.add_argument("--downloads", type=Path,
                        help="pasta onde guardar o áudio baixado de um link (padrão: Música\\ai-charter)")
    parser.add_argument("--video", action="store_true",
                        help="com um link: o vídeo também vira o fundo da música (video.webm, VP8 até 720p)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"),
                        help="pasta onde a música será criada (padrão: ./output)")
    parser.add_argument("--separator", choices=("demucs", "folder", "none"), default="demucs",
                        help="como separar os stems (padrão: demucs)")
    parser.add_argument("--stems", type=Path, help="pasta com stems prontos (use com --separator folder)")
    parser.add_argument("--rules", type=Path, help="arquivo .toml com regras de charting personalizadas")
    genres = list_genres()
    parser.add_argument("--genre", choices=[g.id for g in genres], metavar="GENERO",
                        help="regras ajustadas para um gênero (o --rules vale por cima delas): "
                             + ", ".join(f"{g.id} ({g.name})" for g in genres))
    parser.add_argument("--cifra", type=Path,
                        help="cifra copiada do Cifra Club (.txt): a letra vira a faixa de vocal (se não houver "
                             "--lyrics) e os acordes limitam o reconhecimento de acordes")
    parser.add_argument("--reference", type=Path,
                        help="partitura (.mid ou Guitar Pro .gp3/.gp4/.gp5/.gpx/.gp) que guia a transcrição: corrige "
                             "altura e acordes das notas detectadas e as peças da bateria "
                             "(veja as pistas com: ai-charter-ref arquivo)")
    parser.add_argument("--reference-tracks",
                        help="pistas da partitura para cada instrumento, ex.: 2=guitar,4=bass "
                             "(padrão: palpite pelo nome e instrumento da pista)")
    parser.add_argument("--reference-fidelity", default="seguir",
                        help="quanto a partitura manda nas notas: corrigir (só altura e acordes das notas "
                             "detectadas), completar (o ritmo da partitura nos compassos em que ela confere com "
                             "o áudio) ou seguir (as notas da partitura onde ela tiver notas; padrão). Uma para "
                             "todos ou por instrumento, ex.: drums=seguir,guitar=completar")
    parser.add_argument("--reference-sections", default="",
                        help="pista da partitura escolhida à mão por trecho, quando um instrumento tem várias "
                             "(base, solo...): compassos da partitura=pista, ex.: 1-16=3,17-24=1 (o resto "
                             "continua automático)")
    parser.add_argument("--reference-anchors", default="",
                        help="pontos de alinhamento marcados à mão, quando a partitura se perde: compasso da "
                             "partitura=instante da música em que ele começa, ex.: 1=2.4,33=1:05.2")
    parser.add_argument("--instruments", default=",".join(ALL_INSTRUMENTS),
                        help="instrumentos a gerar, separados por vírgula (padrão: guitar,bass,drums,keys)")
    parser.add_argument("--stem-source", default="",
                        help="outro stem como fonte de uma pista, numa música sem aquele instrumento: "
                             "guitar=other (o resto, onde ficam os sintetizadores), keys=vocals... O áudio troca "
                             "de lugar junto (o stem escolhido vira o guitar.ogg). A bateria não troca")
    parser.add_argument("--tempo", choices=("auto", "constant", "variable"), default="auto",
                        help="mapa de tempo constante ou variável (padrão: auto)")
    parser.add_argument("--lyrics", type=Path,
                        help="letra (texto, uma linha por frase) para gerar a faixa de vocal; "
                             "padrão: um .txt com o mesmo nome da música, se existir")
    parser.add_argument("--auto-lyrics", action="store_true",
                        help="sem letra, transcreve a voz com o Whisper e salva um rascunho "
                             "(letra_automatica.txt) para revisar; precisa de: pip install \"ai-charter[lyrics]\"")
    parser.add_argument("--format", choices=("mid", "chart"), default="mid",
                        help="formato da chart: mid (padrão, tem vocais) ou chart")
    parser.add_argument("--title", help="nome da música (padrão: lido das tags do arquivo)")
    parser.add_argument("--artist", help="artista (padrão: lido das tags do arquivo)")
    parser.add_argument("--zip", action="store_true", help="também cria um .zip da pasta da música")
    parser.add_argument("--keep-work", action="store_true", help="não apaga os arquivos temporários")
    parser.add_argument("--view", action="store_true",
                        help="ao terminar, abre a música no visualizador (navegador)")
    args = parser.parse_args(argv)

    instruments = tuple(i.strip() for i in args.instruments.split(",") if i.strip())
    invalid = set(instruments) - set(ALL_INSTRUMENTS)
    if invalid:
        parser.error(f"instrumento(s) inválido(s): {', '.join(sorted(invalid))}")
    from ai_charter.stem_sources import parse as parse_stem_sources
    try:
        stem_sources = parse_stem_sources(args.stem_source) or None
    except ValueError as exc:
        parser.error(str(exc))
    from ai_charter.youtube import is_url
    url = args.input if is_url(args.input) else None
    if url is None and not Path(args.input).exists():
        parser.error(f"arquivo não encontrado: {args.input}")
    if args.lyrics is not None and not args.lyrics.exists():
        parser.error(f"arquivo de letra não encontrado: {args.lyrics}")
    if args.cifra is not None and not args.cifra.is_file():
        parser.error(f"arquivo de cifra não encontrado: {args.cifra}")
    reference_tracks, reference_fidelity, reference_sections, reference_anchors = None, "seguir", None, None
    if args.reference is not None:
        from ai_charter.reference import parse_anchors, parse_fidelity, parse_mapping, parse_sections, read_score
        try:   # lê já, para um arquivo ruim falhar antes da separação (que demora)
            read_score(args.reference)
            reference_tracks = parse_mapping(args.reference_tracks) if args.reference_tracks else None
            reference_fidelity = parse_fidelity(args.reference_fidelity)
            reference_sections = parse_sections(args.reference_sections) or None
            reference_anchors = parse_anchors(args.reference_anchors) or None
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))

    return Options(
        input=_source(parser, args, url), output_dir=args.output, separator=args.separator, stems_dir=args.stems,
        rules_path=args.rules, genre=args.genre, reference=args.reference, reference_tracks=reference_tracks,
        reference_fidelity=reference_fidelity, reference_sections=reference_sections,
        reference_anchors=reference_anchors, cifra=args.cifra,
        instruments=instruments, stem_sources=stem_sources, tempo_mode=args.tempo,
        title=args.title, artist=args.artist, lyrics=args.lyrics, auto_lyrics=args.auto_lyrics,
        output_format=args.format,
        make_zip=args.zip, keep_work=args.keep_work,
    )


def _source(parser, args, url: str | None) -> Path:
    """O arquivo de música; um link é baixado agora (o mesmo vídeo não é baixado de novo)."""
    if url is None:
        return Path(args.input)
    from ai_charter.youtube import download
    print("Baixando o áudio do vídeo...")
    try:
        return download(url, args.downloads)[0]
    except RuntimeError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> None:
    opts = parse_args(argv)
    given = argv if argv is not None else sys.argv[1:]
    view = "--view" in given
    try:
        result = run(opts)
    except (RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)
    from ai_charter import youtube
    url = next((a for a in given if youtube.is_url(a)), None)
    if url is not None:   # música de um link: a capa e, se pedido, o vídeo de fundo
        if youtube.save_cover(opts.input, result.song_dir):
            print("A miniatura do vídeo virou a capa (album.png)")
        if "--video" in given:
            print("Vídeo de fundo:")
            try:   # o vídeo fica guardado junto do áudio baixado
                youtube.background_video(url, opts.input.parent, result.song_dir, result.lead_in)
            except RuntimeError as exc:
                print(f"    o vídeo não deu certo ({exc}); a chart está pronta sem ele", file=sys.stderr)
    print(f"\nPronto! Música criada em: {result.song_dir}")
    if result.zip_path:
        print(f"Arquivo .zip: {result.zip_path}")
    print("Copie essa pasta para a pasta de músicas do Clone Hero ou do YARG e faça um rescan.")
    if view:
        from ai_charter.view import serve
        serve(result.song_dir)


if __name__ == "__main__":
    main()
