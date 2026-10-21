# ai-charter

Ferramenta para criar charts de **Clone Hero** e **YARG** automaticamente, nos moldes do
[autochart.eu](https://autochart.eu/).

Você entrega um arquivo de música (MP3 de alta qualidade ou FLAC) e o programa:

1. **separa os instrumentos** (stems) com IA: guitarra, baixo, bateria, teclado, voz e o resto;
2. **detecta o andamento** (BPM) e monta o mapa de tempo;
3. **transcreve as notas** de cada instrumento tocável (ataques, altura, acordes, duração)
   e da **voz** (afinação de cada sílaba);
4. **gera as charts** em **Expert, Hard, Medium e Easy**, seguindo regras configuráveis;
5. **exporta uma pasta** pronta para o jogo: `notes.mid`, `song.ini`, `album.png` e
   os áudios separados (`song.ogg`, `guitar.ogg`, `bass.ogg`, `drums.ogg`, `keys.ogg`, `vocals.ogg`).

Instrumentos gerados: **guitarra** (5 lanes), **baixo** (5 lanes), **teclado** (5 lanes),
**bateria** (Pro Drums: bumbo + 4 lanes, com pratos marcados) e **vocal** (com afinação e a
letra dividida em sílabas, como no YARG/Rock Band). Cada chart vem com frases de
**Star Power**, viradas de ativação na bateria e **seções** para o modo prática.
Trechos repetidos da música (riffs, refrões) recebem exatamente a mesma chart.

---

## Instalação

Pré-requisitos: **Python 3.11+** e **ffmpeg** instalado (no PATH).

```bash
git clone https://github.com/tucabrum/ai-charter.git
cd ai-charter
python -m venv .venv
source .venv/bin/activate          # no Windows: .venv\Scripts\activate
pip install -e ".[separation]"     # instala tudo, inclusive a IA de separação (Demucs)
```

> Sem placa de vídeo NVIDIA o Demucs roda na CPU — funciona, só demora alguns minutos por música.
> Para testar sem a IA, instale só `pip install -e .` e use `--separator none`.

## Uso

```bash
ai-charter "Minha Música.flac"
```

A pasta da música aparece em `output/Artista - Título/`. Copie-a para a pasta de músicas
do Clone Hero ou do YARG e faça um *rescan* de músicas no jogo.

Opções úteis:

| Opção | O que faz |
|---|---|
| `-o PASTA` | onde criar a música (padrão: `./output`) |
| `--title`, `--artist` | nome e artista (padrão: lidos das tags do arquivo) |
| `--instruments guitar,drums` | gera só alguns instrumentos |
| `--separator demucs` | separa os stems com IA (padrão) |
| `--separator folder --stems PASTA` | usa stems que você já tem (`guitar.wav`, `bass.wav`, `drums.wav`, `piano.wav`, `vocals.wav`, `other.wav`...) |
| `--separator none` | sem IA, só filtros simples (qualidade bem menor; bom para testes) |
| `--lyrics letra.txt` | letra para a faixa de vocal (padrão: um `.txt` com o mesmo nome da música) |
| `--format chart` | gera `notes.chart` em vez de `notes.mid` (o `.chart` não tem vocal) |
| `--tempo constant` / `variable` | força BPM fixo ou variável (padrão: decide sozinho) |
| `--rules minhas_regras.toml` | usa suas próprias regras de charting |
| `--zip` | também cria um `.zip` da pasta |

## Vocais

A faixa de vocal precisa da **letra** num arquivo de texto, **uma linha por frase cantada**:

```text
No caminho da manhã
o sol desperta devagar
```

O programa divide as palavras em sílabas (português por padrão; `language = "en"` nas regras
para inglês), detecta as notas cantadas e coloca cada sílaba na sua nota. No jogo, cada
sílaba no meio de uma palavra aparece com hífen (`ca-` `mi-` `nho`). Se alguma palavra for
dividida errado, escreva-a já com hífens na letra (ex.: `sa-u-da-de`).

## Várias guitarras (violão + guitarra solo)

O jogo só tem uma faixa de guitarra. Se a música tem mais de uma (por exemplo, `violao.wav`
e `guitarra.wav` com `--separator folder`), o programa escolhe **compasso a compasso** qual
vira a chart: a que está "na frente" (mais alta, tocando notas simples, mais aguda) vence,
então um solo prevalece sobre o violão de base. Para não ficar pulando de uma para outra,
trocar tem um custo e, depois de uma troca, a faixa escolhida fica pelo menos 2 compassos —
uma pausa curta no meio do solo não devolve a chart para o violão. Tudo isso é ajustável em
`[guitar_selection]` nas regras.

> Com a separação por IA (Demucs) só existe um stem de guitarra, então essa escolha vale
> quando você fornece as guitarras separadas (modo `folder`).

## Personalizando as regras

Todas as decisões de charting ficam em [`ai_charter/default_rules.toml`](ai_charter/default_rules.toml),
comentado linha a linha. Para mudar algo, crie um arquivo só com o que quer alterar:

```toml
# minhas_regras.toml - Easy com 4 cores e Hard sem notas fora do grid
[frets.easy]
lanes = 4

[frets.hard]
allow_offgrid = false
```

e rode `ai-charter musica.flac --rules minhas_regras.toml`.

Principais regras por dificuldade (guitarra/baixo/teclado):

| | Expert | Hard | Medium | Easy |
|---|---|---|---|---|
| Lanes (cores) | 5 | 5 | 4 (sem laranja) | 3 (verde, vermelho, amarelo) |
| Notas por acorde | até 3 | até 2 | até 2 | 1 |
| Menor distância entre notas | semicolcheia | colcheia | colcheia | semínima |
| HOPOs automáticos | sim | sim | não (tudo palhetado) | não |
| Notas a menos que a dificuldade acima | — | ≥ 10% | ≥ 15% | ≥ 25% |

## Como funciona (por dentro)

```
ai_charter/
├── cli.py              linha de comando (lê as opções e chama o pipeline)
├── pipeline.py         junta todas as etapas, do áudio até a pasta final
├── config.py           lê as regras do arquivo .toml
├── default_rules.toml  regras padrão de charting
├── separation.py       separação de stems (Demucs, pasta ou filtros simples)
├── analysis/           "ouvir" o áudio
│   ├── tempo.py        detecta as batidas e monta o mapa de tempo (segundos <-> ticks)
│   ├── pitched.py      notas de guitarra/baixo/teclado: ataque, altura, acorde, duração
│   ├── drums.py        bumbo, caixa, chimbal e crash por faixas de frequência
│   ├── vocals.py       notas cantadas (afinação com pYIN, cortes por consoante/afinação)
│   └── sections.py     seções da música (modo prática) e trecho de preview
├── charting/           transformar notas detectadas em chart jogável
│   ├── common.py       grid rítmico, escolha das notas, trechos repetidos, Star Power
│   ├── frets.py        lanes pelo contorno melódico, acordes, sustains, HOPO/strum
│   ├── guitar_select.py qual guitarra vira a chart em cada compasso
│   ├── drums.py        bateria por dificuldade (limite de mãos, pratos Pro Drums)
│   ├── lyrics.py       letra -> sílabas (português e inglês)
│   └── vocals.py       notas da voz no grid, frases e alinhamento da letra
└── export/             escrever os arquivos do jogo
    ├── midi_writer.py  notes.mid (padrão Rock Band, com vocal)
    ├── chart_writer.py notes.chart
    ├── song_ini.py     song.ini (inclui a "nota" de dificuldade 0-6 de cada instrumento)
    └── audio.py        conversão para .ogg, tags e capa (via ffmpeg)
```

Ideias principais:

- **Ticks**: o `.chart` mede o tempo em *ticks* (192 por batida). Detectamos a batida da música e
  convertemos o tempo de cada nota (em segundos) para ticks, encaixando no grid
  (semicolcheias, ou tercinas quando fizer mais sentido).
- **Lanes pelo contorno**: o jogo não tem como mostrar a nota exata, então a chart segue o
  *desenho* da melodia — nota mais aguda vai mais para a direita, mais grave para a esquerda,
  nota repetida fica na mesma lane.
- **Dificuldades**: as notas recebem uma prioridade (força do ataque + importância da posição no
  compasso). Em cada dificuldade escolhemos primeiro as mais importantes, respeitando a distância
  mínima entre notas e o grid permitido.

## Testes

```bash
pip install -e ".[dev]"
pytest
```

Os testes geram duas músicas sintéticas com notas conhecidas e conferem se a chart acerta:

- **música 1** (`tests/synth.py`): melodia, baixo e bateria — ritmo, contorno das lanes,
  acordes, peças da bateria;
- **música 2** (`tests/song2.py`, "Estrada"): violão, guitarra (riff e solo por cima do
  violão), baixo, bateria e uma voz sintetizada cantando uma letra original — qual guitarra
  é escolhida em cada trecho, o solo nota a nota, e o vocal sílaba por sílaba (tempo,
  afinação, frases).

## Limitações atuais e próximos passos

- **Vocais** precisam da letra em texto; ainda não há transcrição automática da letra
  (um modelo como o Whisper poderia fazer isso no futuro).
- **Várias guitarras** só são separadas quando você fornece os stems; o Demucs junta todas
  as guitarras num stem só.
- **Bateria**: tons (azul/verde sem prato) e prato de condução ainda não são detectados separadamente.
- **Fórmula de compasso** assumida como 4/4.
- O arquivo é uma pasta (ou `.zip`); o formato `.sng` ainda não é gerado.
- Uma interface web (upload pelo navegador, como o autochart.eu) pode ser construída por cima do
  `pipeline.run()`.
