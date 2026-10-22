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
| `--auto-lyrics` | sem letra, transcreve a voz com o Whisper e salva um rascunho para revisar |
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

### Letra automática (`--auto-lyrics`)

Sem arquivo de letra, o programa pode **transcrever a voz** com o Whisper (modelo de
reconhecimento de fala de código aberto, que roda no seu computador):

```bash
pip install -e ".[lyrics]"
ai-charter musica.flac --auto-lyrics
```

Ele salva o resultado em `letra_automatica.txt` (dentro da pasta da música) e já gera o vocal
com ela. O Whisper foi treinado principalmente com fala, então **erra mais em canto**: revise o
arquivo e rode de novo com `--lyrics letra_automatica.txt`. Na primeira vez o modelo é baixado
da internet (depois funciona offline). O tamanho do modelo fica em `whisper_model` nas regras
(`tiny` é rápido e erra mais; `large-v3` é o mais preciso e o mais pesado).

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

## Partes "sujas" (sanitização)

Gravações reais têm chiado, cordas soltas soando sem querer, raspadas... Nesses trechos a
transcrição fica confusa (palhetadas faltando, notas fora do ritmo, acordes trocados). O
programa mede, em cada compasso, o **ruído de fundo** e o quanto o ritmo foge do **ritmo típico**
da faixa (aprendido nos compassos limpos). Quando os dois indicam problema, o compasso vira um
**caminho regular**: o ritmo típico com um acorde só (o mais provável entre os acordes que já
apareceram limpos). Em solos (notas simples), só remove as notas fora do grid — nunca inventa notas.
Compassos limpos nunca são mexidos. O programa avisa quais compassos simplificou; os limites ficam
em `[sanitize]` nas regras (`enabled = false` desliga).

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
├── compare.py          compara uma chart gerada com uma de referência (ai-charter-compare)
├── critic.py           aponta defeitos de jogabilidade, sem referência (ai-charter-critic)
├── view.py             visualizador e editor no navegador (ai-charter-view)
├── viewer/index.html   a página do visualizador/editor (funciona sozinha, sem instalar nada)
├── analysis/           "ouvir" o áudio
│   ├── transcribe.py   letra automática com o Whisper (--auto-lyrics)
│   ├── tempo.py        detecta as batidas e monta o mapa de tempo (segundos <-> ticks)
│   ├── pitched.py      notas de guitarra/baixo/teclado: ataque, altura, acorde, duração
│   ├── chords.py       reconhecimento de acordes (C, Am, G5...) pelo chroma
│   ├── drums.py        bateria: descobre os sons do kit e separa bumbo, caixa, tons e pratos
│   ├── vocals.py       notas cantadas (afinação com pYIN, cortes por consoante/afinação)
│   └── sections.py     seções da música (modo prática) e trecho de preview
├── charting/           transformar notas detectadas em chart jogável
│   ├── common.py       grid rítmico, escolha das notas, trechos repetidos, Star Power
│   ├── frets.py        lanes pelo contorno melódico, acordes, sustains, HOPO/strum
│   ├── chords.py       acorde de cada nota (com correção nas trocas) e formato fixo por acorde
│   ├── sanitize.py     compassos confusos (ruído + ritmo bagunçado) viram um caminho regular
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
- **Acordes**: em cada ataque, o programa compara as notas que soam com modelos de acordes
  (maiores, menores, power chords e notas simples, já contando os harmônicos dos instrumentos).
  O mesmo acorde recebe sempre o mesmo formato na chart, e a troca de acorde é corrigida
  quando o acorde anterior ainda está soando.
- **Lanes pelo contorno**: o jogo não tem como mostrar a nota exata, então a chart segue o
  *desenho* da melodia — nota mais aguda vai mais para a direita, mais grave para a esquerda,
  nota repetida fica na mesma lane.
- **Bateria**: como os conversores de bateria para MIDI, primeiro descobre **quais sons o kit
  da gravação tem** (grupos de golpes com o mesmo espectro: o bumbo, a caixa, cada tom, os pratos)
  e só depois marca quando cada um toca. Cada som recebe nome pelo que ele é: a altura nos graves
  (bumbo abaixo de ~95 Hz; um surdo de ~105 Hz que ressoa é tom, não bumbo), o chiado das esteiras
  que continua ~80 ms depois do golpe (caixa; num tom, o estalo da baqueta some) e quanto os agudos
  ainda soam 150-300 ms depois (chimbal x condução x ataque). Caixa e tom de corpo parecido são
  separados golpe a golpe pelo chiado, e um chimbal fraco no golpe de uma caixa é tratado como o
  chiado das esteiras chegando aos agudos.
  Uma peça mascarada por outra (o surdo debaixo do bumbo, o chimbal debaixo da caixa) entra
  como golpe fraco, e só fica na chart se continuar o ritmo que aquela peça está tocando ali
  (golpes firmes dela no mesmo intervalo antes e depois), como um charter faria de ouvido.
  Sons que sempre tocam juntos (bumbo + chimbal) viram as duas peças.
- **Dificuldades**: as notas recebem uma prioridade (força do ataque + importância da posição no
  compasso). Em cada dificuldade escolhemos primeiro as mais importantes, respeitando a distância
  mínima entre notas e o grid permitido.

## Comparando com uma chart de referência

Para medir a qualidade, compare a chart gerada com uma chart feita por uma pessoa (ex.: a
oficial do jogo), das mesmas faixas de áudio:

```bash
ai-charter-compare oficial/notes.mid gerada/notes.mid
```

O relatório não pontua se a cor da nota é a mesma (charts são subjetivas). Ele mede, por
instrumento e dificuldade: quantas notas da referência aparecem no mesmo momento (recall) e quantas
geradas existem na referência (precisão), acorde × nota simples, contorno (subir/descer/repetir
juntas), sustains, densidade, resultados por seção e, nos vocais, tempo, afinação, sílaba e frases.

Uma chart oficial é só UMA opinião de como chartear a música. Para julgar a jogabilidade sem
depender de referência, use o avaliador:

```bash
ai-charter-critic gerada/notes.mid
```

Ele conta acordes de 3+ notas (chart "inflada"), trocas relâmpago (um acorde de uma palhetada só
entre dois grupos estáveis), picos (a nota pula 3+ lanes e volta) e o trecho mais denso.

## Visualizador: ver a chart tocando sem abrir o jogo

```bash
ai-charter-view "output/Phoenix - Lasso"                                  # abre no navegador
ai-charter-view "output/Phoenix - Lasso" --compare "oficial/notes.mid"     # e a oficial ao lado
ai-charter musica.flac --view                                             # gera e já abre
```

A página toca o áudio (todos os stems, com mute por stem no botão **Áudio**) e mostra a pista
como no YARG: notas palhetadas, HOPOs (miolo branco), taps (miolo escuro), sustains, Star Power
(azul-claro), pratos e tons da bateria, batidas, compassos e seções. Em cima ficam a linha do tempo
(clique para pular) e os vocais (afinação e letra, com as sílabas cantadas destacadas).

- **+ Pista** abre outra pista lado a lado (outro instrumento, outra dificuldade ou a chart de
  comparação), e cada pista tem seus próprios seletores.
- **Espaço** toca/pausa; **← →** voltam/avançam 5 s; **Velocidade** 0,5× e 0,75× para trechos difíceis;
  **Pista** muda quantos segundos de notas aparecem.
- Lê `notes.mid` e `notes.chart`. Também dá para abrir `ai_charter/viewer/index.html` direto no
  navegador e usar **Abrir pasta** (ou arrastar a pasta da música para a página).
- Cada pista escolhe a vista: **Pista (YARG)**, com as notas vindo na sua direção, ou **Piano roll**,
  com o tempo da esquerda para a direita e uma linha por cor (verde embaixo), como as notas MIDI no
  FL Studio. Roda do mouse anda na música; Ctrl+roda dá zoom no piano roll.

### Editor

**✎ Editar** liga a edição de guitarra, baixo, guitarra base, teclado e bateria de um `notes.mid`, nas duas vistas:

| O quê | Como |
|---|---|
| pôr nota | clique num lugar vazio (no mesmo tick de outra nota, vira acorde) |
| mover no tempo / trocar de cor | arraste a nota (encaixa na **Grade**: 1/4 … 1/64 ou livre); ← → e ↑ ↓ |
| mudar a cor | 1–5 ou os botões coloridos; Shift põe ou tira a cor no acorde |
| sustain | arraste a borda direita no piano roll, Alt+arraste, ou Shift+← → (cada cor tem o seu) |
| HOPO / strum, tap, Star Power | H, T, S (ou os botões) na seleção |
| bateria | 1 bumbo, 2 vermelho (caixa), 3 amarelo, 4 azul, 5 verde; **C** alterna prato e tom; o bumbo é a linha de baixo do piano roll (na pista 3D, a barra) |
| apagar | botão direito apaga aquela cor; Delete apaga as notas selecionadas |
| selecionar | clique pega só a cor clicada (dá para tirar uma nota de um acorde); duplo clique pega o acorde inteiro; Shift+clique soma; Shift+arraste seleciona uma área; Ctrl+A tudo |
| copiar / colar | Ctrl+C; Ctrl+V cola no ponto em que a música está |
| desfazer / refazer / salvar | Ctrl+Z, Ctrl+Y, Ctrl+S |

Aberto pelo `ai-charter-view`, **Salvar** grava o `notes.mid` na pasta da música; na primeira vez a
original é guardada como `notes.backup.mid`. Aberta direto do arquivo, a página baixa o `notes.mid`
editado. Só as notas que o editor controla são reescritas (as cores e o HOPO/strum forçado das
dificuldades mexidas, tap e Star Power); bateria, vocais, seções e mapa de tempo ficam como estavam.
No MIDI, tap, Star Power e prato x tom da bateria (marcadores 110-112) valem para todas as
dificuldades do instrumento ao mesmo tempo.

### Pedir à IA para revisar o resto da chart

**🤖 Pedir à IA** abre uma caixa de texto para dizer o que fazer no resto da chart (ex.: "faça os
outros refrões como eu fiz no primeiro"). Sem texto, o pedido é seguir o padrão das suas edições.
Ao enviar, a página salva na pasta da música:

- `notes.mid`, com as suas edições;
- `notes.locks.json`, com os **compassos travados**: todo compasso em que você mexeu, naquele
  instrumento e dificuldade (e em todas, se mexeu em tap ou Star Power). Eles aparecem com 🔒 no
  editor, e as travas se acumulam entre um pedido e outro;
- `ai_request.json`, com o texto e o antes/depois de cada compasso editado.

Quem atende o pedido (por exemplo o Claude Code: "processa o pedido de IA da Lasso") lê a chart
e as travas e grava a revisão por um comando que **devolve à força** o conteúdo dos compassos
travados, mesmo que a revisão tenha mexido neles:

```bash
ai-charter-revise show  "output/Phoenix - Lasso"                  # a chart em texto, com [trava]
ai-charter-revise apply "output/Phoenix - Lasso" revisada.mid     # grava respeitando as travas
```

O `apply` guarda a versão anterior em `notes.before-ai.mid`, conta quantas notas desfez nos
trechos travados e arquiva o pedido como `ai_request.done.json`.

Para refazer só a bateria de uma música já gerada (por exemplo, depois de uma melhoria no
transcritor), a partir do `drums.ogg` da pasta, mantendo o mapa de tempo, o Star Power e as travas:

```bash
ai-charter-revise drums "output/Phoenix - Lasso"
```

## Testes

```bash
pip install -e ".[dev]"
pytest
```

Os testes geram duas músicas sintéticas com notas conhecidas e conferem se a chart acerta:

- **música 1** (`tests/synth.py`): melodia, baixo e bateria — ritmo, contorno das lanes,
  acordes, peças da bateria;
- **bateria** (`tests/test_drums.py`): um kit com tons e pratos, e um surdo tocando o ritmo junto
  com o bumbo, que não pode virar bumbo;
- **música 2** (`tests/song2.py`, "Estrada"): violão, guitarra (riff e solo por cima do
  violão), baixo, bateria e uma voz sintetizada cantando uma letra original — qual guitarra
  é escolhida em cada trecho, o solo nota a nota, os acordes do violão, os power chords do riff
  e o vocal sílaba por sílaba (tempo, afinação, frases). Uma versão "suja" da mesma música
  (chiado, cordas soltas, raspadas no verso 2) testa a sanitização.

## Limitações atuais e próximos passos

- **Letra automática** (`--auto-lyrics`) não foi testada com o modelo de verdade (o ambiente de
  desenvolvimento não consegue baixá-lo); erros de transcrição em canto são esperados.
- **Várias guitarras** só são separadas quando você fornece os stems; o Demucs junta todas
  as guitarras num stem só.
- **Bateria**: chimbal logo depois de um prato de ataque costuma ser mascarado por ele; chimbal
  aberto e condução podem se confundir. As regras de `[drum_detection]` mudaram: arquivos de
  regras personalizados com as chaves antigas (`kick_band`, `dominance`...) precisam ser atualizados.
- **Fórmula de compasso** assumida como 4/4.
- O arquivo é uma pasta (ou `.zip`); o formato `.sng` ainda não é gerado.
- Uma interface web (upload pelo navegador, como o autochart.eu) pode ser construída por cima do
  `pipeline.run()`.
