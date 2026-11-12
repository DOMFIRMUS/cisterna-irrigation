# estilo_figuras — estilo comum das figuras dos artigos

Pacote Python com o estilo de todas as figuras dos artigos: fonte, tamanhos,
paleta, larguras de coluna, letras de painel e exportação. Galeria com todos os
elementos em `galeria/galeria.html`; antes e depois de cada projeto em
`exemplos/antes_depois/`.

## Instalação (já feita nesta máquina)

- Python 3.14 global: `pip install -e "D:\DOUTORADO\Artigos produzidos\_estilo_figuras"`.
- Ambiente da micro-CT (`10_TOMOGRAFIA_MESTRADO\.venv`, Python 3.12): arquivo
  `estilo_figuras.pth` em `site-packages` apontando para esta pasta.
- Pacotes extras nos dois: `cmcrameri` (mapas de cor), `adjustText` (rótulos),
  `pyvista` (render 3-D).
- Fonte Source Sans 3 (licença SIL OFL 1.1) instalada nas fontes do usuário do
  Windows e também incluída em `estilo_figuras/fonts/`, que o pacote registra
  sozinho; funciona em outro computador sem instalar a fonte.

Para um ambiente novo: `pip install -e ".[completo]"` nesta pasta.

## Uso

```python
import estilo_figuras as ef

ef.use()                                           # fonte, tamanhos, grade, paleta, mapas de cor
fig, axs = ef.figure("springer-2", 70, ncols=2)    # 174 x 70 mm, constrained layout
axs[0].plot(x, y, **ef.series(0, line=True), label="Calcita")   # cor 1 + marcador + linha
axs[1].scatter(phi, k, color=ef.COLORS[1], marker=ef.MARKERS[1], **ef.RING)
ef.panel_labels(axs)                               # (a), (b)
info = ef.save(fig, saida / "fig03", formats=("pdf", "png", "tiff"), journal="springer-2")
print(info["size_mm"], info["issues"])             # tamanho final e problemas encontrados
```

Sem o pacote, o arquivo de estilo também funciona sozinho:
`plt.style.use(r"...\_estilo_figuras\estilo_figuras\artigo.mplstyle")`
(nesse caso, a fonte precisa estar instalada).

## O que o estilo fixa

| Item | Valor |
|---|---|
| Fonte | Source Sans 3; matemática (`$...$`) na mesma fonte |
| Tamanhos | 8 pt números, legendas e rótulos; 8,5 pt rótulos de eixo; 9 pt títulos; 9,5 pt letras de painel (faixa das revistas: 8–12 pt) |
| Larguras (`WIDTHS_MM`) | `springer-1` 84, `springer-1.5` 129, `springer-2` 174, `elsevier-1` 90, `elsevier-1.5` 140, `elsevier-2` 190 mm |
| Altura máxima | Springer 234 mm, Elsevier 240 mm |
| Eixos e grade | só eixos esquerdo e inferior; grade fina, contínua e clara; nunca dois eixos y |
| Exportação | PDF com fonte TrueType embutida e bytes reprodutíveis; PNG 600 dpi; TIFF 600 dpi RGB, LZW |

### Paleta categórica (`COLORS`, `MARKERS`, `LINESTYLES`)

| # | Cor | Hex | Marcador |
|---|---|---|---|
| 1 | azul | `#2a78d6` | `o` |
| 2 | laranja | `#eb6834` | `s` |
| 3 | água | `#1baf7a` | `^` |
| 4 | amarelo | `#eda100` | `D` |
| 5 | magenta | `#e87ba4` | `v` |
| 6 | verde | `#008300` | `P` |
| 7 | violeta | `#6250d6` | `X` |
| 8 | vermelho | `#e34948` | `h` |

Regras:

- A cor identifica a entidade (amostra, método, grupo) e segue sempre a mesma
  entidade em todas as figuras do artigo. Atribua as posições em ordem, 1, 2, 3…
- Não existe 9ª cor: agrupe o resto em "Outros", separe em painéis ou use marcadores.
- Dispersões, mapas e painéis múltiplos: no máximo três cores (as três primeiras
  passam no teste de daltonismo para todos os pares). Acima disso, marcadores.
- Cor nunca vai sozinha: marcador, estilo de linha ou rótulo direto repete a
  identidade, para a figura sobreviver à impressão em cinza.
- Texto sempre em tinta (`INK`, `INK2`), nunca na cor da série.
- Validação: a ordem passou no validador da skill de visualização sobre papel
  branco (`node validate_palette.js ... --surface "#ffffff"`). O teste
  `test_palette_passes_validator` repete essa checagem.

### Mapas de cor (`ef.cmap(tipo, n=None)`)

| Tipo | Mapa | Para |
|---|---|---|
| `sequential` | `cmc.devon_r` | magnitude (claro → escuro, um matiz) |
| `diverging` | `cmc.vik` | sinal em torno de zero |
| `cyclic` | `cmc.romaO` | ângulos, orientação |
| `multi` | `cmc.batlow` | campos longos que pedem muitos níveis distinguíveis |
| `ct` | `gray` | imagens de tomografia |

`n` devolve a versão em classes (`ef.cmap("sequential", 7)`).

## Funções

| Função | O que faz |
|---|---|
| `use(font_size=None, grid=True)` | aplica o estilo; `grid` pode ser `True`, `"x"`, `"y"` ou `False` |
| `figure(largura, altura_mm, nrows, ncols)` | figura no tamanho da coluna |
| `series(i, line=False, open_marker=False)` | cor + marcador (+ linha) da posição `i` |
| `legend_handles(rotulos)` | entradas de legenda para categorias |
| `tint(cor, força)` | tom claro para áreas (violinos, faixas) |
| `panel_labels(axes, inside=False)` | letras (a), (b), …; `inside=True` para imagens |
| `image(ax, img)` | imagem sem grade, ticks ou moldura |
| `scalebar(ax, comprimento, "1 mm")` | barra de escala em unidades dos dados |
| `label_points(ax, x, y, rotulos)` | rótulos afastados com adjustText e linha de ligação |
| `check(fig, journal)` | lista texto fora de 8–12 pt e arte maior que a coluna |
| `fit_width(fig, journal)` | ajusta o canvas para o arquivo salvo ter a largura exata |
| `save(fig, caminho, formats, journal)` | salva PDF/PNG/TIFF, ajusta a largura e confere |
| `volume.render_mask(mascara, voxel_mm)` | render 3-D (PyVista) de um volume binário |
| `GAP`, `RING` | fresta branca entre barras; anel branco em pontos |

## Testes

```bash
python -X utf8 -m pytest -q tests
```

25 testes: fonte embutida no PDF, coerência entre `.mplstyle` e pacote,
tamanhos 8–12 pt, larguras, largura exata após o corte, ordem fixa da paleta,
validador de daltonismo, PNG/TIFF a 600 dpi, PDF reprodutível, `check`, letras
de painel, rótulos sem sobreposição, mapas de cor, barra de escala e render 3-D.

## Projetos

Os projetos com código já arquivado (Zenodo com DOI) não foram alterados; o
estilo vale para figuras novas e para os projetos ainda em edição. A situação
de cada um está na galeria (`galeria/galeria.html`).

| Projeto | Situação em 31/10/2026 |
|---|---|
| 08 Artigo 3 (revisão) | **aplicado** via `analise/estilo_revisao.py`; 7 figuras regeneradas, dados conferidos idênticos |
| 06, 07, 08 Artigos 1–2 | congelados (código v1.0.0 publicado no Zenodo) |
| 10 micro-CT | congelado por decisão do Jean (rascunho Zenodo 23840902) |
| hotspot (RSASE) | editável, não aplicado: exige refazer o pacote da RSASE |
| deep learning (QREI), paper-tank (WRM) | não aplicados: podem já estar em avaliação |

Para refazer os antes e depois: `python -X utf8 exemplos/rodar_exemplos.py`.
Eles rodam a função de figura de cada projeto duas vezes, com o estilo atual e
com o novo, e gravam tudo em `exemplos/antes_depois/`; nada é escrito nos projetos.

## Publicar o código de um artigo

Se o código de um artigo usa `estilo_figuras` e vai para o GitHub ou o Zenodo,
copie a pasta `estilo_figuras/` para dentro do repositório (com `fonts/` e a
licença OFL), para o código continuar autossuficiente.
