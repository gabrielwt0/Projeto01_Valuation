"""
visualizacao.py

Fase 5: gráficos do relatório de valuation. Funções puras — recebem dado
já calculado (por wacc.py/dcf.py/var.py) e devolvem a Figure; se um
caminho for passado, também salvam e fecham a figura (uso do
scripts/gerar_relatorio.py, que precisa dos PNGs em disco pro markdown).
Sem caminho, quem chamou (ex.: app.py/Streamlit) fica dono da figure e
decide quando fechá-la. Backend 'Agg' forçado no import porque este
projeto roda sem display (servidor/CI) — sem isso, matplotlib tenta abrir
uma janela e falha num ambiente headless.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

CORES = {
    "receita": "#4C72B0",
    "fcff_historico": "#4C72B0",
    "fcff_projetado": "#DD8452",
    "equity": "#55A868",
    "divida": "#C44E52",
    "var_parametrico": "#C44E52",
    "var_historico": "#DD8452",
    "var_monte_carlo": "#8172B2",
}


def _salvar_se_pedido(fig: plt.Figure, caminho: str | None) -> plt.Figure:
    fig.tight_layout()
    if caminho is not None:
        fig.savefig(caminho, dpi=120)
        plt.close(fig)
    return fig


def grafico_receita_historica(anos: list[int], receitas: list[float], caminho: str | None = None) -> plt.Figure:
    """
    Barras de receita líquida por ano (R$ mil, escala nativa do DFP).
    Serve para visualizar de cara o boom/queda que torna o CAGR sensível à
    janela escolhida (ver JANELA_CAGR_RECEITA_ANOS em src/dcf.py).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(a) for a in anos], receitas, color=CORES["receita"])
    ax.set_ylabel("Receita líquida (R$ mil)")
    ax.set_title("Receita líquida histórica")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    return _salvar_se_pedido(fig, caminho)


def grafico_projecao_fcff(
    anos_historico: list[int],
    fcff_historico: list[float],
    anos_projetado: list[int],
    fcff_projetado: list[float],
    caminho: str | None = None,
) -> plt.Figure:
    """
    Linha do FCFF histórico + projeção explícita, com marcador vertical
    separando os dois trechos — deixa claro que a parte projetada é
    premissa (g_explicito), não dado observado.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(anos_historico, fcff_historico, marker="o", color=CORES["fcff_historico"], label="Histórico (DFP)")
    ax.plot(
        [anos_historico[-1], *anos_projetado],
        [fcff_historico[-1], *fcff_projetado],
        marker="o",
        linestyle="--",
        color=CORES["fcff_projetado"],
        label="Projetado (g_explicito)",
    )
    ax.axvline(anos_historico[-1], color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("FCFF (R$ mil)")
    ax.set_title("FCFF: histórico e projeção")
    ax.legend()
    return _salvar_se_pedido(fig, caminho)


def grafico_estrutura_capital_wacc(
    valor_equity: float,
    divida_contabil: float,
    ke: float,
    kd_liquido: float,
    wacc: float,
    caminho: str | None = None,
) -> plt.Figure:
    """
    Barra horizontal única, empilhada, mostrando os pesos de equity/dívida
    na estrutura de capital, com Ke/Kd/WACC anotados ao lado.
    """
    total = valor_equity + divida_contabil
    peso_equity = valor_equity / total
    peso_divida = divida_contabil / total

    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.barh([0], [peso_equity], color=CORES["equity"], label=f"Equity ({peso_equity:.1%}) — Ke {ke:.1%}")
    ax.barh([0], [peso_divida], left=[peso_equity], color=CORES["divida"], label=f"Dívida ({peso_divida:.1%}) — Kd líq. {kd_liquido:.1%}")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_title(f"Estrutura de capital e WACC ({wacc:.1%})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    return _salvar_se_pedido(fig, caminho)


def grafico_sensibilidade_dcf(
    fcff_base: float,
    wacc_central: float,
    g_perpetuidade_central: float,
    calcular_preco_justo,
    caminho: str | None = None,
    n_pontos: int = 5,
    delta_wacc: float = 0.02,
    delta_g: float = 0.01,
) -> plt.Figure:
    """
    Heatmap clássico de sensibilidade do DCF: preço justo por ação variando
    WACC (linhas) x g_perpetuidade (colunas), ao redor dos valores centrais
    usados no relatório.

    calcular_preco_justo(wacc, g_perpetuidade) -> float é injetado pelo
    chamador (scripts/gerar_relatorio.py) porque só ele tem o contexto
    (ticker/ano/dívida líquida/nº de ações) para fechar o cálculo completo
    até preço por ação — este módulo só desenha.
    """
    waccs = np.linspace(wacc_central - delta_wacc, wacc_central + delta_wacc, n_pontos)
    gs = np.linspace(g_perpetuidade_central - delta_g, g_perpetuidade_central + delta_g, n_pontos)

    precos = np.array([[calcular_preco_justo(w, g) for g in gs] for w in waccs])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(precos, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n_pontos), [f"{g:.1%}" for g in gs])
    ax.set_yticks(range(n_pontos), [f"{w:.1%}" for w in waccs])
    ax.set_xlabel("g perpetuidade")
    ax.set_ylabel("WACC")
    ax.set_title("Sensibilidade do preço justo (R$/ação)")

    for i in range(n_pontos):
        for j in range(n_pontos):
            ax.text(j, i, f"{precos[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="R$/ação")
    return _salvar_se_pedido(fig, caminho)


def grafico_var_distribuicao(
    retornos,
    var_parametrico: float,
    var_historico: float,
    var_monte_carlo: float,
    confianca: float,
    caminho: str | None = None,
) -> plt.Figure:
    """
    Histograma dos retornos diários com os três VaR (paramétrico,
    histórico, Monte Carlo) marcados como linhas verticais na cauda
    esquerda (-VaR, já que VaR é reportado como perda positiva).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(retornos, bins=50, color="#8C8C8C", alpha=0.7)

    for nome, var, cor in [
        ("Paramétrico", var_parametrico, CORES["var_parametrico"]),
        ("Histórico", var_historico, CORES["var_historico"]),
        ("Monte Carlo", var_monte_carlo, CORES["var_monte_carlo"]),
    ]:
        ax.axvline(-var, color=cor, linestyle="--", label=f"VaR {nome} ({confianca:.0%}): -{var:.2%}")

    ax.set_xlabel("Retorno diário simples")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de retornos e VaR")
    ax.legend()
    return _salvar_se_pedido(fig, caminho)
