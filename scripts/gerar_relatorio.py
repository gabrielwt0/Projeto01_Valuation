"""
Fase 5: gera o relatório de valuation (WACC + DCF + VaR) de um ticker,
com gráficos, em reports/{ticker}/.

Orquestra data_loader + wacc + dcf + var + visualizacao — nenhuma lógica
de cálculo mora aqui, só a montagem do relatório a partir do que os
outros módulos já expõem.

Uso:
    python -m scripts.gerar_relatorio [TICKER] [ANO]

    (default: JHSF3 2023 — o par usado para validar as Fases 2-4, ver
    tests/test_integration.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_loader import get_dfp, get_precos, get_selic
from src.dcf import (
    CD_CONTA_DIVIDA_CP,
    CD_CONTA_DIVIDA_LP,
    JANELA_CAGR_RECEITA_ANOS,
    JANELA_IPCA_ANOS,
    N_ANOS_EXPLICITO_PADRAO,
    calcular_cagr_receita,
    calcular_divida_liquida,
    calcular_enterprise_value,
    calcular_equity_value,
    calcular_fcff,
    calcular_ipca_medio_anual,
    calcular_receita,
    valor_por_acao,
)
from src.var import (
    calcular_retornos_simples,
    calcular_var_historico,
    calcular_var_monte_carlo,
    calcular_var_parametrico,
)
from src.visualizacao import (
    grafico_estrutura_capital_wacc,
    grafico_projecao_fcff,
    grafico_receita_historica,
    grafico_sensibilidade_dcf,
    grafico_var_distribuicao,
)
from src.wacc import (
    JANELA_HISTORICA_ANOS,
    anualizar_taxa_diaria,
    calcular_beta,
    calcular_wacc,
    custo_capital_proprio,
    custo_divida,
    premio_de_risco,
    valor_mercado_equity,
)

N_ANOS_HISTORICO_GRAFICO = 5  # quantos anos de receita/FCFF mostrar no gráfico histórico


def gerar_wacc(ticker: str, ano: int, inicio: str, fim: str) -> dict:
    beta_info = calcular_beta(ticker, inicio, fim)
    premio = premio_de_risco(inicio, fim)
    rf = anualizar_taxa_diaria(get_selic(inicio, fim).iloc[-1]) / 100
    ke = custo_capital_proprio(beta_info["beta"], rf, premio)
    kd_liquido = custo_divida(ticker, ano)
    valor_equity = valor_mercado_equity(ticker, ano)

    bpp = get_dfp(ticker, ano, "BPP", "con")
    divida_contabil = 1000 * bpp[
        bpp.CD_CONTA.isin([CD_CONTA_DIVIDA_CP, CD_CONTA_DIVIDA_LP]) & (bpp.ORDEM_EXERC == "ÚLTIMO")
    ]["VL_CONTA"].sum()

    wacc = calcular_wacc(ke, kd_liquido, valor_equity, divida_contabil)

    return {
        "beta": beta_info["beta"],
        "r_quadrado": beta_info["r_quadrado"],
        "rf": rf,
        "premio_risco": premio,
        "ke": ke,
        "kd_liquido": kd_liquido,
        "valor_equity": valor_equity,
        "divida_contabil": divida_contabil,
        "wacc": wacc,
    }


def gerar_dcf(ticker: str, ano: int, wacc: float) -> dict:
    fcff_base = calcular_fcff(ticker, ano)
    g_explicito = calcular_cagr_receita(ticker, ano)

    hoje = pd.Timestamp.today().normalize()
    inicio_ipca = (hoje - pd.DateOffset(years=JANELA_IPCA_ANOS)).strftime("%Y-%m-%d")
    g_perpetuidade = calcular_ipca_medio_anual(inicio_ipca, hoje.strftime("%Y-%m-%d"))

    dcf = calcular_enterprise_value(fcff_base, wacc, g_explicito, g_perpetuidade)
    divida_liquida = calcular_divida_liquida(ticker, ano)
    equity_value = calcular_equity_value(dcf["enterprise_value"], divida_liquida)
    preco_justo = valor_por_acao(ticker, ano, equity_value)

    return {
        "fcff_base": fcff_base,
        "g_explicito": g_explicito,
        "g_perpetuidade": g_perpetuidade,
        **dcf,
        "divida_liquida": divida_liquida,
        "equity_value": equity_value,
        "preco_justo": preco_justo,
    }


def gerar_var(ticker: str, inicio: str, fim: str) -> dict:
    precos = get_precos(ticker, inicio, fim)[ticker]
    retornos = calcular_retornos_simples(precos)

    resultado = {"retornos": retornos}
    for confianca in [0.95, 0.99]:
        resultado[confianca] = {
            "parametrico": calcular_var_parametrico(retornos, confianca),
            "historico": calcular_var_historico(retornos, confianca),
            "monte_carlo": calcular_var_monte_carlo(retornos, confianca, seed=42),
        }
    return resultado


def gerar_graficos(ticker: str, ano: int, pasta: Path, wacc_info: dict, dcf_info: dict, var_info: dict) -> None:
    pasta.mkdir(parents=True, exist_ok=True)

    anos_historico = list(range(ano - N_ANOS_HISTORICO_GRAFICO + 1, ano + 1))
    receitas = [calcular_receita(ticker, a) for a in anos_historico]
    grafico_receita_historica(anos_historico, receitas, str(pasta / "receita_historica.png"))

    anos_fcff_historico = list(range(ano - 2, ano + 1))
    fcff_historico = [calcular_fcff(ticker, a) for a in anos_fcff_historico]
    anos_projetado = list(range(ano + 1, ano + 1 + N_ANOS_EXPLICITO_PADRAO))
    grafico_projecao_fcff(
        anos_fcff_historico, fcff_historico, anos_projetado, dcf_info["fluxos_projetados"],
        str(pasta / "projecao_fcff.png"),
    )

    grafico_estrutura_capital_wacc(
        wacc_info["valor_equity"], wacc_info["divida_contabil"], wacc_info["ke"], wacc_info["kd_liquido"],
        wacc_info["wacc"], str(pasta / "estrutura_capital.png"),
    )

    def _preco_justo_para(wacc: float, g_perpetuidade: float) -> float:
        dcf = calcular_enterprise_value(dcf_info["fcff_base"], wacc, dcf_info["g_explicito"], g_perpetuidade)
        equity_value = calcular_equity_value(dcf["enterprise_value"], dcf_info["divida_liquida"])
        return valor_por_acao(ticker, ano, equity_value)

    grafico_sensibilidade_dcf(
        dcf_info["fcff_base"], wacc_info["wacc"], dcf_info["g_perpetuidade"], _preco_justo_para,
        str(pasta / "sensibilidade_dcf.png"),
    )

    grafico_var_distribuicao(
        var_info["retornos"], var_info[0.95]["parametrico"], var_info[0.95]["historico"],
        var_info[0.95]["monte_carlo"], 0.95, str(pasta / "var_distribuicao.png"),
    )


def _montar_markdown(ticker: str, ano: int, preco_atual: float, wacc_info: dict, dcf_info: dict, var_info: dict) -> str:
    upside = dcf_info["preco_justo"] / preco_atual - 1

    linhas = [
        f"# Relatório de Valuation — {ticker} ({ano})",
        "",
        f"Gerado automaticamente por `scripts/gerar_relatorio.py`. Ver `README.md` para a metodologia "
        "completa e as premissas/limitações de cada etapa (documentadas como comentários em `src/`).",
        "",
        "## 1. Custo de capital (WACC)",
        "",
        f"- Beta ({JANELA_HISTORICA_ANOS}a vs. Ibovespa): **{wacc_info['beta']:.3f}** (R² = {wacc_info['r_quadrado']:.2f})",
        f"- Prêmio de risco de mercado: **{wacc_info['premio_risco']:.2%}**",
        f"- Rf (Selic anualizada): **{wacc_info['rf']:.2%}**",
        f"- Ke (CAPM): **{wacc_info['ke']:.2%}**",
        f"- Kd líquido de IR: **{wacc_info['kd_liquido']:.2%}**",
        f"- **WACC: {wacc_info['wacc']:.2%}**",
        "",
        f"![Estrutura de capital]({ticker}/estrutura_capital.png)",
        "",
        "## 2. Receita histórica",
        "",
        f"![Receita histórica]({ticker}/receita_historica.png)",
        "",
        "## 3. FCFF e DCF (Gordon Growth)",
        "",
        f"- FCFF base ({ano}): **R$ {dcf_info['fcff_base']:,.0f} mil**",
        f"- g explícito (CAGR receita, {JANELA_CAGR_RECEITA_ANOS}a): **{dcf_info['g_explicito']:.2%}**",
        f"- g perpetuidade (IPCA médio anualizado, {JANELA_IPCA_ANOS}a): **{dcf_info['g_perpetuidade']:.2%}**",
        f"- Enterprise Value: **R$ {dcf_info['enterprise_value']:,.0f} mil**",
        f"- Dívida líquida: **R$ {dcf_info['divida_liquida']:,.0f} mil**",
        f"- Equity Value: **R$ {dcf_info['equity_value']:,.0f} mil**",
        "",
        f"![Projeção de FCFF]({ticker}/projecao_fcff.png)",
        "",
        "> O gráfico acima mostra o FCFF histórico completo, não só o ano-base: se ele foi negativo em "
        "algum dos anos recentes e só virou positivo no último ano (efeito comum de ΔNWC lumpy em "
        "incorporadoras), o DCF está projetando a partir de um único ano potencialmente anômalo — "
        "ver ressalva em `calcular_fcff` (src/dcf.py).",
        "",
        f"![Sensibilidade do DCF]({ticker}/sensibilidade_dcf.png)",
        "",
        "## 4. Preço justo vs. mercado",
        "",
        f"- **Preço justo (DCF): R$ {dcf_info['preco_justo']:.2f}**",
        f"- Preço de mercado atual: R$ {preco_atual:.2f}",
        f"- Upside/downside implícito: **{upside:+.1%}**",
        "",
        "> Premissas (`g_explicito`, `g_perpetuidade`) são um ponto de partida, não uma recomendação — "
        "ver ressalvas em `src/dcf.py` sobre sensibilidade da janela de CAGR escolhida.",
        "",
        "## 5. Valor em Risco (VaR)",
        "",
        f"![Distribuição de retornos e VaR]({ticker}/var_distribuicao.png)",
        "",
        "| Confiança | Paramétrico | Histórico | Monte Carlo |",
        "|---|---|---|---|",
    ]
    for confianca in [0.95, 0.99]:
        v = var_info[confianca]
        linhas.append(
            f"| {confianca:.0%} | {v['parametrico']:.2%} | {v['historico']:.2%} | {v['monte_carlo']:.2%} |"
        )
    linhas.append("")
    linhas.append("VaR expresso como % do valor exposto, horizonte de 1 dia.")

    return "\n".join(linhas)


def gerar_relatorio(ticker: str, ano: int, reports_dir: Path | None = None) -> Path:
    """
    reports_dir é injetável (default: reports/ na raiz do projeto) para
    permitir testar o pipeline inteiro contra um diretório temporário em
    tests/test_integration.py, sem sobrescrever os relatórios já
    commitados em reports/.
    """
    hoje = pd.Timestamp.today().normalize()
    inicio_precos = (hoje - pd.DateOffset(years=JANELA_HISTORICA_ANOS)).strftime("%Y-%m-%d")
    fim = hoje.strftime("%Y-%m-%d")

    print(f"Calculando WACC de {ticker}...")
    wacc_info = gerar_wacc(ticker, ano, inicio_precos, fim)

    print("Calculando DCF...")
    dcf_info = gerar_dcf(ticker, ano, wacc_info["wacc"])

    print("Calculando VaR...")
    inicio_var = (hoje - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    var_info = gerar_var(ticker, inicio_var, fim)

    if reports_dir is None:
        reports_dir = Path(__file__).resolve().parent.parent / "reports"
    pasta_graficos = reports_dir / ticker

    print("Gerando gráficos...")
    gerar_graficos(ticker, ano, pasta_graficos, wacc_info, dcf_info, var_info)

    preco_atual = get_precos(ticker, (hoje - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), fim)[ticker].iloc[-1]
    markdown = _montar_markdown(ticker, ano, preco_atual, wacc_info, dcf_info, var_info)

    caminho_relatorio = reports_dir / f"{ticker}.md"
    caminho_relatorio.write_text(markdown, encoding="utf-8")
    print(f"Relatório salvo em {caminho_relatorio}")
    return caminho_relatorio


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "JHSF3"
    ano = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
    gerar_relatorio(ticker, ano)
