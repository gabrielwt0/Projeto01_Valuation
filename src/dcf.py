"""
dcf.py

Fase 3: projeção de FCFF (Free Cash Flow to Firm) do zero a partir do DFP,
e valuation por DCF com valor terminal via Gordon Growth.

Fontes de dado: src/data_loader.py (DRE, DFC, BPA, BPP). Este módulo só
calcula — não busca dado bruto de fonte externa. Reaproveita constantes de
src/wacc.py (alíquota de IR, contas de dívida onerosa) em vez de duplicar.
"""

import pandas as pd

from src.data_loader import get_composicao_capital, get_dfp, get_ipca, get_precos
from src.wacc import (
    ALIQUOTA_IR_PADRAO,
    CD_CONTA_DIVIDA_CP,
    CD_CONTA_DIVIDA_LP,
)

N_ANOS_EXPLICITO_PADRAO = 5   # decisão de projeto: documentar no relatório final (Fase 7)
JANELA_CAGR_RECEITA_ANOS = 3  # decisão de projeto, documentar no relatório final (Fase 7):
                               # testado com JHSF3 — janela de 5a (2018->2023) dá CAGR de
                               # 26.8%/ano porque 2018 é base deprimida pós-crise e 2021 é
                               # pico de recuperação pós-pandemia; janela de 2a (2021->2023)
                               # dá -10.9% por comparar só contra o próprio pico. 3a (2020->2023)
                               # é o meio-termo mais defensável: +10.8%, ainda positivo mas sem
                               # herdar o boom completo nem depender de um único ano de pico.
JANELA_IPCA_ANOS = 5          # janela mais curta que JANELA_HISTORICA_ANOS (wacc.py) de
                               # propósito: IPCA de 10a atrás inclui o pico de 2015/2016,
                               # não representativo do regime de meta atual do Bacen.

CD_CONTA_RECEITA_LIQUIDA = "3.01"  # DRE: "Receita de Venda de Bens e/ou Serviços"

# Códigos de conta do DFP (taxonomia CVM), confirmados manualmente em
# JHSF3/2023, igual à ressalva já feita em wacc.py — confira se se mantêm
# para os outros tickers do projeto (EZTC3, CYRE3, LAVV3, CURY3) antes de
# usar em produção.
CD_CONTA_EBIT = "3.05"                        # DRE: "Resultado Antes do Resultado Financeiro e dos Tributos"
CD_CONTA_DEPREC_AMORT = "6.01.01.02"          # DFC_MI: "Depreciação e amortização..." (add-back no caixa operacional)
CD_CONTA_CAPEX_IMOBILIZADO = "6.02.02"        # DFC_MI: "Aquisição de bens do ativo imobilizado..."
CD_CONTA_CAPEX_INTANGIVEL = "6.02.03"         # DFC_MI: "Aquisição de bens do ativo intangível"
CD_CONTA_ATIVO_CIRCULANTE = "1.01"            # BPA
CD_CONTA_CAIXA_EQUIVALENTES = "1.01.01"       # BPA (excluído do NWC operacional)
CD_CONTA_APLICACOES_FINANCEIRAS = "1.01.02"   # BPA (excluído do NWC operacional)
CD_CONTA_PASSIVO_CIRCULANTE = "2.01"          # BPP

# D&A e Capex assumem demonstrativo de fluxo de caixa pelo método indireto
# (DFC_MI) — é o que a JHSF3 publica. Se algum ticker do projeto usar
# método direto (DFC_MD), essas contas não existirão e será preciso migrar.


def calcular_ebit(ticker: str, ano: int) -> float:
    """EBIT contábil = CD_CONTA 3.05 do DRE (ÚLTIMO exercício)."""
    dre = get_dfp(ticker, ano, "DRE", "con")
    linha = dre[(dre.CD_CONTA == CD_CONTA_EBIT) & (dre.ORDEM_EXERC == "ÚLTIMO")]
    return linha["VL_CONTA"].iloc[0]


def calcular_depreciacao_amortizacao(ticker: str, ano: int) -> float:
    """D&A do ano, via add-back no caixa operacional do DFC (método indireto)."""
    dfc = get_dfp(ticker, ano, "DFC_MI", "con")
    linha = dfc[(dfc.CD_CONTA == CD_CONTA_DEPREC_AMORT) & (dfc.ORDEM_EXERC == "ÚLTIMO")]
    return linha["VL_CONTA"].iloc[0]


def calcular_capex(ticker: str, ano: int) -> float:
    """
    Capex do ano = aquisição de imobilizado + aquisição de intangível,
    ambos na atividade de investimento do DFC (método indireto).

    As contas vêm negativas no DFC (saída de caixa) — o retorno aqui é
    positivo (magnitude do investimento), para usar diretamente na
    subtração da fórmula de FCFF.
    """
    dfc = get_dfp(ticker, ano, "DFC_MI", "con")
    linhas = dfc[
        dfc.CD_CONTA.isin([CD_CONTA_CAPEX_IMOBILIZADO, CD_CONTA_CAPEX_INTANGIVEL])
        & (dfc.ORDEM_EXERC == "ÚLTIMO")
    ]
    capex = -linhas["VL_CONTA"].sum()
    assert capex >= 0, f"Capex negativo ({capex}) — confira o sinal das contas do DFC para {ticker}/{ano}."
    return capex


def _nwc_operacional(bpa: pd.DataFrame, bpp: pd.DataFrame, ordem_exerc: str) -> float:
    """
    NWC operacional = (ativo circulante - caixa - aplicações financeiras)
                     - (passivo circulante - dívida onerosa de curto prazo)

    Caixa/aplicações financeiras e dívida onerosa são excluídos porque são
    itens de financiamento, não de operação — variações neles já aparecem
    no custo de dívida (Kd) e na estrutura de capital do WACC, não devem
    ser contadas de novo aqui.
    """
    ativo_circ = bpa[(bpa.CD_CONTA == CD_CONTA_ATIVO_CIRCULANTE) & (bpa.ORDEM_EXERC == ordem_exerc)]["VL_CONTA"].iloc[0]
    caixa = bpa[
        bpa.CD_CONTA.isin([CD_CONTA_CAIXA_EQUIVALENTES, CD_CONTA_APLICACOES_FINANCEIRAS])
        & (bpa.ORDEM_EXERC == ordem_exerc)
    ]["VL_CONTA"].sum()

    passivo_circ = bpp[(bpp.CD_CONTA == CD_CONTA_PASSIVO_CIRCULANTE) & (bpp.ORDEM_EXERC == ordem_exerc)]["VL_CONTA"].iloc[0]
    divida_cp = bpp[(bpp.CD_CONTA == CD_CONTA_DIVIDA_CP) & (bpp.ORDEM_EXERC == ordem_exerc)]["VL_CONTA"].iloc[0]

    return (ativo_circ - caixa) - (passivo_circ - divida_cp)


def calcular_variacao_nwc(ticker: str, ano: int) -> float:
    """
    ΔNWC do ano = NWC operacional do exercício ÚLTIMO menos o do PENÚLTIMO.

    BPA/BPP trazem os dois exercícios no mesmo arquivo (ORDEM_EXERC
    ÚLTIMO/PENÚLTIMO), então uma chamada por demonstrativo já basta — não
    precisa baixar o ano anterior separadamente.
    """
    bpa = get_dfp(ticker, ano, "BPA", "con")
    bpp = get_dfp(ticker, ano, "BPP", "con")

    nwc_ultimo = _nwc_operacional(bpa, bpp, "ÚLTIMO")
    nwc_penultimo = _nwc_operacional(bpa, bpp, "PENÚLTIMO")

    return nwc_ultimo - nwc_penultimo


def calcular_fcff(ticker: str, ano: int, aliquota_ir: float = ALIQUOTA_IR_PADRAO) -> float:
    """
    FCFF = EBIT * (1 - aliquota_ir) + D&A - Capex - ΔNWC

    FCFF (unlevered) porque parte do EBIT, antes do resultado financeiro —
    é o fluxo disponível para todos os provedores de capital (equity +
    dívida), consistente com descontar pelo WACC (não pelo Ke) no DCF.

    LIMITAÇÃO CONHECIDA (documentada na Fase 5, ao plotar o histórico de
    FCFF de JHSF3): o FCFF de um único ano pode ser dominado pelo ΔNWC,
    que é lumpy em incorporadoras (ciclo de banco de terrenos/lançamento).
    JHSF3 teve FCFF NEGATIVO em 2021 e 2022, virando positivo só em 2023
    — e é esse único ano (2023) que calcular_enterprise_value usa como
    base para toda a projeção explícita. Um ano-base anômalo contamina o
    DCF inteiro (efeito parecido com o de calcular_cagr_receita, mas na
    base em vez de no crescimento). Não corrigido de propósito — considerar
    no relatório final (Fase 7) usar uma média de FCFF de 2-3 anos como
    base em vez do valor de um único ano, se a lumpiness for material para
    o ticker analisado.
    """
    ebit = calcular_ebit(ticker, ano)
    da = calcular_depreciacao_amortizacao(ticker, ano)
    capex = calcular_capex(ticker, ano)
    delta_nwc = calcular_variacao_nwc(ticker, ano)

    return ebit * (1 - aliquota_ir) + da - capex - delta_nwc


def calcular_receita(ticker: str, ano: int) -> float:
    """Receita líquida do ano = CD_CONTA 3.01 do DRE (ÚLTIMO exercício)."""
    dre = get_dfp(ticker, ano, "DRE", "con")
    linha = dre[(dre.CD_CONTA == CD_CONTA_RECEITA_LIQUIDA) & (dre.ORDEM_EXERC == "ÚLTIMO")]
    return linha["VL_CONTA"].iloc[0]


def calcular_cagr_receita(ticker: str, ano_final: int, n_anos: int = JANELA_CAGR_RECEITA_ANOS) -> float:
    """
    CAGR de receita entre (ano_final - n_anos) e ano_final:
        CAGR = (receita_final / receita_inicial)^(1/n_anos) - 1

    Usado como g_explicito (crescimento do período explícito do DCF) —
    assume que a empresa mantém o ritmo histórico recente de crescimento
    de receita. Premissa simplificadora a validar manualmente por setor/
    ciclo no relatório final (Fase 7), não uma verdade absoluta — uma
    empresa cíclica pode ter CAGR distorcido por onde a janela começa/termina.
    """
    receita_final = calcular_receita(ticker, ano_final)
    receita_inicial = calcular_receita(ticker, ano_final - n_anos)
    assert receita_inicial > 0, f"receita_inicial não positiva ({receita_inicial}) — CAGR indefinido."
    return (receita_final / receita_inicial) ** (1 / n_anos) - 1


def calcular_ipca_medio_anual(start: str, end: str) -> float:
    """
    IPCA médio anualizado no período [start, end]: composição geométrica
    real das variações mensais (produto dos fatores mensais), anualizada
    pela raiz N-ésima (N = nº de meses na janela).

        ipca_anual = (prod(1 + ipca_mensal_i / 100)) ** (12 / N) - 1

    Faz a composição direta em vez de tirar a média ARITMÉTICA das taxas
    mensais e só então compor por 12 — essa segunda abordagem embute o
    viés de Jensen (média aritmética >= média geométrica) e superestima
    levemente a inflação anualizada. A diferença é pequena com IPCA
    típico (~0.01 p.p. testado numa janela de 5 anos), mas a composição
    direta é a formulação correta, não uma aproximação.

    Usado como base do g_perpetuidade — premissa de que, na perpetuidade,
    o FCFF cresce só pela inflação (crescimento real zero na perpetuidade),
    a mais conservadora possível para o valor terminal (evita superestimar
    o valor terminal, que já domina o EV em DCFs de empresa madura).
    """
    ipca_mensal = get_ipca(start, end)
    n_meses = len(ipca_mensal)
    fator_acumulado = (1 + ipca_mensal / 100).prod()
    return fator_acumulado ** (12 / n_meses) - 1


def projetar_fcff(fcff_base: float, taxa_crescimento: float, n_anos: int = N_ANOS_EXPLICITO_PADRAO) -> list[float]:
    """
    Projeção do FCFF pelo período explícito, com crescimento geométrico
    constante (taxa_crescimento) sobre o fcff_base (ano 0 = último ano
    histórico, calculado via calcular_fcff).

    Retorna uma lista de N_ANOS_EXPLICITO_PADRAO valores, do ano 1 ao N —
    não inclui o próprio fcff_base.
    """
    return [fcff_base * (1 + taxa_crescimento) ** t for t in range(1, n_anos + 1)]


def valor_terminal_gordon(fcff_ultimo_ano_explicito: float, wacc: float, g_perpetuidade: float) -> float:
    """
    Valor terminal (Gordon Growth), na data do último ano explícito:
        VT_n = FCFF_n * (1 + g) / (wacc - g)

    g_perpetuidade deve ser conservador (ex.: perto do crescimento nominal
    de longo prazo da economia) — nunca >= wacc, senão o denominador some
    ou vira negativo, o que não tem sentido econômico.
    """
    assert wacc > g_perpetuidade, (
        f"wacc ({wacc}) precisa ser maior que g_perpetuidade ({g_perpetuidade}), "
        "senão o valor terminal diverge."
    )
    fcff_perpetuidade = fcff_ultimo_ano_explicito * (1 + g_perpetuidade)
    return fcff_perpetuidade / (wacc - g_perpetuidade)


def valor_presente(fluxos: list[float], taxa_desconto: float) -> float:
    """Soma dos fluxos descontados a valor presente, período 1..N (fluxos[0] é o ano 1)."""
    return sum(fluxo / (1 + taxa_desconto) ** t for t, fluxo in enumerate(fluxos, start=1))


def calcular_enterprise_value(
    fcff_base: float,
    wacc: float,
    g_explicito: float,
    g_perpetuidade: float,
    n_anos_explicitos: int = N_ANOS_EXPLICITO_PADRAO,
) -> dict:
    """
    Monta o DCF completo a partir do FCFF base (último ano histórico):
    projeta o período explícito, calcula o valor terminal (Gordon) na
    data do último ano explícito, e traz tudo a valor presente pelo WACC.

    Retorna um dict com o detalhamento (não só o EV final), para poder
    reportar cada etapa no relatório final (Fase 7).
    """
    fluxos = projetar_fcff(fcff_base, g_explicito, n_anos_explicitos)
    vp_fluxos_explicitos = valor_presente(fluxos, wacc)

    valor_terminal = valor_terminal_gordon(fluxos[-1], wacc, g_perpetuidade)
    vp_valor_terminal = valor_terminal / (1 + wacc) ** n_anos_explicitos

    return {
        "fluxos_projetados": fluxos,
        "vp_fluxos_explicitos": vp_fluxos_explicitos,
        "valor_terminal": valor_terminal,
        "vp_valor_terminal": vp_valor_terminal,
        "enterprise_value": vp_fluxos_explicitos + vp_valor_terminal,
    }


def calcular_divida_liquida(ticker: str, ano: int) -> float:
    """
    Dívida líquida = dívida onerosa total (CP + LP, do BPP) - caixa e
    equivalentes - aplicações financeiras (do BPA), todos no exercício
    ÚLTIMO. Usada para ir de Enterprise Value a Equity Value.
    """
    bpp = get_dfp(ticker, ano, "BPP", "con")
    divida_bruta = bpp[
        bpp.CD_CONTA.isin([CD_CONTA_DIVIDA_CP, CD_CONTA_DIVIDA_LP]) & (bpp.ORDEM_EXERC == "ÚLTIMO")
    ]["VL_CONTA"].sum()

    bpa = get_dfp(ticker, ano, "BPA", "con")
    caixa_e_aplicacoes = bpa[
        bpa.CD_CONTA.isin([CD_CONTA_CAIXA_EQUIVALENTES, CD_CONTA_APLICACOES_FINANCEIRAS])
        & (bpa.ORDEM_EXERC == "ÚLTIMO")
    ]["VL_CONTA"].sum()

    return divida_bruta - caixa_e_aplicacoes


def calcular_equity_value(enterprise_value: float, divida_liquida: float) -> float:
    """Equity Value = Enterprise Value - Dívida Líquida."""
    return enterprise_value - divida_liquida


def valor_por_acao(ticker: str, ano: int, equity_value: float) -> float:
    """
    Valor justo por ação = Equity Value / nº de ações em circulação.

    equity_value é esperado em R$ mil (escala nativa do DFP, ESCALA_MOEDA
    == 'MIL' — mesma escala de EBIT/D&A/capex/NWC/dívida líquida usados
    para chegar até aqui), por isso o *1000 antes de dividir pelo nº de
    ações (contagem em unidades, não em milhares) — sem isso o resultado
    fica ~1000x menor que o preço real.

    Nº de ações vem do DFP (mesma fonte/lógica de wacc.valor_mercado_equity,
    não do yfinance) — ver docstring lá para o motivo.
    """
    composicao = get_composicao_capital(ticker, ano).sort_values("DT_REFER")
    ultima = composicao.iloc[-1]
    n_acoes = ultima["QT_ACAO_TOTAL_CAP_INTEGR"] - ultima["QT_ACAO_TOTAL_TESOURO"]
    return (equity_value * 1000) / n_acoes


if __name__ == "__main__":
    # Teste manual rápido (requer internet). Rodar como módulo a partir da
    # raiz do projeto: `python -m src.dcf` — ver a mesma ressalva em
    # wacc.py sobre rodar o arquivo direto quebrar o import `from src...`.
    from src.wacc import (
        JANELA_HISTORICA_ANOS,
        calcular_beta,
        calcular_wacc,
        custo_capital_proprio,
        custo_divida,
        premio_de_risco,
        valor_mercado_equity,
    )
    from src.data_loader import get_selic
    from src.wacc import anualizar_taxa_diaria

    ticker = "JHSF3"
    ano = 2023

    hoje = pd.Timestamp.today().normalize()
    inicio = (hoje - pd.DateOffset(years=JANELA_HISTORICA_ANOS)).strftime("%Y-%m-%d")
    fim = hoje.strftime("%Y-%m-%d")

    # WACC (Fase 2), para descontar o DCF.
    beta_info = calcular_beta(ticker, inicio, fim)
    premio = premio_de_risco(inicio, fim)
    rf = anualizar_taxa_diaria(get_selic(inicio, fim).iloc[-1]) / 100
    ke = custo_capital_proprio(beta_info["beta"], rf, premio)
    kd_liquido = custo_divida(ticker, ano)
    valor_equity_mercado = valor_mercado_equity(ticker, ano)

    bpp = get_dfp(ticker, ano, "BPP", "con")
    # Mesma correção de escala de wacc.py: VL_CONTA está em R$ mil, valor_equity_mercado em R$ cheio.
    divida_contabil = 1000 * bpp[
        bpp.CD_CONTA.isin([CD_CONTA_DIVIDA_CP, CD_CONTA_DIVIDA_LP]) & (bpp.ORDEM_EXERC == "ÚLTIMO")
    ]["VL_CONTA"].sum()

    wacc = calcular_wacc(ke, kd_liquido, valor_equity_mercado, divida_contabil)
    print("WACC:", wacc)

    # FCFF (Fase 3)
    fcff_base = calcular_fcff(ticker, ano)
    print("FCFF base (ano):", fcff_base)

    g_explicito = calcular_cagr_receita(ticker, ano)
    print(f"g_explicito (CAGR receita, {JANELA_CAGR_RECEITA_ANOS}a):", g_explicito)

    inicio_ipca = (hoje - pd.DateOffset(years=JANELA_IPCA_ANOS)).strftime("%Y-%m-%d")
    g_perpetuidade = calcular_ipca_medio_anual(inicio_ipca, fim)
    print(f"g_perpetuidade (IPCA médio anualizado, {JANELA_IPCA_ANOS}a):", g_perpetuidade)

    dcf = calcular_enterprise_value(fcff_base, wacc, g_explicito, g_perpetuidade)
    print("Enterprise Value:", dcf["enterprise_value"])

    divida_liquida = calcular_divida_liquida(ticker, ano)
    equity_value = calcular_equity_value(dcf["enterprise_value"], divida_liquida)
    print("Equity Value:", equity_value)

    preco_justo = valor_por_acao(ticker, ano, equity_value)
    preco_atual = get_precos(ticker, (hoje - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), fim)[ticker].iloc[-1]
    print("Preço justo (DCF):", preco_justo)
    print("Preço de mercado atual:", preco_atual)
