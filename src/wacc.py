"""
wacc.py

Cálculo de WACC do zero: beta via regressão, prêmio de risco, CAPM,
custo de dívida líquido de IR, e ponderação final pelos pesos de mercado.

Fontes de dado: src/data_loader.py (preços, Selic/CDI, DFP). Este módulo
só calcula — não busca dado bruto de fonte externa.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.data_loader import (
    get_cdi,
    get_composicao_capital,
    get_dfp,
    get_ibovespa,
    get_precos,
    get_selic,
)

DIAS_UTEIS_ANO = 252
JANELA_HISTORICA_ANOS = 10  # decisão de projeto: documentar no relatório final
ALIQUOTA_IR_PADRAO = 0.34

# Códigos de conta do DFP (taxonomia CVM), confirmados manualmente em
# JHSF3/2023 antes de escrever este esqueleto. Confira se se mantêm para
# os outros tickers do projeto (EZTC3, CYRE3, LAVV3, CURY3) antes de usar
# em produção — a taxonomia pode variar por empresa/setor.
CD_CONTA_DESPESA_FINANCEIRA = "3.06.02"   # DRE: "Despesas Financeiras"
CD_CONTA_DIVIDA_CP = "2.01.04"            # BPP: Empréstimos e Financiamentos (circulante)
CD_CONTA_DIVIDA_LP = "2.02.01"            # BPP: Empréstimos e Financiamentos (não circulante)


def anualizar_taxa_diaria(taxa_diaria_pct: pd.Series | float) -> pd.Series | float:
    """
    Converte uma taxa diária efetiva (em %, ex.: 0.0508) para taxa anual
    efetiva (em %), via juros compostos com DIAS_UTEIS_ANO dias úteis.

    get_selic/get_cdi (data_loader.py) retornam a taxa DIÁRIA, apesar do
    docstring de lá dizer "% a.a." — essa é a conversão que corrige isso
    no lugar certo (aqui, não no data_loader, que só busca dado bruto).
    """
    taxa_anual_pct = (1 + taxa_diaria_pct / 100) ** DIAS_UTEIS_ANO - 1
    return taxa_anual_pct * 100


def calcular_retornos_diarios(precos: pd.Series) -> pd.Series:
    """
    Retorno diário (log-retorno) a partir de uma série de preços de fechamento.
    Log-retorno escolhido por ser aditivo no tempo, o que facilita tanto a
    regressão de beta quanto o VaR paramétrico/Monte Carlo mais adiante.
    """
    return np.log(precos / precos.shift(1)).dropna()


def calcular_beta(ticker: str, start: str, end: str) -> dict:
    """
    Beta de um ativo via regressão linear dos retornos diários do ativo
    contra os retornos diários do Ibovespa (proxy de mercado).

    beta = slope da regressão simples (x = retorno_ibov, y = retorno_ativo)

    Retorna um dict com beta, r_quadrado e stderr — não só o número puro,
    para poder reportar a qualidade do ajuste no relatório final (Fase 7).
    """
    precos_ativo = get_precos(ticker, start, end)[ticker]
    precos_ibov = get_ibovespa(start, end)

    ret_ativo = calcular_retornos_diarios(precos_ativo)
    ret_ibov = calcular_retornos_diarios(precos_ibov)

    # Alinha as duas séries por data antes de regredir — evita comparar
    # retornos de dias diferentes caso os índices não batam 100%.
    df_alinhado = pd.concat([ret_ativo.rename("ativo"), ret_ibov.rename("ibov")], axis=1).dropna()

    resultado = stats.linregress(df_alinhado["ibov"], df_alinhado["ativo"])
    return {
        "beta": resultado.slope,
        "r_quadrado": resultado.rvalue ** 2,
        "stderr": resultado.stderr,
    }


def premio_de_risco(start: str, end: str) -> float:
    """
    Prêmio de risco de mercado: retorno médio anualizado do Ibovespa menos
    CDI médio anualizado, na mesma janela (start/end — use JANELA_HISTORICA_ANOS
    ao chamar esta função, não hardcode a janela aqui dentro).

    Retorno do Ibovespa: média ARITMÉTICA dos log-retornos diários, anualizada
    via exp(media_diaria * DIAS_UTEIS_ANO) - 1 (consistente com log-retorno
    sendo aditivo no tempo — NÃO usar (1+r)**N-1 aqui, essa fórmula é para
    retorno simples).
    """
    precos_ibov = get_ibovespa(start, end)
    ret_ibov = calcular_retornos_diarios(precos_ibov)
    retorno_medio_diario_ibov = ret_ibov.mean()
    retorno_anual_ibov = np.exp(retorno_medio_diario_ibov * DIAS_UTEIS_ANO) - 1

    cdi_diario = get_cdi(start, end)
    cdi_anual_medio = anualizar_taxa_diaria(cdi_diario.mean()) / 100  # confirmar unidade esperada

    return retorno_anual_ibov - cdi_anual_medio


def custo_capital_proprio(beta: float, rf: float, premio_risco: float) -> float:
    """
    CAPM: Ke = Rf + beta * premio_risco

    rf e premio_risco devem estar na mesma unidade (ambos decimais,
    ex.: 0.10 para 10%, não 10) — confira isso na hora de plugar os
    valores vindos de anualizar_taxa_diaria (que retorna em %, não decimal).

    """
    assert 0 < rf < 1, f"rf parece estar em % em vez de decimal: {rf}"
    assert -0.5 < premio_risco < 0.5, f"premio_risco fora de faixa plausível: {premio_risco}"
    return rf + beta * premio_risco

def custo_divida(ticker: str, ano: int, aliquota_ir: float = ALIQUOTA_IR_PADRAO) -> float:
    """
    Custo de dívida líquido de IR:
        Kd = (despesa_financeira / dívida_onerosa) * (1 - aliquota_ir)

    Onde buscar cada número no DFP (taxonomia CVM, confirmada em JHSF3/2023):
      - despesa financeira: get_dfp(ticker, ano, 'DRE', 'con'), filtrar
        CD_CONTA == CD_CONTA_DESPESA_FINANCEIRA e ORDEM_EXERC == 'ÚLTIMO'.
        Cuidado: filtrar por DS_CONTA == 'Despesas Financeiras' pega DOIS
        níveis da hierarquia (3.06.02 e 3.06.02.02) e duplica o valor —
        por isso o filtro tem que ser por CD_CONTA exato, não pelo texto.
      - dívida onerosa: get_dfp(ticker, ano, 'BPP', 'con'), somar as linhas
        com CD_CONTA em {CD_CONTA_DIVIDA_CP, CD_CONTA_DIVIDA_LP}, também só
        ORDEM_EXERC == 'ÚLTIMO' (o arquivo traz ano atual + anterior juntos).

    """
    dre = get_dfp(ticker, ano, 'DRE', 'con')
    despesa = dre[
                (dre.CD_CONTA == CD_CONTA_DESPESA_FINANCEIRA)
                & (dre.ORDEM_EXERC == 'ÚLTIMO')
            ]['VL_CONTA'].iloc[0]

    bpp = get_dfp(ticker, ano, 'BPP', 'con')
    divida = bpp[
                bpp.CD_CONTA.isin([CD_CONTA_DIVIDA_CP, CD_CONTA_DIVIDA_LP])
                & (bpp.ORDEM_EXERC == 'ÚLTIMO')
            ]['VL_CONTA'].sum()

    kd_bruto = abs(despesa) / divida
    return kd_bruto * (1 - aliquota_ir)


def valor_mercado_equity(ticker: str, ano: int) -> float:
    """
    Valor de mercado do patrimônio líquido (E) = preço atual x nº de ações
    em circulação.

    Nº de ações vem do DFP (dfp_cia_aberta_composicao_capital_{ano}.csv,
    via get_composicao_capital) em vez do yfinance: testado no console —
    yfinance até trouxe 'sharesOutstanding' para JHSF3.SA, mas o TODO
    original já registrava que nem sempre vem preenchido para papéis
    brasileiros, e a CVM é a fonte oficial (facilita reconciliar com o
    resto do módulo, que já é todo baseado em DFP).

    Ações em circulação = total integralizado - ações em tesouraria
    (QT_ACAO_TOTAL_CAP_INTEGR - QT_ACAO_TOTAL_TESOURO). Esse arquivo não
    tem ORDEM_EXERC (BPP/DRE têm); em vez disso, uma mesma empresa pode
    aparecer mais de uma vez no ano se mudou a data de referência (ex.:
    troca de exercício social) — por isso pega a linha de DT_REFER mais
    recente, não a primeira.

    Preço: fechamento mais recente disponível via yfinance (get_precos),
    numa janela curta dos últimos dias — não tem relação com `ano`, que
    é só o ano de referência do nº de ações no DFP.
    """
    composicao = get_composicao_capital(ticker, ano).sort_values("DT_REFER")
    ultima = composicao.iloc[-1]
    n_acoes = ultima["QT_ACAO_TOTAL_CAP_INTEGR"] - ultima["QT_ACAO_TOTAL_TESOURO"]

    hoje = pd.Timestamp.today().normalize()
    inicio = (hoje - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    precos = get_precos(ticker, inicio, hoje.strftime("%Y-%m-%d"))
    preco_atual = precos[ticker].iloc[-1]

    return preco_atual * n_acoes



def calcular_wacc(ke: float, kd_liquido: float, valor_equity: float, divida_contabil: float) -> float:
    """
    WACC = Ke * E/(E+D) + Kd_liquido * D/(E+D)

    valor_equity: valor de mercado do patrimônio líquido (E)
    divida_contabil: dívida onerosa total, contábil, do BPP (D)

    """ 
    total = valor_equity + divida_contabil
    wacc = ke * (valor_equity / total) + kd_liquido * (divida_contabil / total)
    return wacc


if __name__ == "__main__":
    # Teste manual rápido (requer internet). Rodar como módulo a partir da
    # raiz do projeto: `python -m src.wacc` — rodar o arquivo direto
    # (`python src/wacc.py`) quebra o import `from src.data_loader import ...`
    # porque a raiz do projeto não entra no sys.path nesse modo.
    ticker = "JHSF3"
    ano = 2023

    hoje = pd.Timestamp.today().normalize()
    inicio = (hoje - pd.DateOffset(years=JANELA_HISTORICA_ANOS)).strftime("%Y-%m-%d")
    fim = hoje.strftime("%Y-%m-%d")

    beta_info = calcular_beta(ticker, inicio, fim)
    print("beta:", beta_info)

    premio = premio_de_risco(inicio, fim)
    print("prêmio de risco:", premio)

    rf = anualizar_taxa_diaria(get_selic(inicio, fim).iloc[-1]) / 100
    ke = custo_capital_proprio(beta_info["beta"], rf, premio)
    print("Ke (CAPM):", ke)

    kd_liquido = custo_divida(ticker, ano)
    print("Kd líquido:", kd_liquido)

    valor_equity = valor_mercado_equity(ticker, ano)
    print("valor de mercado do equity:", valor_equity)

    bpp = get_dfp(ticker, ano, "BPP", "con")
    divida_contabil = bpp[
        bpp.CD_CONTA.isin([CD_CONTA_DIVIDA_CP, CD_CONTA_DIVIDA_LP])
        & (bpp.ORDEM_EXERC == "ÚLTIMO")
    ]["VL_CONTA"].sum()

    wacc = calcular_wacc(ke, kd_liquido, valor_equity, divida_contabil)
    print("WACC:", wacc)
