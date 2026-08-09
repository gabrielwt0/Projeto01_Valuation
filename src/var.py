"""
var.py

Fase 4: Valor em Risco (VaR) do zero — paramétrico, histórico e Monte
Carlo — para um ativo ou uma carteira de ações brasileiras.

Fontes de dado: src/data_loader.py (preços). Este módulo só calcula — não
busca dado bruto de fonte externa.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.data_loader import get_carteira_precos, get_precos

CONFIANCA_PADRAO = 0.95
HORIZONTE_DIAS_PADRAO = 1
N_SIMULACOES_MONTE_CARLO_PADRAO = 100_000


def calcular_retornos_simples(precos: pd.Series) -> pd.Series:
    """
    Retorno diário SIMPLES (não log-retorno, como em wacc.calcular_retornos_diarios)
    a partir de uma série de preços de fechamento.

    Escolhido simples (não log) porque o retorno de uma carteira é
    EXATAMENTE a média ponderada dos retornos simples de cada ativo — para
    log-retorno isso só vale como aproximação de primeira ordem. Como o
    VaR aqui sempre acaba expresso em % de perda do valor exposto, essa
    exatidão pesa mais do que a aditividade no tempo que fez log-retorno
    ser a escolha certa para beta/CAPM em wacc.py.
    """
    return (precos / precos.shift(1) - 1).dropna()


def calcular_retornos_carteira(precos_carteira: pd.DataFrame, pesos: dict[str, float]) -> pd.Series:
    """
    Retorno diário simples da carteira = soma ponderada dos retornos
    simples de cada ativo. `pesos` deve cobrir as mesmas colunas de
    precos_carteira (ver data_loader.get_carteira_precos) e somar 1.0.
    """
    soma_pesos = sum(pesos.values())
    assert abs(soma_pesos - 1.0) < 1e-6, f"Pesos da carteira somam {soma_pesos}, deveriam somar 1.0."

    retornos = pd.DataFrame({ticker: calcular_retornos_simples(precos_carteira[ticker]) for ticker in pesos})
    return retornos.dropna().mul(pd.Series(pesos)).sum(axis=1)


def _z_score(confianca: float) -> float:
    """z da normal padrão correspondente ao percentil de cauda esquerda (1 - confiança)."""
    return stats.norm.ppf(1 - confianca)


def calcular_var_parametrico(
    retornos: pd.Series,
    confianca: float = CONFIANCA_PADRAO,
    horizonte_dias: int = HORIZONTE_DIAS_PADRAO,
    valor_exposto: float = 1.0,
) -> float:
    """
    VaR paramétrico (variância-covariância): assume retornos ~ Normal(mu, sigma²).

        VaR = -(mu_h + z * sigma_h) * valor_exposto

    mu/sigma diários escalados para o horizonte pela regra da raiz do
    tempo (mu_h = mu*h, sigma_h = sigma*sqrt(h)) — assume retornos i.i.d.,
    o que ignora autocorrelação e clustering de volatilidade (limitação
    conhecida do modelo, documentar no relatório final, Fase 7).

    Retorna a perda no pior caso como número POSITIVO (convenção usual de
    VaR — quanto maior, pior).
    """
    mu = retornos.mean()
    sigma = retornos.std(ddof=1)
    mu_h = mu * horizonte_dias
    sigma_h = sigma * np.sqrt(horizonte_dias)

    z = _z_score(confianca)
    var_pct = -(mu_h + z * sigma_h)
    return var_pct * valor_exposto


def calcular_var_historico(
    retornos: pd.Series,
    confianca: float = CONFIANCA_PADRAO,
    horizonte_dias: int = HORIZONTE_DIAS_PADRAO,
    valor_exposto: float = 1.0,
) -> float:
    """
    VaR histórico: percentil empírico dos retornos passados, sem assumir
    nenhuma distribuição (ao contrário do paramétrico) — captura caudas
    gordas/assimetria já presentes na amostra.

    O percentil de 1 dia é escalado para o horizonte pela raiz do tempo,
    mesma simplificação (e mesma limitação) do VaR paramétrico — a amostra
    não tem retornos de "horizonte_dias" dias em quantidade suficiente
    para tirar o percentil diretamente sem descartar a maior parte dela.
    """
    percentil = np.percentile(retornos, (1 - confianca) * 100)
    var_pct = -percentil * np.sqrt(horizonte_dias)
    return var_pct * valor_exposto


def calcular_var_monte_carlo(
    retornos: pd.Series,
    confianca: float = CONFIANCA_PADRAO,
    horizonte_dias: int = HORIZONTE_DIAS_PADRAO,
    valor_exposto: float = 1.0,
    n_simulacoes: int = N_SIMULACOES_MONTE_CARLO_PADRAO,
    seed: int | None = None,
) -> float:
    """
    VaR Monte Carlo: simula n_simulacoes cenários de retorno ~ Normal(mu_h, sigma_h²)
    (mesmos mu/sigma escalados do paramétrico) e tira o percentil empírico
    dos cenários simulados.

    Com retornos normais, converge para o mesmo valor do VaR paramétrico
    conforme n_simulacoes cresce — a vantagem real do Monte Carlo aparece
    ao trocar a distribuição simulada (ex.: t-Student p/ caudas gordas, ou
    uma cópula para dependência não-linear entre ativos de uma carteira)
    por algo mais realista que a Normal usada aqui como baseline.
    Documentar essa limitação/próximo passo no relatório final (Fase 7).
    """
    mu = retornos.mean()
    sigma = retornos.std(ddof=1)
    mu_h = mu * horizonte_dias
    sigma_h = sigma * np.sqrt(horizonte_dias)

    rng = np.random.default_rng(seed)
    simulados = rng.normal(mu_h, sigma_h, n_simulacoes)

    percentil = np.percentile(simulados, (1 - confianca) * 100)
    var_pct = -percentil
    return var_pct * valor_exposto


if __name__ == "__main__":
    # Teste manual rápido (requer internet). Rodar como módulo a partir da
    # raiz do projeto: `python -m src.var` — mesma ressalva de wacc.py/dcf.py
    # sobre rodar o arquivo direto quebrar o import `from src...`.
    ticker = "JHSF3"
    hoje = pd.Timestamp.today().normalize()
    inicio = (hoje - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    fim = hoje.strftime("%Y-%m-%d")

    precos = get_precos(ticker, inicio, fim)[ticker]
    retornos = calcular_retornos_simples(precos)

    valor_exposto = 100_000.0  # R$ 100 mil de exposição, só para ilustrar em R$

    for confianca in [0.95, 0.99]:
        var_param = calcular_var_parametrico(retornos, confianca, valor_exposto=valor_exposto)
        var_hist = calcular_var_historico(retornos, confianca, valor_exposto=valor_exposto)
        var_mc = calcular_var_monte_carlo(retornos, confianca, valor_exposto=valor_exposto, seed=42)
        print(f"--- {ticker}, confiança {confianca:.0%}, 1 dia, R$ {valor_exposto:,.0f} exposto ---")
        print(f"VaR paramétrico: R$ {var_param:,.2f}")
        print(f"VaR histórico:   R$ {var_hist:,.2f}")
        print(f"VaR Monte Carlo: R$ {var_mc:,.2f}")

    # Exemplo de carteira (2 ativos, pesos iguais)
    tickers = ["JHSF3", "EZTC3"]
    precos_carteira = get_carteira_precos(tickers, inicio, fim)
    pesos = {t: 1 / len(tickers) for t in tickers}
    retornos_carteira = calcular_retornos_carteira(precos_carteira, pesos)

    var_param_carteira = calcular_var_parametrico(retornos_carteira, 0.95, valor_exposto=valor_exposto)
    print(f"\n--- Carteira {tickers} (pesos iguais), confiança 95%, 1 dia ---")
    print(f"VaR paramétrico: R$ {var_param_carteira:,.2f}")
