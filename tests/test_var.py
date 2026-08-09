"""
Testes de src/var.py — funções puras (sem rede), com retornos sintéticos
controlados em vez de preços reais via yfinance.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.var import (
    _z_score,
    calcular_retornos_carteira,
    calcular_retornos_simples,
    calcular_var_historico,
    calcular_var_monte_carlo,
    calcular_var_parametrico,
)


class TestCalcularRetornosSimples:
    def test_retorno_simples_10_por_cento(self):
        precos = pd.Series([100.0, 110.0, 121.0])
        retornos = calcular_retornos_simples(precos)
        assert retornos.tolist() == pytest.approx([0.10, 0.10])

    def test_precos_constantes_dao_retorno_zero(self):
        retornos = calcular_retornos_simples(pd.Series([50.0, 50.0, 50.0]))
        assert (retornos == 0).all()

    def test_queda_de_preco_da_retorno_negativo(self):
        retornos = calcular_retornos_simples(pd.Series([100.0, 90.0]))
        assert retornos.iloc[0] == pytest.approx(-0.10)


class TestCalcularRetornosCarteira:
    def test_dois_ativos_identicos_pesos_iguais_reproduz_o_mesmo_retorno(self):
        precos = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [100.0, 110.0, 121.0]})
        retorno_individual = calcular_retornos_simples(precos["A"])
        retorno_carteira = calcular_retornos_carteira(precos, {"A": 0.5, "B": 0.5})
        # ativos idênticos -> carteira não "inventa" diversificação nenhuma
        assert retorno_carteira.tolist() == pytest.approx(retorno_individual.tolist())

    def test_media_ponderada_com_ativos_diferentes(self):
        precos = pd.DataFrame({
            "A": [100.0, 110.0],   # +10%
            "B": [100.0, 90.0],    # -10%
        })
        retorno_carteira = calcular_retornos_carteira(precos, {"A": 0.5, "B": 0.5})
        assert retorno_carteira.iloc[0] == pytest.approx(0.0)  # +10%*0.5 + -10%*0.5

    def test_pesos_nao_somando_1_dispara_assert(self):
        precos = pd.DataFrame({"A": [100.0, 110.0], "B": [100.0, 90.0]})
        with pytest.raises(AssertionError):
            calcular_retornos_carteira(precos, {"A": 0.5, "B": 0.6})


class TestZScore:
    def test_95_por_cento(self):
        assert _z_score(0.95) == pytest.approx(stats.norm.ppf(0.05))
        assert _z_score(0.95) == pytest.approx(-1.6448536269514722, rel=1e-6)

    def test_50_por_cento_e_zero(self):
        assert _z_score(0.50) == pytest.approx(0.0, abs=1e-9)

    def test_maior_confianca_da_z_mais_negativo(self):
        assert _z_score(0.99) < _z_score(0.95)


@pytest.fixture
def retornos_sinteticos():
    # amostra fixa (não-normal de propósito) para testar as três abordagens
    # com um mu/sigma conhecidos e reproduzíveis.
    return pd.Series([0.01, -0.01, 0.02, -0.02, 0.0, 0.015, -0.03, 0.005, -0.005, 0.01])


class TestCalcularVarParametrico:
    def test_bate_com_formula_manual(self, retornos_sinteticos):
        confianca = 0.95
        mu = retornos_sinteticos.mean()
        sigma = retornos_sinteticos.std(ddof=1)
        z = stats.norm.ppf(1 - confianca)
        esperado = -(mu + z * sigma)

        assert calcular_var_parametrico(retornos_sinteticos, confianca) == pytest.approx(esperado)

    def test_var_e_positivo_para_retornos_com_volatilidade(self, retornos_sinteticos):
        assert calcular_var_parametrico(retornos_sinteticos, 0.95) > 0

    def test_maior_confianca_da_var_maior(self, retornos_sinteticos):
        var_95 = calcular_var_parametrico(retornos_sinteticos, 0.95)
        var_99 = calcular_var_parametrico(retornos_sinteticos, 0.99)
        assert var_99 > var_95

    def test_escala_linearmente_com_valor_exposto(self, retornos_sinteticos):
        var_1 = calcular_var_parametrico(retornos_sinteticos, 0.95, valor_exposto=1.0)
        var_1000 = calcular_var_parametrico(retornos_sinteticos, 0.95, valor_exposto=1000.0)
        assert var_1000 == pytest.approx(var_1 * 1000)

    def test_escala_por_raiz_do_tempo_no_horizonte(self, retornos_sinteticos):
        var_1d = calcular_var_parametrico(retornos_sinteticos, 0.95, horizonte_dias=1)
        var_4d = calcular_var_parametrico(retornos_sinteticos, 0.95, horizonte_dias=4)
        # sigma escala por sqrt(4)=2, mu escala por 4 -- não é simplesmente 2x,
        # mas deve ser estritamente maior que o de 1 dia para essa amostra.
        assert var_4d > var_1d


class TestCalcularVarHistorico:
    def test_bate_com_percentil_numpy(self, retornos_sinteticos):
        confianca = 0.95
        esperado = -np.percentile(retornos_sinteticos, (1 - confianca) * 100)
        assert calcular_var_historico(retornos_sinteticos, confianca) == pytest.approx(esperado)

    def test_sem_escala_no_horizonte_de_1_dia(self, retornos_sinteticos):
        var_hist = calcular_var_historico(retornos_sinteticos, 0.95, horizonte_dias=1)
        percentil = np.percentile(retornos_sinteticos, 5)
        assert var_hist == pytest.approx(-percentil)


class TestCalcularVarMonteCarlo:
    def test_deterministico_com_mesma_seed(self, retornos_sinteticos):
        v1 = calcular_var_monte_carlo(retornos_sinteticos, 0.95, n_simulacoes=10_000, seed=42)
        v2 = calcular_var_monte_carlo(retornos_sinteticos, 0.95, n_simulacoes=10_000, seed=42)
        assert v1 == v2

    def test_seeds_diferentes_dao_resultados_diferentes(self, retornos_sinteticos):
        v1 = calcular_var_monte_carlo(retornos_sinteticos, 0.95, n_simulacoes=1_000, seed=1)
        v2 = calcular_var_monte_carlo(retornos_sinteticos, 0.95, n_simulacoes=1_000, seed=2)
        assert v1 != v2

    def test_converge_para_o_parametrico_com_muitas_simulacoes(self, retornos_sinteticos):
        var_param = calcular_var_parametrico(retornos_sinteticos, 0.95)
        var_mc = calcular_var_monte_carlo(retornos_sinteticos, 0.95, n_simulacoes=500_000, seed=7)
        # mesma distribuição (Normal) do paramétrico -> deve convergir de perto
        assert var_mc == pytest.approx(var_param, rel=0.05)
