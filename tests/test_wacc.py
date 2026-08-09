"""
Testes de src/wacc.py — funções puras (sem rede). Cobrem as fórmulas
(CAPM, WACC, anualização, log-retorno), não a busca de dado (isso é
coberto em tests/test_integration.py, que precisa de internet).
"""

import numpy as np
import pandas as pd
import pytest

from src.wacc import (
    anualizar_taxa_diaria,
    calcular_retornos_diarios,
    calcular_wacc,
    custo_capital_proprio,
)


class TestAnualizarTaxaDiaria:
    def test_taxa_zero_da_zero(self):
        assert anualizar_taxa_diaria(0.0) == 0.0

    def test_round_trip_100_por_cento_ao_ano(self):
        # taxa diária que, composta por 252 dias, dá exatamente 100% a.a.
        taxa_diaria_pct = (2 ** (1 / 252) - 1) * 100
        assert anualizar_taxa_diaria(taxa_diaria_pct) == pytest.approx(100.0)

    def test_monotonica_crescente(self):
        assert anualizar_taxa_diaria(0.05) > anualizar_taxa_diaria(0.04)

    def test_aceita_serie_pandas(self):
        serie = pd.Series([0.0, 0.05, 0.1])
        resultado = anualizar_taxa_diaria(serie)
        assert isinstance(resultado, pd.Series)
        assert resultado.iloc[0] == pytest.approx(0.0)


class TestCalcularRetornosDiarios:
    def test_precos_constantes_dao_retorno_zero(self):
        precos = pd.Series([100.0, 100.0, 100.0])
        retornos = calcular_retornos_diarios(precos)
        assert (retornos == 0).all()

    def test_dobrar_preco_da_log_de_2(self):
        precos = pd.Series([100.0, 200.0])
        retornos = calcular_retornos_diarios(precos)
        assert retornos.iloc[0] == pytest.approx(np.log(2))

    def test_descarta_primeiro_nan(self):
        precos = pd.Series([100.0, 110.0, 121.0])
        retornos = calcular_retornos_diarios(precos)
        assert len(retornos) == len(precos) - 1
        assert not retornos.isna().any()


class TestCustoCapitalProprio:
    def test_beta_zero_da_rf_puro(self):
        assert custo_capital_proprio(beta=0.0, rf=0.10, premio_risco=0.05) == pytest.approx(0.10)

    def test_beta_um_soma_premio_inteiro(self):
        assert custo_capital_proprio(beta=1.0, rf=0.10, premio_risco=0.05) == pytest.approx(0.15)

    def test_beta_maior_que_um_amplifica_premio(self):
        ke_beta_1 = custo_capital_proprio(beta=1.0, rf=0.10, premio_risco=0.05)
        ke_beta_2 = custo_capital_proprio(beta=2.0, rf=0.10, premio_risco=0.05)
        assert ke_beta_2 - ke_beta_1 == pytest.approx(0.05)

    def test_rejeita_rf_em_percentual_em_vez_de_decimal(self):
        # rf=10 (dez, em vez de 0.10) deveria disparar o assert de sanidade
        with pytest.raises(AssertionError):
            custo_capital_proprio(beta=1.0, rf=10.0, premio_risco=0.05)

    def test_rejeita_premio_de_risco_implausivel(self):
        with pytest.raises(AssertionError):
            custo_capital_proprio(beta=1.0, rf=0.10, premio_risco=5.0)


class TestCalcularWacc:
    def test_totalmente_equity_da_ke_puro(self):
        wacc = calcular_wacc(ke=0.15, kd_liquido=0.05, valor_equity=100.0, divida_contabil=0.0)
        assert wacc == pytest.approx(0.15)

    def test_totalmente_divida_da_kd_puro(self):
        wacc = calcular_wacc(ke=0.15, kd_liquido=0.05, valor_equity=0.0, divida_contabil=100.0)
        assert wacc == pytest.approx(0.05)

    def test_pesos_iguais_da_media_simples(self):
        wacc = calcular_wacc(ke=0.20, kd_liquido=0.10, valor_equity=100.0, divida_contabil=100.0)
        assert wacc == pytest.approx(0.15)

    def test_pesos_proporcionais_a_estrutura_de_capital(self):
        # E=75, D=25 -> peso 75%/25%
        wacc = calcular_wacc(ke=0.20, kd_liquido=0.10, valor_equity=75.0, divida_contabil=25.0)
        assert wacc == pytest.approx(0.20 * 0.75 + 0.10 * 0.25)
