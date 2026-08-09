"""
Testes de src/dcf.py — funções puras (sem rede): projeção de FCFF, valor
terminal (Gordon), valor presente e a composição do DCF. As funções que
buscam dado do DFP (calcular_ebit, calcular_fcff, calcular_cagr_receita,
etc.) são cobertas em tests/test_integration.py.
"""

import pytest

from src.dcf import (
    calcular_enterprise_value,
    calcular_equity_value,
    projetar_fcff,
    valor_presente,
    valor_terminal_gordon,
)


class TestProjetarFcff:
    def test_crescimento_zero_repete_o_base(self):
        fluxos = projetar_fcff(fcff_base=100.0, taxa_crescimento=0.0, n_anos=3)
        assert fluxos == pytest.approx([100.0, 100.0, 100.0])

    def test_crescimento_geometrico_composto(self):
        fluxos = projetar_fcff(fcff_base=100.0, taxa_crescimento=0.10, n_anos=2)
        assert fluxos == pytest.approx([110.0, 121.0])

    def test_tamanho_da_lista_igual_a_n_anos(self):
        assert len(projetar_fcff(100.0, 0.05, n_anos=5)) == 5

    def test_nao_inclui_o_proprio_fcff_base(self):
        fluxos = projetar_fcff(fcff_base=100.0, taxa_crescimento=0.5, n_anos=1)
        assert fluxos[0] != 100.0


class TestValorTerminalGordon:
    def test_formula_gordon(self):
        # VT = FCFF_n * (1+g) / (wacc - g)
        vt = valor_terminal_gordon(fcff_ultimo_ano_explicito=100.0, wacc=0.10, g_perpetuidade=0.05)
        assert vt == pytest.approx(100.0 * 1.05 / 0.05)

    def test_g_maior_que_wacc_rejeitado(self):
        with pytest.raises(AssertionError):
            valor_terminal_gordon(100.0, wacc=0.05, g_perpetuidade=0.10)

    def test_g_igual_a_wacc_rejeitado(self):
        with pytest.raises(AssertionError):
            valor_terminal_gordon(100.0, wacc=0.05, g_perpetuidade=0.05)

    def test_g_maior_reduz_denominador_aumenta_vt(self):
        vt_g_baixo = valor_terminal_gordon(100.0, wacc=0.15, g_perpetuidade=0.02)
        vt_g_alto = valor_terminal_gordon(100.0, wacc=0.15, g_perpetuidade=0.10)
        assert vt_g_alto > vt_g_baixo


class TestValorPresente:
    def test_um_fluxo_um_periodo(self):
        vp = valor_presente([110.0], taxa_desconto=0.10)
        assert vp == pytest.approx(100.0)

    def test_fluxos_vazios_da_zero(self):
        assert valor_presente([], taxa_desconto=0.10) == 0

    def test_desconta_por_periodo_crescente(self):
        # ano 1 e ano 2 do mesmo valor nominal -> ano 2 vale menos a valor presente
        vp_ano1 = valor_presente([100.0, 0.0], taxa_desconto=0.10)
        vp_ano2 = valor_presente([0.0, 100.0], taxa_desconto=0.10)
        assert vp_ano1 > vp_ano2

    def test_taxa_zero_nao_desconta(self):
        assert valor_presente([100.0, 100.0], taxa_desconto=0.0) == pytest.approx(200.0)


class TestCalcularEnterpriseValue:
    def test_composicao_bate_com_calculo_manual(self):
        fcff_base, wacc, g_explicito, g_perp, n = 100.0, 0.12, 0.05, 0.03, 3

        resultado = calcular_enterprise_value(fcff_base, wacc, g_explicito, g_perp, n)

        fluxos_esperados = projetar_fcff(fcff_base, g_explicito, n)
        vp_explicito_esperado = valor_presente(fluxos_esperados, wacc)
        vt_esperado = valor_terminal_gordon(fluxos_esperados[-1], wacc, g_perp)
        vp_vt_esperado = vt_esperado / (1 + wacc) ** n

        assert resultado["fluxos_projetados"] == pytest.approx(fluxos_esperados)
        assert resultado["vp_fluxos_explicitos"] == pytest.approx(vp_explicito_esperado)
        assert resultado["valor_terminal"] == pytest.approx(vt_esperado)
        assert resultado["vp_valor_terminal"] == pytest.approx(vp_vt_esperado)
        assert resultado["enterprise_value"] == pytest.approx(vp_explicito_esperado + vp_vt_esperado)

    def test_enterprise_value_positivo_com_premissas_normais(self):
        resultado = calcular_enterprise_value(100.0, wacc=0.12, g_explicito=0.05, g_perpetuidade=0.03)
        assert resultado["enterprise_value"] > 0


class TestCalcularEquityValue:
    def test_sem_divida_liquida_iguala_enterprise_value(self):
        assert calcular_equity_value(enterprise_value=100.0, divida_liquida=0.0) == pytest.approx(100.0)

    def test_subtrai_divida_liquida(self):
        assert calcular_equity_value(enterprise_value=100.0, divida_liquida=30.0) == pytest.approx(70.0)

    def test_caixa_liquido_positivo_aumenta_equity_value(self):
        # dívida líquida negativa = mais caixa do que dívida onerosa
        assert calcular_equity_value(enterprise_value=100.0, divida_liquida=-20.0) == pytest.approx(120.0)
