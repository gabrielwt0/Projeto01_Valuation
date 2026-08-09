"""
Testes de integração — precisam de internet (CVM, Bacen SGS, yfinance).

Fixam ticker=JHSF3, ano=2023 e usam valores conferidos manualmente contra
os CSVs brutos da CVM durante o desenvolvimento das Fases 2-4 (ver
memória do projeto / histórico de conversa). Servem de rede de segurança
contra mudança de formato/taxonomia da CVM ou regressão nas fórmulas —
não contra `hoje` mudar (preço de mercado, WACC "ao vivo" etc. não têm
valor fixo esperado, só são checados por sanidade/tipo/sinal).
"""

import pytest

from src.data_loader import get_dfp, get_ipca, get_selic
from src.dcf import (
    calcular_cagr_receita,
    calcular_capex,
    calcular_depreciacao_amortizacao,
    calcular_divida_liquida,
    calcular_ebit,
    calcular_fcff,
    calcular_receita,
    calcular_variacao_nwc,
)
from src.wacc import custo_divida

TICKER = "JHSF3"
ANO = 2023


class TestSeriesMacro:
    def test_selic_e_taxa_diaria_nao_ja_anualizada(self):
        # 2024-01-02 é dado histórico fechado — não muda mais.
        selic = get_selic("2024-01-02", "2024-01-02")
        assert selic.iloc[0] == pytest.approx(0.043739)
        # se fosse "já anualizada" (~11-12%), o valor estaria entre 5 e 20, não < 1.
        assert 0 < selic.iloc[0] < 1

    def test_ipca_janeiro_2023(self):
        ipca = get_ipca("2023-01-01", "2023-01-31")
        assert ipca.iloc[0] == pytest.approx(0.53)


class TestGetDfp:
    def test_demonstrativo_inexistente_da_erro_claro(self):
        with pytest.raises(ValueError):
            get_dfp(TICKER, ANO, "DEMONSTRATIVO_QUE_NAO_EXISTE", "con")


class TestFcffJhsf32023:
    """Valores conferidos manualmente contra o CSV bruto da CVM (DRE/DFC_MI/BPA/BPP)."""

    def test_ebit(self):
        assert calcular_ebit(TICKER, ANO) == pytest.approx(920857.0)

    def test_depreciacao_amortizacao(self):
        assert calcular_depreciacao_amortizacao(TICKER, ANO) == pytest.approx(55256.0)

    def test_capex(self):
        # 235223 (imobilizado) + 16836 (intangível)
        assert calcular_capex(TICKER, ANO) == pytest.approx(252059.0)

    def test_variacao_nwc(self):
        # NWC_2023 = (2701869 - 318126 - 326173) - (1238472 - 265073) = 1084171
        # NWC_2022 = (3148183 - 269036 - 656655) - (770824 - 135298)  = 1586966
        assert calcular_variacao_nwc(TICKER, ANO) == pytest.approx(-502795.0)

    def test_fcff_bate_com_a_soma_dos_componentes(self):
        ebit = calcular_ebit(TICKER, ANO)
        da = calcular_depreciacao_amortizacao(TICKER, ANO)
        capex = calcular_capex(TICKER, ANO)
        delta_nwc = calcular_variacao_nwc(TICKER, ANO)
        esperado = ebit * (1 - 0.34) + da - capex - delta_nwc

        assert calcular_fcff(TICKER, ANO) == pytest.approx(esperado)
        assert calcular_fcff(TICKER, ANO) == pytest.approx(913757.62)


class TestReceitaECagr:
    def test_receita_2023(self):
        assert calcular_receita(TICKER, 2023) == pytest.approx(1593474.0)

    def test_receita_2020(self):
        assert calcular_receita(TICKER, 2020) == pytest.approx(1170550.0)

    def test_cagr_3_anos_bate_com_calculo_manual(self):
        receita_2023 = calcular_receita(TICKER, 2023)
        receita_2020 = calcular_receita(TICKER, 2020)
        esperado = (receita_2023 / receita_2020) ** (1 / 3) - 1

        assert calcular_cagr_receita(TICKER, 2023, n_anos=3) == pytest.approx(esperado)
        assert calcular_cagr_receita(TICKER, 2023, n_anos=3) == pytest.approx(0.1083, abs=0.001)

    def test_cagr_5_anos_e_bem_mais_alto_por_causa_da_base_deprimida_de_2018(self):
        # regressão de propósito: documenta por que a janela padrão é 3, não 5
        # (ver comentário de JANELA_CAGR_RECEITA_ANOS em src/dcf.py).
        cagr_3a = calcular_cagr_receita(TICKER, 2023, n_anos=3)
        cagr_5a = calcular_cagr_receita(TICKER, 2023, n_anos=5)
        assert cagr_5a > cagr_3a * 2


class TestDividaECustoDeDivida:
    def test_custo_divida_liquido(self):
        assert custo_divida(TICKER, ANO) == pytest.approx(0.0642, abs=0.001)

    def test_divida_liquida(self):
        # divida_bruta (CP+LP) = 3264668; caixa+aplicações = 318126+326173 = 644299
        assert calcular_divida_liquida(TICKER, ANO) == pytest.approx(2620369.0)
