"""
Testes de integração — precisam de internet (CVM, Bacen SGS, yfinance).

Fixam ticker=JHSF3, ano=2023 e usam valores conferidos manualmente contra
os CSVs brutos da CVM durante o desenvolvimento das Fases 2-4 (ver
memória do projeto / histórico de conversa). Servem de rede de segurança
contra mudança de formato/taxonomia da CVM ou regressão nas fórmulas —
não contra `hoje` mudar (preço de mercado, WACC "ao vivo" etc. não têm
valor fixo esperado, só são checados por sanidade/tipo/sinal).
"""

import pandas as pd
import pytest

from src.data_loader import get_carteira_precos, get_composicao_capital, get_dfp, get_ibovespa, get_ipca, get_precos, get_selic
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
from src.wacc import JANELA_HISTORICA_ANOS, calcular_beta, custo_divida, premio_de_risco, valor_mercado_equity

pytestmark = pytest.mark.integration

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


class TestBetaEPremioDeRisco:
    """
    calcular_beta/premio_de_risco/valor_mercado_equity dependem do preço
    de mercado "de hoje" — não têm valor fixo esperado. Testados por
    sanidade de tipo/faixa/sinal, não por igualdade a um número travado
    (ao contrário de TestFcffJhsf32023, que só usa dado histórico fechado).
    """

    @pytest.fixture
    def janela(self):
        # só monta duas strings de data (sem rede) — não precisa de scope="class"
        hoje = pd.Timestamp.today().normalize()
        inicio = (hoje - pd.DateOffset(years=JANELA_HISTORICA_ANOS)).strftime("%Y-%m-%d")
        return inicio, hoje.strftime("%Y-%m-%d")

    def test_beta_tem_as_chaves_esperadas_e_r2_em_faixa_valida(self, janela):
        inicio, fim = janela
        resultado = calcular_beta(TICKER, inicio, fim)
        assert set(resultado) == {"beta", "r_quadrado", "stderr"}
        assert 0 <= resultado["r_quadrado"] <= 1
        assert -5 < resultado["beta"] < 5  # faixa plausível p/ uma ação individual

    def test_premio_de_risco_e_float_em_faixa_plausivel(self, janela):
        inicio, fim = janela
        premio = premio_de_risco(inicio, fim)
        assert isinstance(premio, float)
        assert -0.5 < premio < 0.5

    def test_valor_mercado_equity_e_positivo(self):
        assert valor_mercado_equity(TICKER, ANO) > 0


class TestPrecosECarteira:
    @pytest.fixture
    def janela_curta(self):
        hoje = pd.Timestamp.today().normalize()
        inicio = (hoje - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        return inicio, hoje.strftime("%Y-%m-%d")

    def test_get_precos_traz_coluna_do_ticker(self, janela_curta):
        inicio, fim = janela_curta
        precos = get_precos(TICKER, inicio, fim)
        assert TICKER in precos.columns
        assert (precos[TICKER] > 0).all()

    def test_get_ibovespa_traz_serie_positiva(self, janela_curta):
        inicio, fim = janela_curta
        ibov = get_ibovespa(inicio, fim)
        assert ibov.name == "IBOV"
        assert (ibov > 0).all()

    def test_get_carteira_precos_alinha_ativos_por_data(self, janela_curta):
        inicio, fim = janela_curta
        carteira = get_carteira_precos([TICKER, "EZTC3"], inicio, fim)
        assert list(carteira.columns) == [TICKER, "EZTC3"]
        assert not carteira.isna().any().any()  # dropna já deveria ter alinhado tudo

    def test_get_composicao_capital_traz_acoes_positivas(self):
        composicao = get_composicao_capital(TICKER, ANO)
        assert (composicao["QT_ACAO_TOTAL_CAP_INTEGR"] > 0).all()


class TestGerarRelatorioEndToEnd:
    """
    Teste de fiação: o pipeline inteiro (scripts/gerar_relatorio.py) só
    era exercitado manualmente até aqui — _montar_markdown é testado à
    parte (tests/test_gerar_relatorio.py) com dicts sintéticos, mas nada
    verificava que gerar_wacc/gerar_dcf/gerar_var/gerar_graficos
    produzem dicts com as chaves que _montar_markdown/gerar_graficos
    esperam. reports_dir injetável (ver gerar_relatorio()) evita
    sobrescrever reports/JHSF3.md, que já está commitado.
    """

    def test_pipeline_completo_gera_markdown_e_graficos(self, tmp_path):
        from scripts.gerar_relatorio import gerar_relatorio

        caminho = gerar_relatorio(TICKER, ANO, reports_dir=tmp_path)

        assert caminho == tmp_path / f"{TICKER}.md"
        assert caminho.exists()

        conteudo = caminho.read_text(encoding="utf-8")
        assert f"# Relatório de Valuation — {TICKER} ({ANO})" in conteudo
        assert "**WACC:" in conteudo

        pasta_graficos = tmp_path / TICKER
        for nome in ["receita_historica.png", "projecao_fcff.png", "estrutura_capital.png",
                     "sensibilidade_dcf.png", "var_distribuicao.png"]:
            arquivo = pasta_graficos / nome
            assert arquivo.exists() and arquivo.stat().st_size > 0
