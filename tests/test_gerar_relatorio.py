"""
Testes de scripts/gerar_relatorio.py — só a parte pura (_montar_markdown,
que monta o texto a partir de dicts já calculados). O resto do módulo
(gerar_relatorio, _gerar_wacc/_gerar_dcf/_gerar_var/_gerar_graficos) é
puro encanamento de rede+cálculo já coberto por
tests/test_integration.py e tests/test_visualizacao.py — testar de novo
aqui seria só duplicar chamadas de rede caras sem cobrir lógica nova.
"""

import pytest

from scripts.gerar_relatorio import _montar_markdown

WACC_INFO = {
    "beta": 1.19, "r_quadrado": 0.33, "premio_risco": 0.0237,
    "rf": 0.139, "ke": 0.1672, "kd_liquido": 0.0642, "wacc": 0.1354,
}
DCF_INFO = {
    "fcff_base": 913757.62, "g_explicito": 0.1083, "g_perpetuidade": 0.0571,
    "enterprise_value": 15167262.0, "divida_liquida": 2620369.0,
    "equity_value": 12546893.0, "preco_justo": 18.51,
}
VAR_INFO = {
    0.95: {"parametrico": 0.0352, "historico": 0.0348, "monte_carlo": 0.0355},
    0.99: {"parametrico": 0.0508, "historico": 0.0481, "monte_carlo": 0.0512},
}


class TestMontarMarkdown:
    def test_titulo_com_ticker_e_ano(self):
        md = _montar_markdown("JHSF3", 2023, 10.79, WACC_INFO, DCF_INFO, VAR_INFO)
        assert "# Relatório de Valuation — JHSF3 (2023)" in md

    def test_upside_calculado_corretamente(self):
        md = _montar_markdown("JHSF3", 2023, 10.0, WACC_INFO, DCF_INFO, VAR_INFO)
        # preco_justo=18.51, preco_atual=10.0 -> upside = 85.1%
        assert "+85.1%" in md

    def test_downside_quando_preco_justo_menor_que_mercado(self):
        dcf_info_caro = {**DCF_INFO, "preco_justo": 5.0}
        md = _montar_markdown("JHSF3", 2023, 10.0, WACC_INFO, dcf_info_caro, VAR_INFO)
        assert "-50.0%" in md

    def test_imagens_apontam_para_pasta_do_ticker(self):
        md = _montar_markdown("JHSF3", 2023, 10.79, WACC_INFO, DCF_INFO, VAR_INFO)
        assert "![Estrutura de capital](JHSF3/estrutura_capital.png)" in md
        assert "![Receita histórica](JHSF3/receita_historica.png)" in md
        assert "![Projeção de FCFF](JHSF3/projecao_fcff.png)" in md
        assert "![Sensibilidade do DCF](JHSF3/sensibilidade_dcf.png)" in md
        assert "![Distribuição de retornos e VaR](JHSF3/var_distribuicao.png)" in md

    def test_imagens_usam_ticker_diferente_corretamente(self):
        md = _montar_markdown("EZTC3", 2023, 10.79, WACC_INFO, DCF_INFO, VAR_INFO)
        assert "![Estrutura de capital](EZTC3/estrutura_capital.png)" in md

    def test_tabela_de_var_tem_as_duas_confiancas(self):
        md = _montar_markdown("JHSF3", 2023, 10.79, WACC_INFO, DCF_INFO, VAR_INFO)
        assert "| 95% | 3.52% | 3.48% | 3.55% |" in md
        assert "| 99% | 5.08% | 4.81% | 5.12% |" in md

    def test_ressalva_de_fcff_base_esta_presente(self):
        # a ressalva sobre FCFF/ΔNWC lumpy (achado da Fase 5) precisa
        # continuar aparecendo em qualquer relatório gerado.
        md = _montar_markdown("JHSF3", 2023, 10.79, WACC_INFO, DCF_INFO, VAR_INFO)
        assert "ΔNWC lumpy" in md

    def test_retorna_string_nao_vazia(self):
        md = _montar_markdown("JHSF3", 2023, 10.79, WACC_INFO, DCF_INFO, VAR_INFO)
        assert isinstance(md, str)
        assert len(md) > 0
