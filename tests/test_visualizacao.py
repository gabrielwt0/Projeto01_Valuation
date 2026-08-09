"""
Testes de src/visualizacao.py — sem rede, dado sintético. Não valida o
conteúdo visual do PNG (fora do escopo de um teste automatizado), só que
cada função roda sem erro e produz um arquivo de imagem não-vazio no
caminho pedido — regressão contra erro de sintaxe do matplotlib (eixo
errado, argumento inválido, etc.), não contra "o gráfico ficou bonito".
"""

import pandas as pd
import pytest

from src.visualizacao import (
    grafico_estrutura_capital_wacc,
    grafico_projecao_fcff,
    grafico_receita_historica,
    grafico_sensibilidade_dcf,
    grafico_var_distribuicao,
)


def _e_um_png_nao_vazio(caminho) -> bool:
    return caminho.exists() and caminho.stat().st_size > 0


class TestGraficoReceitaHistorica:
    def test_gera_arquivo(self, tmp_path):
        caminho = tmp_path / "receita.png"
        grafico_receita_historica([2020, 2021, 2022], [100.0, 200.0, 150.0], str(caminho))
        assert _e_um_png_nao_vazio(caminho)


class TestGraficoProjecaoFcff:
    def test_gera_arquivo(self, tmp_path):
        caminho = tmp_path / "fcff.png"
        grafico_projecao_fcff(
            anos_historico=[2021, 2022, 2023],
            fcff_historico=[-100.0, -50.0, 100.0],
            anos_projetado=[2024, 2025],
            fcff_projetado=[110.0, 121.0],
            caminho=str(caminho),
        )
        assert _e_um_png_nao_vazio(caminho)


class TestGraficoEstruturaCapitalWacc:
    def test_gera_arquivo(self, tmp_path):
        caminho = tmp_path / "estrutura.png"
        grafico_estrutura_capital_wacc(
            valor_equity=700.0, divida_contabil=300.0, ke=0.15, kd_liquido=0.06, wacc=0.12,
            caminho=str(caminho),
        )
        assert _e_um_png_nao_vazio(caminho)


class TestGraficoSensibilidadeDcf:
    def test_gera_arquivo_e_chama_callback_n_vezes(self, tmp_path):
        caminho = tmp_path / "sensibilidade.png"
        chamadas = []

        def preco_justo_fake(wacc, g):
            chamadas.append((wacc, g))
            return 10.0 + wacc - g

        grafico_sensibilidade_dcf(
            fcff_base=100.0, wacc_central=0.12, g_perpetuidade_central=0.05,
            calcular_preco_justo=preco_justo_fake, caminho=str(caminho), n_pontos=3,
        )

        assert _e_um_png_nao_vazio(caminho)
        assert len(chamadas) == 3 * 3  # grid n_pontos x n_pontos

    def test_grid_cobre_a_faixa_pedida(self, tmp_path):
        caminho = tmp_path / "sensibilidade2.png"
        waccs_vistos = set()

        def preco_justo_fake(wacc, g):
            waccs_vistos.add(round(wacc, 4))
            return 10.0

        grafico_sensibilidade_dcf(
            fcff_base=100.0, wacc_central=0.10, g_perpetuidade_central=0.03,
            calcular_preco_justo=preco_justo_fake, caminho=str(caminho),
            n_pontos=5, delta_wacc=0.02,
        )
        assert min(waccs_vistos) == pytest.approx(0.08)
        assert max(waccs_vistos) == pytest.approx(0.12)


class TestGraficoVarDistribuicao:
    def test_gera_arquivo(self, tmp_path):
        caminho = tmp_path / "var.png"
        retornos = pd.Series([0.01, -0.02, 0.015, -0.01, 0.0, 0.02, -0.03])
        grafico_var_distribuicao(
            retornos, var_parametrico=0.03, var_historico=0.028, var_monte_carlo=0.031,
            confianca=0.95, caminho=str(caminho),
        )
        assert _e_um_png_nao_vazio(caminho)
