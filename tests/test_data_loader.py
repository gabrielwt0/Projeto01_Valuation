"""Testes de src/data_loader.py — só a parte pura (sem rede)."""

from src.data_loader import _limpar_cnpj


class TestLimparCnpj:
    def test_remove_pontuacao_padrao(self):
        assert _limpar_cnpj("12.345.678/0001-90") == "12345678000190"

    def test_ja_limpo_permanece_igual(self):
        assert _limpar_cnpj("12345678000190") == "12345678000190"

    def test_string_vazia(self):
        assert _limpar_cnpj("") == ""

    def test_none_vira_string_vazia(self):
        assert _limpar_cnpj(None) == ""
