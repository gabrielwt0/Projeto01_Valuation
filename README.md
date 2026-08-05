# valuation-br

Ferramenta de valuation (WACC + DCF) e análise de risco (VaR) para ações
brasileiras, construída do zero sobre dados abertos: CVM (fundamentos),
yfinance (preços) e python-bcb (Selic/CDI/IPCA via SGS do Bacen).

## Motivação

Calcular CAPM, WACC, projeção de FCFF e Valor em Risco (VaR) a partir dos
dados brutos, sem depender de números de beta ou fluxo de caixa já prontos
de terceiros — o objetivo é demonstrar domínio da matemática por trás de
cada fórmula.

## Estrutura

```
src/            código-fonte (data_loader, capm/wacc, dcf, var)
data/raw/       dados brutos baixados (não versionado)
data/processed/ dados tratados (não versionado)
notebooks/      exploração e prototipagem
tests/          testes das fórmulas (pytest)
reports/        relatórios gerados (markdown/gráficos)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fases do projeto

0. Setup do repositório
1. Camada de dados (CVM DFP/ITR, yfinance, python-bcb)
2. WACC do zero (beta via regressão, CAPM, custo de dívida)
3. Projeção de FCFF e DCF (Gordon Growth)
4. VaR (paramétrico, histórico, Monte Carlo)
5. Visualização e relatório
6. Testes
7. README com a documentação matemática (LaTeX)
