# valuation-br

> 🚧 **Em desenvolvimento** — funcional, mas em evolução ativa (ajustes na camada de dados, WACC e DCF ainda em andamento).

Ferramenta de valuation (WACC + DCF) e análise de risco (VaR) para ações
brasileiras, construída do zero sobre dados abertos: CVM (fundamentos),
yfinance (preços) e python-bcb (Selic/CDI/IPCA via SGS do Bacen).

## Motivação

Calcular CAPM, WACC, projeção de FCFF e Valor em Risco (VaR) a partir dos
dados brutos, sem depender de números de beta ou fluxo de caixa já prontos
de terceiros — o objetivo é demonstrar domínio da matemática por trás de
cada fórmula.

## Metodologia

Documentação matemática de cada fórmula implementada, na ordem em que uma
usa o resultado da anterior (CAPM → WACC → FCFF/DCF → VaR). Notação
consistente com os nomes de variável do código — cada seção referencia a
função correspondente em `src/` para quem quiser ver a implementação.

### Log-retorno diário

Usado para beta/CAPM (`wacc.calcular_retornos_diarios`) por ser aditivo
no tempo, o que simplifica a regressão e a anualização geométrica:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

(Para VaR de carteira usa-se retorno SIMPLES em vez de log — ver seção
de VaR abaixo, a razão é outra: exatidão na ponderação por ativo.)

### Beta (regressão OLS)

Beta de um ativo = slope da regressão linear simples dos seus
log-retornos diários contra os do Ibovespa, proxy de mercado
(`wacc.calcular_beta`):

$$r_{ativo,t} = \alpha + \beta \cdot r_{ibov,t} + \varepsilon_t$$

### Prêmio de risco de mercado

Retorno anualizado do Ibovespa menos o CDI médio anualizado, na mesma
janela (`wacc.premio_de_risco`). Retorno do Ibovespa anualizado via
exponencial (consistente com log-retorno, não `(1+r)^n`):

$$R_m = e^{\bar{r}_{ibov} \times 252} - 1 \qquad \text{prêmio} = R_m - R_f$$

### Anualização de taxa diária (Selic/CDI)

A série retornada pelo Bacen (SGS) é a taxa efetiva DIÁRIA, não já
anualizada (ver ressalva em `data_loader.get_selic`) — composta por 252
dias úteis (`wacc.anualizar_taxa_diaria`):

$$i_{anual} = (1+i_{diaria})^{252} - 1$$

### CAPM — custo de capital próprio (Ke)

$$K_e = R_f + \beta \cdot (R_m - R_f)$$

### Custo de dívida líquido de IR (Kd)

Despesa financeira sobre dívida onerosa total, líquido do benefício
fiscal (`wacc.custo_divida`):

$$K_d = \frac{\text{Despesa financeira}}{\text{Dívida onerosa (CP+LP)}} \times (1 - t)$$

### WACC

Média ponderada de Ke e Kd pelos pesos de mercado da estrutura de
capital (`wacc.calcular_wacc`):

$$WACC = K_e \cdot \frac{E}{E+D} + K_d \cdot \frac{D}{E+D}$$

### FCFF (Free Cash Flow to Firm)

Fluxo de caixa livre para todos os provedores de capital, unlevered
porque parte do EBIT — por isso descontado pelo WACC, não pelo Ke
(`dcf.calcular_fcff`):

$$FCFF = EBIT \times (1-t) + D\&A - Capex - \Delta NWC$$

### Projeção do período explícito

Crescimento geométrico constante sobre o FCFF base, à taxa `g_explicito`
(`dcf.projetar_fcff`). `g_explicito` vem do CAGR de receita numa janela
curta (`dcf.calcular_cagr_receita`) — ver ressalva no código sobre por
que a janela é 3 anos, não 5 (sensibilidade a anos de pico/vale):

$$FCFF_t = FCFF_0 \times (1+g)^t, \quad t = 1, \dots, n \qquad g = \left(\frac{Receita_{ano}}{Receita_{ano-n}}\right)^{1/n} - 1$$

### IPCA médio anualizado (base do g de perpetuidade)

Composição geométrica real das variações mensais, não a média
aritmética composta depois — evita o viés de Jensen
(`dcf.calcular_ipca_medio_anual`):

$$IPCA_{anual} = \left(\prod_{i=1}^{N} \left(1+\frac{ipca_i}{100}\right)\right)^{12/N} - 1$$

### Valor terminal (Gordon Growth)

Valor de uma perpetuidade crescente a partir do último ano explícito
(`dcf.valor_terminal_gordon`):

$$VT_n = \frac{FCFF_n \times (1+g_\infty)}{WACC - g_\infty} \qquad (WACC > g_\infty)$$

### Enterprise Value

Soma do valor presente dos fluxos explícitos com o valor presente do
valor terminal, ambos descontados pelo WACC (`dcf.calcular_enterprise_value`):

$$EV = \sum_{t=1}^{n} \frac{FCFF_t}{(1+WACC)^t} + \frac{VT_n}{(1+WACC)^n}$$

### Equity Value e preço justo por ação

$$Equity\ Value = EV - Dívida\ Líquida \qquad P_{justo} = \frac{Equity\ Value}{N^{o}\ de\ ações}$$

### VaR paramétrico (variância-covariância)

Assume retornos ~ Normal($\mu$, $\sigma^2$); escala para o horizonte $h$
pela regra da raiz do tempo (`var.calcular_var_parametrico`):

$$VaR = -\left(\mu_h + z_\alpha \cdot \sigma_h\right) \times V, \qquad \mu_h = \mu h,\ \ \sigma_h = \sigma\sqrt{h}$$

onde $z_\alpha$ é o quantil da normal padrão para o nível de confiança
($z_{0{,}95} \approx -1{,}645$) e $V$ é o valor exposto.

### VaR histórico

Percentil empírico dos retornos observados, sem assumir distribuição
(`var.calcular_var_historico`):

$$VaR = -\text{percentil}_{(1-\alpha)\times 100}(r) \times \sqrt{h} \times V$$

### VaR Monte Carlo

Simula $M$ cenários $r_i \sim \mathcal{N}(\mu_h, \sigma_h^2)$ e toma o
percentil empírico dos cenários simulados (`var.calcular_var_monte_carlo`)
— com Normal, converge para o VaR paramétrico quando $M \to \infty$; a
vantagem real aparece ao trocar a distribuição simulada por algo com
caudas mais gordas.

## Estrutura

```
app.py               UI interativa (Streamlit)
src/                 código-fonte (data_loader, wacc, dcf, var, visualizacao)
scripts/             scripts de orquestração (gerar_relatorio, warm_ticker_cache)
data/raw/            dados brutos baixados (não versionado)
data/processed/      dados tratados, incl. cache parquet do DFP (não versionado)
notebooks/           exploração e prototipagem
tests/               testes das fórmulas (pytest — ver pytest.ini)
reports/             relatórios gerados (markdown/gráficos)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## UI

```bash
streamlit run app.py
```

Interface interativa: escolha ticker/ano na barra lateral e veja WACC, DCF
e VaR calculados na hora, com os mesmos gráficos do relatório em markdown
(`scripts/gerar_relatorio.py`) — os dois usam o mesmo pipeline
(`gerar_wacc`/`gerar_dcf`/`gerar_var`), então não há lógica duplicada.

## Performance

`_download_dfp_ano` (`src/data_loader.py`) baixa o ZIP anual da CVM uma vez
e cacheia em duas camadas para não reparsear os ~20 CSVs a cada chamada
(gargalo real: o heatmap de sensibilidade do DCF e o relatório completo
chamam `get_dfp`/`get_composicao_capital` dezenas de vezes):

- **Memória** (`_CACHE_DFP`): instantâneo, mas só dura o processo.
- **Disco** (`data/processed/dfp_parsed/{ano}/*.parquet`): sobrevive entre
  execuções — ~15x mais rápido que reparsear o CSV do zip.

Resultado: o pipeline completo (WACC+DCF+VaR+gráficos) de um ticker caiu de
minutos para segundos depois da primeira chamada de cada ano.

## Fases do projeto

- [x] 0. Setup do repositório
- [x] 1. Camada de dados (CVM DFP/ITR, yfinance, python-bcb)
- [x] 2. WACC do zero (beta via regressão, CAPM, custo de dívida)
- [x] 3. Projeção de FCFF e DCF (Gordon Growth)
- [x] 4. VaR (paramétrico, histórico, Monte Carlo)
- [x] 5. Visualização e relatório
- [x] 6. Testes
- [x] 7. README com a documentação matemática (LaTeX)
