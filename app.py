"""
UI (Streamlit) do valuation — mesmo pipeline de scripts/gerar_relatorio.py
(WACC + DCF + VaR), só que interativo em vez de gerar reports/{ticker}.md.
Nenhuma lógica de cálculo mora aqui: só chama gerar_wacc/gerar_dcf/gerar_var
e as funções de src/visualizacao.py (que, sem `caminho`, devolvem a Figure
em vez de salvar PNG — evita gravar arquivo em disco a cada interação).

Uso:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from scripts.gerar_relatorio import N_ANOS_HISTORICO_GRAFICO, gerar_dcf, gerar_var, gerar_wacc
from src.data_loader import get_precos
from src.dcf import (
    JANELA_CAGR_RECEITA_ANOS,
    JANELA_IPCA_ANOS,
    N_ANOS_EXPLICITO_PADRAO,
    calcular_enterprise_value,
    calcular_equity_value,
    calcular_fcff,
    calcular_receita,
    valor_por_acao,
)
from src.visualizacao import (
    grafico_estrutura_capital_wacc,
    grafico_projecao_fcff,
    grafico_receita_historica,
    grafico_sensibilidade_dcf,
    grafico_var_distribuicao,
)
from src.wacc import JANELA_HISTORICA_ANOS

st.set_page_config(page_title="Valuation BR", layout="wide")


@st.cache_data(show_spinner=False)
def _computar(ticker: str, ano: int):
    hoje = pd.Timestamp.today().normalize()
    inicio_precos = (hoje - pd.DateOffset(years=JANELA_HISTORICA_ANOS)).strftime("%Y-%m-%d")
    fim = hoje.strftime("%Y-%m-%d")

    wacc_info = gerar_wacc(ticker, ano, inicio_precos, fim)
    dcf_info = gerar_dcf(ticker, ano, wacc_info["wacc"])

    inicio_var = (hoje - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    var_info = gerar_var(ticker, inicio_var, fim)

    preco_atual = get_precos(ticker, (hoje - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), fim)[ticker].iloc[-1]

    return wacc_info, dcf_info, var_info, preco_atual


with st.sidebar:
    st.header("Valuation BR")
    ticker = st.text_input("Ticker", value="JHSF3").strip().upper()
    ano = st.number_input("Ano-base (DFP)", min_value=2010, max_value=pd.Timestamp.today().year - 1, value=2023)
    calcular = st.button("Calcular", type="primary", use_container_width=True)
    st.caption(
        "WACC + DCF (Gordon Growth) + VaR calculados do zero a partir de "
        "CVM/yfinance/Bacen — ver README para a metodologia completa."
    )

if not calcular and "ultimo_resultado" not in st.session_state:
    st.info("Escolha um ticker e ano na barra lateral e clique em Calcular.")
    st.stop()

if calcular:
    try:
        with st.spinner(f"Calculando valuation de {ticker} ({ano})..."):
            resultado = _computar(ticker, int(ano))
        st.session_state["ultimo_resultado"] = (ticker, ano, resultado)
    except Exception as e:
        st.error(f"Erro ao calcular {ticker}/{ano}: {e}")
        st.stop()

ticker, ano, (wacc_info, dcf_info, var_info, preco_atual) = st.session_state["ultimo_resultado"]

st.title(f"Valuation — {ticker} ({ano})")

st.header("1. Custo de capital (WACC)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Beta", f"{wacc_info['beta']:.3f}", help=f"R² = {wacc_info['r_quadrado']:.2f}, vs. Ibovespa em {JANELA_HISTORICA_ANOS}a")
col2.metric("Ke (CAPM)", f"{wacc_info['ke']:.2%}")
col3.metric("Kd líquido de IR", f"{wacc_info['kd_liquido']:.2%}")
col4.metric("WACC", f"{wacc_info['wacc']:.2%}")

fig = grafico_estrutura_capital_wacc(
    wacc_info["valor_equity"], wacc_info["divida_contabil"], wacc_info["ke"], wacc_info["kd_liquido"], wacc_info["wacc"],
)
st.pyplot(fig)

st.header("2. Receita histórica")
anos_historico = list(range(ano - N_ANOS_HISTORICO_GRAFICO + 1, ano + 1))
receitas = [calcular_receita(ticker, a) for a in anos_historico]
st.pyplot(grafico_receita_historica(anos_historico, receitas))

st.header("3. FCFF e DCF (Gordon Growth)")
col1, col2, col3 = st.columns(3)
col1.metric(f"FCFF base ({ano})", f"R$ {dcf_info['fcff_base']:,.0f} mil")
col2.metric(f"g explícito (CAGR {JANELA_CAGR_RECEITA_ANOS}a)", f"{dcf_info['g_explicito']:.2%}")
col3.metric(f"g perpetuidade (IPCA {JANELA_IPCA_ANOS}a)", f"{dcf_info['g_perpetuidade']:.2%}")

col1, col2, col3 = st.columns(3)
col1.metric("Enterprise Value", f"R$ {dcf_info['enterprise_value']:,.0f} mil")
col2.metric("Dívida líquida", f"R$ {dcf_info['divida_liquida']:,.0f} mil")
col3.metric("Equity Value", f"R$ {dcf_info['equity_value']:,.0f} mil")

anos_fcff_historico = list(range(ano - 2, ano + 1))
fcff_historico = [calcular_fcff(ticker, a) for a in anos_fcff_historico]
anos_projetado = list(range(ano + 1, ano + 1 + N_ANOS_EXPLICITO_PADRAO))
st.pyplot(grafico_projecao_fcff(anos_fcff_historico, fcff_historico, anos_projetado, dcf_info["fluxos_projetados"]))
if any(f < 0 for f in fcff_historico):
    st.caption(
        "⚠️ FCFF foi negativo em algum ano recente — o DCF projeta a partir de um único "
        "ano-base potencialmente anômalo (ver ressalva em `calcular_fcff`, src/dcf.py)."
    )


def _preco_justo_para(wacc: float, g_perpetuidade: float) -> float:
    dcf = calcular_enterprise_value(dcf_info["fcff_base"], wacc, dcf_info["g_explicito"], g_perpetuidade)
    equity_value = calcular_equity_value(dcf["enterprise_value"], dcf_info["divida_liquida"])
    return valor_por_acao(ticker, ano, equity_value)


st.pyplot(grafico_sensibilidade_dcf(dcf_info["fcff_base"], wacc_info["wacc"], dcf_info["g_perpetuidade"], _preco_justo_para))

st.header("4. Preço justo vs. mercado")
upside = dcf_info["preco_justo"] / preco_atual - 1
col1, col2, col3 = st.columns(3)
col1.metric("Preço justo (DCF)", f"R$ {dcf_info['preco_justo']:.2f}")
col2.metric("Preço de mercado", f"R$ {preco_atual:.2f}")
col3.metric("Upside/downside", f"{upside:+.1%}")
st.caption(
    "Premissas (`g_explicito`, `g_perpetuidade`) são um ponto de partida, não uma "
    "recomendação — ver ressalvas em `src/dcf.py` sobre sensibilidade da janela de CAGR."
)

st.header("5. Valor em Risco (VaR)")
st.pyplot(
    grafico_var_distribuicao(
        var_info["retornos"], var_info[0.95]["parametrico"], var_info[0.95]["historico"],
        var_info[0.95]["monte_carlo"], 0.95,
    )
)
tabela_var = pd.DataFrame(
    {
        "Confiança": ["95%", "99%"],
        "Paramétrico": [f"{var_info[c]['parametrico']:.2%}" for c in (0.95, 0.99)],
        "Histórico": [f"{var_info[c]['historico']:.2%}" for c in (0.95, 0.99)],
        "Monte Carlo": [f"{var_info[c]['monte_carlo']:.2%}" for c in (0.95, 0.99)],
    }
).set_index("Confiança")
st.dataframe(tabela_var, use_container_width=True)
st.caption("VaR expresso como % do valor exposto, horizonte de 1 dia.")
