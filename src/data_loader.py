"""
data_loader.py

Camada de acesso a dados do projeto de Valuation + VaR.
Fontes:
    - python-bcb (SGS do Bacen): Selic, CDI, IPCA
    - yfinance: preços históricos e Ibovespa
    - CVM Dados Abertos: DFP (anual) e ITR (trimestral) — Balanço, DRE, DFC
    - brapi.dev: resolução de ticker -> nome da empresa (ponte para o CD_CVM)

só busca e organiza dados brutos em DataFrames.
"""

import io
import os
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bcb import sgs
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(exist_ok=True)

CVM_BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS"
CVM_CADASTRO_URL = (
    "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"
)

BRAPI_TOKEN = os.environ.get("BRAPI_TOKEN")
BRAPI_BASE_URL = "https://brapi.dev/api"

# Cache local do mapeamento ticker -> CD_CVM já resolvido (evita chamar a brapi de novo)
TICKER_CVM_CACHE_PATH = DATA_DIR / "ticker_cvm_map.csv"


#Séries macro (Bacen SGS)
# ---------------------------------------------------------------------------
 
def get_selic(start: str, end: str) -> pd.Series:
    """Selic diária (% a.a.), código SGS 11."""
    return sgs.get({"selic": 11}, start=start, end=end)["selic"]
 
 
def get_cdi(start: str, end: str) -> pd.Series:
    """CDI diária (% a.a.), código SGS 12."""
    return sgs.get({"cdi": 12}, start=start, end=end)["cdi"]
 
 
def get_ipca(start: str, end: str) -> pd.Series:
    """IPCA mensal (variação %), código SGS 433."""
    return sgs.get({"ipca": 433}, start=start, end=end)["ipca"]
 
 
# ---------------------------------------------------------------------------
# Preços (yfinance)
# ---------------------------------------------------------------------------
 
def get_precos(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Preços diários ajustados de um ticker brasileiro."""
    serie = _get_close(f"{ticker}.SA", start, end)
    return serie.rename(ticker).to_frame()


def get_ibovespa(start: str, end: str) -> pd.Series:
    """Fechamento diário do Ibovespa, usado como proxy de mercado no CAPM."""
    return _get_close("^BVSP", start, end).rename("IBOV")
 
 
def get_carteira_precos(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Preços de vários ativos alinhados no mesmo DataFrame (para o VaR)."""
    series = [get_precos(t, start, end)[t] for t in tickers]
    df = pd.concat(series, axis=1).dropna()
    return df

def _get_close(ticker_yf: str, start: str, end: str) -> pd.Series:
    """
    Busca o fechamento ajustado de um ticker no yfinance, já pronto
    (com sufixo .SA ou índice tipo ^BVSP, conforme o caso).
    Levanta ValueError se não vier dado nenhum.
    """
    df = yf.Ticker(ticker_yf).history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {ticker_yf} — confira o ticker/período.")
    return df["Close"]
 
# ---------------------------------------------------------------------------
# Fundamentos (CVM Dados Abertos)
# ---------------------------------------------------------------------------
 
def _limpar_cnpj(cnpj: str) -> str:
    """Remove pontuação de um CNPJ, deixando só os 14 dígitos."""
    return re.sub(r"\D", "", cnpj or "")
 
 
def _get_cnpj_via_brapi(ticker: str) -> str:
    """
    Consulta a brapi.dev (módulo summaryProfile) para obter o CNPJ
    associado ao ticker. Requer BRAPI_TOKEN configurado.
    """
    if not BRAPI_TOKEN:
        raise RuntimeError(
            "BRAPI_TOKEN não configurado. Crie um arquivo .env na raiz do "
            "projeto (veja .env.example) com BRAPI_TOKEN=<seu_token>."
        )

    url = f"{BRAPI_BASE_URL}/quote/{ticker}"
    params = {"modules": "summaryProfile"}
    headers = {"Authorization": f"Bearer {BRAPI_TOKEN}"}
 
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
 
    resultados = payload.get("results", [])
    if not resultados:
        raise ValueError(f"brapi.dev não retornou dados para o ticker {ticker}.")
 
    cnpj = resultados[0].get("summaryProfile", {}).get("cnpj")
    if not cnpj:
        raise ValueError(f"Resposta da brapi para {ticker} não trouxe CNPJ: {resultados[0]}")
    return _limpar_cnpj(cnpj)
 
 
def _get_cvm_code(ticker: str) -> str:
    """
    Resolve o CD_CVM a partir do ticker, cruzando o CNPJ retornado pela
    brapi.dev com o cadastro de companhias abertas da CVM (CNPJ_CIA).
    Considera apenas empresas com situação 'ATIVO'. Resultado é cacheado
    em data/ticker_cvm_map.csv para evitar chamadas repetidas.
    """
    # 1. Checa cache local primeiro
    if TICKER_CVM_CACHE_PATH.exists():
        cache = pd.read_csv(TICKER_CVM_CACHE_PATH)
        match = cache[cache["ticker"] == ticker]
        if not match.empty:
            return str(match.iloc[0]["cd_cvm"])
    else:
        cache = pd.DataFrame(columns=["ticker", "cd_cvm", "cnpj"])
 
    # 2. Baixa/cacheia o cadastro CVM
    cad_path = DATA_DIR / "cad_cia_aberta.csv"
    if not cad_path.exists():
        resp = requests.get(CVM_CADASTRO_URL, timeout=30)
        resp.raise_for_status()
        cad_path.write_bytes(resp.content)
    cad = pd.read_csv(cad_path, sep=";", encoding="latin1")
 
    # 3. Filtra só empresas ativas e normaliza CNPJ pra comparação
    cad_ativas = cad[cad["SIT"] == "ATIVO"].copy()
    cad_ativas["_cnpj_limpo"] = cad_ativas["CNPJ_CIA"].apply(_limpar_cnpj)
 
    # 4. Resolve CNPJ via brapi e cruza com o cadastro
    cnpj_brapi = _get_cnpj_via_brapi(ticker)
    match = cad_ativas[cad_ativas["_cnpj_limpo"] == cnpj_brapi]
 
    if match.empty:
        raise ValueError(
            f"Nenhuma correspondência no cadastro CVM (empresas ativas) "
            f"para '{ticker}' (CNPJ brapi: '{cnpj_brapi}')."
        )
 
    cd_cvm = str(match.iloc[0]["CD_CVM"])
 
    # 5. Atualiza cache local
    cache = pd.concat(
        [cache, pd.DataFrame([{"ticker": ticker, "cd_cvm": cd_cvm, "cnpj": cnpj_brapi}])],
        ignore_index=True,
    )
    cache.to_csv(TICKER_CVM_CACHE_PATH, index=False)
 
    return cd_cvm
 
 
def _download_dfp_ano(ano: int) -> dict[str, pd.DataFrame]:
    """
    Baixa e extrai o ZIP anual de DFP da CVM, retornando um dict
    {nome_do_arquivo_csv: DataFrame}. Cacheia o ZIP em data/.
    """
    zip_path = DATA_DIR / f"dfp_cia_aberta_{ano}.zip"
    if not zip_path.exists():
        url = f"{CVM_BASE_URL}/dfp_cia_aberta_{ano}.zip"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
 
    dados = {}
    with zipfile.ZipFile(zip_path) as z:
        for nome_arquivo in z.namelist():
            if nome_arquivo.endswith(".csv"):
                with z.open(nome_arquivo) as f:
                    dados[nome_arquivo] = pd.read_csv(
                        io.TextIOWrapper(f, encoding="latin1"), sep=";"
                    )
    return dados
 
 
def get_dfp(ticker: str, ano: int, demonstrativo: str = "DRE", tipo: str = "con") -> pd.DataFrame:
    """
    Retorna um demonstrativo financeiro (DFP anual) de uma empresa.
 
    ticker: ex. 'JHSF3'
    ano: ano de referência do demonstrativo (ex. 2023)
    demonstrativo: 'BPA' (balanço ativo), 'BPP' (balanço passivo),
                   'DRE', 'DFC_MD' (fluxo de caixa método direto) etc.
    tipo: 'con' (consolidado) ou 'ind' (individual)
    """
    cd_cvm = _get_cvm_code(ticker)
    arquivos = _download_dfp_ano(ano)
 
    chave = f"dfp_cia_aberta_{demonstrativo}_{tipo}_{ano}.csv"
    if chave not in arquivos:
        disponiveis = [k for k in arquivos if demonstrativo in k]
        raise ValueError(f"Arquivo '{chave}' não encontrado. Disponíveis: {disponiveis}")
 
    df = arquivos[chave]
    return df[df["CD_CVM"] == int(cd_cvm)].copy()
 
 
if __name__ == "__main__":
    # Teste rápido manual (rodar localmente, requer internet)
    print(get_selic("2023-01-01", "2024-01-01").tail())
    print(get_precos("JHSF3", "2023-01-01", "2024-01-01").tail())
    print(get_dfp("JHSF3", 2023, "DRE").head())
 
