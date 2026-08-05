"""
Resolve o CD_CVM de todas as ações da B3 (tipo 'stock' na brapi.dev) e
grava o resultado em data/raw/ticker_cvm_map.csv, via _get_cvm_code
(que já cacheia sozinho, uma linha por ticker).

Depois de rodado uma vez, esse CSV pode ser versionado no git — o projeto
deixa de depender da brapi.dev/BRAPI_TOKEN para qualquer ticker que já
tenha sido resolvido, mesmo em uma clonagem nova do repositório.

Tickers com sufixo 'F' (mercado fracionário — o mesmo papel do lote padrão,
sem perfil próprio na brapi) são pulados de saída: nunca resolvem e só
duplicam o trabalho.

Tickers que falharem por outro motivo (404 na brapi, sem CNPJ no perfil,
sem correspondência no cadastro CVM — comum para BDRs, units ou empresas
sem registro ativo) são pulados e listados no final, sem interromper o
restante do lote. Erros de rede transitórios (timeout, conexão recusada)
são tentados de novo uma vez antes de contar como falha — já aconteceu de
a run inteira cair por causa de um único "No route to host".

O script é seguro para rodar de novo: _get_cvm_code cacheia cada ticker
resolvido em ticker_cvm_map.csv assim que resolve, então uma rerodada
pula tudo que já está no cache e só gasta chamadas de API com o que falta.

Uso:
    python -m scripts.warm_ticker_cache
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from src.data_loader import BRAPI_BASE_URL, BRAPI_TOKEN, _get_cvm_code

LIST_URL = f"{BRAPI_BASE_URL}/quote/list"

ERROS_TRANSITORIOS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)
ERROS_DE_TICKER = (requests.HTTPError, ValueError, RuntimeError)


def get_todos_tickers_b3() -> list[str]:
    """Lista todos os tickers do tipo 'stock' (ações) cadastrados na brapi,
    excluindo os do mercado fracionário (sufixo 'F')."""
    headers = {"Authorization": f"Bearer {BRAPI_TOKEN}"} if BRAPI_TOKEN else {}
    resp = requests.get(
        LIST_URL, headers=headers, params={"type": "stock", "limit": 5000}, timeout=30
    )
    resp.raise_for_status()
    tickers = (s["stock"] for s in resp.json()["stocks"])
    return sorted(t for t in tickers if not t.endswith("F"))


def _resolver_com_retry(ticker: str) -> str:
    """Resolve o CD_CVM de um ticker, tentando de novo uma vez se a falha
    for de rede (transitória) em vez de uma resposta de fato negativa."""
    try:
        return _get_cvm_code(ticker)
    except ERROS_TRANSITORIOS:
        time.sleep(2)
        return _get_cvm_code(ticker)


if __name__ == "__main__":
    tickers = get_todos_tickers_b3()
    print(f"{len(tickers)} tickers (excluindo mercado fracionário) encontrados na brapi.")

    falhas = []
    t0 = time.time()
    for i, ticker in enumerate(tickers, start=1):
        try:
            cd_cvm = _resolver_com_retry(ticker)
            print(f"[{i}/{len(tickers)}] {ticker} -> CD_CVM {cd_cvm}")
        except ERROS_TRANSITORIOS + ERROS_DE_TICKER as exc:
            print(f"[{i}/{len(tickers)}] {ticker} -> FALHOU ({exc})")
            falhas.append((ticker, str(exc)))

    print(f"\nConcluído em {time.time() - t0:.0f}s. "
          f"{len(tickers) - len(falhas)} resolvidos, {len(falhas)} falharam.")

    if falhas:
        falhas_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "ticker_cvm_falhas.csv"
        import csv
        with open(falhas_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker", "erro"])
            writer.writerows(falhas)
        print(f"Lista de falhas salva em {falhas_path}")
