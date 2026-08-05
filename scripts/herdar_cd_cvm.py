"""
Segundo passo do mapeamento ticker -> CD_CVM.

warm_ticker_cache.py resolve via brapi.dev, mas a brapi só costuma expor
CNPJ/summaryProfile na classe "principal" de cada empresa (geralmente a ON,
sufixo 3). Classes irmãs do mesmo emissor (PN, units, etc.) têm o MESMO
CNPJ/CD_CVM na CVM, então não precisam de nova chamada de rede: basta
herdar o cd_cvm de qualquer ticker resolvido com o mesmo prefixo alfabético
(ex.: WLMM4 herda de WLMM3; ALUP4 e ALUP11 herdam de ALUP3).

Lê data/raw/ticker_cvm_map.csv (resolvidos) e data/raw/ticker_cvm_falhas.csv
(falhos), tenta herdar, e:
  - acrescenta os herdados em ticker_cvm_map.csv
  - reescreve ticker_cvm_falhas.csv só com o que continua sem solução
    (empresas em que NENHUMA classe resolveu — precisam de outra fonte)

Uso:
    python -m scripts.herdar_cd_cvm
"""

import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MAP_PATH = DATA_DIR / "ticker_cvm_map.csv"
FALHAS_PATH = DATA_DIR / "ticker_cvm_falhas.csv"


def _prefixo(ticker: str) -> str:
    """Prefixo alfabético do ticker (ex.: 'WLMM4' -> 'WLMM')."""
    return re.match(r"^([A-Z]+)", ticker).group(1)


if __name__ == "__main__":
    with open(MAP_PATH, newline="", encoding="utf-8") as f:
        resolvidos = list(csv.DictReader(f))

    with open(FALHAS_PATH, newline="", encoding="utf-8") as f:
        falhas = [row["ticker"] for row in csv.DictReader(f)]

    cd_cvm_por_prefixo = {}
    cnpj_por_prefixo = {}
    for row in resolvidos:
        prefixo = _prefixo(row["ticker"])
        cd_cvm_por_prefixo.setdefault(prefixo, row["cd_cvm"])
        cnpj_por_prefixo.setdefault(prefixo, row["cnpj"])

    herdados = []
    ainda_falhos = []
    for ticker in falhas:
        prefixo = _prefixo(ticker)
        if prefixo in cd_cvm_por_prefixo:
            herdados.append({
                "ticker": ticker,
                "cd_cvm": cd_cvm_por_prefixo[prefixo],
                "cnpj": cnpj_por_prefixo[prefixo],
            })
        else:
            ainda_falhos.append(ticker)

    print(f"{len(herdados)} tickers herdaram cd_cvm de uma classe irmã.")
    print(f"{len(ainda_falhos)} continuam sem solução (nenhuma classe resolveu).")

    if herdados:
        resolvidos.extend(herdados)
        resolvidos.sort(key=lambda r: r["ticker"])
        with open(MAP_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ticker", "cd_cvm", "cnpj"])
            writer.writeheader()
            writer.writerows(resolvidos)
        print(f"{MAP_PATH} atualizado ({len(resolvidos)} tickers no total).")

    with open(FALHAS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker"])
        writer.writerows([[t] for t in ainda_falhos])
    print(f"{FALHAS_PATH} reescrito só com as falhas reais restantes.")
